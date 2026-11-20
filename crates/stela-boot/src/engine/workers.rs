//! Compile workers: host processors that run no PE
//! (`stela_boot_core::compile_workers`). They take the GPUs' deferred
//! compile jobs and the GPUs' Linux work a PE whose guest waits would
//! otherwise run (with `nvk`), and prepare blocks ahead of execution
//! (`translations::prepare_ahead`) when no GPU work is ready.
//!
//! The loader reserves them past the PEs' executors and the APs' startup
//! starts them like a PE's: each has its own Host and joins the shared TLB
//! shootdowns, so the jobs it runs may map and fault as on any processor.
//! Each then loops on its executor: it answers shootdowns, takes a GPU job
//! at once (`gpu::gate::Taker::Worker`) and runs its slices back to back,
//! prepares blocks ahead when no GPU work is ready, reading their pages
//! through its own Host as a waiting PE does, and sleeps until its doorbell
//! while nothing waits. A guest call that waits for a GPU job rings the
//! workers (`gate::job_waiting`), and a PE's processor leaves a job to them
//! for `WORKER_GRACE_MILLIS` (`Taker::with_workers`), so the PEs keep their
//! cores for their guests and a job still runs if every worker is held. A
//! store that queues blocks to prepare ahead rings one sleeping worker
//! ([`nudge`]), and none while every worker is awake.
//!
//! A worker has no slot in the guest's shared clock: it never holds guest
//! time, however long a job runs.
use crate::{apic::Apic, engine::host, native_smp::executor::Processor, output};
use alloc::boxed::Box;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use stela_boot_core::compile_workers::MOST;
use stela_engine::machine::Host as _;

/// Each started worker's APIC ID, the first [`COUNT`] of them.
static APICS: [AtomicU32; MOST] = [const { AtomicU32::new(u32::MAX) }; MOST];
/// Workers started.
static COUNT: AtomicUsize = AtomicUsize::new(0);
/// The workers asleep, or about to be, by their index in [`APICS`]: those
/// [`nudge`] may ring.
static SLEEPING: AtomicU32 = AtomicU32::new(0);

/// Spins a worker makes while a GPU job waits that another processor holds,
/// before it sleeps: about the length of a slice on a 3 GHz core, so that it
/// takes the job back as that slice ends rather than at the next doorbell.
#[cfg(feature = "nvk")]
const SPINS_WHILE_HELD: u32 = 1 << 16;
/// Spin rounds of [`SPINS_WHILE_HELD`] a worker makes in a row before it
/// sleeps until a doorbell: a few dozen slices.
#[cfg(feature = "nvk")]
const HELD_ROUNDS: u32 = 64;

/// Whether any compile worker runs.
#[cfg_attr(not(feature = "nvk"), allow(dead_code))]
pub(in crate::engine) fn present() -> bool {
    COUNT.load(Ordering::Acquire) != 0
}

/// Hands each of `workers`, started executors that run no PE, the worker
/// loop for the rest of the boot, through `sender` with a TSC of `tsc_hz`.
/// A worker whose mailbox refuses stays idle in its executor's loop, and
/// the PEs' processors take its jobs.
pub(in crate::engine) fn start(workers: &[Processor], sender: &Apic, tsc_hz: u64) {
    for processor in workers.iter().take(MOST) {
        let Ok(apic) = u32::try_from(processor.apic_id()) else {
            continue;
        };
        let index = COUNT.load(Ordering::Relaxed);
        match processor.spawn(Box::new(move |owner| run(owner, index)), sender, tsc_hz) {
            Ok(()) => {
                APICS[index].store(apic, Ordering::Relaxed);
                COUNT.store(index + 1, Ordering::Release);
                output::arguments(format_args!(
                    "NATIVE COMPILE WORKER slot={} apic={apic:#x}: takes the GPUs' compile jobs \
                     and prepares blocks ahead\n",
                    processor.slot()
                ));
            }
            Err(error) => output::arguments(format_args!(
                "NATIVE COMPILE WORKER slot={} apic={apic:#x} refused its loop: {error}\n",
                processor.slot()
            )),
        }
    }
}

/// Rings `apics`' doorbells. Safe with interrupts on: the rings write this
/// processor's ICR with them off.
fn ring(apics: impl Iterator<Item = u32>) {
    let flags: u64;
    // SAFETY: CPL0 IF save, restored below on this processor.
    unsafe { core::arch::asm!("pushfq", "pop {}", "cli", out(reg) flags) };
    for apic in apics {
        host::ring(apic);
    }
    // SAFETY: the save above, on this processor.
    unsafe { core::arch::asm!("push {}", "popfq", in(reg) flags) };
}

/// Rings every worker's doorbell: a GPU job waits.
#[cfg(feature = "nvk")]
pub(in crate::engine) fn wake() {
    let count = COUNT.load(Ordering::Acquire);
    ring(
        APICS[..count]
            .iter()
            .map(|apic| apic.load(Ordering::Relaxed)),
    );
}

/// Rings one sleeping worker's doorbell, if any sleeps: a block to
/// prepare ahead waits. The worker marks itself sleeping before it looks at
/// the queue a last time, and the store looks at the marks after it queued,
/// so one of the two sees the other.
pub(in crate::engine) fn nudge() {
    core::sync::atomic::fence(Ordering::SeqCst);
    let sleeping = SLEEPING.load(Ordering::Relaxed);
    if sleeping == 0 {
        return;
    }
    let bit = 1 << sleeping.trailing_zeros();
    if SLEEPING.fetch_and(!bit, Ordering::AcqRel) & bit != 0 {
        let index = bit.trailing_zeros() as usize;
        ring(core::iter::once(APICS[index].load(Ordering::Relaxed)));
    }
}

/// A worker's loop, on its executor with interrupts masked, for the rest of
/// the boot: its Host is its executor's own, which the jobs reach through
/// `host`'s per-processor state. Linux's ready work comes first, as a frame
/// waits on it, then a GPU compile job, then blocks to prepare ahead until
/// a GPU job waits; with none, it sleeps.
fn run(owner: &mut host::HostOwner, index: usize) {
    let bit = 1u32 << index;
    #[cfg(feature = "nvk")]
    let gpu_waits = super::gpu::gate::jobs_pending;
    #[cfg(not(feature = "nvk"))]
    let gpu_waits = || false;
    // Spin rounds left while work another processor holds keeps waiting: a
    // worker whose wait outlasts them sleeps until a doorbell, as a PE that
    // waits for a GPU job or its Linux work rings the workers.
    #[cfg(feature = "nvk")]
    let mut spins_left = HELD_ROUNDS;
    loop {
        crate::native_mmio::shootdown::service();
        #[cfg(feature = "nvk")]
        {
            // The GPUs' Linux work, in the boot PE's place, within the
            // waiting pump's budget: a busy desktop's took about a third
            // of a core.
            super::gpu::pump_waiting_pe(&|| false);
            if super::gpu::gate::run_job(super::gpu::gate::Taker::Worker) {
                spins_left = HELD_ROUNDS;
                continue;
            }
        }
        if super::translations::ahead_pending() {
            // The guest's pages through this processor's own Host, as a
            // waiting PE reads them through its machine's, and at least one
            // job a call: a GPU job another processor holds leaves the
            // queue to drain rather than to spin on.
            let mut host = owner.host();
            let first = core::cell::Cell::new(true);
            super::translations::prepare_ahead(
                &mut |frame, bytes| host.read_ram(frame, bytes),
                &|| !first.replace(false) && gpu_waits(),
            );
            #[cfg(feature = "nvk")]
            {
                spins_left = HELD_ROUNDS;
            }
            continue;
        }
        // Work another processor holds (the pump's turn, a job's slice):
        // take it back as that ends, answering shootdowns meanwhile, for a
        // bounded time. Linux's ready work that no PE waits for is the boot
        // CPU's, and spinning on it for good kept every worker's core busy.
        #[cfg(feature = "nvk")]
        if spins_left != 0 && (super::gpu::linux_ready() || super::gpu::gate::jobs_pending()) {
            spins_left -= 1;
            for _ in 0..SPINS_WHILE_HELD {
                core::hint::spin_loop();
            }
            continue;
        }
        #[cfg(feature = "nvk")]
        {
            spins_left = HELD_ROUNDS;
        }
        SLEEPING.fetch_or(bit, Ordering::SeqCst);
        if super::translations::ahead_pending() {
            SLEEPING.fetch_and(!bit, Ordering::Relaxed);
            continue;
        }
        // SAFETY: interrupts are masked on this executor and its IDT takes
        // the doorbell, which ends the wait even when already pending.
        unsafe { crate::native_idle::wait(|| None) };
        SLEEPING.fetch_and(!bit, Ordering::Relaxed);
    }
}
