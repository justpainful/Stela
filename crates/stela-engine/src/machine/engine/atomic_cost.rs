// SPDX-License-Identifier: GPL-2.0-or-later
//! What atomics, ordered accesses, barriers and exclusives cost in native
//! code, run by hand:
//!
//!     cargo test -p stela-engine --lib atomic_cost -- --ignored --nocapture
//!
//! Each case is a system-mode block compiled as a parallel PE's engine
//! compiles it on this host, run on host memory by one thread, then by 2
//! and 4 at once on the same line. A block holds 16 copies of the case's
//! instructions; `ATOMIC COST` lines give nanoseconds per copy over a block
//! of 16 ADDs (the entry's cost cancels). An exclusive ends its block, so
//! an LDXR/ADD/STXR iteration is three blocks: its line gives the three
//! blocks' time less three one-ADD blocks', and the transitions are the
//! chain's. With several threads the time is wall time per thread.
extern crate std;

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use stela_compiler::{
    GuestAddress, GuestState, TargetCpu,
    block::{BlockProfile, CompilationPlan, ExitKind, Features},
    x86_64::{MemoryChecks, NativeSystem, RangeCheck, block_entry, pinned},
};

const PC: u64 = 0x4000;
const COPIES: usize = 16;
const RUNS: u64 = 1 << 16;
const ADD_X2: u32 = 0x9100_0442; // add x2, x2, #1

/// Executable copies of native code, unmapped on drop.
struct Mapped {
    page: *mut core::ffi::c_void,
    bytes: usize,
}

// SAFETY: read-only executable memory, run by several threads at once.
unsafe impl Sync for Mapped {}

unsafe extern "C" {
    fn mmap(
        address: *mut core::ffi::c_void,
        bytes: usize,
        protection: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut core::ffi::c_void;
    fn mprotect(address: *mut core::ffi::c_void, bytes: usize, protection: i32) -> i32;
    fn munmap(address: *mut core::ffi::c_void, bytes: usize) -> i32;
}

impl Mapped {
    fn new(body: &[u8]) -> Self {
        let code = pinned::entered(body);
        let bytes = code.len().next_multiple_of(4096);
        // SAFETY: private anonymous memory, RX after the copy.
        unsafe {
            let page = mmap(core::ptr::null_mut(), bytes, 3, 0x22, -1, 0);
            assert_ne!(page as usize, usize::MAX);
            core::ptr::copy_nonoverlapping(code.as_ptr(), page.cast(), code.len());
            assert_eq!(mprotect(page, bytes, 5), 0);
            Self { page, bytes }
        }
    }

    /// The exit's PC and kind.
    fn run(&self, state: &mut GuestState) -> (u64, u64) {
        // SAFETY: block ABI code touching only `state`, the stack and the
        // host memory the state's registers address.
        let exit = unsafe { block_entry::enter(self.page.cast(), state, 0, 0) };
        (exit as u64, (exit >> 64) as u64)
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: mapped by `new`, no longer running.
        unsafe { assert_eq!(munmap(self.page, self.bytes), 0) };
    }
}

/// `words` as one block, as a parallel PE compiles it at EL1 with SCTLR.SA
/// set on a four-level host.
fn block(words: &[u32]) -> Mapped {
    let source: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let checks =
        MemoryChecks::system(true, false, true, false).with_range(RangeCheck::four_level(4));
    let options = stela_compiler::Options::new(
        BlockProfile::new(Features::SYSTEM_MODE),
        TargetCpu::detect(),
    )
    .with_memory_checks(checks)
    .with_native_system(NativeSystem {
        exclusive: true,
        ..NativeSystem::NONE
    });
    let block = CompilationPlan::new(&source, GuestAddress(PC), &options)
        .unwrap()
        .compile()
        .unwrap();
    assert_eq!(block.source_bytes, source.len(), "one block");
    Mapped::new(&block.native.x86_64_code)
}

/// Two 64-byte lines: X0 addresses the first, X1 the second.
#[repr(C, align(64))]
struct Lines([u64; 16]);

fn state(lines: &Lines) -> GuestState {
    let mut state = GuestState::default();
    state.x[0] = lines.0.as_ptr() as u64;
    state.x[1] = lines.0[8..].as_ptr() as u64;
    state
}

/// Wall nanoseconds per run of `step` on each of `threads` threads at once,
/// all on `lines`.
fn timed(threads: usize, lines: &Lines, step: &(dyn Fn(&mut GuestState) + Sync)) -> f64 {
    let start = AtomicBool::new(false);
    let ns = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let start = &start;
                scope.spawn(move || {
                    let mut state = state(lines);
                    while !start.load(Ordering::Acquire) {
                        core::hint::spin_loop();
                    }
                    for _ in 0..RUNS / 8 {
                        step(&mut state);
                    }
                    let started = Instant::now();
                    for _ in 0..RUNS {
                        step(&mut state);
                    }
                    started.elapsed().as_nanos() as f64
                })
            })
            .collect();
        start.store(true, Ordering::Release);
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .fold(0.0, f64::max)
    });
    ns / RUNS as f64
}

fn report(what: &str, ns: [f64; 3], base: [f64; 3], per: f64) {
    std::println!(
        "ATOMIC COST {what:<34} 1 PE {:7.2}  2 PEs {:7.2}  4 PEs {:7.2} ns",
        (ns[0] - base[0]) / per,
        (ns[1] - base[1]) / per,
        (ns[2] - base[2]) / per,
    );
}

/// The best of five timings on 1, 2 and 4 threads: the host schedules each
/// timing's threads on its P and E cores as it sees fit.
fn on_each(lines: &Lines, step: &(dyn Fn(&mut GuestState) + Sync)) -> [f64; 3] {
    [1, 2, 4].map(|threads| {
        (0..5)
            .map(|_| timed(threads, lines, step))
            .fold(f64::MAX, f64::min)
    })
}

/// 16 copies of `words`, one block.
fn copies(words: &[u32]) -> Vec<u32> {
    words
        .iter()
        .copied()
        .cycle()
        .take(COPIES * words.len())
        .collect()
}

#[test]
#[ignore = "manual: opt-in"]
fn atomic_costs() {
    let lines = Lines([0; 16]);
    let adds = block(&[ADD_X2; COPIES]);
    let base = on_each(&lines, &|state| {
        adds.run(state);
    });
    std::println!(
        "ATOMIC COST base (16 ADDs a block) {:.2} {:.2} {:.2} ns a run",
        base[0],
        base[1],
        base[2]
    );
    let cases: &[(&str, &[u32])] = &[
        ("LDR X2,[X0]", &[0xf940_0002]),
        ("STR X2,[X0]", &[0xf900_0002]),
        ("LDAR X2,[X0]", &[0xc8df_fc02]),
        ("LDAPR X2,[X0]", &[0xf8bf_c002]),
        ("STLR X2,[X0]", &[0xc89f_fc02]),
        ("STLR W2,[X0]", &[0x889f_fc02]),
        ("STLRB W2,[X0]", &[0x089f_fc02]),
        ("STLR X2,[X0]; LDAR X3,[X1]", &[0xc89f_fc02, 0xc8df_fc23]),
        ("STLR X2,[X0]; LDAPR X3,[X1]", &[0xc89f_fc02, 0xf8bf_c023]),
        ("STR X2,[X0]; LDR X3,[X1]", &[0xf900_0002, 0xf940_0023]),
        ("DMB ISH", &[0xd503_3bbf]),
        ("DMB ISHLD", &[0xd503_39bf]),
        ("DMB ISHST", &[0xd503_3abf]),
        ("STR X2,[X0]; DMB ISH", &[0xf900_0002, 0xd503_3bbf]),
        ("LDADD X1,X2,[X0]", &[0xf821_0002]),
        ("LDADDAL X1,X2,[X0]", &[0xf8e1_0002]),
        ("LDADDL W1,W2,[X0]", &[0xb861_0002]),
        ("STADD X1,[X0]", &[0xf821_001f]),
        ("SWPAL X1,X2,[X0]", &[0xf8e1_8002]),
        ("SWPL WZR,WZR,[X0]", &[0xb87f_801f]),
        ("CAS X1,X2,[X0]", &[0xc8a1_7c02]),
        ("CASAL X1,X2,[X0]", &[0xc8e1_fc02]),
        ("LDSETAL X1,X2,[X0]", &[0xf8e1_3002]),
        ("LDCLRL W1,W2,[X0]", &[0xb861_1002]),
        ("LDSMAX X1,X2,[X0]", &[0xf821_4002]),
        ("CASPAL X2,X3,X4,X5,[X0]", &[0x4862_fc04]),
    ];
    for &(what, words) in cases {
        let code = block(&copies(words));
        let ns = on_each(&lines, &|state| {
            code.run(state);
        });
        report(what, ns, base, (COPIES) as f64);
    }
    // LDXR, ADD and STXR, each its own block, until the store passes.
    let ldxr = block(&[0xc85f_7c02]); // ldxr x2, [x0]
    let ldaxr = block(&[0xc85f_fc02]); // ldaxr x2, [x0]
    let add = block(&[ADD_X2]);
    let stxr = block(&[0xc803_7c02]); // stxr w3, x2, [x0]
    let stlxr = block(&[0xc803_fc02]); // stlxr w3, x2, [x0]
    let one = on_each(&lines, &|state| {
        add.run(state);
    });
    for (what, load, store) in [
        ("LDXR/ADD/STXR iteration", &ldxr, &stxr),
        ("LDAXR/ADD/STLXR iteration", &ldaxr, &stlxr),
    ] {
        let ns = on_each(&lines, &|state| {
            loop {
                assert_eq!(load.run(state).1, ExitKind::Fallthrough as u64);
                add.run(state);
                assert_eq!(store.run(state).1, ExitKind::Fallthrough as u64);
                if state.x[3] == 0 {
                    break;
                }
            }
        });
        report(what, ns, one.map(|one| 3.0 * one), 1.0);
    }
}
