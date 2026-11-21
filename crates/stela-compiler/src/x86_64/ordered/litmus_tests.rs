// SPDX-License-Identifier: GPL-2.0-or-later
//! Arm's store-buffering litmus test (SB) in native code on two host
//! threads: each PE stores 1 to its own word, then loads the other's, in
//! system-mode blocks as the engine compiles them. Where a full barrier
//! orders the store before the load, Arm forbids both loads reading 0,
//! which x86's store buffer gives a plain store and load.
use crate::{
    GuestAddress, GuestState, TargetCpu,
    block::{BlockProfile, CompilationPlan, Features},
    x86_64::{MemoryChecks, NativeSystem, RangeCheck, native, pinned},
};
use alloc::{vec, vec::Vec};
use core::ffi::c_void;
use core::sync::atomic::{
    AtomicU64,
    Ordering::{Acquire, Relaxed, Release},
};

const PC: u64 = 0x4000;
const ROUNDS: u64 = 40_000;
const STR_W2_X0: u32 = 0xb900_0002;
const LDR_W3_X1: u32 = 0xb940_0023;
const DMB_ISH: u32 = 0xd503_3bbf;
const STLR_W2_X0: u32 = 0x889f_fc02;
const LDAR_W3_X1: u32 = 0x88df_fc23;
const LDAPR_W3_X1: u32 = 0xb8bf_c023;
const LDAXR_W3_X1: u32 = 0x885f_fc23;
const STLRB_W2_X0: u32 = 0x089f_fc02;
const LDARB_W3_X1: u32 = 0x08df_fc23;
const SWPL_W2_WZR_X0: u32 = 0xb862_801f;

unsafe extern "C" {
    fn mmap(
        address: *mut c_void,
        bytes: usize,
        protection: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn mprotect(address: *mut c_void, bytes: usize, protection: i32) -> i32;
    fn munmap(address: *mut c_void, bytes: usize) -> i32;
}

/// One block of `words`, executable, unmapped on drop.
struct Mapped {
    page: *mut c_void,
    bytes: usize,
}

// SAFETY: read-only executable memory once mapped.
unsafe impl Sync for Mapped {}

impl Mapped {
    fn new(words: &[u32]) -> Self {
        let source: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let checks =
            MemoryChecks::system(true, false, true, false).with_range(RangeCheck::FIVE_LEVEL);
        let options = crate::Options::new(
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
        assert_eq!(block.source_bytes, source.len(), "{words:08x?}: one block");
        let code = pinned::entered(&block.native.x86_64_code);
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

    fn run(&self, state: &mut GuestState) {
        // SAFETY: block ABI code touching only `state`, the stack and the
        // two words its registers address.
        unsafe { native::enter(self.page.cast(), state, 0, 0) };
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: mapped by `new`, no longer running.
        unsafe { assert_eq!(munmap(self.page, self.bytes), 0) };
    }
}

#[repr(C, align(64))]
struct Line(AtomicU64);

/// Wait until `word` holds `value`, yielding now and then on a busy host.
fn wait(word: &AtomicU64, value: u64) {
    let mut spins = 0u32;
    while word.load(Acquire) != value {
        spins += 1;
        if spins.is_multiple_of(4096) {
            std::thread::yield_now();
        } else {
            core::hint::spin_loop();
        }
    }
}

/// The rounds of SB, each PE running the blocks of `program` in turn
/// (X0 its word, X1 the other's, W2 1, the load into W3), in which both
/// loads read 0.
fn both_zero(program: &[Vec<u32>]) -> u64 {
    let blocks: Vec<Mapped> = program.iter().map(|words| Mapped::new(words)).collect();
    let words = [Line(AtomicU64::new(0)), Line(AtomicU64::new(0))];
    let (ready, go, done, second_read) = (
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    );
    let pe = |index: usize| {
        let mut state = GuestState::default();
        state.x[0] = words[index].0.as_ptr() as u64;
        state.x[1] = words[1 - index].0.as_ptr() as u64;
        state
    };
    let run = |state: &mut GuestState| {
        state.x[2] = 1;
        state.x[3] = 0xdead;
        for block in &blocks {
            block.run(state);
        }
        state.x[3]
    };
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut state = pe(1);
            for round in 1..=ROUNDS {
                ready.store(round, Release);
                wait(&go, round);
                second_read.store(run(&mut state), Relaxed);
                done.store(round, Release);
            }
        });
        let mut state = pe(0);
        let mut forbidden = 0;
        for round in 1..=ROUNDS {
            for word in &words {
                word.0.store(0, Relaxed);
            }
            wait(&ready, round);
            go.store(round, Release);
            let first = run(&mut state);
            wait(&done, round);
            if first == 0 && second_read.load(Relaxed) == 0 {
                forbidden += 1;
            }
        }
        forbidden
    })
}

/// A full DMB between them orders each PE's store before its load. The
/// plain pair shows whether this host shows SB at all.
#[test]
fn a_full_dmb_orders_a_store_before_a_later_load() {
    let plain = both_zero(&[vec![STR_W2_X0, LDR_W3_X1]]);
    std::println!("SB plain STR; LDR: {plain} of {ROUNDS} rounds read 0 twice");
    assert_eq!(both_zero(&[vec![STR_W2_X0, DMB_ISH, LDR_W3_X1]]), 0);
    assert_eq!(both_zero(&[vec![STR_W2_X0], vec![DMB_ISH, LDR_W3_X1]]), 0);
}

/// A store-release and a later load-acquire (RCsc) are ordered, in one
/// block or across blocks, through LDAR or LDAXR. LDAPR (RCpc) after a
/// store-release, which Arm leaves unordered, shows the plain path runs.
#[test]
fn a_store_release_comes_before_a_later_rcsc_load_acquire() {
    let rcpc = both_zero(&[vec![STLR_W2_X0, LDAPR_W3_X1]]);
    std::println!("SB STLR; LDAPR: {rcpc} of {ROUNDS} rounds read 0 twice");
    assert_eq!(both_zero(&[vec![STLR_W2_X0, LDAR_W3_X1]]), 0);
    assert_eq!(both_zero(&[vec![STLR_W2_X0], vec![LDAR_W3_X1]]), 0);
    assert_eq!(both_zero(&[vec![STLR_W2_X0], vec![LDAXR_W3_X1]]), 0);
    // A swap into ZR with release is the same store.
    assert_eq!(both_zero(&[vec![SWPL_W2_WZR_X0, LDAR_W3_X1]]), 0);
    // Bytes take the same path.
    assert_eq!(both_zero(&[vec![STLRB_W2_X0, LDARB_W3_X1]]), 0);
}
