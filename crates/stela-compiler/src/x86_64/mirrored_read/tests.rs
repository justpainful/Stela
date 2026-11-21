use super::*;
use crate::block::{BlockProfile, CompilationPlan, Features};
use crate::x86_64::register_read::Cell;
use crate::{Options, TargetCpu};
use alloc::vec::Vec;

// ADD X0,X0,#1; MRS X2,TPIDR_EL1; MRS X8,TPIDR_EL1; ADD X3,X8,#1;
// MRS XZR,TPIDR_EL1; ADD X4,X4,#1
const WORDS: [u32; 6] = [
    0x9100_0400,
    0xd538_d082,
    0xd538_d088,
    0x9100_0503,
    0xd538_d09f,
    0x9100_0484,
];
const PC: u64 = 0x8000;
/// MRS X0,TPIDR_EL1, the register's word with Rt 0.
const TPIDR_EL1: u32 = 0xd538_d080;
/// MRS X0,TPIDR_EL0, which no cell answers.
const TPIDR_EL0: u32 = 0xd53b_d040;

/// MRS X0,DAIF.
const DAIF: u32 = 0xd53b_4220;

/// The thread pointer's cell is the second, DAIF's the first, whose read
/// takes the masks in the frame.
fn reads() -> Reads {
    let mut reads = Reads::NONE;
    reads.reads[0] = Some(Read {
        word: TPIDR_EL1,
        cell: size_of::<Cell>() as u32,
        source: Source::Value,
    });
    reads.reads[1] = Some(Read {
        word: DAIF,
        cell: 0,
        source: Source::Masks,
    });
    reads
}

fn compile(words: &[u32], mirrored: Reads) -> crate::block::CompiledBlock {
    compile_at(words, PC, mirrored)
}

/// [`compile`] of the block at `pc`.
fn compile_at(words: &[u32], pc: u64, mirrored: Reads) -> crate::block::CompiledBlock {
    let source: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let options = Options::new(
        BlockProfile::new(Features::SYSTEM_MODE),
        TargetCpu::Baseline,
    )
    .with_native_system(crate::x86_64::NativeSystem {
        mirrored,
        ..crate::x86_64::NativeSystem::NONE
    });
    CompilationPlan::new(&source, GuestAddress(pc), &options)
        .unwrap()
        .compile()
        .unwrap()
}

#[test]
fn cells_match_the_register_whatever_rt_and_never_a_write() {
    let reads = reads();
    for rt in [0, 2, 8, 30, 31] {
        assert_eq!(reads.cell(TPIDR_EL1 | rt), Some(16));
    }
    assert_eq!(reads.cell(TPIDR_EL0), None);
    let masks = reads.read(DAIF | 7).expect("DAIF's read");
    assert_eq!((masks.cell, masks.source), (0, Source::Masks));
    // MSR TPIDR_EL1,X2 is not admitted by the parser (L clear); the table
    // matches by register, so the parser checks the direction.
    assert_eq!(Reads::NONE.cell(TPIDR_EL1), None);
    assert_eq!(destination(TPIDR_EL1 | 31), Register::Zero);
    assert_eq!(destination(TPIDR_EL1 | 8), Register::X(8));
}

#[test]
fn blocks_given_the_cells_run_on_past_the_reads() {
    let alone = compile(&WORDS, Reads::NONE);
    assert_eq!(alone.source_bytes, 8, "the first MRS ends the block");
    let inline = compile(&WORDS, reads());
    assert_eq!(inline.source_bytes, 4 * WORDS.len());
    assert_eq!(inline.native.source_map.len(), WORDS.len());
    // Another register's MRS, and MSR of this one, still end the block.
    let other = compile(&[0x9100_0400, TPIDR_EL0 | 2, 0x9100_0400], reads());
    assert_eq!(other.source_bytes, 8);
    let write = compile(&[0x9100_0400, 0xd518_d082, 0x9100_0400], reads());
    assert_eq!(write.source_bytes, 8);
}

/// Each read takes the cell's value while it is readable and leaves at
/// the MRS, with the instructions before it done and none after, while it
/// is not, an MRS of XZR included.
#[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
#[test]
fn reads_complete_from_an_open_cell_and_exit_at_a_closed_one() {
    let code = compile(&WORDS, reads()).native.x86_64_code;
    let mut cells = [Cell::default(); 2];
    let start = |cells: &[Cell; 2]| {
        let mut state = crate::GuestState::default();
        state.anchors[Anchor::Cells] = cells.as_ptr() as u64;
        for register in 0..9 {
            state.x[register] = 0x100 * register as u64;
        }
        state
    };
    cells[1] = Cell {
        readable: 1,
        value: 0xffff_fe10_2030_4050,
    };
    cells[0] = Cell {
        readable: 0,
        value: 0x5555,
    };
    let mut state = start(&cells);
    let exit = run(&code, &mut state);
    assert_eq!(
        exit,
        (PC + 4 * WORDS.len() as u64, ExitKind::Fallthrough as u64)
    );
    assert_eq!(
        (state.x[0], state.x[2], state.x[8], state.x[3], state.x[4]),
        (
            1,
            0xffff_fe10_2030_4050,
            0xffff_fe10_2030_4050,
            0xffff_fe10_2030_4051,
            0x401
        )
    );
    cells[1].readable = 0;
    let mut state = start(&cells);
    let exit = run(&code, &mut state);
    assert_eq!(exit, (PC + 4, ExitKind::System as u64));
    assert_eq!((state.x[0], state.x[2], state.x[3]), (1, 0x200, 0x300));
    // MRS XZR reads nothing, but a closed cell (UNDEFINED, trapped) still
    // leaves for the engine.
    let code = compile(&[0x9100_0400, 0xd538_d09f, 0x9100_0400], reads())
        .native
        .x86_64_code;
    let mut state = start(&cells);
    assert_eq!(run(&code, &mut state), (PC + 4, ExitKind::System as u64));
    assert_eq!(state.x[0], 1);
    cells[1].readable = 1;
    let mut state = start(&cells);
    assert_eq!(
        run(&code, &mut state),
        (PC + 12, ExitKind::Fallthrough as u64)
    );
    assert_eq!(state.x[0], 2);
}

/// MRS DAIF takes the frame's masks at bits 9:6 while its cell is open,
/// whatever the cell's value, into a pinned register, a frame one and XZR
/// alike, clearing every other bit; while it is closed the block leaves at
/// the first read.
#[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
#[test]
fn mask_reads_take_the_frame_masks_from_an_open_cell() {
    // ADD X0,X0,#1; MRS X2,DAIF; MRS X8,DAIF; MRS X9,DAIF; MRS XZR,DAIF;
    // ADD X4,X4,#1
    let words = [
        0x9100_0400,
        DAIF | 2,
        DAIF | 8,
        DAIF | 9,
        DAIF | 31,
        0x9100_0484,
    ];
    let compiled = compile(&words, reads());
    assert_eq!(compiled.source_bytes, 4 * words.len());
    let code = compiled.native.x86_64_code;
    for masks in [0b0000, 0b1010, 0b1111] {
        for readable in [0, 1] {
            let cells = [
                Cell {
                    readable,
                    value: 0x5555,
                },
                Cell::default(),
            ];
            let mut state = crate::GuestState::default();
            state.anchors[Anchor::Cells] = cells.as_ptr() as u64;
            state.daif = masks;
            for register in [2, 8, 9] {
                state.x[register] = u64::MAX;
            }
            let exit = run(&code, &mut state);
            assert_eq!(state.x[0], 1, "the instruction before ran");
            if readable == 0 {
                assert_eq!(exit, (PC + 4, ExitKind::System as u64));
                assert_eq!((state.x[2], state.x[4]), (u64::MAX, 0));
                continue;
            }
            assert_eq!(
                exit,
                (PC + 4 * words.len() as u64, ExitKind::Fallthrough as u64)
            );
            let expected = u64::from(masks) << 6;
            assert_eq!(
                (state.x[2], state.x[8], state.x[9]),
                (expected, expected, expected)
            );
            assert_eq!(state.x[4], 1);
        }
    }
}

/// A conditional branch ending a block with a mirrored read still takes its
/// condition from the host flags its compare left, though the read's exit
/// path lies between them, and branches as the guest flags say.
#[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
#[test]
fn a_terminal_branch_reads_the_flags_its_compare_left() {
    // MRS X8,TPIDR_EL1; CMP X8,#5; B.<cond> +0x40
    const NZCV: usize = core::mem::offset_of!(crate::GuestState, nzcv);
    for (condition, cases) in [
        (8u32, [(4u64, false), (5, false), (6, true)]), // HI
        (9, [(4, true), (5, true), (6, false)]),        // LS
        (0, [(4, false), (5, true), (6, false)]),       // EQ
        (11, [(4, true), (5, false), (6, false)]),      // LT
    ] {
        let words = [0xd538_d088, 0xf100_151f, 0x5400_0200 | condition];
        let code = compile(&words, reads()).native.x86_64_code;
        // No reload of the captured flags from the frame: TEST m8,imm8 at
        // [rdi+nzcv+1] (F6 87 disp32) or MOV EAX,[rdi+nzcv] would show it.
        // MOVZX EAX,WORD [rdi+nzcv] (0F B7 87 disp32) the compound ones.
        let reload = [0xf6, 0x87, (NZCV + 1) as u8, (NZCV >> 8) as u8];
        let restore = [0x0f, 0xb7, 0x87, NZCV as u8, (NZCV >> 8) as u8];
        assert!(
            !code.windows(4).any(|w| w == reload),
            "condition {condition}"
        );
        assert!(
            !code.windows(5).any(|w| w == restore),
            "condition {condition}"
        );
        for (value, taken) in cases {
            let cells = [Cell::default(), Cell { readable: 1, value }];
            let mut state = crate::GuestState::default();
            state.anchors[Anchor::Cells] = cells.as_ptr() as u64;
            let (pc, _) = run(&code, &mut state);
            let expected = if taken { PC + 8 + 0x40 } else { PC + 12 };
            assert_eq!(pc, expected, "condition {condition} value {value}");
            assert_eq!(state.x[8], value);
        }
    }
}

/// Run `code` through `pinned::entered`: the exit's PC and kind.
#[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
fn run(code: &[u8], state: &mut crate::GuestState) -> (u64, u64) {
    use core::ffi::c_void;
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
    #[cfg(target_os = "macos")]
    const MAP_ANONYMOUS: i32 = 0x1000;
    #[cfg(target_os = "linux")]
    const MAP_ANONYMOUS: i32 = 0x20;
    const BYTES: usize = 1 << 16;
    let bytes = crate::x86_64::pinned::entered(code);
    // SAFETY: private anonymous memory, RX after the copy, unmapped after
    // the call; the block touches only `state` and the cells it points at.
    unsafe {
        let page = mmap(core::ptr::null_mut(), BYTES, 3, 2 | MAP_ANONYMOUS, -1, 0);
        assert_ne!(page as usize, usize::MAX);
        assert!(bytes.len() <= BYTES);
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), page.cast(), bytes.len());
        assert_eq!(mprotect(page, BYTES, 5), 0);
        let entry = |state: *mut crate::GuestState| -> u128 {
            crate::x86_64::native::call::<u128>(page as *const u8, state, 0, 0)
        };
        let exit = entry(state);
        assert_eq!(munmap(page, BYTES), 0);
        (exit as u64, (exit >> 64) as u64)
    }
}

/// Host cost of MRS DAIF completed inside its block against the split the
/// engine makes without it: blocks ending at each MRS, each System exit
/// jumping to the thunk the engine installs (`register_read::mask_read_thunk`)
/// and the thunk's exit jumping to the next block. The jumps stand for the
/// engine's links, which also check the chain's account, so the split's
/// cost is a floor. Run by hand: it prints nanoseconds per read, the
/// least of several timings, and asserts only that both ways agree.
/// `cargo test --release -p stela-compiler --lib mirrored_read_costs --
/// --ignored --nocapture`
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
#[ignore = "manual: opt-in"]
fn mirrored_read_costs() {
    use crate::x86_64::chain::{DIRECT_EXIT_BYTES, is_direct_exit, jump, without_failures};
    /// MRS X2,DAIF then ADD X3,X2,X3, so many times a run.
    const READS: usize = 8;
    const RUNS: usize = 200_000;
    let mut words = alloc::vec![0x9100_0400];
    for _ in 0..READS {
        words.extend([DAIF | 2, 0x8b03_0043]);
    }
    let inline = compile(&words, reads());
    assert_eq!(inline.source_bytes, 4 * words.len());
    // The split: each block from its PC to the next MRS, then its thunk.
    let mut body = Vec::new();
    let mut links = Vec::new();
    let mut index = 0;
    while index < words.len() {
        let pc = PC + 4 * index as u64;
        let block = compile_at(&words[index..], pc, Reads::NONE);
        let code = block.native.x86_64_code;
        let at = body.len();
        body.extend_from_slice(&code);
        index += block.source_bytes / 4;
        if index == words.len() {
            break;
        }
        let read = PC + 4 * (index as u64 - 1);
        let site = without_failures(&code).len() - DIRECT_EXIT_BYTES;
        assert!(is_direct_exit(
            &code[site..],
            GuestAddress(read),
            ExitKind::System
        ));
        let thunk = crate::x86_64::register_read::mask_read_thunk(
            GuestAddress(read),
            2,
            crate::Place::new(Anchor::Cells, 0),
        );
        let thunk_at = body.len();
        body.extend_from_slice(&thunk.code);
        links.push((at + site, thunk_at));
        // The thunk's exit to the next block, which starts after it.
        links.push((thunk_at + thunk.next.unwrap(), body.len()));
    }
    for (site, to) in links {
        let patch = jump(site as u64, to as u64).unwrap();
        body[site..site + patch.len()].copy_from_slice(&patch);
    }
    let cells = [
        Cell {
            readable: 1,
            value: 0,
        },
        Cell::default(),
    ];
    let mut results = Vec::new();
    for (name, code) in [("inline", inline.native.x86_64_code), ("split", body)] {
        let mapped = Mapped::new(&code);
        let mut state = crate::GuestState::default();
        state.anchors[Anchor::Cells] = cells.as_ptr() as u64;
        state.daif = 0b0101;
        let mut least = f64::INFINITY;
        let mut exit = 0;
        for _ in 0..5 {
            let started = std::time::Instant::now();
            for _ in 0..RUNS {
                exit = mapped.run(&mut state);
            }
            least = least.min(started.elapsed().as_nanos() as f64 / RUNS as f64);
        }
        assert_eq!(exit as u64, PC + 4 * words.len() as u64, "{name}");
        assert_eq!(state.x[2], 0b0101 << 6, "{name}");
        results.push((name, least, state.x[3]));
        std::println!("MRS COST {name} ns_per_run={least:.2} reads={READS}");
    }
    assert_eq!(results[0].2, results[1].2, "both ways add the same reads");
    std::println!(
        "MRS COST split-inline ns_per_read={:.2}",
        (results[1].1 - results[0].1) / READS as f64
    );
}

/// Code mapped executable in [`crate::x86_64::pinned::entered`], unmapped
/// on drop.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
struct Mapped(*mut core::ffi::c_void, usize);

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
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

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
impl Mapped {
    fn new(body: &[u8]) -> Self {
        let code = crate::x86_64::pinned::entered(body);
        let bytes = code.len().next_multiple_of(4096);
        // SAFETY: private anonymous memory, RX after the copy.
        unsafe {
            let page = mmap(core::ptr::null_mut(), bytes, 3, 0x22, -1, 0);
            assert_ne!(page as usize, usize::MAX);
            core::ptr::copy_nonoverlapping(code.as_ptr(), page.cast(), code.len());
            assert_eq!(mprotect(page, bytes, 5), 0);
            Self(page, bytes)
        }
    }

    /// Run the code on `state`: the exit's PC.
    fn run(&self, state: &mut crate::GuestState) -> u128 {
        // SAFETY: the mapping holds the entered code, which touches only
        // `state` and the cells it points at.
        unsafe { crate::x86_64::native::call::<u128>(self.0.cast(), state, 0, 0) }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: our mapping, nothing runs it any more.
        assert_eq!(unsafe { munmap(self.0, self.1) }, 0);
    }
}
