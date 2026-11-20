//! Host time per guest instruction of small ARM64 kernels, each a loop the
//! system-mode engine compiles as one self-looping EL1 block: the options
//! the engine's regime gives a kernel PE (`MemoryChecks::system` with
//! SCTLR's SA, the two-sided range check of a four-level placement in 2
//! slices, locked atomics, self-loops and regions), run on the host as the
//! engine runs a block (`pinned::entered`), guest addresses being host
//! addresses as in the native window. Run by hand:
//! `cargo test --release -p stela-compiler --lib kernel_costs -- --ignored
//! --nocapture`; `KERNEL_COSTS_DUMP=<dir>` writes each block's code there
//! (`<kernel>.bin`) for a disassembler, and `KERNEL_COSTS_MASKED=1` compiles
//! for a guest that discloses FEAT_CSV3 (every access masked).
use crate::{
    GuestAddress, GuestState, Options, TargetCpu,
    block::{BlockProfile, CompilationPlan, Features},
    state::Limit,
    x86_64::{MemoryChecks, NativeSystem, RangeCheck, pinned},
};
use alloc::{format, vec, vec::Vec};

/// B.NE to `offset` bytes from itself.
const fn b_ne(offset: i32) -> u32 {
    0x5400_0001 | ((offset >> 2) as u32 & 0x7_ffff) << 5
}

const SUBS_X2_1: u32 = 0xf100_0442;

/// One kernel: its words, which loop back to the first while X2 counts
/// down.
struct Kernel {
    name: &'static str,
    words: Vec<u32>,
}

impl Kernel {
    fn new(name: &'static str, body: &[u32]) -> Self {
        let mut words = body.to_vec();
        words.push(SUBS_X2_1);
        words.push(b_ne(-4 * words.len() as i32));
        Self { name, words }
    }
}

fn kernels() -> Vec<Kernel> {
    vec![
        // ldp x3,x4,[x1],#16; stp x3,x4,[x0],#16
        Kernel::new("memcpy_ldp_stp", &[0xa8c1_1023, 0xa881_1003]),
        // ldp x3,x4,[x1]; ldp x5,x6,[x1,#16]; stp x3,x4,[x0];
        // stp x5,x6,[x0,#16]; add x1,x1,#32; add x0,x0,#32
        Kernel::new(
            "memcpy_unrolled",
            &[
                0xa940_1023,
                0xa941_1825,
                0xa900_1003,
                0xa901_1805,
                0x9100_8021,
                0x9100_8000,
            ],
        ),
        // ldrb w3,[x1],#1; strb w3,[x0],#1
        Kernel::new("bytes_ldrb_strb", &[0x3840_1423, 0x3800_1403]),
        // ldr x3,[x0,#8]; add x4,x4,x3; ldr x0,[x0]
        Kernel::new("list_chase", &[0xf940_0403, 0x8b03_0084, 0xf940_0000]),
        // add x3,x3,x4; eor x4,x4,x3,ror #7; mul x5,x3,x4;
        // orr x6,x6,x5,lsr #3; and x7,x6,#0xff; add x9,x9,x7
        Kernel::new(
            "integer_alu",
            &[
                0x8b04_0063,
                0xcac3_1c84,
                0x9b04_7c65,
                0xaa45_0cc6,
                0x9240_1cc7,
                0x8b07_0129,
            ],
        ),
        // ld1 {v0.4s},[x0]; ld1 {v1.4s},[x1],#16; fmla v0.4s,v1.4s,v3.4s;
        // st1 {v0.4s},[x0],#16
        Kernel::new(
            "neon_fmla",
            &[0x4c40_7800, 0x4cdf_7821, 0x4e23_cc20, 0x4c9f_7800],
        ),
        // stp x29,x30,[sp,#-32]!; mov x29,sp; str x19,[sp,#16];
        // add x19,x0,#1; ldr x19,[sp,#16]; ldp x29,x30,[sp],#32
        Kernel::new(
            "stack_frame",
            &[
                0xa9be_7bfd,
                0x9100_03fd,
                0xf900_0bf3,
                0x9100_0413,
                0xf940_0bf3,
                0xa8c2_7bfd,
            ],
        ),
        // adrp x8, #0 (the page of the destination, whose address the block
        // is compiled at); ldr x9,[x8,#16]; add x9,x9,#1; str x9,[x8,#16]:
        // a global counter, as kernel statistics keep them.
        Kernel::new(
            "global_counter",
            &[0x9000_0008, 0xf940_0909, 0x9100_0529, 0xf900_0909],
        ),
        // adrp x8, #0; ldr x9,[x8,#16]; add x10,x10,x9: a global read.
        Kernel::new("global_read", &[0x9000_0008, 0xf940_0909, 0x8b09_014a]),
    ]
}

/// The owner's host placement: four-level paging, Stela's window in 2
/// slices.
const RANGE: RangeCheck = RangeCheck::four_level(2);

/// The engine's options for an EL1 block on this host.
fn options(masked: bool) -> Options {
    let checks = MemoryChecks::system(true, false, true, masked).with_range(RANGE);
    let native = NativeSystem {
        exclusive: true,
        self_loops: true,
        regions: true,
        ..NativeSystem::NONE
    };
    Options::new(
        BlockProfile::new(Features::SYSTEM_MODE),
        TargetCpu::detect(),
    )
    .with_memory_checks(checks)
    .with_native_system(native)
}

/// Guest instructions each timing retires.
const INSTRUCTIONS: u64 = 1 << 24;
/// Bytes of each buffer a kernel reads or writes: L1-resident.
const BUFFER: usize = 16 * 1024;
/// Where the kernels are compiled, but the globals': a kernel PC.
const PC: u64 = 0xffff_fe00_0712_3400;

#[test]
#[ignore = "manual: opt-in"]
fn kernel_costs() {
    let masked = std::env::var_os("KERNEL_COSTS_MASKED").is_some();
    let dump = std::env::var_os("KERNEL_COSTS_DUMP").map(std::path::PathBuf::from);
    let check = RANGE.pinned_code();
    for kernel in kernels() {
        // 16-byte nodes, each the next's address and a payload, in a
        // shuffled ring; the copies' source and destination. Each buffer is
        // 64-byte aligned, so that no run depends on where the allocator
        // put it: LDP and STP take their 16-byte path.
        let mut words = vec![0u64; BUFFER / 8 + 8];
        let skew = aligned(&mut words);
        let base = words[skew..].as_mut_ptr() as u64;
        let mut order: Vec<usize> = (0..BUFFER / 16).collect();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for at in (1..order.len()).rev() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            order.swap(at, seed as usize % (at + 1));
        }
        for (index, &node) in order.iter().enumerate() {
            let next = order[(index + 1) % order.len()];
            words[skew + 2 * node] = base + 16 * next as u64;
            words[skew + 2 * node + 1] = node as u64;
        }
        let mut destination = vec![0u64; BUFFER / 8 + 8];
        let destination_at = {
            let skew = aligned(&mut destination);
            destination[skew..].as_mut_ptr() as u64
        };
        // Normal floats for the FMLA, whose recipe leaves other values to
        // its slow path.
        let mut ones = vec![0x3f80_0000_3f80_0000u64; BUFFER / 8 + 8];
        let ones_at = {
            let skew = aligned(&mut ones);
            ones[skew..].as_ptr() as u64
        };
        let mut stack = vec![0u64; 64];
        let stack_top = (stack.as_mut_ptr() as u64 + 256) & !15;

        let pc = match kernel.name {
            // A page inside the destination.
            "global_counter" | "global_read" => (destination_at + 0xfff) & !0xfff,
            _ => PC,
        };
        let source: Vec<u8> = kernel.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let compiled = CompilationPlan::new(&source, GuestAddress(pc), &options(masked))
            .and_then(|plan| plan.compile())
            .unwrap_or_else(|error| panic!("{}: {error:?}", kernel.name));
        let code = &compiled.native.x86_64_code;
        if let Some(dump) = &dump {
            std::fs::write(dump.join(format!("{}.bin", kernel.name)), code).unwrap();
        }
        let checks = code
            .windows(check.len())
            .filter(|bytes| *bytes == check)
            .count();
        let per_iteration = kernel.words.len() as u64;
        // Each iteration moves at most 16 bytes through each buffer, the
        // unrolled copy 32.
        let stride = if kernel.name == "memcpy_unrolled" {
            32
        } else {
            16
        };
        let iterations = (BUFFER / stride) as u64;
        let entry = super::flag_cost_tests::Mapped::new(&pinned::entered(code));
        let mut control = crate::x86_64::chain::ChainControl::default();
        let mut best = f64::MAX;
        for _ in 0..7 {
            let mut state = GuestState {
                lower_limit: Limit(u64::MAX),
                ..GuestState::default()
            };
            state.anchors[crate::Anchor::Control] = (&raw mut control) as u64;
            state.sp = stack_top;
            state.v[3] = [0x3f80_0000_3f80_0000; 2];
            // FPSR.IXC set, as after any inexact operation: with it clear,
            // an FMLA whose result is exact takes its slow path each time.
            state.fpsr = 0x10;
            let mut chase = base + 16 * order[0] as u64;
            let mut retired = 0;
            let start = std::time::Instant::now();
            while retired < INSTRUCTIONS {
                let (to, from) = match kernel.name {
                    "list_chase" => (chase, 0),
                    "neon_fmla" => (destination_at, ones_at),
                    _ => (destination_at, base),
                };
                state.x[0] = to;
                state.x[1] = from;
                state.x[2] = iterations;
                while state.x[2] != 0 {
                    entry.run(&mut state);
                }
                if kernel.name == "list_chase" {
                    chase = state.x[0];
                }
                retired += iterations * per_iteration;
            }
            let ns = start.elapsed().as_nanos() as f64 / retired as f64;
            best = best.min(ns);
        }
        core::hint::black_box((&destination, &words, &ones, &stack));
        std::println!(
            "KERNEL {:<16} masked={} self_loop={} insns/iter={} x86_bytes={} range_checks={} \
             ns_per_guest_insn={:.3}",
            kernel.name,
            u8::from(masked),
            u8::from(compiled.self_loop),
            per_iteration,
            code.len(),
            checks,
            best
        );
    }
}

/// The index of `buffer`'s first 64-byte aligned word.
fn aligned(buffer: &mut [u64]) -> usize {
    let address = buffer.as_mut_ptr() as usize;
    (address.next_multiple_of(64) - address) / 8
}
