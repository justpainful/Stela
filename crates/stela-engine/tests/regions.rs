//! Regions (`NativeSystem::regions`): blocks that run on past their
//! branches and jump inside themselves end in the state the same code
//! reaches as blocks through links: registers, flags, memory, retired
//! instructions and any data abort's syndrome, address and return address.
//! The bodies have the shapes of the macOS kernel's and libsystem's hot
//! loops that a block alone cannot hold: a 16-byte copy with its byte tail
//! (memmove), a scan with an if/else (character classes), a search with a
//! preheader and two ends (OSArray), nested loops (row copies), a string
//! compare with three ends (strncmp), a rotated loop entered by a forward
//! B, and an accumulation whose flags a select reads after a join.
#![cfg(all(
    feature = "test-support",
    feature = "std",
    target_arch = "x86_64",
    target_os = "linux"
))]
use anyhow::{Result, ensure};
use stela_compiler::{GuestState, TargetCpu, block::ExitKind};
use stela_engine::machine::{Stop, system::Register};
#[path = "support/store_host.rs"]
mod store_host;
use store_host::{Mapping, machine};

const PAGE: u64 = 4096;
const CSV3_SHIFT: u32 = 60;
const BYTES: u64 = 4 * PAGE;
const SUFFIX: u32 = 0x9100_04e7; // add x7,x7,#1
const PREFIX: u32 = 0x9100_04c6; // add x6,x6,#1

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// A program: its words (entry first, the suffix after them) and how a case
/// sets registers and data from `base`, the data mapping.
struct Program {
    name: &'static str,
    words: Vec<u32>,
    setup: fn(&mut Rng, u64, &mut GuestState, &mut [u8]),
}

fn programs() -> Vec<Program> {
    vec![
        Program {
            name: "memmove by 16 bytes, then its byte tail",
            words: vec![
                0xf100_405f, //    cmp x2,#16
                0x5400_00c3, //    b.lo 2f
                0xa8c1_1023, // 1: ldp x3,x4,[x1],#16
                0xa881_1003, //    stp x3,x4,[x0],#16
                0xd100_4042, //    sub x2,x2,#16
                0xf100_405f, //    cmp x2,#16
                0x54ff_ff82, //    b.hs 1b
                0xb400_00a2, // 2: cbz x2,4f
                0x3840_1423, // 3: ldrb w3,[x1],#1
                0x3800_1403, //    strb w3,[x0],#1
                0xf100_0442, //    subs x2,x2,#1
                0x54ff_ffa1, //    b.ne 3b
            ], // 4:
            setup: |rng, base, state, _| {
                state.x[1] = base + rng.below(2 * PAGE);
                state.x[0] = base + rng.below(2 * PAGE);
                state.x[2] = rng.below(2 * PAGE);
            },
        },
        Program {
            name: "count spaces and others by an if/else",
            words: vec![
                0x3840_1408, // 1: ldrb w8,[x0],#1
                0x7100_811f, //    cmp w8,#32
                0x5400_0061, //    b.ne 2f
                0x9100_054a, //    add x10,x10,#1
                0x1400_0002, //    b 3f
                0x9100_056b, // 2: add x11,x11,#1
                0xf100_0442, // 3: subs x2,x2,#1
                0x54ff_ff21, //    b.ne 1b
            ],
            setup: |rng, base, state, data| {
                state.x[0] = base + rng.below(2 * PAGE);
                state.x[2] = 1 + rng.below(3 * PAGE);
                for byte in data.iter_mut() {
                    if *byte & 1 == 0 {
                        *byte = 32;
                    }
                }
            },
        },
        Program {
            name: "search with a preheader and two ends",
            words: vec![
                0xaa1f_03e9, //    mov x9,xzr
                0xb400_00e2, //    cbz x2,2f
                0xf840_8408, // 1: ldr x8,[x0],#8
                0xeb03_011f, //    cmp x8,x3
                0x5400_00a0, //    b.eq 3f
                0x9100_0529, //    add x9,x9,#1
                0xf100_0442, //    subs x2,x2,#1
                0x54ff_ff61, //    b.ne 1b
                0x9280_0009, // 2: mov x9,#-1
                0xaa09_03ea, // 3: mov x10,x9
            ],
            setup: |rng, base, state, data| {
                state.x[0] = base + 8 * rng.below(PAGE / 4);
                state.x[2] = rng.below(PAGE);
                let at = 8 * rng.below(BYTES / 8) as usize;
                state.x[3] = u64::from_le_bytes(data[at..at + 8].try_into().unwrap());
            },
        },
        Program {
            name: "row copy by nested loops",
            words: vec![
                0xaa04_03e5, // 1: mov x5,x4
                0x3840_1426, // 2: ldrb w6,[x1],#1
                0x3800_1406, //    strb w6,[x0],#1
                0xf100_04a5, //    subs x5,x5,#1
                0x54ff_ffa1, //    b.ne 2b
                0x8b03_0000, //    add x0,x0,x3
                0xf100_0442, //    subs x2,x2,#1
                0x54ff_ff21, //    b.ne 1b
            ],
            setup: |rng, base, state, _| {
                state.x[1] = base + rng.below(PAGE);
                state.x[0] = base + rng.below(PAGE);
                state.x[4] = 1 + rng.below(64);
                state.x[3] = rng.below(64);
                state.x[2] = 1 + rng.below(40);
            },
        },
        Program {
            name: "strncmp with three ends",
            words: vec![
                0xb400_0102, //    cbz x2,3f
                0x3840_1403, // 1: ldrb w3,[x0],#1
                0x3840_1424, //    ldrb w4,[x1],#1
                0x6b04_0065, //    subs w5,w3,w4
                0x5400_00a1, //    b.ne 2f
                0x3400_0063, //    cbz w3,3f
                0xf100_0442, //    subs x2,x2,#1
                0x54ff_ff41, //    b.ne 1b
                0x5280_0005, // 3: mov w5,#0
                0x9340_7ca5, // 2: sxtw x5,w5
            ],
            setup: |rng, base, state, data| {
                let (a, b) = (rng.below(2 * PAGE), rng.below(2 * PAGE));
                state.x[0] = base + a;
                state.x[1] = base + b;
                state.x[2] = rng.below(3 * PAGE);
                let same = rng.below(3 * PAGE) as usize;
                for index in 0..same.min(data.len() - a.max(b) as usize) {
                    let byte = data[a as usize + index] | 1;
                    data[a as usize + index] = byte;
                    data[b as usize + index] = byte;
                }
                if rng.below(2) == 0 {
                    let end = (a as usize + same).min(data.len() - 1);
                    data[end] = 0;
                }
            },
        },
        Program {
            name: "fill entered by a forward B",
            words: vec![
                0x1400_0003, //    b 2f
                0xf800_8403, // 1: str x3,[x0],#8
                0x9100_0463, //    add x3,x3,#1
                0xf100_0442, // 2: subs x2,x2,#1
                0x54ff_ffa2, //    b.hs 1b
            ],
            setup: |rng, base, state, _| {
                state.x[0] = base + 8 * rng.below(PAGE / 2);
                state.x[2] = rng.below(PAGE / 2);
            },
        },
        Program {
            name: "carries counted, a select reading flags after a join",
            words: vec![
                0xf840_8408, // 1: ldr x8,[x0],#8
                0xab08_0129, //    adds x9,x9,x8
                0x5400_0063, //    b.lo 2f
                0x9100_054a, //    add x10,x10,#1
                0x1400_0002, //    b 3f
                0x9100_058c, // 2: add x12,x12,#1
                0x9a88_816b, // 3: csel x11,x11,x8,hi
                0xf100_0442, //    subs x2,x2,#1
                0x54ff_ff01, //    b.ne 1b
            ],
            setup: |rng, base, state, _| {
                state.x[0] = base + 8 * rng.below(PAGE / 2);
                state.x[2] = 1 + rng.below(PAGE / 2);
            },
        },
        Program {
            name: "search leaving through a side exit past an SVC",
            words: vec![
                0xf840_8408, // 1: ldr x8,[x0],#8
                0xeb03_011f, //    cmp x8,x3
                0x5400_0100, //    b.eq 3f
                0x3600_0068, //    tbz w8,#0,2f
                0x9100_054a, //    add x10,x10,#1
                0x1400_0002, //    b 5f
                0x9100_056b, // 2: add x11,x11,#1
                0xf100_0442, // 5: subs x2,x2,#1
                0x54ff_ff01, //    b.ne 1b
                0xd400_0001, //    svc #0
                0xaa09_03ec, // 3: mov x12,x9
            ],
            setup: |rng, base, state, data| {
                state.x[0] = base + 8 * rng.below(PAGE / 2);
                state.x[2] = 1 + rng.below(PAGE / 2);
                let at = 8 * rng.below(BYTES / 8) as usize;
                state.x[3] = u64::from_le_bytes(data[at..at + 8].try_into().unwrap());
            },
        },
        Program {
            name: "copy loop after a preheader no run takes, ending at a zero",
            words: vec![
                0x9e67_0061, //    fmov d1,x3
                0xb400_00c2, //    cbz x2,2f
                0xa8c1_2428, // 1: ldp x8,x9,[x1],#16
                0xb400_00c9, //    cbz x9,3f
                0xa881_2408, //    stp x8,x9,[x0],#16
                0xf100_0442, //    subs x2,x2,#1
                0x54ff_ff81, //    b.ne 1b
                0x9100_054a, // 2: add x10,x10,#1
                0xd400_0001, //    svc #0
                0xaa09_03ec, // 3: mov x12,x9
            ],
            setup: |rng, base, state, data| {
                state.x[1] = base + 16 * rng.below(PAGE / 8);
                state.x[0] = base + 16 * rng.below(PAGE / 8);
                state.x[2] = rng.below(PAGE / 4);
                for word in data.as_chunks_mut::<8>().0 {
                    if rng.below(64) == 0 {
                        *word = [0; 8];
                    }
                }
            },
        },
        Program {
            name: "products the engine completes",
            words: vec![
                0xf840_8408, // 1: ldr x8,[x0],#8
                0x3600_0088, //    tbz w8,#0,2f
                0x9e67_0101, //    fmov d1,x8
                0x1e61_0842, //    fmul d2,d2,d1
                0x1400_0002, //    b 3f
                0x9100_054a, // 2: add x10,x10,#1
                0xf100_0442, // 3: subs x2,x2,#1
                0x54ff_ff21, //    b.ne 1b
            ],
            setup: |rng, base, state, data| {
                state.x[0] = base + 8 * rng.below(PAGE / 2);
                state.x[2] = 1 + rng.below(PAGE / 2);
                state.v[2] = [1.5f64.to_bits(), 0];
                // FPCR.DZE, which no product raises: each FMUL exits
                // (`ExitKind::FloatingPoint`) mid-region, for the engine.
                state.fpcr = 1 << 9;
                for word in data.as_chunks_mut::<8>().0 {
                    let value = u64::from_le_bytes(*word);
                    *word = (value & 0x3fff_ffff_ffff_ffff | 0x3ff0_0000_0000_0000).to_le_bytes();
                }
            },
        },
    ]
}

/// What a run ends in.
#[derive(Debug, PartialEq)]
struct Outcome {
    stop: Stop,
    pc: u64,
    state: GuestState,
    retired: u64,
    memory: Vec<u8>,
    abort: Option<(u64, u64, u64)>,
}

/// One case's start: registers, data, the page made to fault, and whether
/// every access masks its address.
struct Case<'a> {
    initial: &'a GuestState,
    contents: &'a [u8],
    protected: Option<u64>,
    masked: bool,
}

fn run(
    words: &[u32],
    code: &Mapping,
    data: &Mapping,
    regions: bool,
    case: &Case,
) -> Result<(Outcome, u64, u64)> {
    let Case {
        initial,
        contents,
        protected,
        masked,
    } = *case;
    let mut machine = machine(code, data, words, TargetCpu::detect())?;
    machine.set_regions(regions);
    // FEAT_CSV3 disclosed: every access masks its address.
    let pfr0 = machine.cpu.registers.get(Register::ID_AA64PFR0_EL1, 0);
    let pfr0 = pfr0 & !(0xf << CSV3_SHIFT) | u64::from(masked) << CSV3_SHIFT;
    machine
        .cpu
        .registers
        .set_raw(Register::ID_AA64PFR0_EL1, 0, pfr0);
    ensure!(data.write(data.0, contents), "write data");
    if let Some(page) = protected {
        data.protect(page as usize * PAGE as usize, PAGE as usize, false)?;
    }
    machine.cpu.state = initial.clone();
    machine.cpu.pc = code.0;
    let retired = machine.cpu.retired;
    let stop = machine.run(machine.statistics.blocks + 200_000);
    if let Some(page) = protected {
        data.protect(page as usize * PAGE as usize, PAGE as usize, true)?;
    }
    // Each machine maps its own vector page: an abort stops at its vector.
    let vector = machine.host.lower_synchronous();
    let (stop, pc) = match stop == Stop::Breakpoint(vector) {
        true => (Stop::Breakpoint(0), machine.cpu.pc.wrapping_sub(vector)),
        false => (stop, machine.cpu.pc),
    };
    let abort = (stop == Stop::Breakpoint(0)).then(|| {
        (
            machine.cpu.registers.get(Register::ESR_EL1, 0),
            machine.cpu.registers.get(Register::FAR_EL1, 0),
            machine.cpu.registers.get(Register::ELR_EL1, 0),
        )
    });
    Ok((
        Outcome {
            stop,
            pc,
            state: machine.cpu.state.clone(),
            retired: machine.cpu.retired - retired,
            memory: data.read(data.0, BYTES as usize).expect("readable data"),
            abort,
        },
        machine.statistics.blocks,
        machine.statistics.exit_kinds[ExitKind::FloatingPoint as usize],
    ))
}

#[test]
fn regions_end_where_their_linked_blocks_do() -> Result<()> {
    let data = Mapping::new()?;
    let code = Mapping::new()?;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (mut cases, mut aborts, mut fewer, mut exits) = (0, 0, 0, 0);
    for program in programs() {
        let (mut program_cases, mut program_fewer) = (0, 0);
        for prefix in [false, true] {
            let words = [
                if prefix { vec![PREFIX] } else { vec![] },
                program.words.clone(),
                vec![SUFFIX],
            ]
            .concat();
            for case in 0..60 {
                let mut initial = GuestState {
                    x: std::array::from_fn(|_| rng.next()),
                    v: std::array::from_fn(|_| [rng.next(), rng.next()]),
                    nzcv: (rng.next() as u32) & 0xf000_0000,
                    ..GuestState::default()
                };
                let mut contents: Vec<u8> = (0..BYTES).map(|_| rng.next() as u8).collect();
                (program.setup)(&mut rng, data.0, &mut initial, &mut contents);
                // A third of the cases fault on a page the loop reaches.
                let protected = (case % 3 == 2).then(|| 1 + rng.below(3));
                // Half mask every access (FEAT_CSV3), as the M2 guest does.
                let masked = case % 2 == 1;
                let start = Case {
                    initial: &initial,
                    contents: &contents,
                    protected,
                    masked,
                };
                let run = |regions| run(&words, &code, &data, regions, &start);
                let (linked, linked_blocks, _) = run(false)?;
                let (region, region_blocks, numerical) = run(true)?;
                let what = format!(
                    "{} prefix {prefix} case {case} masked {masked} protected {protected:?}",
                    program.name
                );
                assert_eq!(
                    (region.stop, region.pc, region.retired, region.abort),
                    (linked.stop, linked.pc, linked.retired, linked.abort),
                    "{what}"
                );
                assert_eq!(region.state, linked.state, "{what}");
                assert!(region.memory == linked.memory, "{what}: memory");
                cases += 1;
                aborts += usize::from(linked.abort.is_some());
                fewer += usize::from(region_blocks < linked_blocks);
                program_cases += 1;
                program_fewer += usize::from(region_blocks < linked_blocks);
                exits += numerical;
            }
        }
        // Each program runs inside its region in most cases.
        assert!(
            program_fewer > program_cases / 2,
            "{}: {program_fewer} of {program_cases} ran fewer blocks",
            program.name
        );
    }
    assert!(fewer > cases / 2, "{fewer} of {cases} ran fewer blocks");
    assert!(aborts > cases / 10, "{aborts} of {cases} aborted");
    // Exits the engine completes at an instruction inside a region.
    assert!(exits > 1000, "{exits} numerical exits");
    Ok(())
}

/// A region whose loop never ends leaves through a direct exit and its
/// polled link after at most a block's instructions per entry, whichever
/// of its paths the iterations take, and counts each path exactly.
#[test]
fn an_endless_region_takes_its_link_within_a_blocks_instructions() -> Result<()> {
    const MAX_BLOCK: u64 = stela_compiler::block::MAX_BLOCK_INSTRUCTIONS as u64;
    let data = Mapping::new()?;
    let code = Mapping::new()?;
    // 0: add x5,x5,#1; 1: tbz x5,#0, 3; 2: b 4; 3: add x6,x6,#1;
    // 4: cbz xzr, 0: four instructions on odd x5, four on even, through a
    // forward B and a join.
    let words = [
        0x9100_04a5,
        0x3600_0045,
        0x1400_0002,
        0x9100_04c6,
        0xb400_0000 | ((-4i32 as u32 & 0x7ffff) << 5) | 31,
    ];
    for regions in [false, true] {
        let mut machine = machine(&code, &data, &words, TargetCpu::detect())?;
        machine.set_regions(regions);
        machine.cpu.pc = code.0;
        let (retired, blocks) = (machine.cpu.retired, machine.statistics.blocks);
        machine.run(blocks + 1000);
        let (retired, blocks) = (
            machine.cpu.retired - retired,
            machine.statistics.blocks - blocks,
        );
        let state = &machine.cpu.state;
        ensure!(blocks >= 1000, "the run spent its budget: {blocks}");
        ensure!(
            retired == 4 * state.x[5] && state.x[6] == state.x[5] / 2,
            "regions {regions}: {retired} instructions for x5 {} x6 {}",
            state.x[5],
            state.x[6]
        );
        ensure!(
            retired <= blocks * MAX_BLOCK,
            "regions {regions}: {retired} instructions in {blocks} transfers"
        );
        if regions {
            ensure!(retired >= blocks * (MAX_BLOCK - 8), "{retired} in {blocks}");
        }
    }
    Ok(())
}

/// A region rewritten by the guest (store, DC CVAU, IC IVAU, ISB) runs its
/// new body the next time round, as blocks through links do.
#[test]
fn a_rewritten_region_runs_its_new_body() -> Result<()> {
    let data = Mapping::new()?;
    let code = Mapping::new()?;
    let words = [
        0x1400_0002,                                 // 0: b 2
        0x9100_04a5,                                 // 1: add x5,x5,#1 (rewritten to #2)
        0xf100_0442,                                 // 2: subs x2,x2,#1
        0x54ff_ffc2,                                 // 3: b.hs 1
        0x3500_0146,                                 // 4: cbnz w6, 14
        0xb900_0149,                                 // 5: str w9,[x10]
        0xd50b_7b2a,                                 // 6: dc cvau, x10
        0xd503_3b9f,                                 // 7: dsb ish
        0xd50b_752a,                                 // 8: ic ivau, x10
        0xd503_3b9f,                                 // 9: dsb ish
        0xd503_3fdf,                                 // 10: isb
        0xaa0b_03e2,                                 // 11: mov x2,x11
        0x5280_0026,                                 // 12: mov w6,#1
        0x1400_0000 | (-13i32 as u32 & 0x03ff_ffff), // 13: b 0
        SUFFIX,                                      // 14
    ];
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for case in 0..8 {
        let (first, second) = (1 + rng.below(700), 1 + rng.below(700));
        let (mut outcomes, mut transfers) = (vec![], vec![]);
        for regions in [false, true] {
            let mut machine = machine(&code, &data, &words, TargetCpu::detect())?;
            machine.set_regions(regions);
            // SCTLR_EL1.UCI: EL0 cleans and invalidates by VA.
            let sctlr = machine.cpu.registers.get(Register::SCTLR_EL1, 0);
            machine
                .cpu
                .registers
                .set_raw(Register::SCTLR_EL1, 0, sctlr | 1 << 26);
            let mut state = GuestState::default();
            state.x[2] = first;
            state.x[9] = 0x9100_08a5; // add x5,x5,#2
            state.x[10] = code.0 + 4;
            state.x[11] = second;
            machine.cpu.state = state;
            machine.cpu.pc = code.0;
            let (retired, blocks) = (machine.cpu.retired, machine.statistics.blocks);
            let stop = machine.run(blocks + 200_000);
            transfers.push(machine.statistics.blocks - blocks);
            // b.hs runs the body once more than the count: x2 wraps below 0.
            ensure!(
                machine.cpu.state.x[5] == first + 2 * second && machine.cpu.state.x[7] == 1,
                "case {case} regions {regions}: x5 {} after {first} and {second}, {stop:?}",
                machine.cpu.state.x[5]
            );
            outcomes.push((
                stop,
                machine.cpu.pc,
                machine.cpu.state.clone(),
                machine.cpu.retired - retired,
            ));
        }
        assert_eq!(outcomes[0], outcomes[1], "case {case}");
        assert!(transfers[1] < transfers[0], "case {case}: {transfers:?}");
    }
    Ok(())
}

/// Host time per guest instruction of each program through the engine,
/// blocks against regions: one machine per mode runs 40 cases 50 times
/// each, the first round compiling. Opt-in: `--ignored --nocapture`, in a
/// release build on a quiet host.
#[test]
#[ignore = "manual: opt-in"]
fn regions_time_their_programs() -> Result<()> {
    let data = Mapping::new()?;
    let code = Mapping::new()?;
    for program in programs() {
        let words = [program.words.clone(), vec![SUFFIX]].concat();
        let mut rng = Rng(0x6a09_e667_f3bc_c908);
        let cases: Vec<(GuestState, Vec<u8>)> = (0..40)
            .map(|_| {
                let mut initial = GuestState {
                    x: std::array::from_fn(|_| rng.next()),
                    ..GuestState::default()
                };
                let mut contents: Vec<u8> = (0..BYTES).map(|_| rng.next() as u8).collect();
                (program.setup)(&mut rng, data.0, &mut initial, &mut contents);
                (initial, contents)
            })
            .collect();
        let mut line = format!("{:58}", program.name);
        for regions in [false, true] {
            let mut machine = machine(&code, &data, &words, TargetCpu::detect())?;
            machine.set_regions(regions);
            // A case that ends in an abort leaves EL1; each starts at EL0.
            let pstate = machine.cpu.pstate;
            let (mut nanos, mut retired, mut blocks) = (0u128, 0u64, 0u64);
            for round in 0..50 {
                for (initial, contents) in &cases {
                    ensure!(data.write(data.0, contents), "write data");
                    machine.cpu.state = initial.clone();
                    machine.cpu.pstate = pstate;
                    machine.cpu.pc = code.0;
                    let (before, transfers) = (machine.cpu.retired, machine.statistics.blocks);
                    let start = std::time::Instant::now();
                    machine.run(machine.statistics.blocks + 1_000_000);
                    if round > 0 {
                        nanos += start.elapsed().as_nanos();
                        retired += machine.cpu.retired - before;
                        blocks += machine.statistics.blocks - transfers;
                    }
                }
            }
            line += &format!(
                " | {}: {:.3} ns/instr {:5.1} instr/transfer",
                if regions { "regions" } else { "blocks" },
                nanos as f64 / retired.max(1) as f64,
                retired as f64 / blocks.max(1) as f64
            );
        }
        std::println!("{line}");
    }
    Ok(())
}
