//! Host timing of the run loop's fixed cost per native run: the schedule,
//! the step's bookkeeping, the fetch of a block the table already holds and
//! the completion of its exit, with native code that costs nothing (the
//! fake host answers each run with the next of a cycle of exits). A guest
//! loop that ends a run at every turn, by a BR (Indirect) or by a System
//! instruction the engine completes and then a BR (System), on one PE
//! alone, on one PE of a parallel hub, and on two PEs of one hub at once,
//! each on its own thread. The loop runs the System instruction twice a
//! turn, from X0 and then from X2, which hold different values, so that a
//! write changes its register each time. The counter reads the time stamp
//! counter, as a host's does.

use super::super::parallel::{Hub, Kick};
use super::*;
use stela_contracts::identity::Identity;

const BR_X1: u32 = 0xd61f_0020; // br x1
const BR_X3: u32 = 0xd61f_0060; // br x3
/// The System instructions timed, each before a BR, with Rt X0.
const SYSTEM: [(&str, u32); 12] = [
    ("mrs_tpidr_el1", 0xd538_d080),   // mrs x0, tpidr_el1
    ("mrs_tpidrro_el0", 0xd53b_d060), // mrs x0, tpidrro_el0
    ("mrs_daif", 0xd53b_4220),        // mrs x0, daif
    ("msr_daifset", 0xd503_42df),     // msr daifset, #2
    ("mrs_cntvct_el0", 0xd53b_e040),  // mrs x0, cntvct_el0
    ("msr_tpidr_el1", 0xd518_d080),   // msr tpidr_el1, x0
    ("msr_contextidr", 0xd518_d020),  // msr contextidr_el1, x0
    ("msr_apibkeylo", 0xd518_2140),   // msr apibkeylo_el1, x0
    ("msr_ttbr0_el1", 0xd518_2000),   // msr ttbr0_el1, x0
    ("tlbi_vale1is", 0xd508_83a0),    // tlbi vale1is, x0
    ("tlbi_vmalle1", 0xd508_871f),    // tlbi vmalle1
    ("dsb_ish", 0xd503_3b9f),         // dsb ish
];
/// What X0 and X2 hold.
const VALUES: [u64; 2] = [0x0000_0000_1234_5000, 0x0000_0000_6789_a000];
/// Writes timed again with X0 and X2 taking a new value of so many at each
/// turn (`FakeHost::spread`): the keys of a few dozen threads, and of more
/// than the PAC service remembers.
const SPREAD: [(&str, u32, u64); 3] = [
    ("msr_apibkeylo_x64", 0xd518_2140, 64),
    ("msr_apibkeylo_x4096", 0xd518_2140, 4096),
    ("msr_ttbr0_el1_x64", 0xd518_2000, 64),
];
/// Accesses timed again with the board's stored registers numbered as a
/// booted kernel's are, [`FILLED_READS`] and [`FILLED_WRITES`] before the
/// register's own: the engine looks the register up among them.
const FILLED: [(&str, u32); 2] = [
    ("mrs_tpidrro_el0_filled", 0xd53b_d060), // mrs x0, tpidrro_el0
    ("msr_par_el1_filled", 0xd518_7400),     // msr par_el1, x0
];
const FILLED_READS: u8 = 96;
const FILLED_WRITES: u8 = 48;
/// Runs a timing makes, after as many again to warm up.
const RUNS: u64 = 1_000_000;

/// A hub whose kicks wake nobody: every PE runs.
struct Quiet;
impl Kick for Quiet {
    fn kick(&self, _: usize) {}
}

/// A machine moved to the thread that runs it: the fake host's raw chain
/// pointer and callbacks keep it from being `Send`.
struct Moved(Machine<FakeHost, NoDevices>);
// SAFETY: each machine runs on one thread at a time, and its fake host's
// chain pointer and callbacks are unset.
unsafe impl Send for Moved {}

impl Moved {
    /// [`time`] on the thread that took it whole.
    fn time(mut self) -> f64 {
        time(&mut self.0)
    }
}

/// The guest loop's words for System instruction `word` with Rt X0 (none:
/// BRs alone), and the cycle of exits its runs end with: `word` at BASE,
/// BR X1 to BASE + 8, `word` from X2 there and BR X3 back.
fn guest(word: Option<u32>) -> ([u32; 4], Vec<NativeExit>) {
    let exit = |next_pc, kind: ExitKind| NativeExit::Exit {
        next_pc,
        kind: kind as u64,
    };
    let Some(word) = word else {
        return (
            [BR_X1, BR_X1, BR_X3, BR_X3],
            vec![
                exit(BASE + 8, ExitKind::Indirect),
                exit(BASE, ExitKind::Indirect),
            ],
        );
    };
    // MRS, MSR and SYS name Rt in bits 4:0; the others keep theirs.
    let second = if word >> 24 == 0xd5 && word & 31 == 0 {
        word | 2
    } else {
        word
    };
    (
        [word, BR_X1, second, BR_X3],
        vec![
            exit(BASE, ExitKind::System),
            exit(BASE + 8, ExitKind::Indirect),
            exit(BASE + 8, ExitKind::System),
            exit(BASE, ExitKind::Indirect),
        ],
    )
}

/// Ready `machine` to loop at BASE, with EL1's PAC keys its own
/// (HCR_EL2.APK and API), as a kernel's are.
fn prepare(machine: &mut Machine<FakeHost, NoDevices>, cycle: Vec<NativeExit>) {
    const HCR_APK_API: u64 = 3 << 40;
    let hcr = machine.cpu.read(SystemRegister::HCR_EL2);
    machine
        .cpu
        .registers
        .set(SystemRegister::HCR_EL2, 0, hcr | HCR_APK_API);
    machine.derived_changed();
    machine.host.cycle = cycle;
    machine.host.stamped = true;
    machine.cpu.pc = BASE;
    machine.cpu.state.x[0] = VALUES[0];
    machine.cpu.state.x[1] = BASE + 8;
    machine.cpu.state.x[2] = VALUES[1];
    machine.cpu.state.x[3] = BASE;
}

/// Number [`FILLED_READS`] and [`FILLED_WRITES`] IMPLEMENTATION DEFINED
/// registers no guest instruction names in `machine`'s board, as a boot
/// does the registers it reads and writes before the timed one.
fn fill(machine: &Machine<FakeHost, NoDevices>) {
    use stela_compiler::aarch64::system::registers::Encoding;
    let encoding = |index: u8| Encoding::new(3, 1 + index / 128, 15, index / 8 % 16, index % 8);
    for index in 0..FILLED_READS {
        assert!(
            machine
                .stored_reads
                .find_or_add(encoding(index), || true)
                .is_some()
        );
    }
    for index in 0..FILLED_WRITES {
        assert!(
            machine
                .stored_writes
                .find_or_add(encoding(index), || true)
                .is_some()
        );
    }
}

/// Nanoseconds per run of `machine`'s loop over [`RUNS`] runs, after as
/// many to warm up.
fn time(machine: &mut Machine<FakeHost, NoDevices>) -> f64 {
    let exceptions = machine.statistics.exceptions;
    let blocks = machine.statistics.blocks;
    assert_eq!(machine.run(blocks + RUNS), Stop::Budget);
    assert_eq!(machine.statistics.exceptions, exceptions, "nothing traps");
    let blocks = machine.statistics.blocks;
    let started = std::time::Instant::now();
    assert_eq!(machine.run(blocks + RUNS), Stop::Budget);
    let runs = machine.statistics.blocks - blocks;
    assert_eq!(runs, RUNS, "one block a run");
    started.elapsed().as_nanos() as f64 / runs as f64
}

/// `pes` machines of one parallel hub, at BASE.
fn parallel(pes: usize, word: Option<u32>) -> Vec<Machine<FakeHost, NoDevices>> {
    let (words, cycle) = guest(word);
    let identities = (0..pes as u64)
        .map(|pe| Identity {
            midr: 0x611f_0320,
            mpidr: 0x8000_0000 | pe << 8,
            clock: None,
            caches: None,
        })
        .collect::<Vec<_>>();
    let mut home = board(&words, NoDevices, identities.clone());
    let mut workers = (1..pes)
        .map(|pe| {
            let prepared = board(&words, NoDevices, identities.clone());
            home.worker_profile()
                .start(prepared.cpu, prepared.host, prepared.devices, pe)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let hub = Hub::new(pes, Quiet);
    home.enter_parallel(Arc::clone(&hub)).unwrap();
    for worker in &mut workers {
        worker
            .start_parallel(Arc::clone(&hub), home.cpu.core_locks.clone(), BASE)
            .unwrap();
    }
    let mut machines = vec![home];
    machines.extend(workers);
    for machine in &mut machines {
        prepare(machine, cycle.clone());
    }
    machines
}

/// `cargo test -p stela-engine --release --lib run_loop_costs --
/// --ignored --nocapture`
#[test]
#[ignore = "manual: opt-in"]
fn run_loop_costs() {
    // RUN_CASE=<name> times that case alone, RUN_SERIAL=1 only on one PE
    // alone: for a profile of one path.
    let only = std::env::var("RUN_CASE").ok();
    let serial = std::env::var_os("RUN_SERIAL").is_some();
    let cases = core::iter::once(("br", None, 0, false))
        .chain(
            SYSTEM
                .iter()
                .map(|&(name, word)| (name, Some(word), 0, false)),
        )
        .chain(
            SPREAD
                .iter()
                .map(|&(name, word, spread)| (name, Some(word), spread, false)),
        )
        .chain(
            FILLED
                .iter()
                .map(|&(name, word)| (name, Some(word), 0, true)),
        )
        .filter(|(name, ..)| only.as_deref().is_none_or(|only| only == *name));
    for (exit, word, spread, filled) in cases {
        let (words, cycle) = guest(word);
        let mut machine = super::machine(&words);
        prepare(&mut machine, cycle);
        machine.host.spread = spread;
        if filled {
            fill(&machine);
        }
        if spread != 0 || filled {
            // Serial alone: the parallel PEs share no keys to spread, and
            // the board's numbers are the same on every PE.
            let ns = (0..3)
                .map(|_| time(&mut machine))
                .fold(f64::INFINITY, f64::min);
            std::println!("RUNS exit={exit} pes=1 serial ns_per_run={ns:.1}");
            continue;
        }
        // The least of three: the host's other work only adds.
        let ns = (0..3)
            .map(|_| time(&mut machine))
            .fold(f64::INFINITY, f64::min);
        std::println!("RUNS exit={exit} pes=1 serial ns_per_run={ns:.1}");
        if serial {
            continue;
        }
        for pes in [1, 2] {
            let machines = parallel(pes, word);
            let start = std::sync::Barrier::new(pes);
            let each = std::thread::scope(|scope| {
                let start = &start;
                let threads = machines
                    .into_iter()
                    .map(|machine| {
                        let moved = Moved(machine);
                        scope.spawn(move || {
                            start.wait();
                            moved.time()
                        })
                    })
                    .collect::<Vec<_>>();
                threads
                    .into_iter()
                    .map(|thread| thread.join().unwrap())
                    .collect::<Vec<_>>()
            });
            let mean = each.iter().sum::<f64>() / each.len() as f64;
            std::println!(
                "RUNS exit={exit} pes={pes} parallel ns_per_run={mean:.1} each={each:.1?}"
            );
        }
    }
}
