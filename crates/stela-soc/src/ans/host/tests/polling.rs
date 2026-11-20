// SPDX-License-Identifier: GPL-2.0-or-later
//! The looks for completions of a host disk that sends no completion signal,
//! timed against a model of the boot's staged disk (`staged_disk` in
//! `stela-boot-core`), in counter ticks of the 24 MHz counter.
//!
//! The staged disk's results reach the model only through a step taken on
//! another processor: a look that finds no result asks for one, which comes
//! once a processor releases the devices' lock outside its run loop, and a
//! step that takes a result calls the home machine, whose next transfer
//! polls again. So a look is what makes a completion seen: one late keeps
//! a completed read from the guest, and one early only asks for a step that
//! finds nothing.
use super::*;

/// Ticks in a microsecond of the 24 MHz counter.
const MICRO: u64 = 24;
/// From a step that took a result to the home machine's poll, in
/// microseconds: a sleeping PE's wake (`PES WAKE` averages about 1 us)
/// and one transfer.
const HOME: u64 = 5;

/// The staged disk's timing: the disk's own time for a read, and the delays
/// of the steps and the home machine's poll after one.
#[derive(Clone, Copy, Debug)]
struct Timing {
    /// What the host disk takes for a read once a step gave it.
    disk: u64,
    /// From a finish that found no result to the step it asked for, on
    /// average: the wait for a processor that releases the devices' lock
    /// outside its run loop.
    step: u64,
    /// How each step's wait is drawn around `step`.
    waits: Waits,
    /// From a step that took a result to the home machine's poll.
    home: u64,
}

#[derive(Clone, Copy, Debug)]
enum Waits {
    /// Evenly from half the average to one and a half.
    Even,
    /// Exponentially, as the time to the next of many independent
    /// releases is: mostly short, sometimes several times the average.
    Memoryless,
}

/// The staged disk's state, shared with the simulation that steps it.
struct Model {
    timing: Timing,
    now: u64,
    /// Started commands no step gave the disk yet: tag and bytes.
    queued: Vec<(u32, usize)>,
    /// Commands on the disk: tag, bytes and when the disk completes them.
    running: Vec<(u32, usize, u64)>,
    /// Results a step took, for the model's finish.
    ready: Vec<(u32, usize)>,
    next: u32,
    /// When the asked step runs: one at a time, as `Staging::called` asks.
    step: Option<u64>,
    /// When the home machine polls after a step that took a result.
    home: Option<u64>,
    /// A deterministic generator for the steps' waits.
    seed: u64,
}

impl Model {
    fn ask_step(&mut self, after: u64) {
        if self.step.is_none() {
            self.step = Some(self.now + after);
        }
    }

    /// A step's wait ([`Waits`]).
    fn step_wait(&mut self) -> u64 {
        self.seed = self
            .seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let average = self.timing.step;
        match self.timing.waits {
            Waits::Even => average / 2 + (self.seed >> 33) % (average + 1),
            Waits::Memoryless => {
                let uniform = ((self.seed >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
                (-uniform.ln() * average as f64) as u64
            }
        }
    }

    /// A step at `now`: the disk's completed commands become results, the
    /// queued ones go to the disk, and a result calls the home machine.
    fn take_step(&mut self) {
        self.step = None;
        let now = self.now;
        let mut took = false;
        self.running.retain(|&(tag, bytes, done)| {
            if done > now {
                return true;
            }
            self.ready.push((tag, bytes));
            took = true;
            false
        });
        let disk = self.timing.disk;
        self.running.extend(
            self.queued
                .drain(..)
                .map(|(tag, bytes)| (tag, bytes, now + disk)),
        );
        if took && self.home.is_none() {
            self.home = Some(now + self.timing.home);
        }
    }
}

struct Staged {
    model: Rc<RefCell<Model>>,
    identity: Identity,
}

impl BlockDevice for Staged {
    fn identity(&self) -> &Identity {
        &self.identity
    }
    fn read(&mut self, _: u64, _: &mut [u8]) -> Result<(), BlockError> {
        Err(BlockError::Unsupported("the model queues every command"))
    }
    fn write(&mut self, _: u64, _: &[u8]) -> Result<(), BlockError> {
        Err(BlockError::Unsupported("the model queues every command"))
    }
    fn depth(&self) -> usize {
        64
    }
    fn room(&self) -> bool {
        true
    }
    /// A start asks for the step that gives the disk its command, which the
    /// processor that rang the doorbell takes as it releases the lock.
    fn start(&mut self, command: Command) -> Result<u32, BlockError> {
        let Command::Read { bytes, .. } = command else {
            return Err(BlockError::Unsupported("the model reads only"));
        };
        let mut model = self.model.borrow_mut();
        let tag = model.next;
        model.next += 1;
        model.queued.push((tag, bytes));
        model.ask_step(MICRO);
        Ok(tag)
    }
    fn finish(&mut self, tag: u32) -> Option<Finished> {
        let mut model = self.model.borrow_mut();
        if let Some(at) = model.ready.iter().position(|&(ready, _)| ready == tag) {
            let (_, bytes) = model.ready.swap_remove(at);
            return Some(Ok(Some(vec![0; bytes])));
        }
        let wait = model.step_wait();
        model.ask_step(wait);
        None
    }
    fn posted(&self) -> bool {
        !self.model.borrow().ready.is_empty()
    }
}

/// What a run of reads saw: the mean time from each read's submission to
/// the poll that completed it, and the looks (polls at the model's own
/// deadline) per read.
#[derive(Debug)]
struct Seen {
    latency_us: u64,
    looks: f64,
}

/// `reads` reads of 16 KiB, `depth` at a time, each submitted as another
/// completes: with `depth` 1, page-ins one after another, where every
/// look's delay is the guest's.
fn simulate(timing: Timing, reads: u32, depth: u32, policy: Option<fn(&HostDisk) -> u64>) -> Seen {
    let model = Rc::new(RefCell::new(Model {
        timing,
        now: 0,
        queued: Vec::new(),
        running: Vec::new(),
        ready: Vec::new(),
        next: 0,
        step: None,
        home: None,
        seed: 0x5eed,
    }));
    let identity = RamDisk::new("", 64, 4096).identity;
    let mut disk = HostDisk::new(Box::new(Staged {
        model: Rc::clone(&model),
        identity,
    }));
    disk.first_look = policy;
    let read = || Request::Read {
        lba: 0,
        bytes: 16384,
    };
    // Warm the averages first; only the later reads count.
    let warm = reads / 4;
    let (mut done, mut total, mut looks) = (0u32, 0u64, 0u64);
    // Each read's submission, by key.
    let mut submitted = Vec::new();
    for key in 0..depth {
        submitted.push(0);
        assert!(disk.submit(key, read(), 0).is_none());
    }
    let mut look = disk.next_look(0);
    while done < reads + warm {
        let (step, home) = {
            let model = model.borrow();
            (
                model.step.unwrap_or(u64::MAX),
                model.home.unwrap_or(u64::MAX),
            )
        };
        let now = look.min(step).min(home);
        model.borrow_mut().now = now;
        if now == step {
            model.borrow_mut().take_step();
            continue;
        }
        if now == home {
            model.borrow_mut().home = None;
        } else if done >= warm {
            looks += 1;
        }
        let completed = disk.poll(now);
        for (key, result) in completed {
            assert!(result.is_ok(), "read {key} failed");
            if done >= warm {
                total += now - submitted[key as usize];
            }
            done += 1;
            let key = submitted.len() as u32;
            submitted.push(now);
            assert!(disk.submit(key, read(), now).is_none());
        }
        look = disk.next_look(now);
    }
    Seen {
        latency_us: total / u64::from(reads) / MICRO,
        looks: looks as f64 / f64::from(reads),
    }
}

/// The earlier schedule, a first look half the latency after the oldest
/// transfer began, against looks from its start.
const SCHEDULES: [(&str, Option<fn(&HostDisk) -> u64>); 2] =
    [("half", Some(|disk| disk.latency / 2)), ("start", None)];

/// A transfer is looked at an eighth of the latency on from its start, and
/// running discards alone half the latency apart.
#[test]
fn a_transfer_is_looked_at_from_its_start() {
    let (mut disk, log) = queued(1, false);
    let read = Request::Read {
        lba: 0,
        bytes: 4096,
    };
    assert!(disk.submit(0, read, 1000).is_none());
    assert_eq!(disk.next_look(1000), 1000 + FIRST_LATENCY / 8);
    release(&log, 0);
    assert_eq!(disk.poll(2000).len(), 1);
    disk.discard(8, 4);
    assert!(disk.poll(2000 + DISCARD_QUIET).is_empty());
    assert!(disk.discarding());
    let latency = disk.latency;
    assert_eq!(
        disk.next_look(3000 + DISCARD_QUIET),
        3000 + DISCARD_QUIET + latency / 2
    );
}

/// A staged disk's result comes a step's wait after the look that asks for
/// it: looks from the start take a read the disk does in less than that
/// wait about one wait, where the first look half the latency on took
/// about two. A read that takes the disk far longer than the wait comes no
/// later.
#[test]
fn looks_from_the_start_take_a_short_read_one_step_wait_after_it() {
    let short = Timing {
        disk: 50 * MICRO,
        step: 160 * MICRO,
        waits: Waits::Even,
        home: HOME * MICRO,
    };
    let [(_, half), (_, start)] = SCHEDULES;
    let (before, after) = (
        simulate(short, 1000, 1, half),
        simulate(short, 1000, 1, start),
    );
    assert!(
        after.latency_us * 4 < before.latency_us * 3,
        "{before:?} {after:?}"
    );
    let long = Timing {
        disk: 400 * MICRO,
        step: 20 * MICRO,
        ..short
    };
    let (before, after) = (
        simulate(long, 1000, 1, half),
        simulate(long, 1000, 1, start),
    );
    assert!(
        after.latency_us <= before.latency_us + before.latency_us / 50,
        "{before:?} {after:?}"
    );
}

/// Reads over a range of the disk's own time and the step's wait, one at
/// a time and eight: the time a read waits past its disk's own is what the
/// looks add. `cargo test -p stela-soc --lib
/// dependent_read_latency -- --ignored --nocapture`.
#[test]
#[ignore = "manual: opt-in"]
fn dependent_read_latency_by_disk_and_step_times() {
    let policies = SCHEDULES;
    for waits in [Waits::Even, Waits::Memoryless] {
        std::println!("{waits:?} step waits, one read at a time:");
        std::println!("disk_us step_us home_us -> per policy latency_us/looks");
        for disk in [50, 100, 200, 400] {
            for step in [20, 80, 160, 400] {
                let timing = Timing {
                    disk: disk * MICRO,
                    step: step * MICRO,
                    waits,
                    home: HOME * MICRO,
                };
                let mut line = std::format!("{disk:>7} {step:>7} {HOME:>7} ->");
                for (name, policy) in policies {
                    let seen = simulate(timing, 4000, 1, policy);
                    line.push_str(&std::format!(
                        " {name}={}/{:.1}",
                        seen.latency_us,
                        seen.looks
                    ));
                }
                std::println!("{line}");
            }
        }
    }
    std::println!("Even step waits, eight reads at a time: disk_us step_us -> latency_us/looks");
    for disk in [100, 400] {
        for step in [20, 160] {
            let timing = Timing {
                disk: disk * MICRO,
                step: step * MICRO,
                waits: Waits::Even,
                home: HOME * MICRO,
            };
            let mut line = std::format!("{disk:>7} {step:>7} ->");
            for (name, policy) in policies {
                let seen = simulate(timing, 4000, 8, policy);
                line.push_str(&std::format!(
                    " {name}={}/{:.2}",
                    seen.latency_us,
                    seen.looks
                ));
            }
            std::println!("{line}");
        }
    }
}
