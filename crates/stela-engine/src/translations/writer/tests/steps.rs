// SPDX-License-Identifier: GPL-2.0-or-later
//! Steps on the host's clock: a boot's session of the size the long ones
//! reach, each step timed against its deadline and each part of a step
//! named ([`Part`]).
use super::*;
use alloc::collections::VecDeque;

/// The clock's ticks a second: nanoseconds.
const NANOS: u64 = 1_000_000_000;

/// The calling thread's CPU time: what a step itself took, as a boot's
/// processor gives it to the step alone, whatever else the host runs.
fn cpu_nanos() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a timespec the call fills.
    let result = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now) };
    assert_eq!(result, 0, "the thread's CPU clock");
    now.tv_sec as u64 * NANOS + now.tv_nsec as u64
}

/// A step's deadline, as the boot gives it ([`STEP_MICROS`] there).
const DEADLINE: u64 = NANOS / 1000;
/// A step past its deadline by more than this is slow ([`Steps::slow`]).
const SLOW: u64 = 4 * DEADLINE;

/// A target that takes each write at once and keeps none: a write costs
/// what it reads and hashes.
struct Discard;

impl Target for Discard {
    type Error = String;
    fn write_at(&mut self, _: u64, _: &[u8]) -> Result<bool, String> {
        Ok(true)
    }
    fn flush(&mut self) -> Result<bool, String> {
        Ok(true)
    }
}

/// A boot's store at a session's pace: a batch of entries at a time, each
/// pressing whatever its size, counted in bytes as the boot's parts count
/// them, and each page's uses.
struct Store {
    added: BTreeMap<Digest, Record>,
    bytes: u64,
    total: u64,
    uses: BTreeMap<Digest, u32>,
    shedding: bool,
    shed: u64,
}

impl Entries for Store {
    fn take_added(&mut self) -> BTreeMap<Digest, Record> {
        self.bytes = 0;
        core::mem::take(&mut self.added)
    }
    fn found_keys(&mut self) -> Vec<Digest> {
        Vec::new()
    }
    fn added(&mut self) -> u64 {
        self.total
    }
    /// A batch presses whatever its size: one waiting for a segment's delay
    /// would leave the session to steps that find nothing to do, whose CPU
    /// time this clock counts.
    fn pressed(&mut self) -> bool {
        !self.added.is_empty()
    }
    fn heap(&mut self) -> Option<Heap> {
        None
    }
    fn shed(&mut self, shed: bool) {
        self.shedding = shed;
    }
    fn added_bytes(&mut self) -> u64 {
        self.bytes
    }
    fn uses_of(&mut self, key: &Digest) -> u32 {
        self.uses.get(key).copied().unwrap_or(0)
    }
}

impl Store {
    /// Add page `page`'s record of one entry at `offset`: its key, when
    /// kept.
    fn add(&mut self, page: u32, offset: u16, entry_bytes: usize) -> Option<Digest> {
        if self.shedding {
            self.shed += 1;
            return None;
        }
        let key = page_key(page);
        let mut record = Record::new(translator_id(), key);
        record.insert(entry(offset, page as u8, entry_bytes));
        self.bytes += record.retained_bytes() as u64;
        self.added.insert(key, record);
        *self.uses.entry(key).or_default() += 1;
        self.total += 1;
        Some(key)
    }
}

/// A session's size: the pages it compiles, in batches that press; each
/// entry's code bytes; one page in `again` compiled again a batch later
/// (its record merged anew), none for 0; the arena, file and limit.
pub(super) struct Session {
    pub records: u32,
    pub batch: u32,
    pub entry_bytes: usize,
    pub again: u32,
    pub arena: usize,
    pub file: u64,
    pub limit: u64,
}

/// What a session's steps took, in nanoseconds: the steps that found work,
/// the longest, those longer than 1, 5 and 20 ms, the time of all, and the
/// writer's stats and events.
#[derive(Debug, Default)]
pub(super) struct Steps {
    pub steps: u64,
    pub longest: u64,
    pub over: [u64; 3],
    /// Steps past their deadline by more than [`SLOW`].
    pub slow: u64,
    pub busy: u64,
    pub stats: Stats,
    pub segments: u64,
    pub merged: u64,
    pub arena_full: bool,
    pub shed: u64,
    /// The writer's state, when the session did not end within [`CAP`].
    pub stuck: Option<String>,
}

/// The most CPU time a session's steps take before it gives up.
const CAP: u64 = 60 * NANOS;

/// What the writer is doing, to say where a session that does not end is.
fn state(writer: &Writer<'_>, store: &Store) -> String {
    let task = match &writer.task {
        Task::Idle => "idle",
        Task::Delta(..) => "delta",
        Task::Merge(..) => "merge",
        Task::Write { .. } => "write",
    };
    std::format!(
        "task={task} compacting={} waiting={} held={} segments={}/{} lists={} arena={:?} \
         arena_full={} pressed={} free={} data={} segment_bytes={} store={} \
         next_delta={} looked={:?} shedding={} pack_full={} merged={} settled={} partial={} \
         failed={}",
        writer.compaction.is_some(),
        writer.waiting.len(),
        writer.held.len(),
        writer.segments.len(),
        writer.count(SEGMENT_BLOB),
        writer.count(USED_BLOB),
        writer.arena(),
        writer.arena_full,
        writer.pressed,
        writer.free(),
        writer.data_bytes(),
        writer.segment_bytes(),
        store.added.len(),
        writer.next_delta,
        writer.looked,
        writer.shedding,
        writer.pack_full,
        writer.merged,
        writer.settled,
        writer.partial,
        writer.failed,
    )
}

impl Steps {
    /// One line per figure, the parts' longest in microseconds.
    fn report(&self, name: &str) {
        let micros = |nanos: u64| nanos / 1000;
        std::println!(
            "STEPS {name}: steps={} longest_us={} over_1ms={} over_5ms={} over_20ms={} slow={} \
             busy_ms={} segments={} merged={} arena_full={} shed={}",
            self.steps,
            micros(self.longest),
            self.over[0],
            self.over[1],
            self.over[2],
            self.slow,
            self.busy / 1_000_000,
            self.segments,
            self.merged,
            self.arena_full,
            self.shed,
        );
        let parts: Vec<String> = Part::ALL
            .iter()
            .map(|part| {
                std::format!(
                    "{}={}",
                    part.name(),
                    micros(self.stats.parts[*part as usize])
                )
            })
            .collect();
        std::println!("STEPS {name}: parts_us {}", parts.join(" "));
        let stats = Stats {
            parts: [0; Part::ALL.len()],
            ..self.stats
        };
        std::println!("STEPS {name}: {stats:?}");
        if let Some(stuck) = &self.stuck {
            std::println!("STEPS {name}: no end within {} s: {stuck}", CAP / NANOS);
        }
    }
}

/// Run `session` on the thread's CPU clock, each step given [`DEADLINE`],
/// to the end of the writer's work, writing to `target`, then stop: the
/// steps, and the keys of the pages it added.
pub(super) fn run<T: Target<Error = String>>(
    session: &Session,
    target: &mut T,
) -> (Steps, BTreeSet<Digest>) {
    let translator = translator_id();
    let mut arena = vec![0u8; session.arena];
    let file = File {
        translator,
        limit: session.limit,
        file_bytes: session.file,
        committed: None,
        segments: Vec::new(),
        used: Vec::new(),
        partial: false,
        arena: Arena::new(&mut arena),
    };
    let mut clock = cpu_nanos;
    let mut writer = Writer::new(file, NANOS);
    let mut store = Store {
        added: BTreeMap::new(),
        bytes: 0,
        total: 0,
        uses: BTreeMap::new(),
        shedding: false,
        shed: 0,
    };
    let mut steps = Steps::default();
    let started = clock();
    let mut added = BTreeSet::new();
    let mut next = 0u32;
    let mut again = VecDeque::new();
    // Steps in a row that found nothing to do, once every page was added.
    let mut quiet = 0;
    while quiet < 2000 {
        if next < session.records && store.added.is_empty() {
            let repeats = if session.again == 0 {
                0
            } else {
                (session.batch / session.again).min(again.len() as u32)
            };
            for _ in 0..repeats {
                let page = again.pop_front().expect("a page to compile again");
                added.extend(store.add(page, 64, session.entry_bytes));
            }
            for _ in repeats..session.batch.min(session.records - next) {
                if session.again != 0 && next.is_multiple_of(session.again) {
                    again.push_back(next);
                }
                added.extend(store.add(next, 0, session.entry_bytes));
                next += 1;
            }
        }
        let idle = writer.idle();
        let from = clock();
        let events = writer
            .step(&mut clock, from + DEADLINE, &mut store, &mut |execution| {
                Some(execution.advance(&mut *target, PIECE_BYTES))
            })
            .unwrap();
        let took = clock() - from;
        if !idle || !writer.idle() || !events.is_empty() {
            steps.steps += 1;
            steps.busy += took;
        }
        steps.longest = steps.longest.max(took);
        for (over, past) in steps.over.iter_mut().zip([1, 5, 20]) {
            *over += u64::from(took > past * DEADLINE);
        }
        steps.slow += u64::from(took > DEADLINE + SLOW);
        for event in &events {
            match event {
                Event::Segment { .. } => steps.segments += 1,
                Event::Merged { .. } => steps.merged += 1,
                Event::ArenaFull { .. } => steps.arena_full = true,
                _ => {}
            }
        }
        let done = next >= session.records && store.added.is_empty();
        quiet = if done && writer.idle() && events.is_empty() {
            quiet + 1
        } else {
            0
        };
        let now = clock();
        if now > started + CAP {
            steps.stuck = Some(std::format!("now={now} {}", state(&writer, &store)));
            break;
        }
    }
    steps.stats = writer.stats();
    steps.shed = store.shed;
    writer
        .stop(&mut store, &mut |execution| {
            while !execution.advance(&mut *target, PIECE_BYTES)? {}
            Ok(())
        })
        .unwrap();
    (steps, added)
}

/// How far past its deadline a step may end at most: the unit of work it
/// began before the deadline takes a few milliseconds at most in a debug
/// build, and a host that runs other work on this one's processor charges
/// a few more to the thread's CPU clock now and then (a WSL machine on a
/// busy PC did, up to 6 ms in a part that does a few comparisons).
const OVERRUN: u64 = 25 * DEADLINE;
/// The steps a session may have of each slow one ([`SLOW`]).
const SLOW_SHARE: u64 = 50;

/// A session that fills its arena, so that it compacts, and whose full
/// merges leave records out to stay within the file's room, so that they
/// rank records by use: each step ends near its deadline on the thread's
/// CPU clock, and the file it leaves reads whole, every record whose bytes
/// a compaction moved among them. Before, in a debug build, a step took up
/// to 33 ms (a full merge's start and its choice of the records it keeps,
/// each 25 ms) and 212 of 760 steps more than 5 ms; now up to 9 ms, and 6
/// of 1025 more than 5 ms.
#[test]
fn every_step_ends_near_its_deadline_and_the_file_reads_whole() {
    let session = Session {
        records: 32_000,
        batch: 4_000,
        entry_bytes: 1 << 10,
        again: 4,
        arena: 24 << 20,
        file: DATA_START + (64 << 20),
        limit: 1 << 30,
    };
    let mut disk = Disk::new(session.file);
    disk.track = false;
    let (steps, added) = run(&session, &mut disk);
    steps.report("bounded");
    let stats = steps.stats;
    assert!(steps.stuck.is_none(), "{steps:?}");
    assert!(stats.compactions > 1 && stats.merges > 0, "{steps:?}");
    assert!(steps.longest < DEADLINE + OVERRUN, "{steps:?}");
    assert!(steps.slow * SLOW_SHARE <= steps.steps, "{steps:?}");
    // Every record the file holds reads whole and is one of a page added.
    let translator = translator_id();
    let file = load(&disk.bytes, u64::MAX);
    let mut kept = 0;
    for pack in &file.segments {
        for (key, bytes) in pack.iter() {
            assert!(added.contains(&key));
            Record::decode(bytes, &translator, &key).expect("a record whole");
            kept += 1;
        }
    }
    assert!(kept > added.len() / 4, "{kept} of {}", added.len());
}

/// A long session's steps: run 20261120's boot wrote 519 435 records in 80
/// segments of 3 to 24 thousand records (about 8 KiB each), merged in full
/// up to 20 thousand records and 354 MB, in an arena of 670 MB it compacted
/// 12 times. Here, within a few hundred megabytes of the host's memory, 500
/// thousand records of half a KiB in segments of 15 thousand, and 24
/// thousand of 8 KiB, each in an arena that fills and a file whose full
/// merges leave records out: what each step took against its 1 ms
/// deadline, and its parts; a part that moves bytes takes in proportion to
/// them. A release build gives the boot's figures.
#[test]
#[ignore = "manual: opt-in"]
fn step_costs_of_a_long_session() {
    let many = Session {
        records: 500_000,
        batch: 15_000,
        entry_bytes: 512,
        again: 8,
        arena: 96 << 20,
        file: DATA_START + (512 << 20),
        limit: 1 << 30,
    };
    run(&many, &mut Discard).0.report("many");
    let large = Session {
        records: 24_000,
        batch: 6_000,
        entry_bytes: 8 << 10,
        again: 4,
        arena: 64 << 20,
        file: DATA_START + (256 << 20),
        limit: 1 << 30,
    };
    run(&large, &mut Discard).0.report("large");
}
