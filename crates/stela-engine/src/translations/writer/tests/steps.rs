// SPDX-License-Identifier: GPL-2.0-or-later
//! Steps on the host's clock: a boot's session of the size the long ones
//! reach, each step timed against its deadline and each part of a step
//! named ([`Part`]).
use super::*;
use alloc::collections::VecDeque;
use std::time::Instant;

/// The clock's ticks a second: nanoseconds.
const NANOS: u64 = 1_000_000_000;
/// A step's deadline, as the boot gives it ([`STEP_MICROS`] there).
const DEADLINE: u64 = NANOS / 1000;

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
/// pressing, counted in bytes as the boot's parts count them, and each
/// page's uses.
struct Store {
    added: BTreeMap<Digest, Record>,
    bytes: u64,
    total: u64,
    batch: usize,
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
    fn pressed(&mut self) -> bool {
        self.added.len() >= self.batch
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
    fn uses(&mut self) -> BTreeMap<Digest, u32> {
        self.uses.clone()
    }
}

impl Store {
    /// Add page `page`'s record of one entry at `offset`.
    fn add(&mut self, page: u32, offset: u16, entry_bytes: usize) {
        if self.shedding {
            self.shed += 1;
            return;
        }
        let key = page_key(page);
        let mut record = Record::new(translator_id(), key);
        record.insert(entry(offset, page as u8, entry_bytes));
        self.bytes += record.retained_bytes() as u64;
        self.added.insert(key, record);
        *self.uses.entry(key).or_default() += 1;
        self.total += 1;
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
    pub busy: u64,
    pub stats: Stats,
    pub segments: u64,
    pub merged: u64,
    pub arena_full: bool,
    pub shed: u64,
}

impl Steps {
    /// One line per figure, the parts' longest in microseconds.
    fn report(&self, name: &str) {
        let micros = |nanos: u64| nanos / 1000;
        std::println!(
            "STEPS {name}: steps={} longest_us={} over_1ms={} over_5ms={} over_20ms={} busy_ms={} \
             segments={} merged={} arena_full={} shed={}",
            self.steps,
            micros(self.longest),
            self.over[0],
            self.over[1],
            self.over[2],
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
    }
}

/// Run `session` on the host's clock, each step given [`DEADLINE`], to the
/// end of the writer's work.
pub(super) fn run(session: &Session) -> Steps {
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
    let started = Instant::now();
    let mut clock = move || started.elapsed().as_nanos() as u64;
    let mut writer = Writer::new(file, NANOS);
    let mut store = Store {
        added: BTreeMap::new(),
        bytes: 0,
        total: 0,
        batch: session.batch as usize,
        uses: BTreeMap::new(),
        shedding: false,
        shed: 0,
    };
    let mut steps = Steps::default();
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
                store.add(page, 64, session.entry_bytes);
            }
            for _ in repeats..session.batch.min(session.records - next) {
                if session.again != 0 && next.is_multiple_of(session.again) {
                    again.push_back(next);
                }
                store.add(next, 0, session.entry_bytes);
                next += 1;
            }
        }
        let idle = writer.idle();
        let from = clock();
        let events = writer
            .step(&mut clock, from + DEADLINE, &mut store, &mut |execution| {
                Some(execution.advance(&mut Discard, PIECE_BYTES))
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
    }
    steps.stats = writer.stats();
    steps.shed = store.shed;
    steps
}

/// A long session's steps: run 20261120's boot wrote 519 435 records in 80
/// segments of 3 to 24 thousand records (about 8 KiB each), merged in full
/// up to 20 thousand records and 354 MB, in an arena of 670 MB it compacted
/// 12 times. Here 500 thousand records of 2 KiB in segments of 15 thousand,
/// and 60 thousand of 8 KiB, each in an arena that fills and a file whose
/// full merges leave records out: what each step took against its 1 ms
/// deadline, and its parts. A release build on a quiet host gives the
/// boot's figures.
#[test]
#[ignore = "manual: opt-in"]
fn step_costs_of_a_long_session() {
    let many = Session {
        records: 500_000,
        batch: 15_000,
        entry_bytes: 2 << 10,
        again: 8,
        arena: 256 << 20,
        file: DATA_START + (2 << 30),
        limit: 2 << 30,
    };
    run(&many).report("many");
    let large = Session {
        records: 60_000,
        batch: 15_000,
        entry_bytes: 8 << 10,
        again: 4,
        arena: 256 << 20,
        file: DATA_START + (1 << 30),
        limit: 2 << 30,
    };
    run(&large).report("large");
}
