//! Host cost of one transfer through each path an exit out of its block
//! takes to a block in another page: a stamped tail, a cell tail whose
//! cell holds the target, a cell tail whose cell holds nothing (its probe
//! goes on to the table probe), and a lookup tail through its jump cache
//! or the table probe for one, a few or many targets, in one native run
//! or in runs of a few dozen transfers as the guest's are. Run by hand: it
//! prints nanoseconds and TSC ticks per transfer, less a loop whose
//! transfer is a direct jump, and asserts only the counts.
use super::super::*;
use super::gate;
use crate::GuestState;
use alloc::{boxed::Box, vec::Vec};
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
    fn munmap(address: *mut c_void, bytes: usize) -> i32;
}

const REGION: usize = 1 << 16;
/// The loop at 0, the tail under test, the probes of BTYPE 0, then the
/// target blocks, [`BLOCK_BYTES`] apart.
const TAIL: usize = 1024;
const PROBES: usize = 4096;
const BLOCKS: usize = 8192;
const BLOCK_BYTES: usize = 32;
const MOST_TARGETS: usize = (REGION - BLOCKS) / BLOCK_BYTES;
/// The target sequence's length, a power of two.
const SEQUENCE: usize = 1 << 12;
const RAN: u32 = 3;
/// The replacement rate of the probes the cases run ([`JUMP_REPLACE`]),
/// which [`jump_replace_costs`] sets.
static REPLACE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(JUMP_REPLACE);

/// The probes of BTYPE 0 the cases run, of [`REPLACE`]'s rate.
fn probes() -> Vec<u8> {
    let replace = REPLACE.load(core::sync::atomic::Ordering::Relaxed);
    dispatcher_with(0, Interruption::Quiet, replace).unwrap()
}
const CONTEXTS: [u64; 2] = [5 << LOOKUP_SALT_BITS, 9 << LOOKUP_SALT_BITS];
const KEY: u64 = block_key(0x8_0040_0000, 1);
/// Guest PCs of the targets: one function entry every 0x44 bytes of a
/// library's text in the upper half, 4-byte aligned as A64 code is.
const fn target_pc(target: usize) -> u64 {
    0xffff_fe00_1234_0000 + 0x44 * target as u64
}

struct Pages(*mut u8, usize);
impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: our mapping, nothing runs it any more.
        assert_eq!(unsafe { munmap(self.0.cast(), self.1) }, 0);
    }
}

fn put(pages: &Pages, offset: usize, bytes: &[u8]) {
    assert!(offset + bytes.len() <= pages.1);
    // SAFETY: inside the mapping; nothing runs it while it is written.
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), pages.0.add(offset), bytes.len()) };
}

/// Where the target of each transfer comes from.
#[derive(Clone, Copy)]
enum Target {
    /// One PC, as a direct exit's `movabs rax`.
    Fixed(u64),
    /// The next of [`SEQUENCE`] PCs at X1, X0 counting them.
    Sequence,
}

/// A mapping for [`fill`].
fn mapped() -> Pages {
    mapped_bytes(REGION)
}

/// A mapping of `bytes`.
fn mapped_bytes(bytes: usize) -> Pages {
    // SAFETY: private anonymous memory, unmapped on drop.
    let base = unsafe { mmap(core::ptr::null_mut(), bytes, 7, 0x22, -1, 0) };
    assert_ne!(base as usize, usize::MAX);
    Pages(base.cast(), bytes)
}

/// The loop in `region`: RSI transfers to the tail with RAX the target,
/// each entered block jumping back; then RAX 1 and EDX 2 return. `direct`
/// replaces the tail with a jump to the first block, the baseline.
fn fill(region: &Pages, target: Target, tail: &[u8], direct: bool) {
    put(region, 0, &[0xcc; REGION]);
    let at = region.0 as u64;
    let mut code = Vec::new();
    // sub rsi, 1; je done (rel32)
    code.extend([0x48, 0x83, 0xee, 0x01, 0x0f, 0x84, 0, 0, 0, 0]);
    let done = code.len();
    match target {
        Target::Fixed(pc) => {
            code.extend([0x48, 0xb8]);
            code.extend(pc.to_le_bytes());
        }
        Target::Sequence => {
            // mov rcx,[rdi]; add rcx,1; and ecx,SEQUENCE-1; mov [rdi],rcx;
            // mov r8,[rdi+8]; mov rax,[r8+rcx*8]
            code.extend([0x48, 0x8b, 0x0f, 0x48, 0x83, 0xc1, 0x01, 0x81, 0xe1]);
            code.extend(((SEQUENCE - 1) as u32).to_le_bytes());
            code.extend([
                0x48, 0x89, 0x0f, 0x4c, 0x8b, 0x47, 0x08, 0x49, 0x8b, 0x04, 0xc8,
            ]);
        }
    }
    let to = if direct { BLOCKS } else { TAIL };
    code.push(0xe9);
    let end = at + code.len() as u64 + 4;
    code.extend(((at + to as u64).wrapping_sub(end) as i32).to_le_bytes());
    let exit = code.len();
    code[done - 4..done].copy_from_slice(&((exit - done) as i32).to_le_bytes());
    // mov eax,1; mov edx,2; then the exit's return.
    code.extend([0xb8, 1, 0, 0, 0, 0xba, 2, 0, 0, 0]);
    code.extend(EXIT_RETURN);
    put(region, 0, &code);
    if !direct {
        put(region, TAIL, tail);
    }
    put(region, PROBES, &probes());
    for block in 0..MOST_TARGETS {
        let offset = BLOCKS + BLOCK_BYTES * block;
        let end = at + offset as u64 + 5;
        let mut jump = vec![0xe9];
        jump.extend((at.wrapping_sub(end) as i32).to_le_bytes());
        put(region, offset, &jump);
    }
}

/// A board's tables with `targets` blocks in the region, each in the
/// table and its page in the ITLB, and a control; each native run's budget
/// outlasts the transfers its loop makes.
struct World {
    table: LookupTable,
    _itlb: LookupTable,
    control: Box<ChainControl>,
}

impl World {
    fn new(region: u64, targets: usize) -> Self {
        Self::with(targets, |target| entry(region, target))
    }

    /// A world whose `targets` blocks are where `entry` says.
    fn with(targets: usize, entry: impl Fn(usize) -> LookupEntry) -> Self {
        let mut table = LookupTable::new();
        let mut itlb = LookupTable::itlb();
        let cells = table.stamped_cells();
        let stamps = table.new_stamps();
        for target in 0..targets {
            let pc = target_pc(target);
            itlb.insert(
                itlb.page_slot(pc, CONTEXTS[1]),
                LookupEntry::page(pc & !0xfff, CONTEXTS[1], KEY),
            );
            table.insert(table.slot(pc, KEY), entry(target));
        }
        let control = Box::new(ChainControl {
            table: table.address(),
            itlb: itlb.address(),
            contexts: CONTEXTS,
            salts: CONTEXTS.map(|context| itlb.salt(context)),
            masks: [table.mask(), itlb.mask()],
            stamps,
            cells,
            ..ChainControl::default()
        });
        World {
            table,
            _itlb: itlb,
            control,
        }
    }

    /// Enter the loop for `transfers` transfers, in native runs of at most
    /// `run` transfers and fewer than [`MAX_BUDGET`], each entered as the
    /// engine enters native code: TSC ticks taken.
    fn run(&mut self, region: &Pages, sequence: &[u64], transfers: u64, run: u64) -> u64 {
        let mut state = Box::new(GuestState::default());
        state.anchors[Anchor::Control] = &raw mut *self.control as u64;
        state.x[1] = sequence.as_ptr() as u64;
        let start = time();
        let mut left = transfers;
        while left > 0 {
            let now = left.min(run).min(MAX_BUDGET - 1);
            left -= now;
            state.account = Account::new(MAX_BUDGET).0;
            // SAFETY: the region holds the loop, a tail, probes and blocks
            // with the block ABI; they touch the state, the control, the
            // tables, the stamps and the sequence, which outlive the call.
            let result = unsafe {
                crate::x86_64::native::enter_counted(region.0 as *const u8, &mut *state, now + 1, 0)
            };
            assert_eq!((result as u64, (result >> 64) as u64), (1, 2));
        }
        time() - start
    }
}

fn entry(region: u64, target: usize) -> LookupEntry {
    LookupEntry {
        pc: target_pc(target),
        key: KEY,
        entry: region + (BLOCKS + BLOCK_BYTES * target) as u64,
        block: 10 + target as u32,
        targets: 0b1111,
    }
}

fn time() -> u64 {
    // SAFETY: RDTSC reads the time stamp counter only.
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// [`SEQUENCE`] PCs of `targets` targets in a pseudo-random order, which no
/// predictor learns.
fn sequence(targets: usize) -> Vec<u64> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    (0..SEQUENCE)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            target_pc((state >> 20) as usize % targets)
        })
        .collect()
}

/// The path a case's tail takes.
#[derive(Clone, Copy, Debug)]
enum Path {
    /// A hot exit's stamped tail, through a stamped cell.
    Stamped,
    /// A cell tail through a cell past the stamped ones, as every exit's
    /// is until its transfers make it hot: the census's cell probes.
    Cell,
    /// A cell tail whose cell holds no entry.
    EmptyCell,
    /// A lookup tail of this many targets in one native run.
    Lookup(usize),
    /// One of this many targets in native runs of this many transfers.
    LookupRuns(usize, u64),
    /// A predicted tail of a BR or BLR site whose prediction never holds,
    /// over this many targets: its miss's call of the probe at this entry
    /// of the dispatcher, as at a site of more targets than the two it
    /// predicts (a vtable call, an `objc_msgSend` tail): the jump probe
    /// ([`JUMP_PROBE_ENTRY`]), or the table probe, which every such miss
    /// took before.
    PredictedMiss(usize, usize),
}

/// Nanoseconds and TSC ticks of one transfer through `path`, less the
/// direct jump's, and the table probe transfers it made per transfer.
fn cost(path: Path) -> (f64, f64, f64) {
    const TRANSFERS: u64 = 1 << 22;
    let pc = target_pc(0);
    let (target, targets, run) = match path {
        Path::Lookup(targets) | Path::PredictedMiss(targets, _) => {
            (Target::Sequence, targets, TRANSFERS)
        }
        Path::LookupRuns(targets, run) => (Target::Sequence, targets, run),
        _ => (Target::Fixed(pc), 1, TRANSFERS),
    };
    let sequence = sequence(targets);
    let word = ExitWord::from(match path {
        Path::Lookup(_) | Path::LookupRuns(..) | Path::PredictedMiss(..) => ExitKind::Indirect,
        _ => ExitKind::Direct,
    });
    // The region's address first: tails jump relative to it.
    let (measured, direct) = (mapped(), mapped());
    let base = measured.0 as u64;
    let mut world = World::new(base, targets);
    let at = base + TAIL as u64;
    let site = 0x1000;
    let tail: Vec<u8> = match path {
        Path::Stamped | Path::Cell => {
            let held = world.table.find(world.table.slot(pc, KEY), pc, KEY);
            match path {
                Path::Stamped => world.table.link_hot_cell(site, pc, Some(held)).unwrap(),
                _ => world.table.link_cell(site, pc, Some(held)),
            };
            let number = world.table.site_cell_number(site).unwrap();
            let cell = LookupTable::cell_slot(&world.table.cells, number) as *const Slot as u64;
            match path {
                Path::Stamped => {
                    world.table.note_stamped(number);
                    let stamping = base + (PROBES + STAMPING_PROBE_ENTRY) as u64;
                    stamped_tail(
                        at, cell, number, pc, word, RAN, held.block, held.entry, stamping,
                    )
                    .unwrap()
                    .to_vec()
                }
                _ => {
                    let probe = base + (PROBES + CELL_PROBE_ENTRY) as u64;
                    cell_tail(at, cell, pc, word, RAN, probe).unwrap().to_vec()
                }
            }
        }
        Path::EmptyCell => {
            world.table.link_cell(site, pc, None);
            let number = world.table.site_cell_number(site).unwrap();
            let cell = LookupTable::cell_slot(&world.table.cells, number) as *const Slot as u64;
            let probe = base + (PROBES + CELL_PROBE_ENTRY) as u64;
            cell_tail(at, cell, pc, word, RAN, probe).unwrap().to_vec()
        }
        Path::Lookup(_) | Path::LookupRuns(..) => {
            let probe = base + (PROBES + JUMP_PROBE_ENTRY) as u64;
            lookup_tail(at, word, RAN, probe, 0).unwrap().to_vec()
        }
        Path::PredictedMiss(_, entry) => {
            // A prediction no target equals: its direct exit never runs.
            let never = GuestAddress(target_pc(MOST_TARGETS));
            let probe = base + (PROBES + entry) as u64;
            predicted_tail(at, never, word, RAN, probe)
                .unwrap()
                .to_vec()
        }
    };
    fill(&measured, target, &tail, false);
    // The baseline: the same loop, whose transfer is a direct jump.
    fill(&direct, target, &tail, true);
    let filled = |world: &World| {
        // SAFETY: the table's stamps, which live while it does.
        let counts = unsafe { stamp_counts(world.control.stamps) };
        counts[StampCount::Filled as usize]
    };
    // Warm: the stamp, the jump cache, the predictors.
    world.run(&measured, &sequence, 1 << 16, run);
    world.run(&direct, &sequence, 1 << 16, run);
    let (mut best, mut baseline) = (u64::MAX, u64::MAX);
    let mut table_probes = 0;
    for _ in 0..7 {
        let before = filled(&world);
        best = best.min(world.run(&measured, &sequence, TRANSFERS, run));
        table_probes = filled(&world) - before;
        baseline = baseline.min(world.run(&direct, &sequence, TRANSFERS, run));
    }
    let ticks = best.saturating_sub(baseline) as f64 / TRANSFERS as f64;
    let hz = tsc_hz();
    (
        ticks * 1e9 / hz,
        ticks,
        table_probes as f64 / TRANSFERS as f64,
    )
}

/// The TSC's rate, from a 50 ms sleep.
fn tsc_hz() -> f64 {
    let (start, at) = (std::time::Instant::now(), time());
    std::thread::sleep(std::time::Duration::from_millis(50));
    (time() - at) as f64 / start.elapsed().as_secs_f64()
}

#[test]
#[ignore = "manual: opt-in"]
fn transfer_path_costs() {
    for path in [
        Path::Stamped,
        Path::Cell,
        Path::EmptyCell,
        Path::Lookup(1),
        Path::Lookup(2),
        Path::Lookup(4),
        Path::Lookup(64),
        Path::Lookup(1024),
        Path::LookupRuns(1, 32),
        Path::LookupRuns(16, 32),
        Path::LookupRuns(64, 32),
        Path::PredictedMiss(4, PROBE_ENTRY),
        Path::PredictedMiss(4, JUMP_PROBE_ENTRY),
        Path::PredictedMiss(64, PROBE_ENTRY),
        Path::PredictedMiss(64, JUMP_PROBE_ENTRY),
        Path::PredictedMiss(1024, PROBE_ENTRY),
        Path::PredictedMiss(1024, JUMP_PROBE_ENTRY),
    ] {
        let (ns, ticks, table) = cost(path);
        std::println!(
            "{path:?}: {ns:.2} ns, {ticks:.1} TSC ticks, {table:.2} table probes per transfer"
        );
    }
}

/// How each block of a [`spread_cost`] ring goes on to the next.
#[derive(Clone, Copy, Debug)]
enum Ring {
    /// `sub rsi,1; je out; jmp next`: the code the ring touches without a
    /// transfer path, the baseline.
    Direct,
    /// A linked slot, as a direct exit to a block in its page.
    Link,
    /// A thunk at the exit's patch point, as a direct exit out of its page
    /// linked to a translated block.
    Thunk,
    /// A hot exit's stamped tail, its stamp written.
    Stamped,
    /// A cell tail, through the cell probe.
    Cell,
    /// An indirect exit to the target in RAX, one per site, through its
    /// lookup tail and the jump probe: this PE's jump cache, or the table
    /// probe.
    Lookup,
    /// [`Ring::Lookup`] whose tail checks the jump cache itself, as every
    /// lookup tail did before the jump probe ([`inline_lookup_tail`]).
    InlineLookup,
    /// An indirect exit through a predicted tail of its one target, whose
    /// direct exit is linked through a stamped tail.
    Predicted,
    /// [`Ring::Lookup`] of sites that each take one of two targets at
    /// random: the next block or the one after it.
    PolyLookup,
    /// [`Ring::Predicted`] of such sites, predicting the next block.
    PolyPredicted,
    /// [`Ring::PolyPredicted`] with the second target predicted too, as
    /// the engine makes a site's first miss its second target.
    PolyPredicted2,
    /// A small switch's BR: sites that each take one of four targets at
    /// random (the next block and the three after it), two of them
    /// predicted as [`Ring::PolyPredicted2`]'s, the others the second
    /// predicted tail's miss, which calls the jump probe.
    Switch,
    /// [`Ring::Switch`] whose miss calls the table probe, as every such
    /// miss did before the jump probe.
    SwitchTable,
}

/// Bytes of each block's tails: a predicted tail and a stamped tail. Not
/// a power of two, so the tails' branches do not share predictor sets.
const RING_TAIL: usize = 272;
/// Where a predicted tail's stamped tail lies in the block's tails.
const RING_STAMPED: usize = 80;
const _: () = assert!(PREDICTED_TAIL_BYTES <= RING_STAMPED);
const _: () = assert!(RING_STAMPED + STAMPED_TAIL_BYTES <= RING_TAIL);
const _: () = assert!(LOOKUP_TAIL_BYTES <= RING_TAIL);

/// Bytes of each block of a ring: its exit and nothing else.
const RING_BLOCK: usize = 80;
/// The ring's blocks start here, after the return at 0 and the probes.
const RING_BLOCKS: usize = 1 << 16;

/// TSC ticks per transfer around a ring of `blocks` blocks in a
/// pseudo-random order, each going on to the next through `ring`: one
/// direct exit per block whose code, tail and stamp are each spread over
/// the code and the stamps as a guest's many hot exits are, which the
/// one-exit [`cost`] loop keeps in its caches. With the stamp counts each
/// run added per transfer.
fn spread_cost(ring: Ring, blocks: usize) -> (f64, [f64; STAMP_COUNTS]) {
    const TRANSFERS: u64 = 1 << 22;
    const RUN: u64 = 1 << 12;
    // A switch's block picks among four targets: twice the bytes.
    let block_bytes = match ring {
        Ring::Switch | Ring::SwitchTable => 2 * RING_BLOCK,
        _ => RING_BLOCK,
    };
    let tails = RING_BLOCKS + blocks * block_bytes;
    let stride = match ring {
        Ring::PolyPredicted2 | Ring::Switch | Ring::SwitchTable => 2 * RING_TAIL,
        _ => RING_TAIL,
    };
    let bytes = (tails + blocks * stride).next_multiple_of(4096);
    let region = mapped_bytes(bytes);
    put(&region, 0, &vec![0xcc; bytes]);
    let base = region.0 as u64;
    let block_at = |block: usize| base + (RING_BLOCKS + block * block_bytes) as u64;
    let tail_at = |block: usize| base + (tails + block * stride) as u64;
    // The return the exhausted budget takes.
    let mut out = vec![0xb8, 1, 0, 0, 0, 0xba, 2, 0, 0, 0];
    out.extend(EXIT_RETURN);
    put(&region, 0, &out);
    put(&region, PROBES, &probes());
    // The ring: a random cycle through every block, from block 0.
    let mut order: Vec<usize> = (1..blocks).collect();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for at in (1..order.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        order.swap(at, (state % (at as u64 + 1)) as usize);
    }
    let mut next = vec![0; blocks];
    let mut from = 0;
    for &block in &order {
        next[from] = block;
        from = block;
    }
    next[from] = 0;
    let mut world = World::with(blocks, |target| LookupEntry {
        entry: block_at(target),
        ..entry(base, target)
    });
    let word = ExitWord::from(ExitKind::Direct);
    for (block, &to) in next.iter().enumerate() {
        let at = block_at(block);
        let (pc, entry) = (target_pc(to), block_at(to));
        let tail = tail_at(block);
        let site = 0x1000 + 0x10 * block as u64;
        let held = world.table.find(world.table.slot(pc, KEY), pc, KEY);
        assert_eq!(held.entry, entry);
        let (code, tail_code) = match ring {
            Ring::Direct => {
                // sub rsi,1; je out; jmp next
                let mut code = vec![0x48, 0x83, 0xee, 0x01, 0x0f, 0x84];
                code.extend((base.wrapping_sub(at + 10) as i32).to_le_bytes());
                code.extend(jump(at + 10, entry).unwrap());
                (code, None)
            }
            Ring::Link => {
                let linking = link(at, LINK_BYTES, RAN, held.block, entry, gate()).unwrap();
                let point = (linking.rest_at - PATCH_BYTES as u64 - at) as usize;
                let mut code = vec![0x90; point];
                code.extend(linking.body());
                assert_eq!(code.len(), LINK_BYTES);
                code.extend(jump(at + LINK_BYTES as u64, base).unwrap());
                (code, None)
            }
            Ring::Thunk => (
                jump(at, tail).unwrap().to_vec(),
                Some(thunk(tail, RAN, held.block, entry, base, gate()).unwrap()),
            ),
            Ring::Stamped => {
                world.table.link_hot_cell(site, pc, Some(held)).unwrap();
                let number = world.table.site_cell_number(site).unwrap();
                world.table.note_stamped(number);
                let cell = LookupTable::cell_slot(&world.table.cells, number) as *const Slot as u64;
                let stamping = base + (PROBES + STAMPING_PROBE_ENTRY) as u64;
                let code = stamped_tail(
                    tail, cell, number, pc, word, RAN, held.block, entry, stamping,
                )
                .unwrap();
                (jump(at, tail).unwrap().to_vec(), Some(code.to_vec()))
            }
            Ring::Cell => {
                world.table.link_cell(site, pc, Some(held));
                let number = world.table.site_cell_number(site).unwrap();
                let cell = LookupTable::cell_slot(&world.table.cells, number) as *const Slot as u64;
                let probe = base + (PROBES + CELL_PROBE_ENTRY) as u64;
                let code = cell_tail(tail, cell, pc, word, RAN, probe).unwrap();
                (jump(at, tail).unwrap().to_vec(), Some(code.to_vec()))
            }
            Ring::Lookup | Ring::PolyLookup => {
                let probe = base + (PROBES + JUMP_PROBE_ENTRY) as u64;
                let code = lookup_tail(tail, ExitKind::Indirect, RAN, probe, 0).unwrap();
                let jump = match ring {
                    Ring::Lookup => indirect_jump(at, pc, tail),
                    _ => either_jump(at, pc, target_pc(next[to]), tail),
                };
                (jump, Some(code.to_vec()))
            }
            Ring::InlineLookup => {
                let probe = base + (PROBES + PROBE_ENTRY) as u64;
                let code = inline_lookup_tail(tail, ExitKind::Indirect.into(), RAN, probe);
                (indirect_jump(at, pc, tail), Some(code))
            }
            Ring::Predicted | Ring::PolyPredicted => {
                let code = predicted(&mut world, base, tail, held, site, None, JUMP_PROBE_ENTRY);
                let jump = match ring {
                    Ring::Predicted => indirect_jump(at, pc, tail),
                    _ => either_jump(at, pc, target_pc(next[to]), tail),
                };
                (jump, Some(code))
            }
            Ring::PolyPredicted2 | Ring::Switch | Ring::SwitchTable => {
                let probe = match ring {
                    Ring::SwitchTable => PROBE_ENTRY,
                    _ => JUMP_PROBE_ENTRY,
                };
                let other = target_pc(next[to]);
                let second = tail + RING_TAIL as u64;
                let mut code = predicted(&mut world, base, tail, held, site, Some(second), probe);
                let held = world.table.find(world.table.slot(other, KEY), other, KEY);
                code.resize(RING_TAIL, 0xcc);
                code.extend(predicted(
                    &mut world,
                    base,
                    second,
                    held,
                    site + 8,
                    None,
                    probe,
                ));
                let jump = match ring {
                    Ring::PolyPredicted2 => either_jump(at, pc, other, tail),
                    _ => {
                        let third = next[next[to]];
                        let targets = [to, next[to], third, next[third]].map(target_pc);
                        any_jump(at, targets, tail)
                    }
                };
                (jump, Some(code))
            }
        };
        assert!(code.len() <= block_bytes);
        put(&region, RING_BLOCKS + block * block_bytes, &code);
        if let Some(tail_code) = tail_code {
            put(&region, tails + block * stride, &tail_code);
        }
    }
    let mut guest = Box::new(GuestState::default());
    guest.anchors[Anchor::Control] = &raw mut *world.control as u64;
    let control = &raw mut *world.control;
    let mut run = |transfers: u64| {
        let start = time();
        let mut left = transfers;
        while left > 0 {
            let now = left.min(RUN);
            left -= now;
            guest.account = Account::new(now).0;
            // SAFETY: the region holds the ring, its tails and probes with
            // the block ABI; they touch the state, the control and the
            // tables, which outlive the call; the budget or RSI ends it.
            unsafe {
                crate::x86_64::native::enter_counted(block_at(0) as *const u8, &mut *guest, now, 0)
            };
        }
        time() - start
    };
    // SAFETY: the table's stamps, which live while it does.
    let counts = || unsafe { stamp_counts((*control).stamps) };
    // Warm: every stamp written, the predictors trained.
    run(8 * blocks as u64 + (1 << 16));
    let mut best = u64::MAX;
    let mut per = [0.0; STAMP_COUNTS];
    for _ in 0..5 {
        let before = counts();
        best = best.min(run(TRANSFERS));
        let after = counts();
        per =
            core::array::from_fn(|count| (after[count] - before[count]) as f64 / TRANSFERS as f64);
    }
    (best as f64 / TRANSFERS as f64, per)
}

/// A predicted tail at `tail` of `held`'s PC, and at [`RING_STAMPED`] after
/// it the stamped tail its direct exit is linked to through the stamped
/// cell of `site`: its miss calls the probe at `entry` of the dispatcher
/// (the engine's the jump probe), or for an open tail jumps to `miss`.
fn predicted(
    world: &mut World,
    base: u64,
    tail: u64,
    held: LookupEntry,
    site: u64,
    miss: Option<u64>,
    entry: usize,
) -> Vec<u8> {
    let (pc, word) = (held.pc, ExitKind::Indirect);
    let probe = base + (PROBES + entry) as u64;
    let mut code = match miss {
        None => predicted_tail(tail, GuestAddress(pc), word, RAN, probe).unwrap(),
        Some(_) => open_predicted_tail(tail, GuestAddress(pc), word).unwrap(),
    }
    .to_vec();
    world.table.link_hot_cell(site, pc, Some(held)).unwrap();
    let number = world.table.site_cell_number(site).unwrap();
    world.table.note_stamped(number);
    let cell = LookupTable::cell_slot(&world.table.cells, number) as *const Slot as u64;
    let stamping = base + (PROBES + STAMPING_PROBE_ENTRY) as u64;
    let stamped_at = tail + RING_STAMPED as u64;
    let stamped = stamped_tail(
        stamped_at, cell, number, pc, word, RAN, held.block, held.entry, stamping,
    )
    .unwrap();
    let mut patch = |point: u64, to: u64| {
        let offset = (point - tail) as usize;
        code[offset..offset + PATCH_BYTES].copy_from_slice(&jump(point, to).unwrap());
    };
    patch(direct_patch(tail + PREDICTED_EXIT as u64), stamped_at);
    if let Some(miss) = miss {
        patch(dynamic_patch(tail + PREDICTED_MISS as u64), miss);
    }
    code.resize(RING_STAMPED, 0xcc);
    code.extend(stamped);
    code
}

/// `movabs rax, pc; jmp tail` at `at`: a block whose indirect branch's
/// target is `pc`, its exit's point linked to `tail`.
fn indirect_jump(at: u64, pc: u64, tail: u64) -> Vec<u8> {
    let mut code = vec![0x48, 0xb8];
    code.extend(pc.to_le_bytes());
    code.extend(jump(at + 10, tail).unwrap());
    code
}

/// A block at `at` whose indirect branch takes `first` or `second` at
/// random, by a xorshift of RSI, its exit's point linked to `tail`.
fn either_jump(at: u64, first: u64, second: u64, tail: u64) -> Vec<u8> {
    let mut code = Vec::new();
    for (shift, op) in [(13u8, 4u8), (7, 5), (17, 4)] {
        // mov rax, rsi; shl/shr rax, shift; xor rsi, rax
        code.extend([0x48, 0x89, 0xf0, 0x48, 0xc1, 0xe0 | op << 3 & 0x38, shift]);
        code.extend([0x48, 0x31, 0xc6]);
    }
    // bt esi, 20; movabs rax, first; movabs r8, second; cmovc rax, r8
    code.extend([0x0f, 0xba, 0xe6, 20, 0x48, 0xb8]);
    code.extend(first.to_le_bytes());
    code.extend([0x49, 0xb8]);
    code.extend(second.to_le_bytes());
    code.extend([0x49, 0x0f, 0x42, 0xc0]);
    let end = at + code.len() as u64;
    code.extend(jump(end, tail).unwrap());
    code
}

/// The lookup tail at `at` of a branch that sets BTYPE 0 as every one was
/// before the jump probe: the jump cache's check in the tail, its hit
/// entered by the tail's own `jmp [entry]`, and any other through the
/// plain tail's call of the table probe at `probe`; 224 bytes with its pad.
fn inline_lookup_tail(at: u64, word: ExitWord, ran: u32, probe: u64) -> Vec<u8> {
    let mut code = Code::new();
    let mut asm = code.asm();
    asm.place(Rcx, CONTROL);
    let mut misses = Vec::new();
    let entry = jump_cache_check(&mut asm, 0, &mut misses);
    misses.push(stop_check(&mut asm, Interruption::Quiet, true));
    count(&mut asm, transfer_change(ran).unwrap());
    asm.alu_imm8(Alu::Add, Qword, Mem::base(Rcx, HITS), 1);
    asm.mov_load(Dword, R8, entry(JUMP_BLOCK));
    asm.mov_store(Qword, Mem::base(Rcx, BLOCK), R8);
    asm.jmp_rm(entry(JUMP_CODE));
    for miss in misses {
        asm.bind(miss);
    }
    super::super::tail(&mut code, at, word, ran, probe).unwrap();
    let mut bytes = code.into_bytes();
    assert!(bytes.len() <= 224);
    bytes.resize(224, 0xcc);
    bytes
}

/// A block at `at` whose indirect branch takes one of `targets` at random,
/// by bits 20 and 21 of a xorshift of RSI, its exit's point linked to
/// `tail`: a switch's BR.
fn any_jump(at: u64, targets: [u64; 4], tail: u64) -> Vec<u8> {
    let mut code = Vec::new();
    for (shift, op) in [(13u8, 4u8), (7, 5), (17, 4)] {
        // mov rax, rsi; shl/shr rax, shift; xor rsi, rax
        code.extend([0x48, 0x89, 0xf0, 0x48, 0xc1, 0xe0 | op << 3 & 0x38, shift]);
        code.extend([0x48, 0x31, 0xc6]);
    }
    // movabs rax, t0; movabs r8, t1; bt esi, 20; cmovc rax, r8
    code.extend([0x48, 0xb8]);
    code.extend(targets[0].to_le_bytes());
    code.extend([0x49, 0xb8]);
    code.extend(targets[1].to_le_bytes());
    code.extend([0x0f, 0xba, 0xe6, 20, 0x49, 0x0f, 0x42, 0xc0]);
    // movabs r8, t2; movabs r9, t3; cmovc r8, r9 (bit 20 still in CF)
    code.extend([0x49, 0xb8]);
    code.extend(targets[2].to_le_bytes());
    code.extend([0x49, 0xb9]);
    code.extend(targets[3].to_le_bytes());
    code.extend([0x4d, 0x0f, 0x42, 0xc1]);
    // bt esi, 21; cmovc rax, r8
    code.extend([0x0f, 0xba, 0xe6, 21, 0x49, 0x0f, 0x42, 0xc0]);
    let end = at + code.len() as u64;
    code.extend(jump(end, tail).unwrap());
    code
}

#[test]
#[ignore = "manual: opt-in"]
fn spread_transfer_costs() {
    for blocks in [64, 1024, 4096, 16384] {
        for ring in [
            Ring::Direct,
            Ring::Link,
            Ring::Thunk,
            Ring::Stamped,
            Ring::Cell,
            Ring::Lookup,
            Ring::Predicted,
            Ring::PolyLookup,
            Ring::PolyPredicted,
            Ring::PolyPredicted2,
        ] {
            let (ticks, counts) = spread_cost(ring, blocks);
            std::println!(
                "{blocks} blocks, {ring:?}: {ticks:.1} TSC ticks per transfer, stamp counts {counts:.2?}"
            );
        }
    }
}

/// The jump probe against what it replaced, side by side: lookup tails
/// that check the jump cache themselves ([`Ring::InlineLookup`]) or call
/// the jump probe ([`Ring::Lookup`]), and a small switch's misses through
/// the table probe ([`Ring::SwitchTable`]) or the jump probe
/// ([`Ring::Switch`]). Each pair is measured in turn three times and the
/// best of each kept, as the host's other load moves between runs.
#[test]
#[ignore = "manual: opt-in"]
fn jump_probe_spread_costs() {
    for blocks in [64, 256, 1024, 4096, 16384] {
        for pair in [
            [Ring::InlineLookup, Ring::Lookup],
            [Ring::SwitchTable, Ring::Switch],
        ] {
            let mut best = [f64::MAX; 2];
            for _ in 0..3 {
                for (at, ring) in pair.into_iter().enumerate() {
                    best[at] = best[at].min(spread_cost(ring, blocks).0);
                }
            }
            std::println!(
                "{blocks} blocks: {:?} {:.1}, {:?} {:.1} TSC ticks per transfer",
                pair[0],
                best[0],
                pair[1],
                best[1]
            );
        }
    }
}

/// A table probe transfer replacing another PC's jump cache entry at
/// every transfer (as before [`JUMP_REPLACE`]) and at one in four, side by
/// side: lookups over more targets than the cache holds, a predicted
/// tail's misses over many targets and lookup sites spread over many
/// blocks. Each pair is measured in turn three times and the best of each
/// kept.
#[test]
#[ignore = "manual: opt-in"]
fn jump_replace_costs() {
    let rates = [1, JUMP_REPLACE];
    let set = |replace| REPLACE.store(replace, core::sync::atomic::Ordering::Relaxed);
    for path in [
        Path::Lookup(64),
        Path::Lookup(256),
        Path::Lookup(1024),
        Path::LookupRuns(1024, 32),
        Path::PredictedMiss(1024, JUMP_PROBE_ENTRY),
    ] {
        let mut best = [(f64::MAX, 0.0); 2];
        for _ in 0..3 {
            for (at, replace) in rates.into_iter().enumerate() {
                set(replace);
                let (ns, _, table) = cost(path);
                if ns < best[at].0 {
                    best[at] = (ns, table);
                }
            }
        }
        std::println!(
            "{path:?}: every miss {:.2} ns ({:.2} table probes), one in {JUMP_REPLACE} {:.2} ns ({:.2})",
            best[0].0,
            best[0].1,
            best[1].0,
            best[1].1
        );
    }
    for blocks in [1024, 4096, 16384] {
        let mut best = [f64::MAX; 2];
        for _ in 0..3 {
            for (at, replace) in rates.into_iter().enumerate() {
                set(replace);
                best[at] = best[at].min(spread_cost(Ring::Lookup, blocks).0);
            }
        }
        std::println!(
            "{blocks} lookup sites: every miss {:.1}, one in {JUMP_REPLACE} {:.1} TSC ticks per transfer",
            best[0],
            best[1]
        );
    }
    set(JUMP_REPLACE);
}
