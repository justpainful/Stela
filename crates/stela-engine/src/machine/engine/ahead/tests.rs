use super::super::stored::tests::MemoryStore;
use super::super::tests::{BASE, machine};
use super::super::{Fetched, StoreEvent};
use super::*;
use crate::machine::test_preparation;

/// `cbz x0, #8; add x0, x0, #1; svc #0`: the first block ends at the CBZ,
/// whose successors are the ADD and the SVC, both in its page.
const WORDS: [u32; 3] = [0xb400_0040, 0x9100_0400, 0xd400_0001];

fn job(pc: u64) -> AheadJob {
    let preparation = Arc::new(test_preparation());
    let page = pc & !(FETCH_PAGE - 1);
    let mut bytes = [0u8; PAGE_BYTES];
    bytes[..4].copy_from_slice(&0xd65f_03c0u32.to_le_bytes());
    let context = preparation.context(page);
    AheadJob::new(
        Digest([7; 32]),
        page,
        pc,
        context,
        Arc::new(bytes),
        preparation,
    )
    .unwrap()
}

#[test]
fn the_queue_is_bounded_dedupes_what_it_holds_and_never_waits() {
    let queue = AheadQueue::<2>::new();
    let [first, second, third] = [0x4000, 0x4004, 0x4008].map(job);
    assert_eq!(queue.push(first.clone()), Pushed::Queued);
    assert_eq!(queue.push(first.clone()), Pushed::Repeated);
    assert_eq!(queue.push(second.clone()), Pushed::Queued);
    assert_eq!(queue.push(third.clone()), Pushed::Dropped, "full");
    // A worker holding the queue makes a PE drop, not wait.
    queue.held.store(true, Ordering::Relaxed);
    assert_eq!(queue.push(third.clone()), Pushed::Dropped);
    queue.held.store(false, Ordering::Relaxed);
    let taken = queue.take().unwrap();
    assert_eq!(taken.pc(), first.pc());
    // Taken but not finished: still being translated.
    assert_eq!(queue.push(first.clone()), Pushed::Repeated);
    queue.finish(&taken);
    assert_eq!(queue.push(first.clone()), Pushed::Queued);
    assert_eq!(queue.take().unwrap().pc(), second.pc());
    assert_eq!(queue.take().unwrap().pc(), first.pc());
    assert!(queue.take().is_none() && !queue.pending());
    let counts = queue.counts();
    assert_eq!((counts.queued, counts.repeated, counts.dropped), (3, 2, 2));
    // Neither another page's PC nor another context makes a job.
    let model = job(0x4000);
    assert!(
        AheadJob::new(
            model.key,
            model.page,
            model.page + FETCH_PAGE,
            model.context,
            model.bytes.clone(),
            model.preparation.clone()
        )
        .is_none()
    );
    assert!(
        AheadJob::new(
            model.key,
            model.page,
            model.pc,
            Digest([1; 32]),
            model.bytes.clone(),
            model.preparation.clone()
        )
        .is_none()
    );
}

/// The successors a PE hands on, translated ahead, are the blocks it would
/// compile, and its fetch of one installs it instead of compiling it.
#[test]
fn a_successor_translated_ahead_is_the_pes_block_and_saves_its_compile() {
    let mut first = machine(&WORDS);
    first.host.store = Some(MemoryStore {
        ahead: Some(Vec::new()),
        ..MemoryStore::default()
    });
    assert!(matches!(
        first.fetch_block(BASE, false),
        Ok(Fetched::Block(_))
    ));
    let jobs = first.host.store.as_mut().unwrap().ahead.take().unwrap();
    let mut pcs: Vec<u64> = jobs.iter().map(AheadJob::pc).collect();
    pcs.sort_unstable();
    assert_eq!(pcs, [BASE + 4, BASE + 8]);
    let job = jobs.iter().find(|job| job.pc() == BASE + 4).unwrap();
    let ahead = job.translate().unwrap();
    assert!(ahead.ahead());

    // The block a PE compiles there itself.
    let mut compiler = machine(&WORDS);
    compiler.host.store = Some(MemoryStore::default());
    assert!(matches!(
        compiler.fetch_block(BASE + 4, false),
        Ok(Fetched::Block(_))
    ));
    let compiled = compiler.host.store.as_mut().unwrap().offers.pop().unwrap();
    assert_eq!(
        (ahead.key(), ahead.pc(), ahead.context()),
        (compiled.key(), compiled.pc(), compiled.context())
    );
    assert_eq!(ahead.input(), compiled.input());
    assert_eq!(
        ahead.block().native.x86_64_code,
        compiled.block().native.x86_64_code
    );

    // Shared, it is installed, and only its first install is a hit.
    let shared = ahead.clone().shared();
    for (install, hits) in [(0, 1), (1, 1)] {
        let mut store = MemoryStore::default();
        store
            .prepared
            .insert((ahead.key(), ahead.pc(), ahead.context()), shared.clone());
        let mut receiver = machine(&WORDS);
        receiver.host.store = Some(store);
        assert!(matches!(
            receiver.fetch_block(BASE + 4, false),
            Ok(Fetched::Block(_))
        ));
        assert_eq!(
            (receiver.statistics.shared, receiver.statistics.compiled),
            (1, 0),
            "install {install}"
        );
        let store = receiver.host.store.as_ref().unwrap();
        assert_eq!(store.used_ahead, if install == 0 { hits } else { 0 });
    }
}

/// A block translated ahead from a page the guest wrote since is refused
/// at its install, counted stale, and compiled anew.
#[test]
fn a_block_translated_ahead_from_a_page_written_since_is_refused() {
    let mut first = machine(&WORDS);
    first.host.store = Some(MemoryStore {
        ahead: Some(Vec::new()),
        ..MemoryStore::default()
    });
    assert!(matches!(
        first.fetch_block(BASE, false),
        Ok(Fetched::Block(_))
    ));
    let jobs = first.host.store.as_mut().unwrap().ahead.take().unwrap();
    let job = jobs.iter().find(|job| job.pc() == BASE + 4).unwrap();
    let ahead = job.translate().unwrap().shared();

    // The ADD became a SUB: the page has another key, whose lookup the
    // store answers with the old block, as a colliding index would.
    let written = [WORDS[0], 0xd100_0400, WORDS[2]];
    let mut probe = machine(&written);
    probe.host.store = Some(MemoryStore::default());
    assert!(matches!(
        probe.fetch_block(BASE + 4, false),
        Ok(Fetched::Block(_))
    ));
    let now = probe.host.store.as_mut().unwrap().offers.pop().unwrap();
    assert_ne!(now.key(), ahead.offer().key());
    let mut store = MemoryStore::default();
    store
        .prepared
        .insert((now.key(), now.pc(), now.context()), ahead);
    let mut receiver = machine(&written);
    receiver.host.store = Some(store);
    assert!(matches!(
        receiver.fetch_block(BASE + 4, false),
        Ok(Fetched::Block(_))
    ));
    assert_eq!(
        (receiver.statistics.shared, receiver.statistics.compiled),
        (0, 1)
    );
    let store = receiver.host.store.as_ref().unwrap();
    assert!(store.events.contains(&StoreEvent::AheadStale));
    assert_eq!(store.used_ahead, 0);
    assert_eq!(
        store.offers.last().unwrap().block().native.x86_64_code,
        now.block().native.x86_64_code
    );
}
