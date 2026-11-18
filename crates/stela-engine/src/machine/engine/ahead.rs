//! Translation ahead of execution: a PE that compiles a block hands the
//! static successors of it that lie in the same fetch page to processors
//! that run no PE (the host's compile workers), which compile them with the
//! PE's own [`Preparation`] from the page bytes the PE already read and
//! hashed, and share them through the store as any PE's block
//! ([`super::Translations::prepared`]). The PE's later fetch of a successor then
//! installs it instead of compiling it.
//!
//! Only the PE installs: a worker never writes the code arena, its slots or
//! its thunks, and never posts the IC IVAU a compile posts
//! ([`super::parallel`]'s `compiling`), as no PE has fetched the page
//! through it. A block translated ahead is keyed, as every shared block, by
//! the SHA-256 of its page's bytes, its PC and its entry context, and an
//! install takes it only when its admitted input equals the bytes the PE
//! reads then ([`SharedTranslation::matches`]): a page written after its
//! snapshot was taken gives another key, or a block whose input no longer
//! matches, and the PE compiles it anew.
//!
//! The queue is bounded and never makes a PE wait: a PE that finds it full,
//! or held for a moment by a worker, drops its jobs; a job already queued
//! or being compiled is not queued twice. A worker that finds the store
//! already holding a job's block drops it too.
use super::stored::{Emission, StoredPage};
use super::{
    Devices, FETCH_PAGE, Host, INSTRUCTION, MAX_BLOCK_BYTES, Machine, Preparation, Prepared,
    SharedTranslation,
};
use crate::translations::{Digest, PAGE_BYTES, StoredRecord};
use alloc::sync::Arc;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Successors a compile hands on at most: a block has a taken and a
/// fall-through successor, or a call's target and its continuation.
pub const AHEAD_PER_MISS: usize = 2;

/// A fetch page's bytes, as a PE read and hashed them.
pub type PageBytes = [u8; PAGE_BYTES];

/// A block to translate ahead of execution: its PC, the page it lies in,
/// that page's bytes, key and entry context as the PE that compiled its
/// predecessor had them, and that PE's preparation.
#[derive(Debug, Clone)]
pub struct AheadJob {
    key: Digest,
    page: u64,
    pc: u64,
    context: Digest,
    bytes: Arc<PageBytes>,
    preparation: Arc<Preparation>,
}

// A worker takes jobs a PE queued: everything a job holds is immutable.
const _: () = {
    const fn shared<T: Send + Sync>() {}
    shared::<AheadJob>();
    shared::<super::Offer>();
};

impl AheadJob {
    /// The job of `pc`, in the page at `page` whose bytes are `bytes`, or
    /// none when `pc` is not an instruction of that page or `context` is
    /// not `preparation`'s there.
    pub fn new(
        key: Digest,
        page: u64,
        pc: u64,
        context: Digest,
        bytes: Arc<PageBytes>,
        preparation: Arc<Preparation>,
    ) -> Option<Self> {
        (pc & !(FETCH_PAGE - 1) == page
            && page == pc & !(PAGE_BYTES as u64 - 1)
            && pc.is_multiple_of(INSTRUCTION)
            && preparation.context(page) == context)
            .then_some(Self {
                key,
                page,
                pc,
                context,
                bytes,
                preparation,
            })
    }
    pub fn key(&self) -> &Digest {
        &self.key
    }
    pub fn pc(&self) -> u64 {
        self.pc
    }
    pub fn context(&self) -> &Digest {
        &self.context
    }

    /// The block the PE would compile at the job's PC from its page's
    /// bytes, as an offer to share ([`super::Translations::offer`]'s, marked
    /// [`super::Offer::ahead`]); none when its first instruction is one
    /// the engine checks or traps, or the compiler refuses it, which the
    /// PE meets itself. The PE prepares the same window the same way
    /// (`Machine::compile_window`): admitted, to the end of the page and at
    /// most a block's, with no host breakpoint or device site in a page the
    /// store keys.
    pub fn translate(&self) -> Option<super::Offer> {
        let offset = (self.pc - self.page) as usize;
        let window = super::prepare::window(self.pc);
        let mut staging = [0; MAX_BLOCK_BYTES as usize];
        let source = staging.get_mut(..window)?;
        source.copy_from_slice(self.bytes.get(offset..offset + window)?);
        self.preparation.admit(source);
        let Ok(Prepared::Block {
            compiled,
            input,
            successors,
            ..
        }) = self.preparation.prepare(source, self.pc, window)
        else {
            return None;
        };
        Some(super::Offer {
            key: self.key,
            page: self.page,
            pc: self.pc,
            context: self.context,
            emission: Arc::new(Emission::new(
                source.get(..input)?.to_vec(),
                compiled,
                self.preparation.clone(),
            )),
            successors,
            quiet: false,
            ahead: true,
        })
    }

    /// A word of the job's key, PC and context, never zero: equal for the
    /// same block, and what [`AheadQueue`] dedupes by.
    fn mark(&self) -> u64 {
        let word = |digest: &Digest| {
            u64::from_le_bytes(digest.0[..8].try_into().expect("a digest's word"))
        };
        let mixed = [word(&self.key), self.pc, word(&self.context)]
            .into_iter()
            .fold(0u64, |hash, word| {
                (hash ^ word)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .rotate_left(23)
            });
        mixed | 1
    }
}

/// What [`AheadQueue::push`] did with a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pushed {
    Queued,
    /// The same block is queued or being translated.
    Repeated,
    /// The queue was full, or a worker held it that moment.
    Dropped,
}

/// What became of a job a worker took ([`AheadQueue::note`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AheadEvent {
    /// The store already held its block, or could not take one.
    Present,
    /// Translated, in `ticks` of the worker's counter.
    Translated {
        ticks: u64,
    },
    /// No block: its first instruction is one the engine checks or traps,
    /// or the compiler refused it.
    Refused,
    /// The store took its block, or did not.
    Published,
    Unpublished,
    /// A PE installed its block for the first time
    /// ([`SharedTranslation::claim`]), or found its page changed.
    Hit,
    Stale,
}

/// What a queue and its workers did ([`AheadQueue::counts`]).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AheadCounts {
    pub queued: u64,
    pub repeated: u64,
    pub dropped: u64,
    pub present: u64,
    pub translated: u64,
    pub refused: u64,
    pub published: u64,
    pub unpublished: u64,
    pub hits: u64,
    pub stale: u64,
    /// Worker counter ticks spent translating.
    pub ticks: u64,
}

/// Slots of the table of jobs queued or being translated.
const FLIGHT_SLOTS: usize = 256;

/// A bounded queue of [`AheadJob`]s, its first out first: PEs push without
/// waiting, workers take. A lock held for a few stores guards the ring; a
/// direct-mapped table of the jobs queued or being translated
/// ([`AheadJob::mark`]) keeps a block from being queued twice, a collision
/// only letting one through again.
pub struct AheadQueue<const N: usize> {
    held: AtomicBool,
    ring: UnsafeCell<[Option<AheadJob>; N]>,
    /// The ring's oldest job; `len` jobs follow it.
    head: UnsafeCell<usize>,
    len: AtomicUsize,
    flight: [AtomicU64; FLIGHT_SLOTS],
    counts: [AtomicU64; 11],
}

// SAFETY: `held` excludes every other access to `ring` and `head`; the jobs
// are Send and Sync.
unsafe impl<const N: usize> Sync for AheadQueue<N> {}

impl<const N: usize> Default for AheadQueue<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> AheadQueue<N> {
    pub const fn new() -> Self {
        Self {
            held: AtomicBool::new(false),
            ring: UnsafeCell::new([const { None }; N]),
            head: UnsafeCell::new(0),
            len: AtomicUsize::new(0),
            flight: [const { AtomicU64::new(0) }; FLIGHT_SLOTS],
            counts: [const { AtomicU64::new(0) }; 11],
        }
    }

    fn count(&self, at: usize, by: u64) {
        self.counts[at].fetch_add(by, Ordering::Relaxed);
    }

    fn flight(&self, job: &AheadJob) -> (&AtomicU64, u64) {
        let mark = job.mark();
        (&self.flight[(mark >> 32) as usize % FLIGHT_SLOTS], mark)
    }

    /// Queue `job` unless it is queued or being translated, the queue is
    /// full or a worker holds it: never waits.
    pub fn push(&self, job: AheadJob) -> Pushed {
        let (slot, mark) = self.flight(&job);
        if slot.load(Ordering::Relaxed) == mark {
            self.count(1, 1);
            return Pushed::Repeated;
        }
        if self
            .held
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            self.count(2, 1);
            return Pushed::Dropped;
        }
        let len = self.len.load(Ordering::Relaxed);
        let queued = len < N;
        if queued {
            // SAFETY: `held` is ours until released below.
            let (ring, head) = unsafe { (&mut *self.ring.get(), &*self.head.get()) };
            // The slot past the last job is empty: nothing drops under the
            // lock.
            ring[(*head + len) % N] = Some(job);
            slot.store(mark, Ordering::Relaxed);
            self.len.store(len + 1, Ordering::Release);
        }
        self.held.store(false, Ordering::Release);
        self.count(if queued { 0 } else { 2 }, 1);
        if queued {
            Pushed::Queued
        } else {
            Pushed::Dropped
        }
    }

    /// The oldest job, if any. A worker waits for the lock: it holds no
    /// guest.
    pub fn take(&self) -> Option<AheadJob> {
        if self.len.load(Ordering::Acquire) == 0 {
            return None;
        }
        while self
            .held
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        let len = self.len.load(Ordering::Relaxed);
        let job = if len == 0 {
            None
        } else {
            // SAFETY: `held` is ours until released below.
            let (ring, head) = unsafe { (&mut *self.ring.get(), &mut *self.head.get()) };
            let job = ring[*head].take();
            *head = (*head + 1) % N;
            self.len.store(len - 1, Ordering::Release);
            job
        };
        self.held.store(false, Ordering::Release);
        job
    }

    /// `job`, taken, is done with: the same block may be queued again.
    pub fn finish(&self, job: &AheadJob) {
        let (slot, mark) = self.flight(job);
        let _ = slot.compare_exchange(mark, 0, Ordering::Relaxed, Ordering::Relaxed);
    }

    /// Whether a job waits.
    pub fn pending(&self) -> bool {
        self.len.load(Ordering::Acquire) != 0
    }

    pub fn note(&self, event: AheadEvent) {
        match event {
            AheadEvent::Present => self.count(3, 1),
            AheadEvent::Translated { ticks } => {
                self.count(4, 1);
                self.count(10, ticks);
            }
            AheadEvent::Refused => self.count(5, 1),
            AheadEvent::Published => self.count(6, 1),
            AheadEvent::Unpublished => self.count(7, 1),
            AheadEvent::Hit => self.count(8, 1),
            AheadEvent::Stale => self.count(9, 1),
        }
    }

    pub fn counts(&self) -> AheadCounts {
        let load = |at: usize| self.counts[at].load(Ordering::Relaxed);
        AheadCounts {
            queued: load(0),
            repeated: load(1),
            dropped: load(2),
            present: load(3),
            translated: load(4),
            refused: load(5),
            published: load(6),
            unpublished: load(7),
            hits: load(8),
            stale: load(9),
            ticks: load(10),
        }
    }
}

impl<H: Host, D: Devices> Machine<H, D> {
    /// Hand the store the successors of the block just compiled at `pc`,
    /// in the store page at `physical` whose bytes hash to `key`, that lie
    /// in that page and that neither the page's record (in `preparation`'s
    /// context) nor this PE's block table holds, to translate ahead of execution
    /// ([`super::Translations::ahead`]).
    pub(super) fn translate_ahead(
        &mut self,
        pc: u64,
        physical: u64,
        key: Digest,
        record: Option<&StoredRecord>,
        preparation: &Arc<Preparation>,
        successors: &[u64],
    ) {
        if !self
            .host
            .translations()
            .is_some_and(|translations| translations.translates_ahead())
        {
            return;
        }
        let page = pc & !(FETCH_PAGE - 1);
        let context = preparation.context(page);
        let mut wanted = [0u64; AHEAD_PER_MISS];
        let mut count = 0;
        for &successor in successors {
            if count == AHEAD_PER_MISS {
                break;
            }
            if successor == pc
                || successor & !(FETCH_PAGE - 1) != page
                || wanted[..count].contains(&successor)
                || record.is_some_and(|record| record.holds((successor - page) as u16, &context))
            {
                continue;
            }
            // Installed on this PE already, through the page's ITLB entry.
            if let Some(entry) = self.current_key(successor) {
                let table = &self.code.table;
                let target = table.find(table.slot(successor, entry), successor, entry);
                if target.pc == successor && target.key == entry {
                    continue;
                }
            }
            wanted[count] = successor;
            count += 1;
        }
        if count == 0 || self.host.memory_low() {
            return;
        }
        let Some(StoredPage {
            key: kept,
            bytes,
            snapshot,
            ..
        }) = self.pages.get_mut(&(physical & !(FETCH_PAGE - 1)))
        else {
            return;
        };
        if *kept != key {
            return;
        }
        let bytes = snapshot.get_or_insert_with(|| Arc::new(**bytes)).clone();
        let Some(translations) = self.host.translations() else {
            return;
        };
        let mut jobs = wanted[..count].iter().filter_map(|&successor| {
            AheadJob::new(
                key,
                page,
                successor,
                context,
                bytes.clone(),
                preparation.clone(),
            )
        });
        translations.ahead(&mut jobs);
    }

    /// Note an install of `shared` from the store, or its refusal: the
    /// first install of a block translated ahead is a miss it saved, and a
    /// refusal of one a page changed since.
    pub(super) fn note_ahead(&mut self, shared: &Arc<SharedTranslation>, installed: bool) {
        if !shared.ahead() {
            return;
        }
        let Some(translations) = self.host.translations() else {
            return;
        };
        if !installed {
            translations.note(super::StoreEvent::AheadStale);
        } else if shared.claim() {
            translations.used_ahead(shared);
        }
    }
}

#[cfg(test)]
mod tests;
