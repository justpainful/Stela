//! Host processors that run no PE and take the GPUs' deferred compile jobs.
//!
//! A shader compile runs on a PE's host core between two slices of its
//! guest, in slices of a few milliseconds, one job at a time under the
//! compile lock. Core Animation's generic Vfx shader alone took 1.86 s of one
//! core in 1498 slices on an i7-12700KF, while its four efficiency cores and
//! the second thread of each performance core were parked. A compile worker
//! is one of those processors started as an executor like a PE's, which runs
//! no PE: it takes each job at once and runs its slices back to back, so the
//! job's wait shrinks and the PEs keep their cores for their guests.
//!
//! Only an efficiency core is taken, and only one the capture selects right
//! after the PEs' processors: it shares no core with a PE, as an SMT sibling
//! of one would, and the PEs keep the prefix their placement expects.
//!
//! A worker also runs the GPUs' Linux work in a waiting PE's place, which a
//! busy desktop kept about a third of a core on. Jobs run one at a time, but
//! a job's slice (up to 17 ms seen) would hold that work up on a worker that
//! does both, so at least two run: one free for Linux while the other
//! compiles. Workers also prepare blocks ahead of their first run, as PEs
//! whose guest waits do (`stela_engine`'s `ahead`). An earlier design that
//! translated every successor of every block on them kept all four
//! efficiency cores of an i7-12700KF busy without emptying its queue (151
//! million queued in 5.5 h, 14% of them ever run), and the PEs on the
//! performance cores beside them lost a quarter of their speed: system exits
//! took 4.4 us against 2.5 us with two workers, their table walks sharing
//! the busier last-level cache. Two work by default.
use crate::topology::CoreClass;

/// Workers a boot starts when `--compile-workers` names no count.
pub const AUTO: usize = 2;
/// The most workers a boot starts.
pub const MOST: usize = 4;

/// The compile workers to start for `runners` PE processors, from the core
/// classes of the processors the capture selects, in its order: the
/// efficiency cores right after the runners, up to `requested` (else
/// [`AUTO`]) and [`MOST`]. None with fewer than two runners: a serialized or
/// single-processor boot runs its jobs on its one core.
pub fn count(requested: Option<usize>, runners: usize, classes: &[CoreClass]) -> usize {
    let wanted = requested.unwrap_or(AUTO).min(MOST);
    if wanted == 0 || runners < 2 {
        return 0;
    }
    classes
        .iter()
        .skip(runners)
        .take(wanted)
        .take_while(|&&class| class == CoreClass::Efficiency)
        .count()
}

#[cfg(test)]
mod tests;
