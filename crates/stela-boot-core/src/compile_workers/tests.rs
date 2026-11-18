use super::*;
use CoreClass::{Efficiency as E, Performance as P, Unknown as U};

/// An i7-12700KF as the capture selects it: the eight performance cores'
/// first threads, the four efficiency cores, then the performance cores'
/// second threads.
const ALDER_LAKE: [CoreClass; 20] = [P, P, P, P, P, P, P, P, E, E, E, E, P, P, P, P, P, P, P, P];

#[test]
fn a_hybrid_host_gets_every_efficiency_core_after_the_runners_by_default() {
    assert_eq!(AUTO, 4);
    assert_eq!(count(None, 8, &ALDER_LAKE), 4);
    // One efficiency core after the runners gives one.
    assert_eq!(count(None, 11, &ALDER_LAKE), 1);
}

#[test]
fn a_count_takes_that_many_efficiency_cores_up_to_the_most() {
    assert_eq!(count(Some(2), 8, &ALDER_LAKE), 2);
    assert_eq!(count(Some(4), 8, &ALDER_LAKE), 4);
    assert_eq!(count(Some(9), 8, &ALDER_LAKE), MOST);
}

#[test]
fn zero_starts_none() {
    assert_eq!(count(Some(0), 8, &ALDER_LAKE), 0);
}

#[test]
fn a_sibling_of_a_pe_or_an_unknown_class_is_never_a_worker() {
    // The efficiency cores end at slot 12: the rest are siblings.
    assert_eq!(count(Some(4), 10, &ALDER_LAKE), 2);
    assert_eq!(count(Some(4), 12, &ALDER_LAKE), 0);
    // A host that is not hybrid reports no class.
    assert_eq!(count(None, 8, &[U; 16]), 0);
    assert_eq!(count(None, 8, &[P; 16]), 0);
}

#[test]
fn efficiency_cores_first_leave_none_after_the_runners() {
    // Firmware that numbers the efficiency cores first puts them in the
    // runners' prefix; the processor after it is a performance core.
    let mut classes = [P; 20];
    classes[..4].fill(E);
    assert_eq!(count(None, 8, &classes), 0);
}

#[test]
fn a_serialized_or_short_capture_gets_none() {
    assert_eq!(count(None, 1, &ALDER_LAKE), 0);
    assert_eq!(count(None, 8, &ALDER_LAKE[..8]), 0);
}
