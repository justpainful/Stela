// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 Stela contributors
//! The watchdog's timing, on a clock in milliseconds.
use super::*;
use alloc::vec::Vec;

/// A link that sent and received at `at`.
fn moving(at: u64) -> Watchdog {
    let mut watchdog = Watchdog::default();
    watchdog.sent(at);
    watchdog.received(at);
    watchdog
}

/// A frame refused by a full ring for Linux's 6 s with carrier is a hung
/// transmit; a moment less, or without carrier, is not, and a frame the
/// ring takes ends the wait.
#[test]
fn a_ring_full_for_the_transmit_timeout_with_carrier_is_due() {
    let mut watchdog = moving(1_000);
    for turn in (2_000..2_000 + TRANSMIT_TIMEOUT_MS).step_by(7) {
        watchdog.busy(turn);
        assert_eq!(watchdog.due(turn, true), None, "{turn}");
    }
    let at = 2_000 + TRANSMIT_TIMEOUT_MS;
    watchdog.busy(at);
    assert_eq!(
        watchdog.due(at, true),
        Some(Stall::Transmit {
            waited_ms: TRANSMIT_TIMEOUT_MS
        })
    );
    assert_eq!(watchdog.busy_ms(at), Some(TRANSMIT_TIMEOUT_MS));
    assert_eq!(watchdog.due(at, false), None);
    // The ring took the frame: no wait is left.
    watchdog.sent(at);
    assert_eq!(watchdog.due(at + 60_000, true), None);
    assert_eq!(watchdog.busy_ms(at), None);
    // A frame dropped without the ring (no carrier) no longer waits.
    watchdog.busy(at + 1);
    watchdog.dropped();
    assert_eq!(watchdog.due(at + 60_000, true), None);
}

/// Restarts are at least 30 s apart, and a frame still refused after one
/// waits its full timeout again.
#[test]
fn restarts_are_spaced_and_the_wait_starts_again_after_one() {
    let mut watchdog = moving(0);
    watchdog.busy(1_000);
    let first = 1_000 + TRANSMIT_TIMEOUT_MS;
    let stall = watchdog.due(first, true).unwrap();
    watchdog.recovered(first, stall);
    assert_eq!(watchdog.recoveries(), 1);
    assert_eq!(watchdog.busy_ms(first), None);
    // Still refused right after the restart: due only once both its own
    // timeout and the spacing have passed.
    watchdog.busy(first + 1);
    assert_eq!(watchdog.due(first + 1 + TRANSMIT_TIMEOUT_MS, true), None);
    assert_eq!(watchdog.due(first + RECOVERY_SPACING_MS - 1, true), None);
    assert!(matches!(
        watchdog.due(first + RECOVERY_SPACING_MS, true),
        Some(Stall::Transmit { .. })
    ));
}

/// The receive path counts as stopped only for a link that received
/// before, whose ring was found empty for 30 s while at least 16 frames
/// left in that time and the last of them recently, with carrier.
#[test]
fn a_silent_receive_ring_while_sending_is_due_only_after_earlier_traffic() {
    let sending = |watchdog: &mut Watchdog, from: u64, until: u64| {
        for at in (from..=until).step_by(1_000) {
            watchdog.sent(at);
            watchdog.empty(at);
        }
    };
    // Never received: a quiet network, not a stopped ring.
    let mut quiet = Watchdog::default();
    sending(&mut quiet, 0, 120_000);
    assert_eq!(quiet.due(120_000, true), None);
    let mut watchdog = moving(0);
    sending(&mut watchdog, 1_000, RECEIVE_TIMEOUT_MS);
    assert_eq!(watchdog.due(RECEIVE_TIMEOUT_MS, true), None);
    sending(
        &mut watchdog,
        RECEIVE_TIMEOUT_MS + 1_000,
        RECEIVE_TIMEOUT_MS + 1_000,
    );
    let at = RECEIVE_TIMEOUT_MS + 1_000;
    assert_eq!(
        watchdog.due(at, true),
        Some(Stall::Receive {
            silent_ms: at,
            sent: 31
        })
    );
    assert_eq!(watchdog.due(at, false), None);
    // A guest that stopped sending is idle, not unanswered.
    assert_eq!(watchdog.due(at + RECEIVE_TIMEOUT_MS, true), None);
    // Too few frames: a couple of announcements to a quiet network.
    let mut few = moving(0);
    for at in [10_000, 20_000, 40_000] {
        few.sent(at);
        few.empty(at);
    }
    assert_eq!(few.due(40_000, true), None);
    // A transmit wait takes precedence and holds the receive check back.
    watchdog.busy(at);
    assert_eq!(watchdog.due(at, true), None);
}

/// A receive restart after which nothing arrived waits twice as long
/// before the next, up to 8 minutes; a frame that arrives resets it.
#[test]
fn receive_restarts_that_did_not_help_back_off() {
    let mut watchdog = moving(0);
    let mut at = 0;
    let mut spacings = Vec::new();
    let mut last = None;
    while spacings.len() < 7 {
        at += 1_000;
        watchdog.sent(at);
        watchdog.empty(at);
        if let Some(stall) = watchdog.due(at, true) {
            assert!(matches!(stall, Stall::Receive { .. }), "{stall}");
            if let Some(last) = last {
                spacings.push(at - last);
            }
            last = Some(at);
            watchdog.recovered(at, stall);
        }
    }
    assert_eq!(
        spacings,
        [60_000, 120_000, 240_000, 480_000, 480_000, 480_000, 480_000]
    );
    watchdog.received(at);
    watchdog.sent(at);
    for later in (at + 1_000..=at + 31_000).step_by(1_000) {
        watchdog.sent(later);
        watchdog.empty(later);
    }
    assert!(watchdog.due(at + 31_000, true).is_some());
}

/// What the log line says of each stall.
#[test]
fn stalls_say_what_stopped() {
    assert_eq!(
        alloc::format!("{}", Stall::Transmit { waited_ms: 6_010 }),
        "a transmit ring that took no frame for 6010 ms"
    );
    assert_eq!(
        alloc::format!(
            "{}",
            Stall::Receive {
                silent_ms: 31_000,
                sent: 40
            }
        ),
        "40 frames sent and none received for 31000 ms"
    );
}
