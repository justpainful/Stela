// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 Stela contributors
//! When a link's rings stopped moving, for [`Link::recover`](crate::Link::recover).
//!
//! Linux's transmit watchdog (`dev_watchdog` in net/sched/sch_generic.c)
//! calls a driver's `ndo_tx_timeout` once a queue the driver stopped has
//! had carrier and taken no frame for the driver's `watchdog_timeo`, which
//! r8169 sets to `RTL8169_TX_TIMEOUT`, 6 s. [`Watchdog`] keeps that rule
//! for a caller that keeps a frame a full ring refused (`LinkError::Busy`)
//! and offers it again, and adds a receive check Linux does not make,
//! kept conservative: a link that took frames before, with carrier, whose
//! ring was looked at and found empty for [`RECEIVE_TIMEOUT_MS`] while at
//! least [`RECEIVE_SENDS`] frames left, none answered. Either restart is
//! at most one per [`RECOVERY_SPACING_MS`], and a receive restart after
//! which still nothing arrived waits twice as long before the next one,
//! up to [`RECEIVE_BACKOFF`] doublings.
//!
//! Pure: the caller says what each send and look at the ring did, and the
//! time, in milliseconds of one clock.
use core::fmt;

/// Linux r8169's `RTL8169_TX_TIMEOUT`: a stopped transmit queue with
/// carrier this long is a hung chip.
pub const TRANSMIT_TIMEOUT_MS: u64 = 6_000;
/// How long a link that received before may find its ring empty while it
/// sends before the receive path counts as stopped.
pub const RECEIVE_TIMEOUT_MS: u64 = 30_000;
/// The frames sent without one received that a receive stall needs: a
/// guest whose lookups, ARP and TCP retries go unanswered, not one that
/// sent an announcement or two to a quiet network.
pub const RECEIVE_SENDS: u64 = 16;
/// The shortest time between two restarts.
pub const RECOVERY_SPACING_MS: u64 = 30_000;
/// The most doublings of [`RECOVERY_SPACING_MS`] between receive restarts
/// that each left the ring as silent: 8 minutes.
pub const RECEIVE_BACKOFF: u32 = 4;

/// Why the rings count as stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stall {
    /// A frame waited this long on a transmit ring that took nothing.
    Transmit { waited_ms: u64 },
    /// Nothing arrived for this long while this many frames left.
    Receive { silent_ms: u64, sent: u64 },
}

impl fmt::Display for Stall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transmit { waited_ms } => {
                write!(f, "a transmit ring that took no frame for {waited_ms} ms")
            }
            Self::Receive { silent_ms, sent } => {
                write!(f, "{sent} frames sent and none received for {silent_ms} ms")
            }
        }
    }
}

/// What a link's sends and receives did, and its restarts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Watchdog {
    /// Since when a frame waits on a full transmit ring.
    busy_since: Option<u64>,
    last_sent: Option<u64>,
    last_received: Option<u64>,
    /// Since when the receive ring was found empty, from the first look
    /// after the last frame.
    empty_since: Option<u64>,
    /// Frames sent since the last one received.
    sent_since_received: u64,
    last_recovery: Option<u64>,
    /// Receive restarts in a row after which nothing arrived.
    receive_unanswered: u32,
    recoveries: u32,
}

impl Watchdog {
    /// The ring took a frame.
    pub fn sent(&mut self, now: u64) {
        self.last_sent = Some(now);
        self.busy_since = None;
        self.sent_since_received = self.sent_since_received.saturating_add(1);
    }

    /// The ring was full and the frame stays with the caller.
    pub fn busy(&mut self, now: u64) {
        self.busy_since.get_or_insert(now);
    }

    /// The frame was dropped without the ring (no carrier, a refused
    /// frame): it no longer waits.
    pub fn dropped(&mut self) {
        self.busy_since = None;
    }

    /// A frame arrived.
    pub fn received(&mut self, now: u64) {
        self.last_received = Some(now);
        self.empty_since = None;
        self.sent_since_received = 0;
        self.receive_unanswered = 0;
    }

    /// The receive ring was looked at and held no frame.
    pub fn empty(&mut self, now: u64) {
        self.empty_since.get_or_insert(now);
    }

    /// Why the link should be restarted now, if it should: never without
    /// `carrier`, as Linux's watchdog runs only while `netif_carrier_ok`.
    pub fn due(&self, now: u64, carrier: bool) -> Option<Stall> {
        if !carrier {
            return None;
        }
        let since = |at: Option<u64>| at.map(|at| now.saturating_sub(at));
        let spaced = |spacing: u64| since(self.last_recovery).is_none_or(|ago| ago >= spacing);
        if let Some(waited_ms) = since(self.busy_since) {
            return (waited_ms >= TRANSMIT_TIMEOUT_MS && spaced(RECOVERY_SPACING_MS))
                .then_some(Stall::Transmit { waited_ms });
        }
        let silent_ms = since(self.last_received)?;
        let backoff = RECOVERY_SPACING_MS << self.receive_unanswered.min(RECEIVE_BACKOFF);
        (since(self.empty_since).is_some_and(|empty| empty >= RECEIVE_TIMEOUT_MS)
            && silent_ms >= RECEIVE_TIMEOUT_MS
            && self.sent_since_received >= RECEIVE_SENDS
            && since(self.last_sent).is_some_and(|ago| ago < RECEIVE_TIMEOUT_MS)
            && spaced(backoff))
        .then_some(Stall::Receive {
            silent_ms,
            sent: self.sent_since_received,
        })
    }

    /// The link was restarted (or tried to be) for `stall` at `now`: the
    /// next is at least [`RECOVERY_SPACING_MS`] later, and a frame still
    /// refused waits its timeout anew.
    pub fn recovered(&mut self, now: u64, stall: Stall) {
        self.last_recovery = Some(now);
        self.recoveries = self.recoveries.saturating_add(1);
        self.busy_since = None;
        if let Stall::Receive { .. } = stall {
            self.receive_unanswered = self.receive_unanswered.saturating_add(1);
            self.empty_since = None;
            self.sent_since_received = 0;
        }
    }

    /// The restarts so far.
    pub fn recoveries(&self) -> u32 {
        self.recoveries
    }

    /// How long a frame has waited on a full ring.
    pub fn busy_ms(&self, now: u64) -> Option<u64> {
        self.busy_since.map(|at| now.saturating_sub(at))
    }

    /// How long since the ring last took a frame.
    pub fn since_sent(&self, now: u64) -> Option<u64> {
        self.last_sent.map(|at| now.saturating_sub(at))
    }

    /// How long since a frame last arrived.
    pub fn since_received(&self, now: u64) -> Option<u64> {
        self.last_received.map(|at| now.saturating_sub(at))
    }
}

#[cfg(test)]
mod tests;
