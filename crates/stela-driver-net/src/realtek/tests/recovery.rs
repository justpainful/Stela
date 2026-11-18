// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 Stela contributors
//! The restart a stopped transmit ring gets: Linux `rtl_reset_work`.
use super::*;
use crate::{
    bus::dma::TestRegion,
    watchdog::{Stall, TRANSMIT_TIMEOUT_MS, Watchdog},
};

const STATION: Mac = Mac([2, 3, 4, 5, 6, 7]);

/// An RTL8168H (XID 541) as the start leaves it: the MAC configured, both
/// rings published and the chip enabled, in `dma`.
fn started(time: &Time, dma: &mut TestRegion) -> (Model, Ring, Ring) {
    let model = Model::modern(time);
    let mut rx = Ring::receive_ring(dma, true).unwrap();
    let mut tx = Ring::new(dma, true).unwrap();
    rx.initialize(true);
    tx.initialize(false);
    {
        let mut hw = Hardware::new(&model, time, revision(46), 0);
        hw.initialize_modern().unwrap();
        hw.reset().unwrap();
        hw.configure_mac(None).unwrap();
        publish(&hw, &rx, &tx, STATION).unwrap();
        hw.interrupts_on(SIGNALLED_EVENTS);
    }
    model.state.borrow_mut().writes.clear();
    (model, rx, tx)
}

/// Each descriptor's buffer address, in ring order.
fn buffers(ring: &Ring) -> Vec<u64> {
    (0..DESCRIPTORS)
        .map(|index| ring.descriptors.read::<u64>(index * DESCRIPTOR + 8))
        .collect()
}

/// A chip that never hands its transmit descriptors back fills the ring:
/// every later send is refused. Once a frame has waited Linux's 6 s with
/// carrier the watchdog says so, and the restart stops and resets the MAC
/// before either ring is touched, sets both rings in the same DMA buffers,
/// configures and enables the chip again with the same descriptor bases,
/// and turns the signal's events back on. Sends and receives then move,
/// and the watchdog has nothing left to say.
#[test]
fn a_transmit_ring_the_chip_never_returns_is_restarted_after_the_timeout() {
    let time = Time::default();
    let mut dma = TestRegion::new(DESCRIPTORS * (RECEIVE_BUFFER + BUFFER) + (1 << 18));
    let (model, mut rx, mut tx) = started(&time, &mut dma);
    let (rx_buffers, tx_buffers) = (buffers(&rx), buffers(&tx));
    let mut watchdog = Watchdog::default();
    for sequence in 0..DESCRIPTORS {
        tx.transmit(&[sequence as u8; 60]).unwrap();
        watchdog.sent(0);
    }
    // Some frames arrived and are waiting in the ring.
    rx.deliver(&[0x11; 60]);
    let mut ms = 0;
    while watchdog.due(ms, true).is_none() {
        assert_eq!(tx.transmit(&[0xee; 60]), Err(LinkError::Busy), "{ms}");
        watchdog.busy(ms);
        assert!(ms <= TRANSMIT_TIMEOUT_MS, "{ms}");
        ms += 10;
    }
    assert_eq!(ms, TRANSMIT_TIMEOUT_MS);
    let stall = watchdog.due(ms, true).unwrap();
    assert_eq!(
        stall,
        Stall::Transmit {
            waited_ms: TRANSMIT_TIMEOUT_MS
        }
    );
    assert_eq!(tx.device_owned(), DESCRIPTORS);
    let began = time.now();
    let mut hw = Hardware::new(&model, &time, revision(46), 0);
    assert_eq!(
        restart(&mut hw, &mut rx, &mut tx, STATION, true).map_err(|e| format!("{e}")),
        Ok(())
    );
    // About 8 ms of the model's waits: the stop's 2 ms, its FIFO polls,
    // the reset and the configuration's command completions.
    let waited = time.now() - began;
    assert!(waited < 20_000, "{waited} us");
    watchdog.recovered(ms, stall);
    {
        let s = model.state.borrow();
        let at = |found: &dyn Fn(usize, u32) -> bool| {
            s.writes
                .iter()
                .position(|&(offset, value, _)| found(offset, value))
                .unwrap()
        };
        let last = |found: &dyn Fn(usize, u32) -> bool| {
            s.writes
                .iter()
                .rposition(|&(offset, value, _)| found(offset, value))
                .unwrap()
        };
        let masked = at(&|offset, value| offset == INTERRUPT_MASK && value == 0);
        let closed = at(&|offset, _| offset == RX_CONFIG);
        let reset = at(&|offset, value| offset == COMMAND && value as u8 & RESET != 0);
        let base = at(&|offset, _| offset == TX_HIGH);
        let enabled = last(&|offset, value| offset == COMMAND && value as u8 == ENABLE);
        let events =
            last(&|offset, value| offset == INTERRUPT_MASK && value == u32::from(SIGNALLED_EVENTS));
        assert!(
            masked < closed && closed < reset && reset < base,
            "{:?}",
            s.writes
        );
        assert!(base < enabled && enabled < events, "{:?}", s.writes);
        // The MAC configuration ran again: the RXDV gate the stop closed
        // is open, and the H start's TX idle timer is set.
        assert_eq!(Model::raw32(&s, MISC) & RXDV_GATED, 0);
        assert_eq!(s.mac[&0xe048], 1546);
        assert_eq!(s.bytes[CONFIG_LOCK], CONFIG_LOCKED);
    }
    assert_eq!(model.read8(COMMAND), ENABLE);
    assert_eq!(model.read32(RX_CONFIG), receive_config(46) | RX_ACCEPT);
    assert_eq!(model.read32(TX_LOW), tx.descriptors.physical() as u32);
    assert_eq!(model.read32(RX_LOW), rx.descriptors.physical() as u32);
    // The same buffers, every transmit slot the driver's, every receive
    // slot offered: nothing allocated, nothing lost.
    assert_eq!((buffers(&rx), buffers(&tx)), (rx_buffers, tx_buffers));
    assert_eq!((tx.next(), tx.device_owned()), (0, 0));
    assert_eq!((rx.next(), rx.device_owned()), (0, DESCRIPTORS));
    // The frame that waited is taken, and the ring takes a full pass.
    for sequence in 0..DESCRIPTORS {
        tx.transmit(&[sequence as u8; 60]).unwrap();
        watchdog.sent(ms);
    }
    assert_eq!(tx.transmit(&[0; 60]), Err(LinkError::Busy));
    for frame in [[0x21; 60], [0x22; 60]] {
        rx.deliver(&frame);
        assert_eq!(rx.receive(), Some(frame.to_vec()));
    }
    assert_eq!(watchdog.recoveries(), 1);
    assert_eq!(watchdog.due(ms + TRANSMIT_TIMEOUT_MS, true), None);
}

/// A stop that never sees its MAC reset complete fails the restart and
/// leaves both rings exactly as they were: the chip may still read and
/// write them, so no descriptor goes back to the driver.
#[test]
fn a_restart_whose_reset_never_completes_leaves_both_rings_with_the_chip() {
    let time = Time::default();
    let mut dma = TestRegion::new(DESCRIPTORS * (RECEIVE_BUFFER + BUFFER) + (1 << 18));
    let (mut model, mut rx, mut tx) = started(&time, &mut dma);
    for _ in 0..DESCRIPTORS {
        tx.transmit(&[0x5a; 60]).unwrap();
    }
    rx.deliver(&[0x33; 60]);
    let before = (
        rx.descriptors.bytes().to_vec(),
        tx.descriptors.bytes().to_vec(),
    );
    model.stuck_reset = true;
    let mut hw = Hardware::new(&model, &time, revision(46), 0);
    assert_eq!(
        restart(&mut hw, &mut rx, &mut tx, STATION, true),
        Err(Error::Device(BlockError::Timeout))
    );
    assert_eq!(
        (
            rx.descriptors.bytes().to_vec(),
            tx.descriptors.bytes().to_vec()
        ),
        before
    );
    assert!(
        !model
            .state
            .borrow()
            .writes
            .iter()
            .any(|(offset, _, _)| matches!(*offset, TX_HIGH | TX_LOW | RX_HIGH | RX_LOW))
    );
}

/// A MAC configuration that fails after the reset resets the MAC again,
/// as the start does, and leaves the chip disabled.
#[test]
fn a_restart_whose_configuration_fails_leaves_the_chip_reset_and_disabled() {
    let time = Time::default();
    let mut dma = TestRegion::new(DESCRIPTORS * (RECEIVE_BUFFER + BUFFER) + (1 << 18));
    let (mut model, mut rx, mut tx) = started(&time, &mut dma);
    model.fault = Some(ERI_ADDRESS);
    let mut hw = Hardware::new(&model, &time, revision(46), 0);
    assert_eq!(
        restart(&mut hw, &mut rx, &mut tx, STATION, true),
        Err(Error::Device(BlockError::Timeout))
    );
    let s = model.state.borrow();
    let resets = s
        .writes
        .iter()
        .filter(|&&(offset, value, _)| offset == COMMAND && value as u8 & RESET != 0)
        .count();
    assert_eq!(resets, 2);
    assert_eq!(s.bytes[COMMAND] & ENABLE, 0);
    assert!(
        !s.writes
            .iter()
            .any(|(offset, _, _)| matches!(*offset, TX_HIGH | TX_LOW | RX_HIGH | RX_LOW))
    );
}
