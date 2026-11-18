use super::*;
use crate::{
    dwc3::xhci::tests::host,
    usb::{
        Speed,
        control::{
            GET_DESCRIPTOR, SET_ADDRESS, SET_CONFIGURATION,
            tests::{setup, transfer},
        },
        hub::{Hub, Ports},
    },
};
use stela_drivers::{audio::Direction, usb::audio::Sync as StreamSync};

fn configured() -> Audio {
    let mut a = Audio::new();
    transfer(&mut a, setup(0, SET_ADDRESS, 4, 0, 0), &[]).unwrap();
    transfer(&mut a, setup(0, SET_CONFIGURATION, 1, 0, 0), &[]).unwrap();
    a
}

/// A class request to entity `entity` of the control interface.
fn entity(request_type: u8, request: u8, selector: u8, entity: u8, length: u16) -> [u8; 8] {
    setup(
        request_type,
        request,
        u16::from(selector) << 8,
        u16::from(entity) << 8,
        length,
    )
}

fn stream(a: &mut Audio, alternate: u16) {
    transfer(a, setup(0x01, SET_INTERFACE, alternate, 1, 0), &[]).unwrap();
}

fn capture(a: &mut Audio, alternate: u16) {
    transfer(a, setup(0x01, SET_INTERFACE, alternate, 2, 0), &[]).unwrap();
}

/// One streaming setting's (interface, alternate, terminal link, channels,
/// endpoint address, attributes, max packet, interval).
type Setting = (u8, u8, u8, u8, u8, u8, u16, u8);

/// One speed's configuration, walked by its descriptors' lengths.
struct Walk {
    /// The AC interface's terminals: (subtype, ID, type).
    terminals: Vec<(u8, u8, u16)>,
    header: Vec<u8>,
    /// Bytes of the AC interface's class-specific descriptors.
    units: usize,
    features: Vec<Vec<u8>>,
    settings: Vec<Setting>,
}

fn walk(config: &[u8]) -> Walk {
    assert_eq!(
        usize::from(u16::from_le_bytes([config[2], config[3]])),
        config.len()
    );
    let mut w = Walk {
        terminals: Vec::new(),
        header: Vec::new(),
        units: 0,
        features: Vec::new(),
        settings: Vec::new(),
    };
    let mut interface = (0xff, 0xff);
    let mut link = (0, 0);
    // Every descriptor's length walks the configuration exactly.
    let mut at = 0;
    while at < config.len() {
        let d = &config[at..at + usize::from(config[at])];
        assert!(d.len() >= 3, "descriptor at {at}");
        match (d[1], interface.0) {
            (0x0b, _) => assert_eq!(d, [8, 0x0b, 0, 3, 1, 0, 0x20, 0], "association"),
            (INTERFACE, _) => interface = (d[2], d[3]),
            // The streaming interfaces' subtypes mean other things.
            (CS_INTERFACE, 0) => {
                w.units += d.len();
                match d[2] {
                    HEADER => w.header = d.to_vec(),
                    INPUT_TERMINAL | OUTPUT_TERMINAL => {
                        w.terminals
                            .push((d[2], d[3], u16::from_le_bytes([d[4], d[5]])));
                    }
                    FEATURE_UNIT => w.features.push(d.to_vec()),
                    _ => {}
                }
            }
            (CS_INTERFACE, _) if d[2] == AS_GENERAL => link = (d[3], d[10]),
            (ENDPOINT, _) => w.settings.push((
                interface.0,
                interface.1,
                link.0,
                link.1,
                d[2],
                d[3],
                u16::from_le_bytes([d[4], d[5]]),
                d[6],
            )),
            _ => {}
        }
        at += d.len();
    }
    assert_eq!(at, config.len());
    w
}

#[test]
fn descriptors_describe_a_uac2_headset_with_a_mono_microphone() {
    let mut a = configured();
    let config = transfer(&mut a, setup(0x80, GET_DESCRIPTOR, 0x200, 0, 512), &[]).unwrap();
    assert_eq!(config[4], 3, "control, playback and capture interfaces");
    let w = walk(&config);
    // bcdADC 2.00, category headset, wTotalLength the AC descriptors.
    assert_eq!(&w.header[3..6], [0x00, 0x02, 0x04]);
    assert_eq!(
        usize::from(u16::from_le_bytes([w.header[6], w.header[7]])),
        w.units
    );
    // USB streaming in, speaker out; microphone in, USB streaming out.
    assert_eq!(
        w.terminals,
        [
            (INPUT_TERMINAL, INPUT, 0x0101),
            (OUTPUT_TERMINAL, OUTPUT, 0x0301),
            (INPUT_TERMINAL, MIC, 0x0201),
            (OUTPUT_TERMINAL, MIC_OUTPUT, 0x0101),
        ]
    );
    // The microphone: unassociated, one channel on the shared clock,
    // through unit 6 to the streaming terminal.
    let at = config
        .windows(4)
        .position(|d| d == [17, CS_INTERFACE, INPUT_TERMINAL, MIC])
        .unwrap();
    assert_eq!(config[at + 6..at + 9], [0, CLOCK, 1]);
    let at = config
        .windows(4)
        .position(|d| d == [12, CS_INTERFACE, OUTPUT_TERMINAL, MIC_OUTPUT])
        .unwrap();
    assert_eq!(config[at + 7..at + 9], [MIC_FEATURE, CLOCK]);
    // Master mute and volume on both units, sized for their channels.
    assert_eq!(w.features.len(), 2);
    assert_eq!(
        w.features[0][..6],
        [18, CS_INTERFACE, FEATURE_UNIT, FEATURE, INPUT, 0x0f]
    );
    assert_eq!(
        w.features[1],
        [
            14,
            CS_INTERFACE,
            FEATURE_UNIT,
            MIC_FEATURE,
            MIC,
            0x0f,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ]
    );
    // Adaptive stereo OUT and asynchronous mono IN, 49 frames a packet.
    assert_eq!(
        w.settings,
        [
            (1, 1, INPUT, 2, STREAM, 0x09, 49 * 2 * 2, 4),
            (1, 2, INPUT, 2, STREAM, 0x09, 49 * 2 * 3, 4),
            (2, 1, MIC_OUTPUT, 1, CAPTURE, 0x05, 49 * 2, 4),
            (2, 2, MIC_OUTPUT, 1, CAPTURE, 0x05, 49 * 3, 4),
        ]
    );
    // Full speed: the same function, its endpoints polled every frame.
    let other = transfer(&mut a, setup(0x80, GET_DESCRIPTOR, 0x700, 0, 512), &[]).unwrap();
    assert_eq!(other.len(), config.len());
    assert_eq!(other[4], 3);
    let o = walk(&other);
    assert_eq!(o.terminals, w.terminals);
    assert_eq!(o.header, w.header);
    assert_eq!(o.features, w.features);
    let full: Vec<Setting> = w
        .settings
        .iter()
        .map(|&(i, a, t, c, e, m, p, _)| (i, a, t, c, e, m, p, 1))
        .collect();
    assert_eq!(o.settings, full);
}

#[test]
fn clock_and_feature_unit_requests_answer_as_uac2_defines() {
    let mut a = configured();
    assert_eq!(
        transfer(&mut a, entity(0xa1, CUR, SAM_FREQ, CLOCK, 4), &[]).unwrap(),
        48_000u32.to_le_bytes()
    );
    // A host reads the subrange count first, then the whole block.
    assert_eq!(
        transfer(&mut a, entity(0xa1, RANGE, SAM_FREQ, CLOCK, 2), &[]).unwrap(),
        [1, 0]
    );
    let block = transfer(&mut a, entity(0xa1, RANGE, SAM_FREQ, CLOCK, 14), &[]).unwrap();
    assert_eq!(&block[2..6], 48_000u32.to_le_bytes());
    assert_eq!(&block[6..10], 48_000u32.to_le_bytes());
    assert_eq!(
        transfer(&mut a, entity(0xa1, CUR, CLOCK_VALID, CLOCK, 1), &[]).unwrap(),
        [1]
    );
    transfer(
        &mut a,
        entity(0x21, CUR, SAM_FREQ, CLOCK, 4),
        &48_000u32.to_le_bytes(),
    )
    .unwrap();
    assert!(
        transfer(
            &mut a,
            entity(0x21, CUR, SAM_FREQ, CLOCK, 4),
            &44_100u32.to_le_bytes()
        )
        .is_err()
    );
    // Volume: range, clamping and rounding toward the minimum.
    assert_eq!(
        transfer(&mut a, entity(0xa1, RANGE, VOLUME, FEATURE, 8), &[]).unwrap(),
        [1, 0, 0x00, 0xc4, 0, 0, 0x80, 0]
    );
    for (asked, kept) in [
        (-6 * 256, -6 * 256),
        (-77, -128),
        (100, 0),
        (-30_000, -60 * 256),
    ] {
        transfer(
            &mut a,
            entity(0x21, CUR, VOLUME, FEATURE, 2),
            &i16::to_le_bytes(asked),
        )
        .unwrap();
        assert_eq!(
            transfer(&mut a, entity(0xa1, CUR, VOLUME, FEATURE, 2), &[]).unwrap(),
            i16::to_le_bytes(kept)
        );
    }
    transfer(&mut a, entity(0x21, CUR, MUTE, FEATURE, 1), &[1]).unwrap();
    assert_eq!(
        transfer(&mut a, entity(0xa1, CUR, MUTE, FEATURE, 1), &[]).unwrap(),
        [1]
    );
    // Per-channel controls, unknown entities and selectors STALL.
    assert!(transfer(&mut a, setup(0xa1, CUR, 0x0201, 0x0300, 2), &[]).is_err());
    assert!(transfer(&mut a, entity(0xa1, CUR, VOLUME, 9, 2), &[]).is_err());
    assert!(transfer(&mut a, entity(0xa1, CUR, 7, FEATURE, 2), &[]).is_err());
}

#[test]
fn streamed_samples_reach_the_ring_msb_aligned_with_gain_applied() {
    let mut a = configured();
    // No setting selected: the endpoint does not exist.
    assert_eq!(a.packet(STREAM, Token::Out, &mut [0; 4]), Ok(Reply::Stall));
    stream(&mut a, 1);
    assert!(a.streaming());
    let mut packet = [0x34, 0x12, 0xcc, 0xed, 0xff, 0x7f, 0x00, 0x80];
    assert_eq!(a.packet(STREAM, Token::Out, &mut packet), Ok(Reply::Ack(8)));
    let mut out = [[0; 2]; 4];
    assert_eq!(a.take(&mut out), 2);
    assert_eq!(out[0], [0x1234_0000, 0xedcc_0000u32 as i32]);
    assert_eq!(out[1], [0x7fff_0000, i32::MIN]);
    stream(&mut a, 2);
    let mut packet = [0x56, 0x34, 0x12, 0x00, 0x00, 0x80, 0xff];
    a.packet(STREAM, Token::Out, &mut packet).unwrap();
    assert_eq!(a.take(&mut out), 1);
    assert_eq!(out[0], [0x1234_5600, i32::MIN]);
    assert_eq!(a.status().counters.split, 1);
    // -6 dB halves the amplitude to within the step table's rounding.
    transfer(
        &mut a,
        entity(0x21, CUR, VOLUME, FEATURE, 2),
        &i16::to_le_bytes(-6 * 256),
    )
    .unwrap();
    let mut packet = [0, 0, 0x40, 0, 0, 0x40];
    a.packet(STREAM, Token::Out, &mut packet).unwrap();
    a.take(&mut out);
    let half = f64::from(out[0][0]) / f64::from(0x4000_0000);
    assert!((half - 0.501).abs() < 0.001, "{half}");
    transfer(&mut a, entity(0x21, CUR, MUTE, FEATURE, 1), &[1]).unwrap();
    a.packet(STREAM, Token::Out, &mut packet).unwrap();
    a.take(&mut out);
    assert_eq!(out[0], [0, 0]);
}

#[test]
fn a_full_ring_drops_its_oldest_frames_and_asks_for_service() {
    let mut a = configured();
    stream(&mut a, 1);
    let mut first = true;
    // 300 ms of 1 ms packets, frame n carrying n.
    for packet in 0..300u32 {
        let mut bytes = Vec::new();
        for n in packet * 48..packet * 48 + 48 {
            bytes.extend_from_slice(&(n as u16).to_le_bytes());
            bytes.extend_from_slice(&(n as u16).to_le_bytes());
        }
        a.packet(STREAM, Token::Out, &mut bytes).unwrap();
        if first {
            // One 1 ms packet is under the 5 ms that asks for service.
            assert!(!a.take_due());
            first = false;
        }
    }
    assert!(a.take_due());
    assert!(!a.take_due());
    assert_eq!(a.queued(), RING_FRAMES);
    let status = a.status();
    assert_eq!(status.counters.received, 300 * 48);
    assert_eq!(status.counters.overflowed, (300 * 48 - RING_FRAMES) as u64);
    let mut out = [[0; 2]; 1];
    a.take(&mut out);
    let oldest = (300 * 48 - RING_FRAMES) as u16;
    assert_eq!(out[0][0], i32::from(oldest as i16) << 16);
    // Leaving the setting keeps what is queued; reconfiguring drops it.
    stream(&mut a, 0);
    assert!(!a.streaming());
    assert_eq!(a.queued(), RING_FRAMES - 1);
    transfer(&mut a, setup(0, SET_CONFIGURATION, 1, 0, 0), &[]).unwrap();
    assert_eq!(a.queued(), 0);
}

#[test]
fn the_microphone_unit_and_capture_interface_answer_like_the_speakers() {
    let mut a = configured();
    // Unit 6: the speaker unit's volume range, clamping and rounding.
    assert_eq!(
        transfer(&mut a, entity(0xa1, RANGE, VOLUME, MIC_FEATURE, 8), &[]).unwrap(),
        [1, 0, 0x00, 0xc4, 0, 0, 0x80, 0]
    );
    for (asked, kept) in [
        (-6 * 256, -6 * 256),
        (-77, -128),
        (100, 0),
        (-30_000, -60 * 256),
    ] {
        transfer(
            &mut a,
            entity(0x21, CUR, VOLUME, MIC_FEATURE, 2),
            &i16::to_le_bytes(asked),
        )
        .unwrap();
        assert_eq!(
            transfer(&mut a, entity(0xa1, CUR, VOLUME, MIC_FEATURE, 2), &[]).unwrap(),
            i16::to_le_bytes(kept)
        );
    }
    transfer(&mut a, entity(0x21, CUR, MUTE, MIC_FEATURE, 1), &[1]).unwrap();
    assert_eq!(
        transfer(&mut a, entity(0xa1, CUR, MUTE, MIC_FEATURE, 1), &[]).unwrap(),
        [1]
    );
    // The speaker's unit is its own.
    assert_eq!(
        transfer(&mut a, entity(0xa1, CUR, MUTE, FEATURE, 1), &[]).unwrap(),
        [0]
    );
    assert_eq!(
        transfer(&mut a, entity(0xa1, CUR, VOLUME, FEATURE, 2), &[]).unwrap(),
        [0, 0]
    );
    let status = a.capture_status();
    assert!(status.muted);
    assert_eq!(status.volume, -60 * 256);
    assert!(!a.status().muted);
    // Per-channel controls and unknown selectors STALL.
    assert!(transfer(&mut a, setup(0xa1, CUR, 0x0201, 0x0600, 2), &[]).is_err());
    assert!(transfer(&mut a, entity(0xa1, CUR, 7, MIC_FEATURE, 2), &[]).is_err());
    // Interface 2's settings.
    let get =
        |a: &mut Audio, interface| transfer(a, setup(0x81, GET_INTERFACE, 0, interface, 1), &[]);
    assert_eq!(get(&mut a, 2).unwrap(), [0]);
    assert!(!a.capturing());
    capture(&mut a, 2);
    assert_eq!(get(&mut a, 2).unwrap(), [2]);
    assert_eq!(get(&mut a, 1).unwrap(), [0], "playback stays idle");
    assert!(a.capturing());
    assert_eq!(a.capture_status().bits, 24);
    capture(&mut a, 1);
    assert_eq!(a.capture_status().bits, 16);
    assert!(transfer(&mut a, setup(0x01, SET_INTERFACE, 3, 2, 0), &[]).is_err());
    assert!(get(&mut a, 3).is_err());
    assert_eq!(
        transfer(&mut a, setup(0x81, GET_STATUS, 0, 2, 2), &[]).unwrap(),
        [0, 0]
    );
    assert!(transfer(&mut a, setup(0x81, GET_STATUS, 0, 3, 2), &[]).is_err());
    // Endpoint 0x82 has no halt, like the OUT endpoint.
    assert_eq!(
        transfer(&mut a, setup(0x82, GET_STATUS, 0, 0x82, 2), &[]).unwrap(),
        [0, 0]
    );
    transfer(
        &mut a,
        setup(0x02, CLEAR_FEATURE, ENDPOINT_HALT, 0x82, 0),
        &[],
    )
    .unwrap();
    assert!(transfer(&mut a, setup(0x82, GET_STATUS, 0, 0x83, 2), &[]).is_err());
    // Reconfiguring idles capture.
    transfer(&mut a, setup(0, SET_CONFIGURATION, 1, 0, 0), &[]).unwrap();
    assert!(!a.capturing());
    assert_eq!(get(&mut a, 2).unwrap(), [0]);
}

/// A configured function capturing at `alternate` with `samples` and
/// zeros after them to the FIFO's target.
fn capturing(alternate: u16, samples: &[i32]) -> Audio {
    let mut a = configured();
    capture(&mut a, alternate);
    a.give(samples);
    a.give(&vec![0; MIC_TARGET - samples.len()]);
    a
}

#[test]
fn captured_samples_leave_msb_first_with_the_microphone_level_applied() {
    let mut a = configured();
    // No setting selected: the endpoint does not exist.
    assert_eq!(a.packet(CAPTURE, Token::In, &mut [0; 98]), Ok(Reply::Stall));
    capture(&mut a, 1);
    // Under the target the FIFO fills while the packets carry nothing.
    a.give(&[0x1234_5678, i32::MIN, i32::MAX]);
    let mut packet = [0u8; 147];
    assert_eq!(
        a.packet(CAPTURE, Token::In, &mut packet[..98]),
        Ok(Reply::Ack(0))
    );
    a.give(&vec![0; MIC_TARGET - 3]);
    // At the target: the nominal 48 frames.
    assert_eq!(
        a.packet(CAPTURE, Token::In, &mut packet[..98]),
        Ok(Reply::Ack(96))
    );
    assert_eq!(packet[..6], [0x34, 0x12, 0x00, 0x80, 0xff, 0x7f]);
    let counters = a.capture_status().counters;
    assert_eq!((counters.empty, counters.sent), (1, 48));
    // 24-bit: three bytes of each sample.
    let mut a = capturing(2, &[0x1234_5678, 0x7f80_0100]);
    assert_eq!(
        a.packet(CAPTURE, Token::In, &mut packet),
        Ok(Reply::Ack(144))
    );
    assert_eq!(packet[..6], [0x56, 0x34, 0x12, 0x01, 0x80, 0x7f]);
    // -6 dB on unit 6 halves the amplitude within the step table's
    // rounding; mute silences.
    let mut a = capturing(1, &[0x4000_0000]);
    transfer(
        &mut a,
        entity(0x21, CUR, VOLUME, MIC_FEATURE, 2),
        &i16::to_le_bytes(-6 * 256),
    )
    .unwrap();
    a.packet(CAPTURE, Token::In, &mut packet[..98]).unwrap();
    let half = f64::from(i16::from_le_bytes([packet[0], packet[1]])) / f64::from(0x4000);
    assert!((half - 0.501).abs() < 0.001, "{half}");
    let mut a = capturing(1, &[0x4000_0000]);
    transfer(&mut a, entity(0x21, CUR, MUTE, MIC_FEATURE, 1), &[1]).unwrap();
    a.packet(CAPTURE, Token::In, &mut packet[..98]).unwrap();
    assert_eq!(packet[..2], [0, 0]);
    // The speaker's level does not touch the microphone.
    let mut a = capturing(1, &[0x4000_0000]);
    transfer(&mut a, entity(0x21, CUR, MUTE, FEATURE, 1), &[1]).unwrap();
    a.packet(CAPTURE, Token::In, &mut packet[..98]).unwrap();
    assert_eq!(packet[..2], [0x00, 0x40]);
}

/// The frame counts of IN packets until the first zero-length one, each
/// packet's 16-bit samples checked to continue the ramp from `next`.
fn drain(a: &mut Audio, next: &mut i32) -> Vec<usize> {
    let mut sizes = Vec::new();
    loop {
        let mut packet = [0u8; 98];
        let Ok(Reply::Ack(n)) = a.packet(CAPTURE, Token::In, &mut packet) else {
            panic!("capture packet");
        };
        if n == 0 {
            return sizes;
        }
        for &bytes in packet[..n].as_chunks::<2>().0 {
            assert_eq!(i16::from_le_bytes(bytes), *next as i16);
            *next += 1;
        }
        sizes.push(n / 2);
    }
}

#[test]
fn capture_packets_follow_the_fifo_fill_and_overflow_drops_the_oldest() {
    let mut a = configured();
    // Sample n carries n in its top 16 bits.
    let ramp = |from: i32, count: usize| -> Vec<i32> {
        (from..from + count as i32).map(|n| n << 16).collect()
    };
    // Given while the guest is idle: a full FIFO drops its oldest, and
    // starting capture keeps only the target's worth, the newest.
    a.give(&ramp(0, MIC_FRAMES + 10));
    assert_eq!(a.capture_queued(), MIC_FRAMES);
    assert_eq!(a.capture_status().counters.dropped, 10);
    capture(&mut a, 1);
    assert_eq!(a.capture_queued(), MIC_TARGET);
    let counters = a.capture_status().counters;
    assert_eq!(counters.given, (MIC_FRAMES + 10) as u64);
    assert_eq!(counters.dropped, (MIC_FRAMES + 10 - MIC_TARGET) as u64);
    // From the target the stream drains: nominal packets within 1 ms of
    // it, one frame fewer below, then the dry FIFO's short packet.
    let mut next = (MIC_FRAMES + 10 - MIC_TARGET) as i32;
    let sizes = drain(&mut a, &mut next);
    assert_eq!(sizes.iter().sum::<usize>(), MIC_TARGET);
    assert_eq!(sizes[..2], [48, 48]);
    assert!(
        sizes[2..sizes.len() - 1].iter().all(|&n| n == 47),
        "{sizes:?}"
    );
    assert!(*sizes.last().unwrap() < 47);
    let counters = a.capture_status().counters;
    assert_eq!(counters.underruns, 1);
    assert_eq!(counters.ahead, 0);
    assert_eq!(counters.behind as usize, sizes.len() - 2);
    assert_eq!(counters.empty, 1);
    // Dry, the stream waits for the target again rather than trickling.
    a.give(&ramp(next, MIC_TARGET - 1));
    assert_eq!(
        a.packet(CAPTURE, Token::In, &mut [0; 98]),
        Ok(Reply::Ack(0))
    );
    assert_eq!(a.capture_status().counters.empty, 2);
    // Over the target, packets carry a frame more until the fill is back
    // within 1 ms of it.
    a.give(&ramp(
        next + MIC_TARGET as i32 - 1,
        MIC_FRAMES - MIC_TARGET + 1,
    ));
    assert_eq!(a.capture_queued(), MIC_FRAMES);
    let sizes = drain(&mut a, &mut next);
    assert_eq!(sizes.iter().sum::<usize>(), MIC_FRAMES);
    let ahead = sizes.iter().filter(|&&n| n == 49).count();
    // 49-frame packets take the fill from 60 ms to within 1 ms of 20 ms.
    assert_eq!(ahead, (MIC_FRAMES - MIC_TARGET - 48).div_ceil(49));
    assert!(sizes.is_sorted_by(|a, b| a >= b), "{sizes:?}");
    assert_eq!(
        a.capture_status().counters.sent,
        (MIC_TARGET + MIC_FRAMES) as u64
    );
}

struct One(Audio);
impl Ports for One {
    fn count(&self) -> usize {
        1
    }
    fn speed(&self, _: usize) -> Speed {
        Speed::High
    }
    fn function(&mut self, _: usize) -> &mut dyn Function {
        &mut self.0
    }
    fn function_ref(&self, _: usize) -> &dyn Function {
        &self.0
    }
}

/// The host's own UAC2 class driver binds the function behind the native
/// hub on the modeled xHCI and plays PCM through its isochronous endpoint:
/// the descriptors, the class requests and the isochronous OUT path.
#[test]
fn a_uac2_host_driver_plays_through_the_isochronous_endpoint() {
    let hub = Hub::new(One(Audio::new()), "HUB").unwrap();
    let mut bench = host::bench(hub);
    let mut found = host::run(&mut bench);
    assert!(found.failures.is_empty(), "{:?}", found.failures);
    assert_eq!(found.audio.len(), 1, "{:?}", found.unused);
    let mut audio = found.audio.remove(0);
    let formats: Vec<_> = audio
        .binding()
        .formats
        .iter()
        .map(|f| (f.direction, f.channels, f.subslot, f.bits, f.sync.clone()))
        .collect();
    // The asynchronous microphone is, to Linux's rules, a synchronous
    // capture: its packets carry the rate.
    assert_eq!(
        formats,
        [
            (Direction::Playback, 2, 2, 16, StreamSync::Adaptive),
            (Direction::Playback, 2, 3, 24, StreamSync::Adaptive),
            (Direction::Capture, 1, 2, 16, StreamSync::Synchronous),
            (Direction::Capture, 1, 3, 24, StreamSync::Synchronous),
        ]
    );
    for (index, subslot) in [(1usize, 3usize), (0, 2)] {
        // 100 ms of a ramp, each sample's subslot bytes distinct.
        let frames = 4800usize;
        let mut pcm = Vec::new();
        for n in 0..frames * 2 {
            let value = (n as u32).wrapping_mul(2_654_435_761);
            pcm.extend_from_slice(&value.to_le_bytes()[..subslot]);
        }
        audio.start_playback(index, RATE, &pcm, true).unwrap();
        let mut drained = false;
        for _ in 0..20_000 {
            if audio.poll().unwrap().playback_drained {
                drained = true;
                break;
            }
        }
        assert!(drained, "{:?}", audio.progress());
        let model = &mut bench.model.borrow_mut().function.ports.0;
        // A finite stream ends by selecting the zero-bandwidth setting.
        assert!(!model.streaming());
        let mut out = vec![[0i32; 2]; frames + 1];
        assert_eq!(model.take(&mut out), frames);
        for (n, frame) in out[..frames].iter().enumerate() {
            for (c, sample) in frame.iter().enumerate() {
                let value = ((n * 2 + c) as u32).wrapping_mul(2_654_435_761);
                let mut word = [0u8; 4];
                word[4 - subslot..].copy_from_slice(&value.to_le_bytes()[..subslot]);
                assert_eq!(*sample, i32::from_le_bytes(word), "frame {n}");
            }
        }
        assert_eq!(model.status().counters.overflowed, 0);
    }
    // The host's mute and volume reach the feature unit.
    audio.set_mute(FEATURE, 0, true).unwrap();
    audio.set_volume(FEATURE, 0, -12 * 256).unwrap();
    let status = bench.model.borrow().function.ports.0.status();
    assert!(status.muted);
    assert_eq!(status.volume, -12 * 256);
}

/// The host's own UAC2 class driver captures the function's microphone
/// through the isochronous IN endpoint: samples the owner gives arrive in
/// order in each setting's sample size, and the packet sizes follow the
/// capture FIFO's fill.
#[test]
fn a_uac2_host_driver_captures_through_the_isochronous_in_endpoint() {
    let hub = Hub::new(One(Audio::new()), "HUB").unwrap();
    let mut bench = host::bench(hub);
    let mut found = host::run(&mut bench);
    assert!(found.failures.is_empty(), "{:?}", found.failures);
    assert_eq!(found.audio.len(), 1, "{:?}", found.unused);
    let mut audio = found.audio.remove(0);
    // The owner picks a host microphone by these.
    assert_eq!(audio.identity(), (0x1209, 0x0003));
    let captures: Vec<(usize, usize)> = audio
        .binding()
        .formats
        .iter()
        .enumerate()
        .filter(|(_, f)| f.direction == Direction::Capture)
        .map(|(index, f)| (index, usize::from(f.subslot)))
        .collect();
    assert_eq!(captures.len(), 2);
    for &(index, subslot) in captures.iter().rev() {
        let before = bench.model.borrow().function.ports.0.capture_status();
        audio.start_capture(index, RATE).unwrap();
        // 50 ms of the host microphone's samples, over the 20 ms target.
        let frames = 2400usize;
        let sample = |n: usize| (n as u32).wrapping_mul(2_654_435_761) as i32;
        {
            let model = &mut bench.model.borrow_mut().function.ports.0;
            assert!(model.capturing());
            assert_eq!(model.capture_status().bits as usize, subslot * 8);
            model.give(&(0..frames).map(sample).collect::<Vec<_>>());
        }
        let mut pcm = Vec::new();
        let mut chunk = vec![0u8; 49 * subslot * 4];
        for _ in 0..20_000 {
            audio.poll().unwrap();
            let n = audio.read_capture(&mut chunk).unwrap();
            pcm.extend_from_slice(&chunk[..n]);
            if pcm.len() >= frames * subslot {
                break;
            }
        }
        assert_eq!(pcm.len(), frames * subslot, "{:?}", audio.progress());
        for (n, bytes) in pcm.chunks_exact(subslot).enumerate() {
            assert_eq!(bytes, &sample(n).to_le_bytes()[4 - subslot..], "frame {n}");
        }
        audio.stop_capture().unwrap();
        let model = &bench.model.borrow().function.ports.0;
        assert!(!model.capturing());
        let (was, now) = (before.counters, model.capture_status().counters);
        assert_eq!(now.sent - was.sent, frames as u64);
        assert_eq!(now.dropped, was.dropped);
        // Over the target the packets ran a frame long, under it short.
        assert!(now.ahead > was.ahead, "{now:?}");
        assert!(now.behind > was.behind, "{now:?}");
    }
    // The host's mute and volume reach the microphone's feature unit.
    audio.set_mute(MIC_FEATURE, 0, true).unwrap();
    audio.set_volume(MIC_FEATURE, 0, -12 * 256).unwrap();
    let model = &bench.model.borrow().function.ports.0;
    let status = model.capture_status();
    assert!(status.muted);
    assert_eq!(status.volume, -12 * 256);
    assert!(!model.status().muted);
}
