use super::*;
use alloc::vec;

/// An ALC897's pins as a B760 desktop board's firmware describes them
/// (Linux `alsa-info` of such boards): rear line out 0x14, rear mic 0x18,
/// front mic 0x19, rear line in 0x1a, front headphone 0x1b, and the unused
/// 0x15-0x17 and 0x1e.
const ALC897: [(u8, u32); 6] = [
    (0x14, 0x0101_4010),
    (0x15, 0x4000_0000),
    (0x16, 0x4111_11f0),
    (0x17, 0x4111_11f0),
    (0x1b, 0x0221_4020),
    (0x1e, 0x4111_11f0),
];

#[test]
fn the_front_headphone_pin_plays_the_guest() {
    assert_eq!(output_rank(0x0221_4020), Some(0));
    assert_eq!(output_rank(0x0101_4010), Some(4));
    // No physical connection, or no configuration at all.
    assert_eq!(output_rank(0x4111_11f0), None);
    assert_eq!(output_rank(u32::MAX), None);
    // Inputs are no output.
    assert_eq!(output_rank(0x02a1_9030), None);
    assert_eq!(choose_output(&ALC897, None), Some(4));
    // Without a front headphone pin, the rear line out.
    assert_eq!(choose_output(&ALC897[..4], None), Some(0));
    // An internal speaker before a rear line out; a front line out first.
    let laptop = [(0x14, 0x9017_0110), (0x21, 0x0421_1020)];
    assert_eq!(choose_output(&laptop, None), Some(1));
    assert_eq!(choose_output(&laptop[..1], None), Some(0));
    assert_eq!(
        choose_output(&[(0x14, 0x0101_4010), (0x15, 0x0201_4010)], None),
        Some(1)
    );
    // The owner's pin, when named, whatever it is; none when absent.
    assert_eq!(choose_output(&ALC897, Some(0x14)), Some(0));
    assert_eq!(choose_output(&ALC897, Some(0x30)), None);
    assert_eq!(choose_output(&[], None), None);
}

#[test]
fn samples_are_encoded_as_the_stream_takes_them() {
    let frames = [[0x1234_5678, -0x1234_5678], [i32::MAX, i32::MIN]];
    let mut out = Vec::new();
    encode(&frames, 16, &mut out);
    assert_eq!(out, [0x34, 0x12, 0xcb, 0xed, 0xff, 0x7f, 0x00, 0x80]);
    out.clear();
    encode(&frames[..1], 24, &mut out);
    assert_eq!(out, [0x78, 0x56, 0x34, 0x12, 0x88, 0xa9, 0xcb, 0xed]);
    assert_eq!(frame_bytes(16), 4);
    assert_eq!(frame_bytes(24), 8);
}

/// Every write lands ahead of the controller and every played slot is
/// zeroed once, over several laps.
#[test]
fn the_ring_keeps_its_fill_and_cleans_what_played() {
    let n = 1000;
    let mut ring = OutputRing::new(n, 10, 300);
    let mut slots = vec![0u64; n];
    // Slot contents as the absolute frame written there, 0 for silence.
    let mut absolute = 0u64;
    let mut position = 0usize;
    for _ in 0..100 {
        // The controller plays 97 frames between services.
        position = (position + 97) % n;
        ring.observe(position, 97, true);
        for (start, len) in ring.take_clean() {
            slots[start..start + len].fill(0);
        }
        let room = ring.room();
        let written = ring.written;
        for (start, len) in ring.write_spans(room) {
            for slot in &mut slots[start..start + len] {
                absolute += 1;
                *slot = absolute;
            }
        }
        ring.commit(room);
        assert_eq!(ring.fill(), 300);
        assert!(written >= ring.played + 10);
        // Nothing the controller already played is left for its next lap.
        let behind = (0..n - 300)
            .map(|k| (ring.written as usize + k) % n)
            .filter(|&slot| slots[slot] != 0)
            .count();
        assert_eq!(behind, 0);
    }
    assert_eq!(ring.underruns, 0);
}

#[test]
fn a_late_service_is_an_underrun_and_writes_go_on_past_the_fifo() {
    let mut ring = OutputRing::new(1000, 10, 300);
    ring.observe(0, 0, true);
    ring.commit(ring.room());
    assert_eq!(ring.fill(), 300);
    // The controller ran past the fresh frames into silence.
    ring.observe(400, 400, true);
    assert_eq!(ring.underruns, 1);
    assert_eq!(ring.fill(), 10);
    assert_eq!(ring.write_spans(5), [(410, 5), (0, 0)]);
    // Running dry again without fresh frames counts nothing more.
    ring.observe(600, 200, true);
    assert_eq!(ring.underruns, 1);
    // Nor does the end of the guest's sound.
    ring.commit(ring.room());
    ring.observe(900, 300, false);
    assert_eq!(ring.underruns, 1);
    assert_eq!(ring.lost, 0);
}

#[test]
fn a_gap_of_a_lap_zeroes_the_whole_ring() {
    let mut ring = OutputRing::new(1000, 10, 300);
    ring.commit(ring.room());
    // Two laps and 100 frames by the clock: the position alone says 100.
    ring.observe(100, 2100, true);
    assert_eq!(ring.lost, 1);
    assert_eq!(ring.underruns, 1);
    assert_eq!(ring.played, 2100);
    let cleaned: usize = ring.take_clean().iter().map(|(_, len)| len).sum();
    assert_eq!(cleaned, 1000 - 10);
    assert_eq!(ring.write_spans(1), [(110, 1), (0, 0)]);
}

#[test]
fn writes_wrap_in_two_spans() {
    let mut ring = OutputRing::new(100, 4, 40);
    ring.observe(90, 90, false);
    assert_eq!(ring.write_spans(ring.room()), [(94, 6), (0, 30)]);
}

fn ramp(from: i32, count: usize) -> Vec<Frame> {
    (0..count as i32).map(|n| [from + n, -(from + n)]).collect()
}

/// A Q32 step of `ratio` input frames per output frame.
fn step_of(ratio: f64) -> u64 {
    (ratio * (1u64 << 32) as f64).round() as u64
}

/// The sizes of a run of writes, varied: 1 to `most` by a fixed LCG.
fn sizes(seed: &mut u64, most: usize) -> usize {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*seed >> 33) as usize % most + 1
}

const AMPLITUDE: f64 = 1.0e9;

/// A 1 kHz sine at 48 kHz, its right channel the cosine, at input frame
/// `time`.
fn sine(time: f64) -> Frame {
    let angle = 2.0 * core::f64::consts::PI * 1000.0 * time / RATE as f64;
    [
        (AMPLITUDE * angle.sin()).round() as i32,
        (AMPLITUDE * angle.cos()).round() as i32,
    ]
}

#[test]
fn a_ratio_of_one_passes_frames_bit_for_bit() {
    let mut resampler = StereoResampler::default();
    let mut seed = 7;
    let input: Vec<Frame> = (0..20_000)
        .map(|_| {
            sizes(&mut seed, 1 << 30);
            [(seed >> 32) as i32, seed as i32]
        })
        .chain([[i32::MIN, i32::MAX], [i32::MAX, i32::MIN], [0, -1]])
        .collect();
    let mut queue = VecDeque::new();
    let mut out = Vec::new();
    let mut fed = 0;
    let (mut passed, mut made) = (0, 0);
    while fed < input.len() {
        let chunk = sizes(&mut seed, 300).min(input.len() - fed);
        queue.extend(&input[fed..fed + chunk]);
        fed += chunk;
        let (p, m) = resampler.pull(&mut queue, sizes(&mut seed, 400), &mut out);
        passed += p;
        made += m;
    }
    let (p, m) = resampler.pull(&mut queue, usize::MAX, &mut out);
    passed += p;
    made += m;
    // All but the last, which waits for the frame after it.
    assert_eq!(out, input[..input.len() - 1]);
    assert_eq!((passed, made), (input.len() - 1, input.len() - 1));
    assert!(queue.is_empty());
}

/// The control law at rest (queue and fill on target) asks for a ratio of
/// one, and the guest's frames pass untouched through the host queue.
#[test]
fn a_guest_on_time_is_played_untouched() {
    let mut pending = Pending::default();
    let step = 5 * FRAMES_PER_MS;
    // With each write's frames added, the queue stands at its target; the
    // one more is the frame the resampler holds.
    let first = ramp(1_000, PENDING_MS * FRAMES_PER_MS - step + 1);
    pending.add(&first);
    let mut out = Vec::new();
    let mut next = first.len() as i32 + 1_000;
    for _ in 0..100 {
        pending.add(&ramp(next, step));
        next += step as i32;
        pending.take(step, TARGET_MS * FRAMES_PER_MS, &mut out);
        assert_eq!(pending.step(), 1 << 32);
    }
    assert!(out.iter().zip(1_000..).all(|(f, n)| *f == [n, -n]));
    assert_eq!(out.len(), 100 * step);
    assert_eq!(pending.stretched, 0);
}

/// A 1 kHz sine resampled at nine and eleven tenths, through writes and
/// reads of every size, is the sine at the new times: within the error of
/// linear interpolation (a 1 kHz sine at 48 kHz bends at most
/// (2 pi / 48)^2 / 8 of its amplitude, 0.214%, between two frames; 0.2137%
/// measured) and with
/// no break at any write's boundary.
#[test]
fn a_sine_resampled_at_a_tenth_off_stays_a_sine() {
    for ratio in [0.9, 1.1, 7.0 / 8.0, 9.0 / 8.0] {
        let step = step_of(ratio);
        let mut resampler = StereoResampler::new(step);
        let mut queue = VecDeque::new();
        let mut out = Vec::new();
        let mut seed = 42;
        let mut fed = 0usize;
        while out.len() < 200_000 {
            let chunk = sizes(&mut seed, 500);
            queue.extend((fed..fed + chunk).map(|n| sine(n as f64)));
            fed += chunk;
            resampler.pull(&mut queue, sizes(&mut seed, 500), &mut out);
        }
        let mut worst = 0f64;
        for (k, frame) in out.iter().enumerate() {
            let time = (k as u128 * u128::from(step)) as f64 / (1u64 << 32) as f64;
            let ideal = sine(time);
            for channel in 0..2 {
                let error = (f64::from(frame[channel]) - f64::from(ideal[channel])).abs();
                worst = worst.max(error);
            }
        }
        assert!(worst < AMPLITUDE * 0.0023, "{ratio}: {worst}");
    }
}

/// Frames in against frames out over a million frames, through writes and
/// reads of every size, often running the input dry: the frames passed are
/// the outputs times the ratio, rounded down, so within a frame of it, and
/// the resampler holds only the frame it interpolates from.
#[test]
fn a_million_frames_keep_the_ratio_to_a_frame() {
    for ratio in [0.9, 1.0, 1.1, 7.0 / 8.0, 9.0 / 8.0] {
        let step = step_of(ratio);
        let mut resampler = StereoResampler::new(step);
        let mut queue = VecDeque::new();
        let mut out = Vec::new();
        let mut seed = 3;
        let (mut passed, mut made, mut fed) = (0usize, 0usize, 0usize);
        while made < 1_000_000 {
            let chunk = sizes(&mut seed, 700);
            queue.extend(ramp(fed as i32, chunk));
            fed += chunk;
            out.clear();
            let wanted = sizes(&mut seed, 700).min(1_000_000 - made);
            let (p, m) = resampler.pull(&mut queue, wanted, &mut out);
            passed += p;
            made += m;
            assert_eq!(fed, passed + 1 + queue.len());
        }
        // The time may have run past the input; a frame more catches up.
        queue.push_back([0; 2]);
        let (p, m) = resampler.pull(&mut queue, 0, &mut out);
        passed += p;
        assert_eq!(m, 0);
        let exact = (made as u128 * u128::from(step)) >> 32;
        assert_eq!(passed as u128, exact, "{ratio}: {passed} for {made}");
        let expected = made as f64 * ratio;
        assert!(
            (passed as f64 - expected).abs() <= 1.0,
            "{ratio}: {passed} for {made}"
        );
        assert!(((step as f64 / (1u64 << 32) as f64) - ratio).abs() < 1e-9);
    }
}

#[test]
fn extremes_resample_without_overflow() {
    for step in [1, step_of(0.9), step_of(1.1), u64::MAX] {
        let mut resampler = StereoResampler::new(step);
        let mut queue: VecDeque<Frame> = (0..1000)
            .map(|n| {
                if n % 2 == 0 {
                    [i32::MIN, i32::MAX]
                } else {
                    [i32::MAX, i32::MIN]
                }
            })
            .collect();
        let mut out = Vec::new();
        resampler.pull(&mut queue, 5000, &mut out);
        assert!(!out.is_empty());
        assert_eq!(out[0], [i32::MIN, i32::MAX]);
    }
    assert_eq!(StereoResampler::new(0).step(), 1);
    assert_eq!(
        StereoResampler::new(u64::MAX).step(),
        StereoResampler::MAX_STEP
    );
    // A fill far past anything real asks for no more than an eighth.
    let mut pending = Pending::default();
    pending.add(&ramp(0, 2000));
    let mut out = Vec::new();
    for _ in 0..100 {
        pending.take(800, 1 << 40, &mut out);
    }
    assert_eq!(pending.step(), (1 << 32) / 8 * 9);
}

/// A sudden error eases in while the sound runs on: the ratio moves at
/// most 1/128 a write and settles exactly on the law's, one included. Once
/// the input ran dry, the next ratio is taken at once.
#[test]
fn the_ratio_eases_toward_the_law() {
    let mut resampler = StereoResampler::default();
    let mut queue: VecDeque<Frame> = ramp(0, 1000).into();
    let mut out = Vec::new();
    let fast = (1u64 << 32) / 8 * 9;
    // The first write, with nothing playing yet, takes its ratio at once.
    resampler.steer(1 << 32);
    assert_eq!(resampler.pull(&mut queue, 1, &mut out), (1, 1));
    let mut last = resampler.step();
    for _ in 0..100 {
        resampler.steer(fast);
        assert!(resampler.step().abs_diff(last) <= 1 << 25);
        last = resampler.step();
        resampler.pull(&mut queue, 1, &mut out);
    }
    assert_eq!(resampler.step(), fast);
    for _ in 0..100 {
        resampler.steer(1 << 32);
        resampler.pull(&mut queue, 1, &mut out);
    }
    assert_eq!(resampler.step(), 1 << 32);
    // Asked for more than the input holds, then steered: all the way.
    resampler.pull(&mut queue, 10_000, &mut out);
    resampler.steer(fast);
    assert_eq!(resampler.step(), fast);
}

/// The guest a tenth fast or slow, and turning so midway, in a 1 kHz sine:
/// what plays is smooth, every frame's second difference within what a
/// sine of at most 1.125 kHz has (A (2 pi f / 48 kHz)^2, 2.2% of A) and the
/// interpolation's bends (2.35% of A measured), under 3%; one dropped or
/// repeated frame alone would make one of 13% of A, 2 pi / 48.
#[test]
fn a_drifting_guest_plays_without_breaks() {
    let mut pending = Pending::default();
    let mut ring = OutputRing::new(RING_MS * FRAMES_PER_MS, 32, TARGET_MS * FRAMES_PER_MS);
    let mut played = Vec::new();
    let mut out = Vec::new();
    let mut fed = 0usize;
    let mut position = 0;
    let mut seed = 11;
    let step = 5 * FRAMES_PER_MS;
    for round in 0..3000 {
        let arrived = match round {
            0..1000 => step * 11 / 10,
            1000..2000 => step * 9 / 10,
            _ => step,
        } + sizes(&mut seed, 9)
            - 5;
        pending.add(
            &(fed..fed + arrived)
                .map(|n| sine(n as f64))
                .collect::<Vec<_>>(),
        );
        fed += arrived;
        position = (position + step) % (RING_MS * FRAMES_PER_MS);
        ring.observe(position, step as u64, true);
        out.clear();
        pending.take(ring.room(), ring.fill(), &mut out);
        ring.commit(out.len());
        played.extend_from_slice(&out);
    }
    assert_eq!(pending.overruns, 0);
    assert_eq!(ring.underruns, 0);
    let worst = played
        .windows(3)
        .flat_map(|w| {
            (0..2).map(move |c| {
                (i64::from(w[0][c]) - 2 * i64::from(w[1][c]) + i64::from(w[2][c])).abs()
            })
        })
        .max()
        .unwrap_or(0);
    assert!((worst as f64) < AMPLITUDE * 0.03, "{worst}");
}

/// A guest a tenth fast against the controller: the queue settles and
/// nothing overruns.
#[test]
fn a_fast_guest_is_followed_without_overruns() {
    let mut pending = Pending::default();
    let mut ring = OutputRing::new(RING_MS * FRAMES_PER_MS, 32, TARGET_MS * FRAMES_PER_MS);
    let mut out = Vec::new();
    let mut next = 0;
    let mut position = 0;
    let step = 5 * FRAMES_PER_MS;
    for _ in 0..2000 {
        let arrived = step * 11 / 10;
        pending.add(&ramp(next, arrived));
        next += arrived as i32;
        position = (position + step) % (RING_MS * FRAMES_PER_MS);
        ring.observe(position, step as u64, true);
        out.clear();
        pending.take(ring.room(), ring.fill(), &mut out);
        assert!(out.len() <= ring.room());
        ring.commit(out.len());
    }
    assert_eq!(pending.overruns, 0);
    assert!(
        pending.len() < (PENDING_MS + 10) * FRAMES_PER_MS,
        "{}",
        pending.len()
    );
    // Settled near eleven tenths: a tenth of all it played taken past it.
    assert!(
        pending.step().abs_diff(step_of(1.1)) < 1 << 26,
        "{}",
        pending.step()
    );
    assert!(pending.stretched > 0);
    assert_eq!(ring.underruns, 0);
}

/// A guest a tenth slow: frames are spread, so the fill holds; the order of
/// the frames played is kept.
#[test]
fn a_slow_guest_is_stretched() {
    let mut pending = Pending::default();
    let mut ring = OutputRing::new(RING_MS * FRAMES_PER_MS, 32, TARGET_MS * FRAMES_PER_MS);
    let mut out = Vec::new();
    let mut next = 0;
    let mut position = 0;
    let mut last = -1;
    let step = 5 * FRAMES_PER_MS;
    for round in 0..2000 {
        let arrived = step * 9 / 10;
        pending.add(&ramp(next, arrived));
        next += arrived as i32;
        position = (position + step) % (RING_MS * FRAMES_PER_MS);
        ring.observe(position, step as u64, true);
        out.clear();
        pending.take(ring.room(), ring.fill(), &mut out);
        for frame in &out {
            assert!(frame[0] >= last);
            assert!(frame[0] - last <= 2 || last < 0);
            last = frame[0];
        }
        ring.commit(out.len());
        if round > 200 {
            assert!(ring.fill() > 10 * FRAMES_PER_MS, "{}", ring.fill());
        }
    }
    assert!(
        pending.step().abs_diff(step_of(0.9)) < 1 << 26,
        "{}",
        pending.step()
    );
    assert!(pending.stretched < 0);
    assert_eq!(ring.underruns, 0);
}

#[test]
fn a_full_queue_drops_its_oldest() {
    let mut pending = Pending::default();
    let most = MAX_PENDING_MS * FRAMES_PER_MS;
    pending.add(&ramp(0, most + 1));
    assert_eq!(pending.len(), PENDING_MS * FRAMES_PER_MS);
    assert_eq!(
        pending.overruns as usize,
        most + 1 - PENDING_MS * FRAMES_PER_MS
    );
    let mut out = Vec::new();
    pending.take(1, 0, &mut out);
    assert_eq!(out[0][0] as usize, most + 1 - PENDING_MS * FRAMES_PER_MS);
}

#[test]
fn the_fifine_is_the_microphone_of_choice() {
    assert_eq!(choose_mic(&[]), None);
    assert_eq!(choose_mic(&[(0x046d, false)]), None);
    assert_eq!(choose_mic(&[(0x046d, true), (0x3142, true)]), Some(1));
    // A fifine with no capture format is passed over.
    assert_eq!(choose_mic(&[(0x3142, false), (0x046d, true)]), Some(1));
    assert_eq!(choose_mic(&[(0x0d8c, false), (0x046d, true)]), Some(1));
}

#[test]
fn mono_and_narrow_capture_formats_come_first() {
    // (channels, subslot): stereo 16, mono 24, mono 16, an empty and a
    // five-byte one the converter cannot decode.
    let formats = [(2, 2), (1, 3), (1, 2), (0, 2), (1, 5)];
    assert_eq!(capture_order(&formats), [2, 1, 0]);
    assert_eq!(MIC_RATES[0], RATE);
}

#[test]
fn equal_rates_pass_samples_through_a_sample_late() {
    let mut resampler = Resampler::new(48_000, 48_000);
    let mut out = Vec::new();
    resampler.push(&[10, 20, 30], &mut out);
    resampler.push(&[40], &mut out);
    assert_eq!(out, [10, 20, 30]);
}

#[test]
fn a_lower_rate_is_interpolated_up() {
    let mut resampler = Resampler::new(16_000, 48_000);
    let mut out = Vec::new();
    resampler.push(&[0, 300, 600], &mut out);
    // Three outputs per input, on the line between them.
    assert_eq!(out, [0, 100, 200, 300, 400, 500]);
    // 44.1 kHz for a second makes 48 kHz for a second, within a sample.
    let mut resampler = Resampler::new(44_100, 48_000);
    out.clear();
    for chunk in vec![1000; 44_100].chunks(441) {
        resampler.push(chunk, &mut out);
    }
    assert!(out.len().abs_diff(48_000) <= 2, "{}", out.len());
    assert!(out.iter().all(|&sample| sample == 1000));
}

#[test]
fn extremes_interpolate_without_overflow() {
    let mut resampler = Resampler::new(32_000, 48_000);
    let mut out = Vec::new();
    resampler.push(&[i32::MIN, i32::MAX, i32::MIN], &mut out);
    assert_eq!(out[0], i32::MIN);
    assert!(out.iter().all(|&s| (i32::MIN..=i32::MAX).contains(&s)));
    assert!(out[1] > i32::MIN && out[1] < i32::MAX);
}

#[test]
fn stereo_captures_are_averaged_into_msb_aligned_mono() {
    let mut converter = MicConverter::new(2, 2, 48_000);
    let mut out = Vec::new();
    // Left 0x1000, right 0x3000, then a frame split across two reads.
    converter.convert(&[0x00, 0x10, 0x00, 0x30, 0x00, 0xf0], &mut out);
    converter.convert(&[0x00, 0xf0], &mut out);
    converter.convert(&[0, 0, 0, 0], &mut out);
    assert_eq!(out, [0x2000_0000, -0x1000_0000]);
    // Mono 24-bit samples keep their three bytes on top.
    let mut converter = MicConverter::new(1, 3, 48_000);
    out.clear();
    converter.convert(&[0x56, 0x34, 0x12, 0xaa, 0xbb, 0xcc, 0, 0, 0], &mut out);
    assert_eq!(out, [0x1234_5600, 0xccbb_aa00_u32 as i32]);
}
