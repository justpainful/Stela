//! The arithmetic of the guest's sound on the host: which analog output pin
//! of an HD Audio codec plays it, how the guest's 48 kHz stereo frames are
//! laid out in the controller's cyclic DMA buffer, how far ahead of the
//! controller they are written, and how the guest's clock is matched to the
//! controller's. `stela-boot` (`engine::audio`) takes the guest's frames from
//! its USB headset (`stela_soc::usb::audio`) and does the writes; this keeps
//! the decisions, which run on the host in tests.
//!
//! Evidence: HDA 1.0a 7.3.3.31 for the configuration default's fields (port
//! connectivity 31:30, location 29:24, default device 23:20); 3.3.37 for
//! LPIB, the link position in the cyclic buffer, and 3.3.39 for the stream's
//! FIFO size, what the controller may have fetched past LPIB; Linux's HDA
//! driver keeps 20- and 24-bit samples MSB-justified in 32-bit containers
//! (`SNDRV_PCM_FORMAT_S32_LE` with `msbits`), as the guest model keeps them.
//!
//! The controller plays its cyclic buffer without end. Fresh frames are
//! written a [`TARGET_MS`] fill ahead of its position, and each slot it
//! played is zeroed once it passes, so a late service makes it play silence
//! rather than the last lap's sound: an underrun, after which the writes go
//! on from the position plus the FIFO. Only a service gap of a whole lap can
//! replay old sound, and that is detected from the elapsed time and the
//! buffer zeroed.
//!
//! The guest's frames arrive at its own clock, which may run up to a tenth
//! fast while the shared count repays lag (`stela_engine` `FAST_SLEW_PARTS`),
//! or slow. They wait in a host queue; each write takes a few more or fewer
//! of them than it plays, as the queue and fill stand above or below their
//! targets, at most an eighth either way. The frames are not dropped or
//! repeated for it, which clicks, but resampled ([`StereoResampler`]):
//! interpolated linearly at that ratio, whose changes are eased in over a
//! few writes, with the phase and the last frame carried from one write to
//! the next, so the sound runs on unbroken at a pitch a fraction off. A
//! queue past [`MAX_PENDING_MS`] drops its oldest frames at once: an
//! overrun.
//!
//! The guest's microphone is a host USB microphone ([`choose_mic`]), read
//! in the first of its formats ([`capture_order`]) and [`MIC_RATES`] it
//! takes, and made the guest's mono 48 kHz samples by [`MicConverter`]:
//! its channels averaged and its rate interpolated linearly. Its clock
//! needs no correction here: the guest's capture endpoint is asynchronous,
//! and its packet sizes follow the host's samples (`usb::audio`).
use alloc::{collections::VecDeque, vec::Vec};

/// The only rate: the guest's headset clock and the stream's.
pub const RATE: u32 = 48_000;
/// Frames in a millisecond.
pub const FRAMES_PER_MS: usize = RATE as usize / 1000;
/// One stereo frame, each sample MSB-aligned in 32 bits.
pub type Frame = [i32; 2];

/// The fill of fresh frames kept ahead of the controller.
pub const TARGET_MS: usize = 30;
/// The guest frames the host queue keeps beside the fill.
pub const PENDING_MS: usize = 10;
/// The queue past which its oldest frames are dropped at once.
pub const MAX_PENDING_MS: usize = 120;
/// The cyclic buffer: whole [`PERIOD_MS`] periods, long enough that a
/// service gap of a few starved turns plays silence, not old sound.
pub const RING_MS: usize = 256;
pub const PERIOD_MS: usize = 4;
/// The most a write stretches or shrinks the guest's frames: an eighth.
const STRETCH_PARTS: usize = 8;
/// The share of the way to its new ratio a write's resampling moves: a
/// quarter, so a sudden change of the error eases in over a few writes.
const EASE_PARTS: i64 = 4;
/// The most a write's ratio moves from the last write's: 1/128, about 13
/// cents of pitch, so even a jump of the whole eighth is spread over 16
/// writes or more (~80 ms at a turn every 5 ms) rather than heard as a
/// step.
const MAX_EASE: i64 = (ONE >> 7) as i64;

// Configuration default fields.
const CONNECTION_NONE: u32 = 1;
const DEVICE_LINE_OUT: u32 = 0;
const DEVICE_SPEAKER: u32 = 1;
const DEVICE_HEADPHONE: u32 = 2;
/// Location 29:24: gross location 0 (external on the primary chassis),
/// geometric location 2 (front).
const FRONT: u32 = 0x02;

/// How well an analog output pin of configuration default `config` suits a
/// headset: lower is better, `None` for a pin that is no analog output. The
/// owner's headset is in the case's front jack, which firmware describes
/// as an external front headphone pin; a front line out, any headphone pin,
/// a speaker and a rear line out follow.
pub fn output_rank(config: u32) -> Option<u8> {
    if config == u32::MAX || config >> 30 == CONNECTION_NONE {
        return None;
    }
    let front = (config >> 24) & 0x3f == FRONT;
    match ((config >> 20) & 15, front) {
        (DEVICE_HEADPHONE, true) => Some(0),
        (DEVICE_LINE_OUT, true) => Some(1),
        (DEVICE_HEADPHONE, false) => Some(2),
        (DEVICE_SPEAKER, _) => Some(3),
        (DEVICE_LINE_OUT, false) => Some(4),
        _ => None,
    }
}

/// The candidate of `candidates`, (pin node, configuration default), that
/// plays the guest: `forced`'s pin when given, else the best ranked, the
/// first of equals (the codec's own association and sequence order).
pub fn choose_output(candidates: &[(u8, u32)], forced: Option<u8>) -> Option<usize> {
    if let Some(pin) = forced {
        return candidates.iter().position(|&(node, _)| node == pin);
    }
    candidates
        .iter()
        .enumerate()
        .filter_map(|(index, &(_, config))| Some((output_rank(config)?, index)))
        .min()
        .map(|(_, index)| index)
}

/// `frames` as the stream's samples, appended to `out`: 16-bit samples
/// take each sample's top half; 20-, 24- and 32-bit ones its 32-bit
/// container whole, MSB-justified.
pub fn encode(frames: &[Frame], bits: u8, out: &mut Vec<u8>) {
    for sample in frames.iter().flatten() {
        if bits == 16 {
            out.extend_from_slice(&((sample >> 16) as i16).to_le_bytes());
        } else {
            out.extend_from_slice(&sample.to_le_bytes());
        }
    }
}

/// Bytes of one stereo frame in a stream of `bits`.
pub const fn frame_bytes(bits: u8) -> usize {
    if bits == 16 { 4 } else { 8 }
}

/// Where the fresh frames stand against the controller's position in its
/// cyclic buffer, all in frames, as absolute counts from the start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputRing {
    /// The buffer's length.
    frames: usize,
    /// What the controller may have fetched past its position: its FIFO.
    guard: usize,
    /// The fill kept ahead of the position.
    target: usize,
    /// The position last read, in the buffer.
    position: usize,
    /// Frames the controller played.
    played: u64,
    /// The end of the frames written.
    written: u64,
    /// Every slot before this holds silence or fresh frames.
    clean: u64,
    /// Whether frames were written since the last underrun.
    fresh: bool,
    pub underruns: u64,
    /// Service gaps of a whole lap or more, each also an underrun.
    pub lost: u64,
}

/// A span of the buffer: (first frame, frames).
pub type Span = (usize, usize);

impl OutputRing {
    /// A buffer of `frames`, all silence, that the controller starts
    /// playing at its first slot; `guard` frames past its position may be
    /// fetched already.
    pub fn new(frames: usize, guard: usize, target: usize) -> Self {
        let guard = guard.min(frames / 4);
        Self {
            frames,
            guard,
            target: target.clamp(guard + 1, frames / 2),
            position: 0,
            played: 0,
            written: guard as u64,
            clean: frames as u64,
            fresh: false,
            underruns: 0,
            lost: 0,
        }
    }

    /// The controller is at `position` (frames into the buffer), `elapsed`
    /// frames of time after the last observation; `streaming` while the
    /// guest plays, so the end of its sound is no underrun.
    pub fn observe(&mut self, position: usize, elapsed: u64, streaming: bool) {
        let n = self.frames as u64;
        let position = position % self.frames;
        let delta = (position + self.frames - self.position) as u64 % n;
        self.position = position;
        if elapsed + self.guard as u64 >= n {
            // A lap or more went by unseen: whole laps by the time, and every
            // slot may hold sound already played.
            let laps = (elapsed.saturating_sub(delta) + n / 2) / n;
            self.played += delta + laps * n;
            self.clean = self.played;
            self.lost += 1;
            self.resync(true);
            return;
        }
        self.played += delta;
        if self.written < self.played + self.guard as u64 {
            self.resync(streaming);
        }
    }

    fn resync(&mut self, count: bool) {
        if self.fresh && count {
            self.underruns += 1;
        }
        self.fresh = false;
        self.written = self.played + self.guard as u64;
    }

    /// Fresh frames ahead of the controller.
    pub fn fill(&self) -> usize {
        self.written.saturating_sub(self.played) as usize
    }

    /// Frames to write to bring the fill to its target.
    pub fn room(&self) -> usize {
        self.target.saturating_sub(self.fill())
    }

    /// The spans `frames` fresh frames go to, in order: one, or two where
    /// they wrap.
    pub fn write_spans(&self, frames: usize) -> [Span; 2] {
        self.spans(self.written, frames)
    }

    fn spans(&self, start: u64, frames: usize) -> [Span; 2] {
        let first = (start % self.frames as u64) as usize;
        let head = frames.min(self.frames - first);
        [(first, head), (0, frames - head)]
    }

    /// `frames` fresh frames went to [`Self::write_spans`]'s spans.
    pub fn commit(&mut self, frames: usize) {
        self.written += frames as u64;
        self.fresh |= frames != 0;
    }

    /// The spans the controller played since the last cleaning, past the
    /// fresh frames, to zero now; they are counted clean.
    pub fn take_clean(&mut self) -> [Span; 2] {
        let start = self.clean.max(self.written);
        let end = self.played + self.frames as u64;
        self.clean = self.clean.max(end);
        self.spans(start, end.saturating_sub(start) as usize)
    }
}

/// The guest's frames on their way to the controller: the host queue, the
/// resampler that follows the controller's clock, and what was dropped
/// from the queue.
#[derive(Debug, Default)]
pub struct Pending {
    queue: VecDeque<Frame>,
    resampler: StereoResampler,
    /// Frames dropped at a full queue.
    pub overruns: u64,
    /// The guest's frames taken past those played (positive) or played
    /// past those taken (negative) to follow the controller's clock, in
    /// all: what drops and repeats once made up, now the sum over writes
    /// of the frames the resampler passed less those it made.
    pub stretched: i64,
}

impl Pending {
    /// Queues the guest's `frames`; past [`MAX_PENDING_MS`] the oldest go.
    /// The resampler's last frame stays, so the first frame after the cut
    /// is reached from it within one output: the overrun's one break.
    pub fn add(&mut self, frames: &[Frame]) {
        self.queue.extend(frames.iter().copied());
        let most = MAX_PENDING_MS * FRAMES_PER_MS;
        if self.queue.len() > most {
            let excess = self.queue.len() - PENDING_MS * FRAMES_PER_MS;
            self.queue.drain(..excess);
            self.overruns += excess as u64;
        }
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// The resampler's ratio now, input frames per output frame, in Q32.
    pub fn step(&self) -> u64 {
        self.resampler.step()
    }

    /// Up to `room` frames to play after `fill` fresh ones, made from the
    /// queue to bring queue and fill toward their targets ([`TARGET_MS`]
    /// and [`PENDING_MS`]): a write that would once have dropped or
    /// repeated an eighth of the error, at most an eighth of the write,
    /// now asks the resampler for that ratio, `room` plus that drift over
    /// `room`, and the resampler eases toward it
    /// ([`StereoResampler::steer`]). A queue that runs dry makes fewer than
    /// `room`: the ring underruns as it would have.
    pub fn take(&mut self, room: usize, fill: usize, out: &mut Vec<Frame>) {
        if room == 0 {
            return;
        }
        let target = (TARGET_MS + PENDING_MS) * FRAMES_PER_MS;
        let error = (self.queue.len() + fill) as i64 - target as i64;
        let bound = (room / STRETCH_PARTS) as i64;
        let drift = (error / STRETCH_PARTS as i64).clamp(-bound, bound);
        // `room + drift` is at least seven eighths of `room`, so the ratio
        // is within an eighth of ONE.
        let wanted = (room as i64 + drift).max(0) as u128;
        let ratio = (wanted * u128::from(ONE) / room as u128) as u64;
        self.resampler.steer(ratio);
        let (passed, made) = self.resampler.pull(&mut self.queue, room, out);
        self.stretched += passed as i64 - made as i64;
    }
}

/// Linear interpolation of the guest's stereo frames at a ratio that
/// follows the controller's clock, pulled from the host queue one write
/// at a time: each output frame lies between the two input frames around
/// its time, and the time runs on across writes, the phase and the last
/// frame carried, so a write's boundary is no boundary in the sound.
///
/// At a ratio of exactly one from the start the phase stays 0 and the
/// frames pass bit for bit, a frame late (the frame after is needed to
/// interpolate toward). After a correction the phase is left where it
/// ended, rather than snapped back to a frame, which would be a jump of up
/// to a frame in time: a fixed fractional delay, a slight softening of the
/// highest frequencies, is the lesser change.
///
/// A new ratio is eased in ([`Self::steer`]) while the sound runs on; once
/// a pull ran out of input, the sound broke there anyway (the ring plays
/// less than was asked, or silence), and the next ratio is taken at once,
/// as the drops and repeats took it: so a guest that starts a tenth slow
/// with nothing banked is slowed from its first write, not after a dozen
/// writes of underruns.
#[derive(Clone, Debug)]
pub struct StereoResampler {
    /// Input frames per output frame, in Q32.
    step: u64,
    /// The next output's place past `previous`, in Q32 of an input frame;
    /// one or more frames past it when the queue ran out mid-step.
    phase: u64,
    previous: Option<Frame>,
    /// Whether the last pull made fewer frames than asked, or none was
    /// asked yet: the next ratio is taken at once.
    dry: bool,
}

impl Default for StereoResampler {
    fn default() -> Self {
        Self::new(ONE)
    }
}

impl StereoResampler {
    /// The most a step may be: sixteen input frames per output, far past
    /// any correction, so the phase never nears the top of its 64 bits.
    pub const MAX_STEP: u64 = ONE << 4;

    /// At `step` input frames per output frame, in Q32.
    pub const fn new(step: u64) -> Self {
        Self {
            step: clamp_step(step),
            phase: 0,
            previous: None,
            dry: true,
        }
    }

    /// Input frames per output frame now, in Q32.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Moves the step toward `target`: a quarter of the way
    /// ([`EASE_PARTS`]), at most [`MAX_EASE`], and the last few Q32 units,
    /// which a quarter rounds to none, at once, so a steady target is
    /// reached exactly, one included; all the way after a dry pull.
    pub fn steer(&mut self, target: u64) {
        let target = clamp_step(target);
        if self.dry {
            self.step = target;
            return;
        }
        let distance = target as i64 - self.step as i64;
        let mut change = (distance / EASE_PARTS).clamp(-MAX_EASE, MAX_EASE);
        if change == 0 {
            change = distance;
        }
        self.step = clamp_step(self.step.saturating_add_signed(change));
    }

    /// Up to `outputs` frames made from the front of `input`, appended to
    /// `out`; the input frames the time passed are taken from it, as soon
    /// as it passes them. Each output needs the input frame after its
    /// time, so a short input makes fewer. Returns (input frames passed,
    /// output frames made): with input to spare, the frames passed in all
    /// are the outputs made in all times the step, rounded down; the first
    /// frame ever taken starts the time and is not counted.
    pub fn pull(
        &mut self,
        input: &mut VecDeque<Frame>,
        outputs: usize,
        out: &mut Vec<Frame>,
    ) -> (usize, usize) {
        let mut passed = 0;
        let mut made = 0;
        loop {
            // Bring the time between `previous` and the input's front.
            while self.phase >= ONE {
                let Some(frame) = input.pop_front() else {
                    break;
                };
                self.previous = Some(frame);
                self.phase -= ONE;
                passed += 1;
            }
            if self.previous.is_none() {
                self.previous = input.pop_front();
            }
            // A phase still past a frame leaves the input empty.
            let (Some(previous), Some(&next)) = (self.previous, input.front()) else {
                break;
            };
            if made == outputs {
                break;
            }
            out.push([
                lerp(previous[0], next[0], self.phase),
                lerp(previous[1], next[1], self.phase),
            ]);
            made += 1;
            self.phase += self.step;
        }
        self.dry = made < outputs;
        (passed, made)
    }
}

/// `step` within one Q32 unit and [`StereoResampler::MAX_STEP`].
const fn clamp_step(step: u64) -> u64 {
    if step == 0 {
        1
    } else if step > StereoResampler::MAX_STEP {
        StereoResampler::MAX_STEP
    } else {
        step
    }
}

/// The sample `phase` (Q32, below one) of the way from `previous` to
/// `next`, rounded to the nearest: `previous` itself at phase 0, and
/// never outside the two, so the narrowing is exact. The full span of two
/// samples times a phase needs 65 bits.
fn lerp(previous: i32, next: i32, phase: u64) -> i32 {
    let delta = i128::from(next) - i128::from(previous);
    let offset = (delta * i128::from(phase) + (1 << 31)) >> 32;
    (i128::from(previous) + offset) as i32
}

/// The microphone rates asked of a host USB microphone, in order: the
/// guest's own first, then the common ones a converter resamples from.
pub const MIC_RATES: [u32; 4] = [48_000, 44_100, 32_000, 16_000];
/// A USB Audio Class device's idVendor that names the owner's microphone
/// (fifine), preferred over any other with a capture format.
pub const PREFERRED_MIC_VENDOR: u16 = 0x3142;

/// The device of `devices`, (idVendor, has a capture format), whose
/// microphone feeds the guest: the first fifine, else the first that
/// captures.
pub fn choose_mic(devices: &[(u16, bool)]) -> Option<usize> {
    let capturing = |&(_, (_, capture)): &(usize, &(u16, bool))| *capture;
    devices
        .iter()
        .enumerate()
        .filter(capturing)
        .find(|(_, (vendor, _))| *vendor == PREFERRED_MIC_VENDOR)
        .or_else(|| devices.iter().enumerate().find(capturing))
        .map(|(index, _)| index)
}

/// The order the capture formats `formats`, (channels, subslot bytes),
/// are tried in: those the converter decodes (1 to 4 byte subslots, at
/// least one channel), fewest channels first, then the narrowest samples,
/// as the guest takes mono and the bus carries less.
pub fn capture_order(formats: &[(u8, u8)]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..formats.len())
        .filter(|&index| {
            let (channels, subslot) = formats[index];
            channels != 0 && (1..=4).contains(&subslot)
        })
        .collect();
    order.sort_by_key(|&index| formats[index]);
    order
}

/// Linear interpolation from one rate to another, of a mono stream: each
/// output sample lies between the two input samples around its time.
#[derive(Clone, Debug)]
pub struct Resampler {
    /// Input samples per output sample, in Q32.
    step: u64,
    /// The next output's place past `previous`, in Q32 of an input sample.
    phase: u64,
    previous: Option<i32>,
}

const ONE: u64 = 1 << 32;

impl Resampler {
    /// From `from` Hz to `to` Hz.
    pub fn new(from: u32, to: u32) -> Self {
        Self {
            // Rounded up, so whole ratios such as 16 to 48 kHz stay exact.
            step: (u64::from(from) << 32).div_ceil(u64::from(to.max(1))),
            phase: 0,
            previous: None,
        }
    }

    /// The output samples `input` completes, appended to `out`.
    pub fn push(&mut self, input: &[i32], out: &mut Vec<i32>) {
        for &next in input {
            let Some(previous) = self.previous else {
                self.previous = Some(next);
                continue;
            };
            while self.phase < ONE {
                out.push(lerp(previous, next, self.phase));
                self.phase += self.step;
            }
            self.phase -= ONE;
            self.previous = Some(next);
        }
    }
}

/// A host microphone's capture bytes as the guest's mono 48 kHz samples,
/// MSB-aligned in 32 bits: little-endian subslots (Frmts20 2.3.1), the
/// channels averaged, resampled linearly. A frame split across reads is
/// kept for the next.
#[derive(Clone, Debug)]
pub struct MicConverter {
    channels: usize,
    subslot: usize,
    carry: Vec<u8>,
    mono: Vec<i32>,
    resampler: Resampler,
}

impl MicConverter {
    /// For frames of `channels` samples in `subslot` bytes each (1 to 4),
    /// at `rate` Hz.
    pub fn new(channels: u8, subslot: u8, rate: u32) -> Self {
        Self {
            channels: usize::from(channels.max(1)),
            subslot: usize::from(subslot.clamp(1, 4)),
            carry: Vec::new(),
            mono: Vec::new(),
            resampler: Resampler::new(rate, RATE),
        }
    }

    /// The guest's samples `bytes` complete, appended to `out`.
    pub fn convert(&mut self, bytes: &[u8], out: &mut Vec<i32>) {
        let frame = self.channels * self.subslot;
        self.carry.extend_from_slice(bytes);
        let whole = self.carry.len() / frame * frame;
        self.mono.clear();
        for samples in self.carry[..whole].chunks_exact(frame) {
            let sum: i64 = samples
                .chunks_exact(self.subslot)
                .map(|raw| {
                    let mut word = [0u8; 4];
                    word[4 - self.subslot..].copy_from_slice(raw);
                    i64::from(i32::from_le_bytes(word))
                })
                .sum();
            self.mono.push((sum / self.channels as i64) as i32);
        }
        self.carry.drain(..whole);
        self.resampler.push(&self.mono, out);
    }
}

#[cfg(test)]
mod tests;
