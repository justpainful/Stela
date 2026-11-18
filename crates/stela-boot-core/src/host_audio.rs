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
//! of them than it plays, by evenly spread drops or repeats ([`stretch`]),
//! as the queue and fill stand above or below their targets, at most an
//! eighth either way. A queue past [`MAX_PENDING_MS`] drops its oldest
//! frames at once: an overrun.
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

/// The guest's frames on their way to the controller: the host queue, and
/// what was dropped from it.
#[derive(Debug, Default)]
pub struct Pending {
    queue: VecDeque<Frame>,
    /// Frames dropped at a full queue.
    pub overruns: u64,
    /// Frames dropped (positive) or repeated (negative) to follow the
    /// controller's clock, in all.
    pub stretched: i64,
}

impl Pending {
    /// Queues the guest's `frames`; past [`MAX_PENDING_MS`] the oldest go.
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

    /// Up to `room` frames to play after `fill` fresh ones, taken from the
    /// queue to bring queue and fill toward their targets
    /// ([`TARGET_MS`] and [`PENDING_MS`]): an eighth of the error a write,
    /// as evenly spread drops or repeats, at most an eighth of the write.
    pub fn take(&mut self, room: usize, fill: usize, out: &mut Vec<Frame>) {
        let target = (TARGET_MS + PENDING_MS) * FRAMES_PER_MS;
        let error = (self.queue.len() + fill) as i64 - target as i64;
        let bound = (room / STRETCH_PARTS) as i64;
        let drift = (error / STRETCH_PARTS as i64).clamp(-bound, bound);
        let wanted = (room as i64 + drift).max(0) as usize;
        let inputs = wanted.min(self.queue.len());
        // The frames played: all of `room`, or in proportion to the inputs
        // the queue had.
        let outputs = if inputs == wanted {
            room
        } else {
            inputs * room / wanted
        };
        let inputs = if outputs == 0 { 0 } else { inputs };
        stretch(self.queue.make_contiguous(), inputs, outputs, out);
        self.queue.drain(..inputs);
        self.stretched += inputs as i64 - outputs as i64;
    }
}

/// `outputs` frames from the first `inputs` of `input`, appended to `out`:
/// output `i` is input `i * inputs / outputs`, so drops or repeats spread
/// evenly over the span.
pub fn stretch(input: &[Frame], inputs: usize, outputs: usize, out: &mut Vec<Frame>) {
    let inputs = inputs.min(input.len());
    if inputs == 0 {
        return;
    }
    out.extend((0..outputs).map(|i| input[i * inputs / outputs]));
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
                // Rounded to the nearest; the full span of two samples
                // times a phase needs 65 bits.
                let delta = i128::from(next) - i128::from(previous);
                let offset = (delta * i128::from(self.phase) + (1 << 31)) >> 32;
                out.push((i128::from(previous) + offset) as i32);
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
