//! The guest's USB headset (`stela_soc::usb::audio`) on the host's sound:
//! what the guest plays goes to an analog output of the host's HD Audio
//! codec, the front headphone jack where firmware describes one
//! (`stela_boot_core::host_audio::choose_output`).
//!
//! The service ([`HostAudio::serve`]) runs as a part of the host's service
//! turn (`service`), on whichever PE's core serves the host: the guest's
//! frames are taken under the devices' lock, a copy of at most the model's
//! 200 ms ring, and the controller is written outside it. A PE that waits
//! serves the host, and the headset asks for a turn once 5 ms of frames
//! arrived (`Audio::take_due`, which wakes a sleeping PE), so a turn comes
//! every few milliseconds while the guest plays; with every PE busy one
//! comes at least every 50 ms (`service::STARVED_MILLIS`). The controller
//! is kept [`TARGET_MS`] ahead in a [`RING_MS`] cyclic buffer whose played
//! slots are zeroed, so a turn later than the fill plays silence, an
//! underrun, and one later than the whole buffer is detected and zeroed;
//! neither stops the stream. The guest's clock is followed by evenly
//! spread drops or repeats (`host_audio::Pending`).
//!
//! The guest's mute and volume are applied by its headset as the frames are
//! taken (`Audio::take`); the codec's path is left at 0 dB.
//!
//! The guest's microphone is a host USB microphone probe found
//! (`host_audio::choose_mic`, the owner's fifine first). It captures while
//! the guest has a capture setting selected (`Audio::capturing`): the first
//! turn that sees the selection starts it, in the first format and rate it
//! takes (`host_audio::capture_order`, `MIC_RATES`), and the first turn
//! that sees it cleared stops it. Each turn in between polls it, converts
//! what it read to the guest's mono 48 kHz samples (`MicConverter`) and
//! gives them to the headset under the devices' lock (`Audio::give`), whose
//! asynchronous packet sizes follow the microphone's clock. The guest's
//! microphone mute and volume apply as the headset sends; the host
//! microphone's own controls are left as it started. A microphone that
//! arrives after probe is not used until the next boot, as other late USB
//! devices are not (`usb`).
//!
//! The controller's registers and DMA pages are reached through the
//! loader's identity root, so each write runs in it (`storage::RestoreRoot`),
//! and never waits for the host drivers' lock: while another processor
//! holds it the frames wait in the host queue for the next turn.
use super::{host, output, storage::RestoreRoot};
use alloc::{string::String, vec, vec::Vec};
use stela_boot_core::host_audio::{
    self, FRAMES_PER_MS, Frame, MIC_RATES, MicConverter, OutputRing, PERIOD_MS, Pending, RATE,
    RING_MS, TARGET_MS,
};
use stela_drivers::{
    audio::{
        Direction, PcmFormat,
        hda::{AnalogEndpoint, PinOptions, StreamId},
    },
    probe::native::hda::Host,
    usb::audio::{Error as UsbAudioError, UsbAudio},
};
use stela_soc::machine::{AppleDevices, UsbAudioCaptureStatus, UsbAudioStatus};

/// Pin capabilities: headphone drive, EAPD (HDA 1.0a 7.3.4.9).
const PIN_HEADPHONE: u32 = 1 << 3;
const PIN_EAPD: u32 = 1 << 16;
/// The configuration default's headphone device.
const DEVICE_HEADPHONE: u32 = 2;
/// Frames past the controller's FIFO kept unwritten, as the frames it takes
/// between reading LPIB and the write.
const GUARD_SLACK_FRAMES: usize = FRAMES_PER_MS;
/// The guest's ring: what one take can move.
const TAKE_FRAMES: usize = stela_soc::usb::audio::RING_FRAMES;
/// The microphone's frames one read moves: 10 ms at 48 kHz.
const MIC_READ_FRAMES: usize = 10 * FRAMES_PER_MS;

/// The running analog output.
struct Output {
    /// Index into the hosts.
    host: usize,
    stream: StreamId,
    bits: u8,
    ring: OutputRing,
    /// The corrected count at the last position read (`host::COUNTER_HZ`).
    read_at: u64,
    /// Frames written to the controller in all.
    played: u64,
}

pub struct HostAudio {
    /// Every HD Audio controller probe attached, kept for the boot.
    hosts: Vec<Host>,
    output: Option<Output>,
    /// Whether the output was looked for: at the first turn, in the root.
    opened: bool,
    pending: Pending,
    root: u64,
    /// The guest's frames taken in one turn.
    taken: Vec<Frame>,
    /// Frames to write in one turn.
    frames: Vec<Frame>,
    bytes: Vec<u8>,
    /// The guest's frames dropped because no output plays them.
    discarded: u64,
    /// Why no output plays them, repeated in each report: a
    /// boot log that overwrote its first lines still names the cause.
    no_output: Option<String>,
    status: Option<(bool, UsbAudioStatus)>,
    capture: Option<(bool, UsbAudioCaptureStatus)>,
    mic: Microphone,
}

/// The host microphone behind the guest's.
#[derive(Default)]
struct Microphone {
    /// Every USB audio function probe found, kept for the boot.
    devices: Vec<UsbAudio>,
    /// The one that feeds the guest, once chosen.
    chosen: Option<usize>,
    /// Whether one was looked for: at the first turn, in the root.
    opened: bool,
    /// No microphone, or it failed: the guest gets no samples.
    off: bool,
    /// While it captures: its converter and frame bytes.
    running: Option<(MicConverter, usize)>,
    /// Whether its format was reported, once.
    reported: bool,
    buffer: Vec<u8>,
    /// The guest's samples of this turn, for [`HostAudio::give`].
    samples: Vec<i32>,
    /// The guest's samples made in all.
    made: u64,
    /// Packets whose data a USB error lost, as the driver last counted.
    errors: u64,
}

/// The output `host`'s codecs offer, best first by `choose_output`: the
/// codec's graph, the endpoint and the sample size.
fn candidates(hosts: &[Host]) -> Vec<(usize, usize, AnalogEndpoint, u8)> {
    let mut found = Vec::new();
    // 24-bit first, so the first of equal pins is the wider stream.
    for bits in [24, 16] {
        let format = PcmFormat {
            rate: RATE,
            bits,
            channels: 2,
        };
        for (h, host) in hosts.iter().enumerate() {
            for (c, codec) in host.codecs.iter().enumerate() {
                if codec.setup.is_err() {
                    continue;
                }
                let Ok(endpoints) = codec.graph.analog_endpoints(Direction::Playback, format)
                else {
                    continue;
                };
                found.extend(endpoints.into_iter().map(|e| (h, c, e, bits)));
            }
        }
    }
    found
}

impl HostAudio {
    /// The guest's sound on the best analog output of `hosts`, or nowhere,
    /// which the first turn chooses and names in a line, or says why there
    /// is none; `root` is the loader's.
    pub fn new(hosts: Vec<Host>, microphones: Vec<UsbAudio>, root: u64) -> Self {
        Self {
            hosts,
            output: None,
            opened: false,
            pending: Pending::default(),
            root,
            taken: vec![[0; 2]; TAKE_FRAMES],
            frames: Vec::new(),
            bytes: Vec::new(),
            discarded: 0,
            no_output: None,
            status: None,
            capture: None,
            mic: Microphone {
                devices: microphones,
                ..Microphone::default()
            },
        }
    }

    fn open(hosts: &mut [Host]) -> Result<Output, String> {
        if hosts.is_empty() {
            return Err(String::from("no HD Audio controller started"));
        }
        let found = candidates(hosts);
        let pins: Vec<(u8, u32)> = found
            .iter()
            .map(|(_, _, endpoint, _)| (endpoint.route.pin(), endpoint.configuration.raw))
            .collect();
        let Some(chosen) = host_audio::choose_output(&pins, None) else {
            return Err(alloc::format!(
                "no analog output pin among {} candidate routes",
                found.len()
            ));
        };
        let (h, c, endpoint, bits) = found[chosen].clone();
        let host = &mut hosts[h];
        let graph = host.codecs[c].graph.clone();
        let pin = graph
            .widget(endpoint.route.pin())
            .and_then(|widget| widget.pin_capabilities)
            .unwrap_or(0);
        let config = endpoint.configuration;
        let headphone = pin & PIN_HEADPHONE != 0 && (config.raw >> 20) & 15 == DEVICE_HEADPHONE;
        let options = PinOptions::Playback {
            headphone,
            eapd: (pin & PIN_EAPD != 0).then_some(true),
        };
        let format = PcmFormat {
            rate: RATE,
            bits,
            channels: 2,
        };
        let frame = host_audio::frame_bytes(bits);
        let period = PERIOD_MS * FRAMES_PER_MS * frame;
        let periods = RING_MS / PERIOD_MS;
        let describe = || {
            alloc::format!(
                "{} codec {} ({:08x}) pin {:#04x} config {:08x} {:?} location {:#04x} \
                 converter {:#04x} path {:x?} {bits}-bit",
                host.location,
                graph.address,
                graph.vendor_id,
                endpoint.route.pin(),
                config.raw,
                config.device(),
                config.location(),
                endpoint.route.converter(),
                endpoint.route.nodes(),
            )
        };
        let line = describe();
        let stream = host
            .start_output(&graph, &endpoint.route, format, period, periods, options)
            .map_err(|error| alloc::format!("{line}: start failed: {error:?}"))?;
        let fifo = host
            .output_fifo_bytes(stream)
            .map_err(|error| alloc::format!("{line}: FIFO size: {error:?}"))?;
        let guard = usize::from(fifo).div_ceil(frame) + 1 + GUARD_SLACK_FRAMES;
        output::arguments(format_args!(
            "AUDIO out {line} headphone={headphone} ring_ms={RING_MS} target_ms={TARGET_MS} \
             fifo={fifo}\n"
        ));
        Ok(Output {
            host: h,
            stream,
            bits,
            ring: OutputRing::new(RING_MS * FRAMES_PER_MS, guard, TARGET_MS * FRAMES_PER_MS),
            read_at: host::fresh_counter(),
            played: 0,
        })
    }

    /// Takes what the guest played, under the devices' lock: the copy is
    /// all the lock is held for.
    pub fn take(&mut self, devices: &mut AppleDevices) {
        let taken = devices.audio_take(&mut self.taken);
        self.status = devices.audio_status();
        self.capture = devices.audio_capture_status();
        if self.output.is_some() || !self.opened {
            self.pending.add(&self.taken[..taken]);
        } else {
            self.discarded += taken as u64;
        }
    }

    /// Writes what waits ahead of the controller, outside the devices'
    /// lock and without waiting for the host drivers'.
    pub fn play(&mut self) {
        if self.opened && self.output.is_none() {
            return;
        }
        // SAFETY: the loader root `storage::open` captured, which maps the
        // controller's registers and DMA pages for the whole run.
        let Some(_root) = (unsafe { RestoreRoot::try_enter(self.root) }) else {
            return;
        };
        if !self.opened {
            self.opened = true;
            match Self::open(&mut self.hosts) {
                Ok(output) => self.output = Some(output),
                Err(why) => {
                    output::arguments(format_args!("AUDIO no output: {why}\n"));
                    self.no_output = Some(why);
                    self.discarded += self.pending.len() as u64;
                    self.pending = Pending::default();
                    return;
                }
            }
        }
        let Some(output) = self.output.as_mut() else {
            return;
        };
        let streaming = self.status.is_some_and(|(streaming, _)| streaming);
        let host = &mut self.hosts[output.host];
        let result = (|| {
            let frame = host_audio::frame_bytes(output.bits);
            let position = host.output_position(output.stream)? as usize / frame;
            let now = host::fresh_counter();
            let elapsed = u128::from(now.wrapping_sub(output.read_at)) * u128::from(RATE)
                / u128::from(host::COUNTER_HZ);
            output.read_at = now;
            output.ring.observe(
                position,
                elapsed.min(u128::from(u64::MAX)) as u64,
                streaming,
            );
            for (start, frames) in output.ring.take_clean() {
                write_zero(
                    host,
                    output.stream,
                    start * frame,
                    frames * frame,
                    &mut self.bytes,
                )?;
            }
            self.frames.clear();
            self.pending
                .take(output.ring.room(), output.ring.fill(), &mut self.frames);
            let mut done = 0;
            for (start, frames) in output.ring.write_spans(self.frames.len()) {
                if frames == 0 {
                    continue;
                }
                self.bytes.clear();
                host_audio::encode(
                    &self.frames[done..done + frames],
                    output.bits,
                    &mut self.bytes,
                );
                host.write_output(output.stream, start * frame, &self.bytes)?;
                done += frames;
            }
            output.ring.commit(done);
            output.played += done as u64;
            Ok::<(), stela_drivers::probe::native::hda::RetirementError>(())
        })();
        if let Err(error) = result {
            output::arguments(format_args!(
                "AUDIO FAILED {}: {error:?}; the guest's sound is dropped from now on\n",
                host.location
            ));
            self.output = None;
            self.discarded += self.pending.len() as u64;
            self.pending = Pending::default();
        }
    }

    /// Starts, stops and reads the host microphone as the guest's capture
    /// stands, outside the devices' lock and without waiting for the host
    /// drivers'; the samples go to the guest in [`Self::give`].
    pub fn record(&mut self) {
        let mic = &mut self.mic;
        mic.samples.clear();
        let capturing = self.capture.is_some_and(|(capturing, _)| capturing);
        if mic.off || (!capturing && mic.running.is_none() && mic.opened) {
            return;
        }
        // SAFETY: as in `play`; the microphone's xHCI is reached there.
        let Some(_root) = (unsafe { RestoreRoot::try_enter(self.root) }) else {
            return;
        };
        if !mic.opened {
            mic.opened = true;
            if let Err(why) = mic.open() {
                output::arguments(format_args!("AUDIO no input: {why}\n"));
                mic.off = true;
                return;
            }
        }
        if let Err(why) = mic.turn(capturing) {
            output::arguments(format_args!("AUDIO no input: {why}\n"));
            mic.off = true;
            mic.running = None;
        }
    }

    /// Gives the guest's headset this turn's microphone samples, under the
    /// devices' lock; whether there were any.
    pub fn give(&mut self, devices: &mut AppleDevices) {
        if !self.mic.samples.is_empty() {
            devices.audio_give(&self.mic.samples);
            self.mic.samples.clear();
        }
    }

    /// Whether [`Self::give`] has samples to give.
    pub fn has_samples(&self) -> bool {
        !self.mic.samples.is_empty()
    }

    /// The service's report, every few seconds.
    pub fn report(&self) {
        let (streaming, status) = match self.status {
            Some((streaming, status)) => (streaming, Some(status)),
            None => (false, None),
        };
        let counters = status.map(|status| status.counters).unwrap_or_default();
        let capture = self
            .capture
            .map(|(_, status)| status.counters)
            .unwrap_or_default();
        let (out, underruns, lost, ring_ms) = match &self.output {
            Some(output) => (
                output.played,
                output.ring.underruns,
                output.ring.lost,
                output.ring.fill() / FRAMES_PER_MS,
            ),
            None => (0, 0, 0, 0),
        };
        output::arguments(format_args!(
            "AUDIO out={out} underruns={underruns} lost_laps={lost} overruns={} ring_ms={ring_ms} \
             pending_ms={} stretched={} discarded={} guest_streaming={streaming} guest_bits={} \
             guest_muted={} guest_volume_db256={} guest_frames={} guest_overflowed={} \
             in={} mic_drops={} mic_underruns={} mic_running={} mic_made={} mic_errors={}\n",
            self.pending.overruns,
            self.pending.len() / FRAMES_PER_MS,
            self.pending.stretched,
            self.discarded,
            status.map_or(0, |status| status.bits),
            status.is_some_and(|status| status.muted),
            status.map_or(0, |status| status.volume),
            counters.received,
            counters.overflowed,
            capture.given,
            capture.dropped,
            capture.underruns,
            self.mic.running.is_some(),
            self.mic.made,
            self.mic.errors,
        ));
        if let Some(why) = &self.no_output
            && self.discarded != 0
        {
            output::arguments(format_args!("AUDIO no output: {why}\n"));
        }
    }
}

/// The microphone functions probe found are dropped in the loader's root:
/// their drop stops their endpoints through an xHCI only that root maps.
impl Drop for HostAudio {
    fn drop(&mut self) {
        let devices = core::mem::take(&mut self.mic.devices);
        super::network::in_loader_root(self.root, || drop(devices));
    }
}

/// Whether a capture start's refusal leaves the function usable for the
/// next format or rate.
fn retryable(error: &UsbAudioError) -> bool {
    matches!(
        error,
        UsbAudioError::RateUnsupported
            | UsbAudioError::Busy
            | UsbAudioError::Format
            | UsbAudioError::Unsupported(_)
            | UsbAudioError::ClockInvalid
    )
}

impl Microphone {
    /// Chooses the microphone and names it, or says why there is none.
    fn open(&mut self) -> Result<(), String> {
        let offered: Vec<(u16, bool)> = self
            .devices
            .iter()
            .map(|device| {
                let capture = device
                    .binding()
                    .formats
                    .iter()
                    .any(|format| format.direction == Direction::Capture);
                (device.identity().0, capture)
            })
            .collect();
        let Some(chosen) = host_audio::choose_mic(&offered) else {
            return Err(alloc::format!(
                "no USB microphone among {} USB audio functions probe found",
                self.devices.len()
            ));
        };
        let device = &self.devices[chosen];
        let (vendor, product) = device.identity();
        let formats: Vec<(u8, u8, u8)> = capture_formats(device)
            .map(|(_, format)| (format.channels, format.subslot, format.bits))
            .collect();
        output::arguments(format_args!(
            "AUDIO in {} {vendor:04x}:{product:04x} capture formats (channels, subslot, bits) \
             {formats:?}, of {} USB audio functions\n",
            device.location(),
            self.devices.len()
        ));
        self.chosen = Some(chosen);
        Ok(())
    }

    /// One turn of the chosen microphone, with the guest `capturing` or not.
    fn turn(&mut self, capturing: bool) -> Result<(), String> {
        let Some(chosen) = self.chosen else {
            return Ok(());
        };
        let device = &mut self.devices[chosen];
        let location = String::from(device.location());
        let failed = |what: &str, error: UsbAudioError| {
            alloc::format!("{location} {what} failed: {error:?}")
        };
        match (capturing, self.running.is_some()) {
            (true, false) => {
                let (converter, frame, line) = start(device)?;
                if !self.reported {
                    self.reported = true;
                    output::arguments(format_args!("AUDIO in started {line}\n"));
                }
                self.running = Some((converter, frame));
            }
            (false, true) => {
                self.running = None;
                device.stop_capture().map_err(|e| failed("stop", e))?;
            }
            (false, false) => {}
            (true, true) => {
                let progress = device.poll().map_err(|e| failed("poll", e))?;
                self.errors = progress.packet_errors;
                let Some((converter, frame)) = self.running.as_mut() else {
                    return Ok(());
                };
                self.buffer.resize(MIC_READ_FRAMES * *frame, 0);
                loop {
                    let read = device
                        .read_capture(&mut self.buffer)
                        .map_err(|e| failed("read", e))?;
                    let before = self.samples.len();
                    converter.convert(&self.buffer[..read], &mut self.samples);
                    self.made += (self.samples.len() - before) as u64;
                    if read < self.buffer.len() {
                        break;
                    }
                }
            }
        }
        Ok(())
    }
}

/// `device`'s capture formats, with their indices among its formats.
fn capture_formats(
    device: &UsbAudio,
) -> impl Iterator<Item = (usize, &stela_drivers::usb::audio::Format)> {
    device
        .binding()
        .formats
        .iter()
        .enumerate()
        .filter(|(_, format)| format.direction == Direction::Capture)
}

/// Starts `device`'s capture in the first format and rate it takes: its
/// converter, frame bytes and a description.
fn start(device: &mut UsbAudio) -> Result<(MicConverter, usize, String), String> {
    let formats: Vec<(usize, u8, u8, u8)> = capture_formats(device)
        .map(|(index, format)| (index, format.channels, format.subslot, format.bits))
        .collect();
    let shapes: Vec<(u8, u8)> = formats
        .iter()
        .map(|&(_, channels, subslot, _)| (channels, subslot))
        .collect();
    let mut refused = Vec::new();
    for order in host_audio::capture_order(&shapes) {
        let (index, channels, subslot, bits) = formats[order];
        for rate in MIC_RATES {
            match device.start_capture(index, rate) {
                Ok(()) => {
                    return Ok((
                        MicConverter::new(channels, subslot, rate),
                        usize::from(channels) * usize::from(subslot),
                        alloc::format!(
                            "{} {rate} Hz {channels} channels {bits}-bit in {subslot} bytes",
                            device.location()
                        ),
                    ));
                }
                Err(error) if retryable(&error) => refused.push((rate, error)),
                Err(error) => {
                    return Err(alloc::format!(
                        "{} capture start at {rate} Hz failed: {error:?}",
                        device.location()
                    ));
                }
            }
        }
    }
    Err(alloc::format!(
        "{} takes none of its capture formats at 48, 44.1, 32 or 16 kHz: {refused:?}",
        device.location()
    ))
}

/// Zeroes `bytes` of the output's ring from `offset`.
fn write_zero(
    host: &mut Host,
    stream: StreamId,
    offset: usize,
    bytes: usize,
    scratch: &mut Vec<u8>,
) -> Result<(), stela_drivers::probe::native::hda::RetirementError> {
    if bytes == 0 {
        return Ok(());
    }
    scratch.clear();
    scratch.resize(bytes, 0);
    host.write_output(stream, offset, scratch)
}
