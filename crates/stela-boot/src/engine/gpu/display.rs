//! The guest's display on a bound GPU's output.
//!
//! Binding a GPU ends the firmware's framebuffer on it, so Stela's console
//! moves to the output's front buffer at once, where the modeled DCP's frames
//! follow it. A GPU driving no display leaves the firmware's scanout alone.
//!
//! The DCP model presents under the board's device lock, on whichever PE
//! made the device access. A DRM request from a PE other than the boot CPU
//! waits for the boot CPU to run Linux, and the boot CPU may be waiting for
//! that lock, so a present only copies rows into a hidden buffer there and
//! queues its page flip. The flip is requested on the boot CPU: by its own
//! presents, or by the display's flush at the next vblank, which the home
//! machine's device deadline brings there.
//!
//! The DCP offers each frame it composes outside the devices' lock to the
//! GPU first ([`Scanout::try_present_layers`]): the Metal device of the GPU
//! showing the display composes it into the hidden buffer
//! ([`stela_drivers::gpu::metal::compose`]), in a turn of the Metal
//! service, and the buffer is flipped only once the GPU's marker shows the
//! frame done. What needs the host's kernel (exporting the scanout
//! buffers, importing guest RAM) is done only on the boot CPU, where it
//! completes at once; elsewhere a frame needing it is left to the CPU. A
//! frame the GPU does not compose exactly is the CPU's, as before.
//!
//! A second display: where the guest has two ([`want_second_display`]),
//! [`take`] splits the GPU's outputs between them ([`Display::take_split`]):
//! the main display keeps the output it would take alone, in its own size,
//! and the guest's second display runs the next connected output in that
//! monitor's own mode, each its own scanout ([`scanouts`]) with its own
//! buffers, refresh, vertical blanks, flips and hot-plug. Both share the
//! node and the GPU's Metal device, whose compose marker and timeline run
//! one frame at a time across both: the tickets are the device's
//! ([`COMPOSE_TICKET`]), so a frame of one display the GPU is still
//! composing leaves the other's next frame to the CPU.
//!
//! Hot-plug: once a second the same flush on the boot CPU reads the
//! connectors as the driver last detected them, which its hot-plug
//! interrupts keep current. A display taken at start is set again when it
//! returns after being unplugged, and shows on an output connected later
//! when that output has its size or a larger native mode to scale it into.
//! When the firmware's screen is gone and no
//! display was taken, the bound GPUs are looked on until one shows a display
//! in the guest's mode, and probed anew every [`PROBE_LOOKS`] looks, as
//! Linux's output polling and fbdev helper take a display that appears late.
use super::client;
use crate::console::{self, Framebuffer};
use crate::engine::host;
use crate::{Result, output};
use alloc::{boxed::Box, format, string::String, vec::Vec};
use core::ops::Range;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use core::{cell::UnsafeCell, ffi::c_int, ffi::c_void, ptr::NonNull};
use stela_contracts::refuse;
use stela_drivers::gpu::metal::compose;
use stela_drivers::gpu::metal::kms::{self, Display, Drm, Error as DisplayError, Watched};
use stela_soc::machine::{MonitorId, Scanout, ScanoutFormat, ScanoutMode};
use stela_soc::scanout::{GpuFrame, GpuPending};

/// A GPU's primary node, the DRM master.
struct Node(c_int);

// SAFETY: Linux's DRM ioctls honor their UAPI bounds. `client::map` retains
// each mapping for the boot, and different dumb objects have distinct backing.
unsafe impl Drm for Node {
    fn control(&mut self, request: u32, argument: *mut c_void) -> i64 {
        client::control(self.0, request, argument)
    }

    fn map(&mut self, offset: u64, bytes: usize) -> Option<NonNull<u8>> {
        client::map(self.0, offset, bytes)
    }

    fn unmap(&mut self, pixels: NonNull<u8>, _: usize) {
        client::unmap(pixels);
    }

    fn immediate(&self) -> bool {
        on_boot_cpu()
    }
}

fn on_boot_cpu() -> bool {
    host::processor() == super::BOOT_CPU.load(Ordering::Relaxed)
}

/// How often the scanout reads its GPUs' connectors as the driver last
/// detected them ([`kms::detected`]).
const LOOK_SECONDS: u64 = 1;
/// Every this many looks, GPUs with no display taken have their connectors
/// probed anew, as Linux's output polling does every 10 s for connectors
/// whose hot-plug it is not told of.
const PROBE_LOOKS: u32 = 10;

/// The host scanout the modeled DCP presents to: a bound GPU's display, or,
/// when none was taken, the bound GPUs' open nodes, looked on until one
/// shows a display in the guest's mode. Looks run on the boot CPU only,
/// where a DRM request completes at once.
struct KmsScanout {
    /// The guest's mode, fixed when the DCP was placed.
    mode: ScanoutMode,
    refresh_millihertz: u32,
    size_mm: Option<(u32, u32)>,
    /// The identity of the monitor the guest's display was made for: the
    /// taken output's, or the firmware screen's while one is awaited.
    monitor: Option<MonitorId>,
    /// The GPU showing the display, and the display.
    display: Option<(String, Display<Node>)>,
    /// Without a display: the GPUs' open nodes to look on.
    waiting: Vec<(String, c_int)>,
    /// A frame was presented: a display taken later shows the guest's
    /// frames rather than Stela's console.
    presented: bool,
    /// A display was taken after a frame was presented and has shown none
    /// yet: the DCP model presents the frame on screen again, whole
    /// ([`Scanout::wants_frame`]), as macOS draws nothing new on a still
    /// screen until input.
    blank: bool,
    /// The counter at which the next look is due, and the looks so far.
    next_look: u64,
    looks: u32,
    tsc_hz: u64,
    /// A refused look was reported.
    reported: bool,
    /// The counter at the display's last vertical blank as the last look
    /// read it, when it could ([`Display::last_vblank_ns`]).
    vblank_tsc: Option<u64>,
    /// Why the GPU does not compose the guest's frames was reported.
    gpu_reported: bool,
    /// That frames are composed on the GPU was reported.
    gpu_announced: bool,
    /// The bound GPU, by its index, whose output shows the display, when
    /// it was taken at start: the one whose Metal device composes.
    gpu_index: Option<usize>,
    /// The frame the GPU is composing; its ticket is the device's
    /// ([`COMPOSE_TICKET`]).
    gpu: Option<GpuInFlight>,
    /// A frame the GPU composed was given up: the CPU composes every frame.
    gpu_failed: bool,
    /// Frames the GPU did not finish in [`GPU_PATIENCE_SECONDS`], each given
    /// up: [`GPU_TIMEOUTS`] of them end GPU compose for the boot.
    gpu_timeouts: u32,
}

impl Scanout for KmsScanout {
    fn mode(&self) -> ScanoutMode {
        self.mode
    }

    fn refresh_millihertz(&self) -> u32 {
        self.refresh_millihertz
    }

    fn physical_size_mm(&self) -> Option<(u32, u32)> {
        self.size_mm
    }

    fn monitor(&self) -> Option<MonitorId> {
        self.monitor
    }

    fn present(&mut self, rows: &[u8]) -> Result<()> {
        self.presented = true;
        let Some((_, display)) = &mut self.display else {
            // Shown nowhere: a display taken later asks every row of the
            // next frame, its buffers holding none.
            self.look();
            return Ok(());
        };
        console::release();
        let presented = display.present(rows);
        self.blank &= presented.is_err();
        finish_present(display, presented)
    }

    /// Without a display a frame is taken whole and shown nowhere, its rows
    /// unasked, as in [`Self::present`].
    fn try_present_rows(
        &mut self,
        damage: Option<Range<usize>>,
        fill: &mut dyn FnMut(usize, &mut [u8]) -> Result<()>,
    ) -> Result<bool> {
        self.presented = true;
        let Some((_, display)) = &mut self.display else {
            self.look();
            return Ok(true);
        };
        let prepared = display.prepare_rows(damage, fill)?;
        console::release();
        let presented = prepared.present();
        self.blank &= presented.is_err();
        finish_present(display, presented)?;
        Ok(true)
    }

    /// The DCP offers each frame it composes outside the devices' lock.
    /// Every check runs, and the GPU has the frame, before the display is
    /// touched: a frame declined or refused leaves it as it was, and the
    /// CPU composes it. The first reason a frame is declined is reported
    /// once; a frame the GPU took is flipped only once it is done
    /// ([`Self::gpu_pending_done`]).
    fn try_present_layers(&mut self, frame: &GpuFrame<'_>) -> Result<Option<GpuPending>> {
        match self.compose_on_gpu(frame) {
            Ok(GpuTake::Pending(pending)) => Ok(Some(pending)),
            Ok(GpuTake::Declined) => Ok(None),
            Ok(GpuTake::Unfit(reason)) => {
                if let Some((name, _)) = &self.display
                    && !core::mem::replace(&mut self.gpu_reported, true)
                {
                    output::arguments(format_args!(
                        "GPU {name}: display frames are composed on the CPU: {reason}\n"
                    ));
                }
                Ok(None)
            }
            Err(error) => {
                // A refusal is not retried each frame (an export and an
                // import on the boot CPU per frame): the CPU composes the
                // rest of the boot's frames.
                self.gpu_failed = true;
                if let Some((name, _)) = &self.display {
                    output::arguments(format_args!(
                        "GPU {name}: display frames are composed on the CPU from now: {error}\n"
                    ));
                }
                Err(error)
            }
        }
    }

    /// Polled at each look of the display: the marker the GPU writes after
    /// the frame is read with no driver call, and once it shows the frame
    /// its flip is queued, requested at once on the boot CPU. A frame not
    /// done within [`GPU_PATIENCE_SECONDS`], or whose device was lost, is
    /// given up: nothing of it shows, the GPU composes no more frames, and
    /// the DCP shows the frame on screen again whole ([`Self::wants_frame`]).
    fn gpu_pending_done(&mut self, pending: &GpuPending) -> bool {
        let Some(inflight) = self.gpu else {
            return true;
        };
        if inflight.ticket != pending.ticket {
            return true;
        }
        let shown = self
            .gpu_index
            .and_then(metal_service)
            .and_then(|service| service.compose_shows(inflight.ticket));
        let patience = self.tsc_hz.saturating_mul(GPU_PATIENCE_SECONDS);
        let Some((name, display)) = &mut self.display else {
            self.gpu = None;
            return true;
        };
        match shown {
            Some(true) => {
                self.gpu = None;
                let finished = display.finish_gpu_frame();
                if let Err(error) = finish_present(display, finished)
                    && !core::mem::replace(&mut self.reported, true)
                {
                    output::arguments(format_args!("GPU {name}: {error}\n"));
                }
                true
            }
            Some(false) if host::rdtsc().wrapping_sub(inflight.began) < patience => false,
            _ => {
                // Given up: nothing of it shows, and the DCP shows the frame
                // on screen again whole. A device lost or reopened imports
                // its buffers anew, so GPU compose goes on; frames not done
                // in time end it after [`GPU_TIMEOUTS`].
                self.gpu = None;
                display.abandon_gpu_frame();
                self.blank = true;
                if shown.is_some() {
                    self.gpu_timeouts += 1;
                    self.gpu_failed = self.gpu_timeouts >= GPU_TIMEOUTS;
                }
                output::arguments(format_args!(
                    "GPU {name}: a display frame composed on the GPU was {} ({} of {GPU_TIMEOUTS} \
                     timeouts){}\n",
                    match shown {
                        None => "lost with its Metal device",
                        _ => "not done in time",
                    },
                    self.gpu_timeouts,
                    match self.gpu_failed {
                        true => "; the CPU composes every frame from now on",
                        false => "",
                    }
                ));
                true
            }
        }
    }

    fn refresh_rates(&self) -> Vec<u32> {
        match &self.display {
            Some((_, display)) => display.refresh_rates(),
            None => alloc::vec![self.refresh_millihertz],
        }
    }

    /// Where a request must wait, the next flush on the boot CPU sets the
    /// mode; the vblanks take the new period at once.
    fn set_refresh(&mut self, millihertz: u32) -> Result<()> {
        let Some((_, display)) = &mut self.display else {
            return Err(refuse!("no GPU display runs the guest's mode"));
        };
        display.set_refresh(millihertz).map_err(|error| {
            refuse!(
                "the GPU's display refused {millihertz} mHz: {error}",
                error = format!("{error:?}")
            )
        })?;
        self.refresh_millihertz = millihertz;
        // The blank read before runs on the old period.
        self.vblank_tsc = None;
        Ok(())
    }

    /// The display's vertical blanks run on from the one the last look
    /// read; the model takes the age past whole periods.
    fn vblank_age_ns(&self) -> Option<u64> {
        let tsc = self.vblank_tsc?;
        let elapsed = host::rdtsc().saturating_sub(tsc);
        u64::try_from(u128::from(elapsed) * 1_000_000_000 / u128::from(self.tsc_hz.max(1))).ok()
    }

    fn queued(&self) -> bool {
        let looking = self.display.is_some() || !self.waiting.is_empty();
        (looking && host::rdtsc() >= self.next_look)
            || self
                .display
                .as_ref()
                .is_some_and(|(_, display)| display.queued())
    }

    fn flush(&mut self) -> Result<()> {
        if let Some((_, display)) = &mut self.display {
            let flushed = display.flush();
            finish_present(display, flushed)?;
        }
        self.look();
        Ok(())
    }

    /// Display sleep turns the outputs off, and a wake lights them again;
    /// where a request must wait, the next flush on the boot CPU does it.
    /// With no display taken nothing of this scanout is lit.
    fn wants_frame(&self) -> bool {
        self.blank
    }

    fn set_power(&mut self, on: bool) -> Result<()> {
        let Some((_, display)) = &mut self.display else {
            return Ok(());
        };
        display.set_power(on).map_err(|error| {
            refuse!(
                "the GPU's display refused to turn its outputs {state}: {error}",
                state = if on { "on" } else { "off" },
                error = format!("{error:?}")
            )
        })
    }
}

impl KmsScanout {
    fn new(
        mode: ScanoutMode,
        refresh_millihertz: u32,
        size_mm: Option<(u32, u32)>,
        monitor: Option<MonitorId>,
        tsc_hz: u64,
    ) -> Self {
        Self {
            mode,
            refresh_millihertz,
            size_mm,
            monitor,
            display: None,
            waiting: Vec::new(),
            presented: false,
            blank: false,
            next_look: host::rdtsc().saturating_add(tsc_hz.saturating_mul(LOOK_SECONDS)),
            looks: 0,
            tsc_hz,
            reported: false,
            vblank_tsc: None,
            gpu_reported: false,
            gpu_announced: false,
            gpu_index: None,
            gpu: None,
            gpu_failed: false,
            gpu_timeouts: 0,
        }
    }

    /// When a look is due and this is the boot CPU: watch the display's
    /// connector, else look for a display on the waiting GPUs.
    fn look(&mut self) {
        let now = host::rdtsc();
        if now < self.next_look || !on_boot_cpu() {
            return;
        }
        self.next_look = now.saturating_add(self.tsc_hz.saturating_mul(LOOK_SECONDS));
        self.looks = self.looks.wrapping_add(1);
        match &mut self.display {
            Some((name, display)) => {
                // Linux's clock and the counter read together date the
                // blank on the counter.
                self.vblank_tsc = display.last_vblank_ns().and_then(|blank| {
                    let (tsc, now) = (host::rdtsc(), client::now_ns());
                    let age = u128::from(now.checked_sub(blank)?);
                    let ticks = age * u128::from(self.tsc_hz) / 1_000_000_000;
                    Some(tsc.saturating_sub(u64::try_from(ticks).ok()?))
                });
                let watched = display.watch();
                report_watched(name, display, watched);
                report_listings(name, display);
                if let Some(error) = watched.error
                    && !core::mem::replace(&mut self.reported, true)
                {
                    output::arguments(format_args!(
                        "GPU {name}: display request refused: {error:?}; the other outputs \
                         are still read and set\n"
                    ));
                }
            }
            None => self.attach(self.looks.is_multiple_of(PROBE_LOOKS)),
        }
    }

    /// Take the first waiting GPU's display in the guest's mode, reading
    /// each GPU's connectors as last detected unless `probe`.
    fn attach(&mut self, probe: bool) {
        let wanted = (self.mode.width, self.mode.height);
        let mut index = 0;
        while let Some((name, fd)) = self.waiting.get(index) {
            let fd = *fd;
            if !probe && kms::detected(&mut Node(fd)) != Ok(true) {
                index += 1;
                continue;
            }
            match Display::take_fitting(
                Node(fd),
                kms::Wanted::Size(wanted),
                &stela_soc::dcp::modes::advertisable,
            ) {
                Ok(mut display) if display.size() == wanted => {
                    output::arguments(format_args!("GPU {name}: display detected after start\n"));
                    report_ports(name, &display);
                    report_outputs(name, &display);
                    report_listings(name, &mut display);
                    let screen = screen(&display);
                    match self.presented {
                        // The guest's frame on screen shows here at once,
                        // presented again, and a stop's reason.
                        true => console::parked_on(screen),
                        false => crate::status::take_screen(Some(screen)),
                    }
                    self.blank = self.presented;
                    let (name, _) = self.waiting.swap_remove(index);
                    for (_, fd) in self.waiting.drain(..) {
                        client::close_descriptor(fd);
                    }
                    self.display = Some((name, display));
                    self.reported = false;
                    return;
                }
                Ok(mut display) => {
                    // No output runs the guest's size itself, which the
                    // output that gives the picture's timing must; the
                    // outputs are turned off, their buffers and the node
                    // given back, and the GPU is looked on no more.
                    let (width, height) = display.size();
                    output::arguments(format_args!(
                        "GPU {name}: display detected after start has no {}x{} mode, \
                         which macOS draws, only {width}x{height}: not shown\n",
                        wanted.0, wanted.1
                    ));
                    report_listings(name, &mut display);
                    let Node(fd) = display.release();
                    client::close_descriptor(fd);
                    self.waiting.swap_remove(index);
                }
                Err(DisplayError::NoOutput) => index += 1,
                Err(error) => {
                    if !core::mem::replace(&mut self.reported, true) {
                        output::arguments(format_args!(
                            "GPU {name}: display look refused: {error:?}\n"
                        ));
                    }
                    index += 1;
                }
            }
        }
    }
}

/// What a look read of `display`'s connectors, one line for each change.
fn report_watched(name: &str, display: &Display<Node>, watched: Watched) {
    let lines = [
        (watched.lost, "display disconnected"),
        (watched.restored, "display connected again"),
        (watched.joined, "display added on a new output"),
        (
            watched.scaled,
            "added or returned output has no mode of the display's size: picture scaled into \
             its native mode",
        ),
        (
            watched.unshown,
            "connected output still has no free mode or controller in the display's size: \
             left dark",
        ),
        (
            watched.modeless,
            "display connected again has no mode of the display's size: left dark",
        ),
        (
            watched.smaller,
            "dark output's native mode is smaller than the display's size, and the picture is \
             only scaled up",
        ),
        (
            watched.refused,
            "display connected again refused its mode: left dark",
        ),
    ];
    for (count, what) in lines {
        if count != 0 {
            output::arguments(format_args!(
                "GPU {name}: {what} ({count}); {} outputs run it\n",
                display.outputs()
            ));
        }
    }
}

/// Each connector the take or the last look read, one line for each: every
/// mode it lists, the one it runs and why each other was passed over, so a
/// size or a rate macOS was not offered is told from one the monitor's
/// output never listed. A take names every connected output once; a look
/// names an output when it shows the picture or is given up on.
fn report_listings(name: &str, display: &mut Display<Node>) {
    for listing in display.listings() {
        output::arguments(format_args!("GPU {name}: output {listing}\n"));
    }
}

/// Whether a display taken shows on a laptop's built-in panel, whose
/// backlight is set apart from the picture ([`panel_shown`]).
static PANEL: AtomicBool = AtomicBool::new(false);

/// Whether a GPU's display shows on a built-in panel (eDP, LVDS or DSI).
pub(in crate::engine) fn panel_shown() -> bool {
    PANEL.load(Ordering::Relaxed)
}

/// The connectors `display` shows on, the one that paces it first, which
/// tells a laptop panel on this GPU (eDP) from an external monitor.
fn report_ports(name: &str, display: &Display<Node>) {
    let mut ports = String::new();
    for port in display.ports() {
        if !ports.is_empty() {
            ports.push_str(", ");
        }
        let _ = core::fmt::write(&mut ports, format_args!("{port}"));
        if port.panel() {
            PANEL.store(true, Ordering::Relaxed);
            ports.push_str(" (built-in panel)");
        }
    }
    output::arguments(format_args!("GPU {name}: display shows on {ports}\n"));
}

/// How many outputs show `display` and how many connected ones cannot,
/// when that is more than the one output, then each output that scales the
/// picture and why others stay dark.
fn report_outputs(name: &str, display: &Display<Node>) {
    let (shown, unshown) = (display.outputs(), display.unshown());
    if shown <= 1 && unshown == 0 {
        return;
    }
    let (width, height) = display.size();
    output::arguments(format_args!(
        "GPU {name}: display on {shown} outputs, {unshown} connected without a free \
         {width}x{height} mode or controller\n"
    ));
    for kms::Scaled { native, picture } in display.scaled() {
        output::arguments(format_args!(
            "GPU {name}: an output has no {width}x{height} mode: it runs {}x{} and the picture \
             fills {}x{} of it\n",
            native.0, native.1, picture.0, picture.1
        ));
    }
    let smaller = display.smaller();
    if smaller != 0 {
        output::arguments(format_args!(
            "GPU {name}: {smaller} dark outputs have a native mode smaller than \
             {width}x{height}; the picture is only scaled up\n"
        ));
    }
    if let Some(error) = display.refused() {
        output::arguments(format_args!(
            "GPU {name}: scaling an output was refused: {error:?}; it stays dark\n"
        ));
    }
}

fn finish_present(display: &Display<Node>, presented: Result<(), DisplayError>) -> Result<()> {
    // A stop's reason shows on the buffer that does.
    console::parked_on(screen(display));
    presented.map_err(|error| {
        refuse!(
            "the GPU's display refused a frame: {error}",
            error = format!("{error:?}")
        )
    })
}

/// Whether the guest's frames are offered to the GPU at all
/// ([`Scanout::try_present_layers`]); off, the CPU composes every frame.
const GPU_COMPOSE: bool = true;

/// How long a frame the GPU composes may take before it is given up
/// ([`Scanout::gpu_pending_done`]): it waits behind the Metal work already
/// on the queue, which a busy guest can make long.
const GPU_PATIENCE_SECONDS: u64 = 2;

/// Frames given up for [`GPU_PATIENCE_SECONDS`] before GPU compose ends for
/// the boot: a guest's long submission on the shared queue delays a frame
/// without any fault, and each frame given up costs one frame's picture.
const GPU_TIMEOUTS: u32 = 3;

/// How long a compose waits for the Metal service's turn, which a guest's
/// call may hold, before the CPU composes the frame instead: a fraction of
/// the CPU's own compose of a large frame.
const GPU_TURN_MICROS: u64 = 2000;

/// The ticket of the last display frame queued on the GPU's Metal device,
/// whichever display it was for. The device keeps one compose marker and
/// one timeline, whose values only rise, one frame at a time: the main and
/// the second display's frames take their tickets from this one count,
/// under the Metal service's turn, so neither's ticket repeats or falls
/// behind the other's.
static COMPOSE_TICKET: AtomicU64 = AtomicU64::new(0);

/// A frame the GPU is composing: its ticket and the counter at its
/// submission.
#[derive(Clone, Copy)]
struct GpuInFlight {
    ticket: u64,
    began: u64,
}

/// What [`KmsScanout::compose_on_gpu`] did with a frame.
enum GpuTake {
    /// The GPU has it.
    Pending(GpuPending),
    /// Not now (the service busy, a plane or buffer not imported off the
    /// boot CPU): the CPU composes it, and nothing is reported.
    Declined,
    /// Not one the GPU composes, reported once.
    Unfit(String),
}

/// The Metal service of the bound GPU `index`, when it serves one.
fn metal_service(index: usize) -> Option<&'static super::gate::SharedDevice> {
    super::gate::shared()
        .iter()
        .enumerate()
        .find(|&(service, _)| super::gate::served_gpu(service) == Some(index))
        .map(|(_, service)| &**service)
}

impl KmsScanout {
    /// Hands `frame` to the Metal device of the GPU showing the display,
    /// when it can compose it: the planes and the scanout buffer imported
    /// (imported here only on the boot CPU, where the host kernel calls
    /// they make complete at once, so a PE never waits for the boot CPU
    /// while the DCP may wait for it), the frame's arguments written and
    /// its work queued; only then is the hidden buffer the GPU's.
    fn compose_on_gpu(&mut self, frame: &GpuFrame<'_>) -> Result<GpuTake> {
        if !GPU_COMPOSE {
            return Ok(GpuTake::Unfit(String::from("GPU compose is turned off")));
        }
        if self.gpu_failed || self.gpu.is_some() {
            return Ok(GpuTake::Declined);
        }
        let Some((_, display)) = &mut self.display else {
            return Ok(GpuTake::Declined);
        };
        let Some(index) = self.gpu_index else {
            return Ok(GpuTake::Unfit(String::from(
                "the display was taken after start, on a GPU not matched to a Metal device",
            )));
        };
        let Some(service) = metal_service(index) else {
            return Ok(GpuTake::Unfit(String::from(
                "the GPU showing the display serves no Metal device",
            )));
        };
        let target = display.gpu_target(frame.damage.clone());
        let rows = target.rows.start as u32..target.rows.end as u32;
        let plan = match gpu_plan(self.mode, frame, rows) {
            Ok(plan) => plan,
            Err(reason) => return Ok(GpuTake::Unfit(reason)),
        };
        let Ok(pitch) = u32::try_from(target.pitch) else {
            return Ok(GpuTake::Unfit(String::from(
                "the scanout pitch is too large",
            )));
        };
        let boot = on_boot_cpu();
        let wait = self.tsc_hz / 1_000_000 * GPU_TURN_MICROS;
        let submitted = service.with_device(wait, |device| {
            // The turn holds the device: the ticket is the next of every
            // display's, taken only once the frame is queued.
            let ticket = COMPOSE_TICKET.load(Ordering::Relaxed) + 1;
            let job = compose::Job {
                target: target.handle,
                pitch,
                plan: &plan,
                ticket,
                import: boot,
            };
            let submitted = match device.compose(&job) {
                Ok(compose::Submitted::NoTarget) if boot => {
                    let fd = display.export_buffer(target.handle).map_err(|error| {
                        refuse!(
                            "the display refused to export its scanout buffer: {error}",
                            error = format!("{error:?}")
                        )
                    })?;
                    if let Err(error) =
                        device.import_compose_target(target.handle, fd, target.bytes as u64)
                    {
                        client::close_descriptor(fd);
                        return Err(refuse!(
                            "the Metal device refused the scanout buffer: {error}",
                            error = format!("{error:?}")
                        ));
                    }
                    device.compose(&job).map_err(metal_refusal)
                }
                result => result.map_err(metal_refusal),
            };
            if let Ok(compose::Submitted::Queued) = submitted {
                COMPOSE_TICKET.store(ticket, Ordering::Relaxed);
            }
            submitted.map(|submitted| (submitted, ticket))
        });
        match submitted {
            None | Some(Ok((compose::Submitted::NoTarget | compose::Submitted::NotReady, _))) => {
                Ok(GpuTake::Declined)
            }
            Some(Err(error)) => Err(error),
            Some(Ok((compose::Submitted::Queued, ticket))) => {
                console::release();
                // The GPU writes the hidden buffer from now: no flip shows
                // it until the frame is done.
                if let Err(error) = display.begin_gpu_frame(frame.damage.clone()) {
                    // Nothing else writes it while a GPU frame is pending,
                    // which the check above ruled out.
                    return Err(refuse!(
                        "the display refused the GPU's frame: {error}",
                        error = format!("{error:?}")
                    ));
                }
                self.presented = true;
                self.blank = false;
                self.gpu = Some(GpuInFlight {
                    ticket,
                    began: host::rdtsc(),
                });
                if !core::mem::replace(&mut self.gpu_announced, true)
                    && let Some((name, _)) = &self.display
                {
                    output::arguments(format_args!(
                        "GPU {name}: the guest's display frames are composed on this GPU\n"
                    ));
                }
                Ok(GpuTake::Pending(GpuPending { ticket }))
            }
        }
    }
}

fn metal_refusal(error: stela_drivers::gpu::metal::error::Error) -> stela_contracts::Refusal {
    refuse!(
        "the Metal device refused to compose a frame: {error}",
        error = format!("{error:?}")
    )
}

/// The GPU compose of `frame` on a display in `mode`, writing `rows`, or
/// why it is the CPU's: only 8:8:8:8 colour layers and two-plane
/// extended-range ones whose planes the GPU reads whole are composed
/// exactly ([`compose::plan`]); a 10-bit layer is the CPU's.
fn gpu_plan(
    mode: ScanoutMode,
    frame: &GpuFrame<'_>,
    rows: Range<u32>,
) -> core::result::Result<compose::Plan, String> {
    if frame.mode != mode || mode.format != ScanoutFormat::Bgra8888 {
        return Err(format!(
            "the frame's mode {:?} is not the display's",
            frame.mode
        ));
    }
    let mut layers = Vec::with_capacity(frame.layers.len());
    for layer in frame.layers {
        let format = match layer.format {
            ScanoutFormat::Bgra8888 => compose::Format::Bgra8888,
            ScanoutFormat::Rgba8888 => compose::Format::Rgba8888,
            ScanoutFormat::Xr10A8 { .. } => compose::Format::Xr10A8,
            other => {
                return Err(format!(
                    "a layer's pixel format ({other:?}) is composed on the CPU"
                ));
            }
        };
        let plane = |index: usize| {
            layer.planes.get(index).map(|plane| compose::Plane {
                ranges: plane.ranges.clone(),
                offset: plane.offset,
                stride: plane.stride,
            })
        };
        let Some(color) = plane(0) else {
            return Err(String::from("a layer has no colour plane"));
        };
        let rect = |rect: stela_soc::scanout::Rect| compose::Rect {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
        };
        layers.push(compose::Layer {
            format,
            color,
            alpha: plane(1),
            source: rect(layer.source),
            destination: rect(layer.destination),
            premultiplied: layer.premultiplied,
        });
    }
    compose::plan(mode.width, mode.height, rows, &layers)
        .map_err(|decline| format!("the frame is not one the GPU composes exactly ({decline:?})"))
}

/// `display`'s front buffer as a screen.
fn screen(display: &Display<Node>) -> Framebuffer {
    let (width, height) = display.size();
    let (pixels, pitch) = display.front();
    Framebuffer::linear(
        pixels.as_ptr() as usize,
        width as usize,
        height as usize,
        pitch / size_of::<u32>(),
    )
}

/// What [`take`] left, until the board's DCPs take it: the displays taken,
/// else the nodes of the GPUs that had none.
struct Taken(UnsafeCell<Displays>);

/// The displays a GPU took, each with the name its lines carry, by the
/// guest display they show ([`MAIN`] and [`SECOND`]), and the named nodes
/// of the GPUs still without one. The second display is the main's GPU's,
/// on the same node.
type Displays = ([Option<(String, Display<Node>)>; 2], Vec<(String, c_int)>);

/// The guest's main display and its second, as [`Displays`] holds them.
const MAIN: usize = 0;
const SECOND: usize = 1;

// SAFETY: only the boot CPU's start reaches it, once each way.
unsafe impl Sync for Taken {}

static TAKEN: Taken = Taken(UnsafeCell::new(([None, None], Vec::new())));

/// The guest has a second display ([`want_second_display`]).
static SECOND_WANTED: AtomicBool = AtomicBool::new(false);

/// Whether [`take`] gives the guest's second display an output of its own:
/// set before [`take`] (that is, before `gpu::serve`) when the guest has a
/// second display. The main display then keeps the output it would take
/// alone, in its own size, no longer mirrored to the others, and the next
/// connected output of the same GPU shows the second display in its own
/// monitor's mode ([`Display::take_split`]); the GPU awaited after its
/// reset is also awaited, as long, for a second monitor.
pub(in crate::engine) fn want_second_display(wanted: bool) {
    SECOND_WANTED.store(wanted, Ordering::Relaxed);
}

/// The main display and, where one was asked for and an output can show
/// it, the second, on `fd`, for `wanted`.
type Took = (
    Display<Node>,
    Option<core::result::Result<Display<Node>, DisplayError>>,
);

fn take_on(
    fd: c_int,
    wanted: kms::Wanted,
    second: bool,
) -> core::result::Result<Took, DisplayError> {
    let fits = &stela_soc::dcp::modes::advertisable;
    match second {
        true => Display::take_split(
            Node(fd),
            Node(fd),
            wanted,
            kms::Wanted::Native((0, 0)),
            fits,
        )
        .map(|kms::Split { main, second }| (main, Some(second))),
        false => Display::take_fitting(Node(fd), wanted, fits).map(|display| (display, None)),
    }
}
/// The bound GPU whose output [`take`] took, or `usize::MAX`.
static TAKEN_INDEX: AtomicUsize = AtomicUsize::new(usize::MAX);

/// The bound GPU, by its index, whose output took the connected display.
pub(super) fn taken_index() -> Option<usize> {
    let index = TAKEN_INDEX.load(Ordering::Relaxed);
    (index != usize::MAX).then_some(index)
}

/// Which GPU showed the firmware's picture.
#[derive(Clone, Copy)]
pub(in crate::engine) enum Shown<'a> {
    /// The firmware left no screen.
    Nothing,
    /// A screen whose PCI function was not found.
    Unknown,
    /// This function's.
    By(&'a str),
}

/// How long the GPU that showed the firmware's picture may take to detect
/// its display again. The GPU's reset ends the signal, and a monitor that
/// then searches its inputs drops hot-plug detect for some seconds; Linux
/// takes such a display when its hot-plug event arrives. A display later
/// still is taken by the scanout's looks. The same time is given to an
/// output the firmware lit while another output of the GPU already detects
/// its monitor ([`kms::returning`]): the guest's size and rates are set
/// once, by the take, and are the firmware-lit monitor's when it returns
/// in this time.
const RETURN_SECONDS: u64 = 8;
/// The pause between two detections in that time.
const RETURN_STEP_MILLIS: u64 = 100;

/// Moves the display to the first bound GPU with an output, the GPUs that
/// draw first and then those that only show (`claim_all`'s order), in the
/// monitor's own mode that macOS can be told of, else in the mode of
/// `wanted` pixels, the firmware's, if it has it: the guest's mode is not
/// set yet and follows the display taken. The GPU that showed the firmware's
/// picture, or the one GPU bound when that is unknown, has a display known
/// or likely to exist, so it is awaited within [`RETURN_SECONDS`], once.
/// What stops one GPU is reported with what its connectors detected, and
/// its node is kept for the scanout's looks.
pub(in crate::engine) fn take(
    wanted: (usize, usize),
    firmware: Shown<'_>,
    size: Option<(u32, u32)>,
) {
    let Some(gpu) = super::installed() else {
        return;
    };
    let wanted = (wanted.0 as u32, wanted.1 as u32);
    let split = SECOND_WANTED.load(Ordering::Relaxed);
    let bound = &gpu.start.bound;
    // SAFETY: the one boot CPU, before any reader.
    let (taken, waiting) = unsafe { &mut *TAKEN.0.get() };
    for (index, bound_gpu) in bound.iter().enumerate() {
        let name = &bound_gpu.name;
        let Some(fd) = client::minor(index).and_then(client::open_node) else {
            continue;
        };
        let began = host::rdtsc();
        let awaited = match firmware {
            Shown::By(function) => function == name.as_str(),
            // Not one below a bridge that forwards none of the screen.
            Shown::Unknown => bound.len() == 1 && bound_gpu.shows_screen,
            Shown::Nothing => false,
        };
        let patience = match awaited {
            true => RETURN_SECONDS.saturating_mul(gpu.tsc_hz),
            false => 0,
        };
        let pause = || {
            let step = host::rdtsc();
            while host::rdtsc().wrapping_sub(step) < gpu.tsc_hz / 1000 * RETURN_STEP_MILLIS {
                core::hint::spin_loop();
            }
        };
        // `absent`: an output the firmware lit still detected no display
        // when the wait for it ended.
        let (result, absent) = loop {
            let patient = host::rdtsc().wrapping_sub(began) < patience;
            // The monitor the firmware showed on gives the picture its
            // size, its rates and its name: while it is not detected again
            // another monitor of this GPU does not give them in its place.
            let absent = patience != 0 && kms::returning(&mut Node(fd)) == Ok(true);
            if absent && patient {
                pause();
                continue;
            }
            // A second display's monitor searches its inputs after the
            // reset as the firmware's does, often detected after it: the
            // split waits as long for it, so the second display is not
            // left without an output for the boot.
            let alone = split
                && patient
                && matches!(kms::connected_outputs(&mut Node(fd)), Ok(count) if count < 2);
            if alone {
                pause();
                continue;
            }
            let wanted = match size {
                Some(size) => kms::Wanted::Size(size),
                None => kms::Wanted::Native(wanted),
            };
            match take_on(fd, wanted, split) {
                Err(DisplayError::NoOutput) if patient => pause(),
                result => break (result, absent),
            }
        };
        let waited_ms = host::rdtsc().wrapping_sub(began) / (gpu.tsc_hz / 1000).max(1);
        match result {
            Ok((mut display, second)) => {
                if absent {
                    output::arguments(format_args!(
                        "GPU {name}: an output the firmware lit detected no display in \
                         {waited_ms} ms, so another output gives the picture its size\n"
                    ));
                } else if waited_ms >= RETURN_STEP_MILLIS {
                    output::arguments(format_args!(
                        "GPU {name}: display detected after {waited_ms} ms\n"
                    ));
                }
                let (width, height) = display.size();
                output::arguments(format_args!(
                    "GPU {name}: display runs {width}x{height} at {} mHz\n",
                    display.refresh_millihertz()
                ));
                crate::engine::ident::set(
                    crate::engine::ident::Kind::Display,
                    format_args!(
                        "{width}x{height} at {} mHz on an output of GPU {name}",
                        display.refresh_millihertz()
                    ),
                );
                if size.is_none() && !display.preferred_listed() {
                    output::arguments(format_args!(
                        "GPU {name}: the output lists no preferred mode (its EDID marks none, or \
                         Linux left out the monitor's own mode as past the link's clock), so it \
                         runs its first listed mode, as Linux does\n"
                    ));
                }
                report_ports(name, &display);
                // A GPU that serves no Metal device shows what another
                // draws: the display is wired to it (a laptop's panel on its
                // integrated GPU), and each frame is copied into its buffer.
                let chosen = super::gate::metal_choice(bound, Some(index));
                if !chosen.serve.contains(&index)
                    && let Some(&drawing) = chosen.serve.first()
                {
                    output::arguments(format_args!(
                        "GPU {name}: the connected display is on this GPU, so it shows the \
                         frames GPU {} draws, each copied into its buffer and flipped on its \
                         own vblank\n",
                        bound[drawing].name
                    ));
                }
                TAKEN_INDEX.store(index, Ordering::Relaxed);
                report_outputs(name, &display);
                report_listings(name, &mut display);
                crate::status::take_screen(Some(screen(&display)));
                for (_, fd) in waiting.drain(..) {
                    client::close_descriptor(fd);
                }
                taken[MAIN] = Some((name.clone(), display));
                taken[SECOND] = second.and_then(|second| take_second(name, second, waited_ms));
                return;
            }
            Err(error) => {
                output::arguments(format_args!(
                    "GPU {name}: no display after {waited_ms} ms: {error:?}\n"
                ));
                match kms::survey(&mut Node(fd)) {
                    Ok(seen) => {
                        for seen in seen {
                            output::arguments(format_args!("GPU {name}: output {seen}\n"));
                        }
                    }
                    Err(error) => output::arguments(format_args!(
                        "GPU {name}: outputs unreadable: {error:?}\n"
                    )),
                }
                waiting.push((name.clone(), fd));
            }
        }
    }
}

/// The second display [`take`] split off GPU `name`'s outputs, reported,
/// with the name its lines carry; `None`, reported, when no output could
/// show it: the guest's second display then shows nowhere.
fn take_second(
    name: &str,
    second: core::result::Result<Display<Node>, DisplayError>,
    waited_ms: u64,
) -> Option<(String, Display<Node>)> {
    let mut display = match second {
        Ok(display) => display,
        Err(error) => {
            let why = match error {
                DisplayError::NoOutput => {
                    format!("no other connected output with a free controller after {waited_ms} ms")
                }
                error => format!("the driver refused it: {error:?}"),
            };
            output::arguments(format_args!(
                "GPU {name}: no second display: {why}; the guest's second display shows nowhere\n"
            ));
            return None;
        }
    };
    let (width, height) = display.size();
    let mut ports = String::new();
    for port in display.ports() {
        if !ports.is_empty() {
            ports.push_str(", ");
        }
        let _ = core::fmt::write(&mut ports, format_args!("{port}"));
    }
    output::arguments(format_args!(
        "GPU {name}: second display runs {width}x{height} at {} mHz on {ports}\n",
        display.refresh_millihertz()
    ));
    let name = format!("{name} (second display)");
    report_outputs(&name, &display);
    report_listings(&name, &mut display);
    Some((name, display))
}

/// The mode of the display [`take`] took, before the board's DCP takes it.
pub(in crate::engine) fn taken_size() -> Option<(usize, usize)> {
    taken_size_of(MAIN)
}

/// The mode of the guest's second display [`take`] took, before the
/// board's DCP takes it: its monitor's own ([`want_second_display`]).
pub(in crate::engine) fn second_taken_size() -> Option<(usize, usize)> {
    taken_size_of(SECOND)
}

fn taken_size_of(slot: usize) -> Option<(usize, usize)> {
    // SAFETY: the one boot CPU, after `take` and before `scanout`.
    let taken = unsafe { &(*TAKEN.0.get()).0[slot] };
    taken.as_ref().map(|(_, display)| {
        let (width, height) = display.size();
        (width as usize, height as usize)
    })
}

/// Give up the display [`take`] took and the nodes it kept, as amdgpu's
/// reset of their GPU left them: the reset unmapped the display's buffers
/// and holds the display's lock until it ends. Nothing is released, as a
/// release calls the driver that waits for that end, and the console stops
/// drawing on the unmapped buffer, which a draw would fault on.
pub(in crate::engine) fn abandon() {
    // SAFETY: the one boot CPU, after `take`.
    let (taken, waiting) = unsafe { &mut *TAKEN.0.get() };
    let had = taken[MAIN].is_some();
    for slot in taken.iter_mut() {
        if let Some(display) = slot.take() {
            core::mem::forget(display);
        }
    }
    let nodes = waiting.len();
    core::mem::forget(core::mem::take(waiting));
    if had && !crate::console::forget_screen() {
        output::arguments(format_args!(
            "GPU display: the console was busy as its screen was given up\n"
        ));
    }
    output::arguments(format_args!(
        "GPU display given up after the GPU's reset: display={had} nodes={nodes}\n"
    ));
}

/// The scanout of the display [`take`] took, in its own mode, for the
/// board's DCP: the guest's main display.
pub(in crate::engine) fn scanout() -> Option<Box<dyn Scanout>> {
    scanout_of(MAIN)
}

/// The scanout of the guest's second display [`take`] took, in its own
/// monitor's mode, refresh and identity (its EDID), for the board's second
/// DCP; `None` when no second display was asked for or no output could
/// show it. It shares the main display's GPU and Metal device, and has its
/// own buffers, vertical blanks, flips, hot-plug and frames composed on the
/// GPU.
pub(in crate::engine) fn second_scanout() -> Option<Box<dyn Scanout>> {
    scanout_of(SECOND)
}

/// The scanouts of the displays [`take`] took, the main display's first,
/// then the second's where there is one: index 0 and 1 of the board's
/// scanouts. Empty when no display was taken.
pub(in crate::engine) fn scanouts() -> Vec<Box<dyn Scanout>> {
    [scanout(), second_scanout()]
        .into_iter()
        .flatten()
        .collect()
}

fn scanout_of(slot: usize) -> Option<Box<dyn Scanout>> {
    let gpu = super::installed()?;
    // SAFETY: the one boot CPU, after `take`.
    let (name, display) = unsafe { (*TAKEN.0.get()).0[slot].take() }?;
    let (width, height) = display.size();
    let mut scanout = KmsScanout::new(
        ScanoutMode {
            width,
            height,
            format: ScanoutFormat::Bgra8888,
        },
        display.refresh_millihertz(),
        display.size_mm(),
        display.edid().and_then(MonitorId::from_edid),
        gpu.tsc_hz,
    );
    scanout.gpu_index = taken_index();
    scanout.display = Some((name, display));
    Some(Box::new(scanout))
}

/// Whether a GPU is bound, so it draws macOS's frames, while [`scanout`]
/// found no display taken: only the firmware's screen, on an adapter no
/// bound GPU's start ended, can show them, each frame copied.
pub(in crate::engine) fn drawn_without_display() -> bool {
    super::installed().is_some_and(|gpu| !gpu.start.bound.is_empty())
}

/// With no display taken and the firmware's `screen` gone, or with no
/// firmware screen at all ([`guest_screen::DEFAULT`]), a scanout in the
/// guest's mode that shows the first display a bound GPU detects later in
/// it. None when no GPU's node is open.
///
/// [`guest_screen::DEFAULT`]: stela_boot_core::guest_screen::DEFAULT
pub(in crate::engine) fn awaited(screen: Option<Framebuffer>) -> Option<Box<dyn Scanout>> {
    let gpu = super::installed()?;
    // SAFETY: the one boot CPU, after `take`.
    let waiting = core::mem::take(unsafe { &mut (*TAKEN.0.get()).1 });
    if waiting.is_empty() {
        return None;
    }
    use stela_boot_core::guest_screen;
    let ((width, height), refresh, size_mm, monitor) = match screen {
        Some(screen) => (
            screen.resolution(),
            screen.refresh_millihertz(),
            screen.size_mm(),
            screen.monitor(),
        ),
        None => (
            guest_screen::DEFAULT,
            guest_screen::DEFAULT_REFRESH_MILLIHERTZ,
            None,
            None,
        ),
    };
    let mut scanout = KmsScanout::new(
        ScanoutMode {
            width: width as u32,
            height: height as u32,
            format: ScanoutFormat::Bgra8888,
        },
        refresh,
        size_mm,
        monitor,
        gpu.tsc_hz,
    );
    scanout.waiting = waiting;
    output::arguments(format_args!(
        "DISPLAY no GPU display yet: looking for one in {width}x{height} once a second\n"
    ));
    Some(Box::new(scanout))
}
