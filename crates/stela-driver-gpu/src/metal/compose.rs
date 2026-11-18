//! A display frame composed on the GPU: what the modeled DCP's layers ask
//! of the compose, checked and clipped before any GPU work, the arguments
//! the backend's compose kernel reads (`stela_metal_compose_kernel`,
//! `compiler/metal_abi.h`), and the exact per-pixel result it must give.
//!
//! The DCP model composes on the CPU (`stela_soc::scanout::compose`): its
//! layers bottom to top over opaque black, each destination pixel taking
//! the source pixel under its centre (nearest neighbour), blended
//! source-over with its colour premultiplied or not, clipped to the mode,
//! into a blue-low 8:8:8:8 scanout with alpha opaque. A GPU compose of the
//! same frame must give the same bytes, so [`plan`] takes only what it can
//! give exactly: 8:8:8:8 colour layers and two-plane extended-range ones
//! ([`Format`]) whose planes the GPU reaches whole. Anything else is
//! declined ([`Decline`]) and the CPU composes the frame as before.
//! [`compose_pixel`] is the composition one destination pixel at a time,
//! in integers, as the kernel computes it.
//!
//! A layer's planes are guest RAM the backend imports ([`Job`]); the
//! backend's compose writes a dumb buffer of the display the host GPU
//! scans out, which it imported from the display's driver.
use alloc::vec::Vec;
use core::ops::Range;
use stela_contracts::gate::MemoryRange;

/// The most layers a frame has: the DCP's four slots.
pub const LAYERS: usize = 4;

/// The largest source or destination side composed: the kernel's nearest
/// neighbour mapping multiplies two sides in 32 bits.
pub const MAX_SIDE: u32 = 1 << 14;

/// A layer's pixel format, the ones composed on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Blue in the low byte (`'BGRA'`), the scanout's own order.
    Bgra8888,
    /// Red in the low byte (`'RGBA'`).
    Rgba8888,
    /// `RGB10A8_2P_XR10` (`'&b38'`): blue, green and red extended-range
    /// codes from bit 0 of a word, each shown as `(code - 383) / 2`
    /// clamped to [0, 255] (the DCP's `XR10_DISPLAY`), and 8-bit alpha in
    /// a second plane ([`Layer::alpha`]).
    Xr10A8,
}

/// A rectangle in pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// One plane of a layer: the guest physical runs that hold its source
/// rows, read as one span, whose byte `offset` is the source rectangle's
/// top-left pixel, `stride` bytes a row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plane {
    pub ranges: Vec<MemoryRange>,
    pub offset: u64,
    pub stride: u32,
}

impl Plane {
    /// The bytes the runs hold.
    pub fn bytes(&self) -> u64 {
        self.ranges
            .iter()
            .fold(0u64, |bytes, range| bytes.saturating_add(range.bytes))
    }
}

/// One layer as the DCP gives it: a source rectangle of its colour plane
/// (and, for [`Format::Xr10A8`], its alpha plane) shown in `destination`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layer {
    pub format: Format,
    pub color: Plane,
    pub alpha: Option<Plane>,
    pub source: Rect,
    pub destination: Rect,
    pub premultiplied: bool,
}

/// A layer the GPU composes: as given, with the screen pixels it covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    pub layer: Layer,
    /// Its on-screen columns and rows, half open.
    pub columns: Range<u32>,
    pub rows: Range<u32>,
}

/// A frame the GPU can compose: the rows it writes and the layers on
/// screen, bottom to top.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub width: u32,
    pub height: u32,
    pub rows: Range<u32>,
    pub layers: Vec<Placed>,
}

/// Why a frame is left to the CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// More layers than the DCP has slots.
    Layers,
    /// A layer with an empty source or destination.
    Empty,
    /// A side past [`MAX_SIDE`].
    Size,
    /// The rows asked lie outside the screen, or the screen is empty.
    Rows,
    /// A layer's source rows reach past one of its planes, its addresses
    /// overflow, or it lacks a plane its format has: the GPU would read
    /// memory the layer does not name.
    Plane,
}

/// The screen part of `rect` on a `width` by `height` screen.
fn on_screen(rect: Rect, width: u32, height: u32) -> Option<(Range<u32>, Range<u32>)> {
    let end = |start: u32, length: u32, limit: u32| {
        (u64::from(start) + u64::from(length)).min(u64::from(limit)) as u32
    };
    let columns = rect.x..end(rect.x, rect.width, width);
    let rows = rect.y..end(rect.y, rect.height, height);
    (!columns.is_empty() && !rows.is_empty()).then_some((columns, rows))
}

/// Whether `plane`'s rows of `source`, `size` bytes a pixel, lie within
/// its runs: from its top-left pixel through its last row's last pixel,
/// with a row no longer than the stride.
fn within(plane: &Plane, source: Rect, size: u64) -> bool {
    let row = u64::from(source.width) * size;
    let last = u64::from(source.height - 1);
    last.checked_mul(u64::from(plane.stride))
        .and_then(|span| span.checked_add(row))
        .and_then(|span| plane.offset.checked_add(span))
        .is_some_and(|end| end <= plane.bytes() && row <= u64::from(plane.stride))
}

/// The frame of `layers` on a `width` by `height` screen, writing `rows`,
/// or why the GPU does not compose it. A layer wholly off screen is left
/// out, as the CPU leaves it.
pub fn plan(width: u32, height: u32, rows: Range<u32>, layers: &[Layer]) -> Result<Plan, Decline> {
    if layers.len() > LAYERS {
        return Err(Decline::Layers);
    }
    if width == 0 || height == 0 || rows.is_empty() || rows.end > height {
        return Err(Decline::Rows);
    }
    if width > MAX_SIDE || height > MAX_SIDE {
        return Err(Decline::Size);
    }
    let mut placed = Vec::with_capacity(layers.len());
    for layer in layers {
        let (source, destination) = (layer.source, layer.destination);
        let sides = [
            source.width,
            source.height,
            destination.width,
            destination.height,
        ];
        if sides.contains(&0) {
            return Err(Decline::Empty);
        }
        if sides.iter().any(|&side| side > MAX_SIDE) {
            return Err(Decline::Size);
        }
        let alpha = match (layer.format, &layer.alpha) {
            (Format::Xr10A8, Some(alpha)) => within(alpha, source, 1),
            (Format::Xr10A8, None) => false,
            (_, _) => true,
        };
        if !alpha || !within(&layer.color, source, 4) {
            return Err(Decline::Plane);
        }
        if let Some((columns, on_rows)) = on_screen(destination, width, height) {
            placed.push(Placed {
                layer: layer.clone(),
                columns,
                rows: on_rows,
            });
        }
    }
    Ok(Plan {
        width,
        height,
        rows,
        layers: placed,
    })
}

/// `value * weight / 255`, rounded, as the CPU compose scales.
fn scale(value: u32, weight: u32) -> u32 {
    (value * weight + 127) / 255
}

/// The source column or row a destination one at `at` pixels into a
/// destination of `to` pixels takes from a source of `from`: the one under
/// its centre.
fn nearest(at: u32, from: u32, to: u32) -> u32 {
    ((2 * u64::from(at) + 1) * u64::from(from) / (2 * u64::from(to))) as u32
}

/// An extended-range code's display value: the DCP's `XR10_DISPLAY`.
fn xr10(code: u32) -> u32 {
    if code < 385 {
        0
    } else {
        ((code - 383) / 2).min(255)
    }
}

/// The scanout pixel (blue-low, alpha opaque) at `x`, `y` of `plan`,
/// reading a layer's planes through `fetch` (the layer's index, the plane,
/// 0 colour or 1 alpha, and the byte offset in that plane's span; an alpha
/// plane's byte is the low one of what it answers): the layers bottom to
/// top over opaque black, each blended source-over in 8-bit integers
/// exactly as the CPU compose does.
pub fn compose_pixel(
    plan: &Plan,
    x: u32,
    y: u32,
    mut fetch: impl FnMut(usize, usize, u64) -> u32,
) -> u32 {
    // Blue, green, red.
    let mut out = [0u32; 3];
    for (index, placed) in plan.layers.iter().enumerate() {
        if !placed.columns.contains(&x) || !placed.rows.contains(&y) {
            continue;
        }
        let layer = &placed.layer;
        let (source, destination) = (layer.source, layer.destination);
        let column = nearest(x - destination.x, source.width, destination.width);
        let row = nearest(y - destination.y, source.height, destination.height);
        let color = &layer.color;
        let word = fetch(
            index,
            0,
            color.offset + u64::from(row) * u64::from(color.stride) + u64::from(column) * 4,
        );
        let byte = |shift: u32| word >> shift & 0xff;
        let (pixel, alpha) = match (layer.format, &layer.alpha) {
            (Format::Bgra8888, _) => ([byte(0), byte(8), byte(16)], word >> 24),
            (Format::Rgba8888, _) => ([byte(16), byte(8), byte(0)], word >> 24),
            (Format::Xr10A8, alpha) => {
                let at = alpha.as_ref().map_or(0, |alpha| {
                    alpha.offset + u64::from(row) * u64::from(alpha.stride) + u64::from(column)
                });
                let code = |shift: u32| xr10(word >> shift & 0x3ff);
                ([code(0), code(10), code(20)], fetch(index, 1, at) & 0xff)
            }
        };
        for (out, value) in out.iter_mut().zip(pixel) {
            let value = if layer.premultiplied {
                value
            } else {
                scale(value, alpha)
            };
            *out = (value + scale(*out, 255 - alpha)).min(255);
        }
    }
    out[0] | out[1] << 8 | out[2] << 16 | 0xff00_0000
}

/// `ComposeArguments`: what the compose kernel reads at the push constants'
/// arguments address (`STELA_METAL_COMPOSE_*` in `compiler/metal_abi.h`).
pub struct Arguments;

impl Arguments {
    pub const DESTINATION: usize = 0;
    pub const COUNT: usize = 8;
    pub const PITCH: usize = 12;
    pub const WIDTH: usize = 16;
    pub const FIRST_ROW: usize = 20;
    pub const LAYER_COUNT: usize = 24;
    pub const LAYERS: usize = 32;
    pub const LAYER_BYTES: usize = 64;
    pub const BYTES: usize = Self::LAYERS + LAYERS * Self::LAYER_BYTES;
    /// A layer's fields.
    pub const SOURCE: usize = 0;
    pub const STRIDE: usize = 8;
    pub const FLAGS: usize = 12;
    pub const SOURCE_WIDTH: usize = 16;
    pub const SOURCE_HEIGHT: usize = 20;
    pub const X: usize = 24;
    pub const Y: usize = 28;
    pub const WIDTH_SHOWN: usize = 32;
    pub const HEIGHT_SHOWN: usize = 36;
    pub const COLUMNS_END: usize = 40;
    pub const ROWS_END: usize = 44;
    pub const ALPHA: usize = 48;
    pub const ALPHA_STRIDE: usize = 56;
    /// A layer's flags.
    pub const RGBA: u32 = 1;
    pub const PREMULTIPLIED: u32 = 2;
    pub const XR10: u32 = 4;
    /// Threads per workgroup of the kernel.
    pub const WORKGROUP_SIZE: u32 = 64;

    /// The arguments of `plan` written to the buffer at `destination`,
    /// `pitch` bytes a row, each layer's planes at the GPU addresses
    /// `planes` gives for it (colour, then alpha or 0): their spans'
    /// first bytes, which the layer's offsets are counted from.
    pub fn encode(plan: &Plan, destination: u64, pitch: u32, planes: &[(u64, u64)]) -> Vec<u8> {
        let mut bytes = alloc::vec![0u8; Self::BYTES];
        let put32 = |bytes: &mut [u8], at: usize, value: u32| {
            bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        };
        let put64 = |bytes: &mut [u8], at: usize, value: u64| {
            bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
        };
        let rows = plan.rows.end - plan.rows.start;
        put64(&mut bytes, Self::DESTINATION, destination);
        put32(&mut bytes, Self::COUNT, plan.width * rows);
        put32(&mut bytes, Self::PITCH, pitch);
        put32(&mut bytes, Self::WIDTH, plan.width);
        put32(&mut bytes, Self::FIRST_ROW, plan.rows.start);
        put32(&mut bytes, Self::LAYER_COUNT, plan.layers.len() as u32);
        for (index, (placed, &(color, alpha))) in plan.layers.iter().zip(planes).enumerate() {
            let at = Self::LAYERS + index * Self::LAYER_BYTES;
            let layer = &placed.layer;
            let flags = match layer.format {
                Format::Bgra8888 => 0,
                Format::Rgba8888 => Self::RGBA,
                Format::Xr10A8 => Self::XR10,
            } | if layer.premultiplied {
                Self::PREMULTIPLIED
            } else {
                0
            };
            put64(&mut bytes, at + Self::SOURCE, color + layer.color.offset);
            put32(&mut bytes, at + Self::STRIDE, layer.color.stride);
            put32(&mut bytes, at + Self::FLAGS, flags);
            put32(&mut bytes, at + Self::SOURCE_WIDTH, layer.source.width);
            put32(&mut bytes, at + Self::SOURCE_HEIGHT, layer.source.height);
            put32(&mut bytes, at + Self::X, layer.destination.x);
            put32(&mut bytes, at + Self::Y, layer.destination.y);
            put32(&mut bytes, at + Self::WIDTH_SHOWN, layer.destination.width);
            put32(
                &mut bytes,
                at + Self::HEIGHT_SHOWN,
                layer.destination.height,
            );
            put32(&mut bytes, at + Self::COLUMNS_END, placed.columns.end);
            put32(&mut bytes, at + Self::ROWS_END, placed.rows.end);
            if let Some(plane) = &layer.alpha {
                put64(&mut bytes, at + Self::ALPHA, alpha + plane.offset);
                put32(&mut bytes, at + Self::ALPHA_STRIDE, plane.stride);
            }
        }
        bytes
    }

    /// The workgroups that compose `plan`'s rows.
    pub fn groups(plan: &Plan) -> u32 {
        (plan.width * (plan.rows.end - plan.rows.start)).div_ceil(Self::WORKGROUP_SIZE)
    }
}

/// One frame for the backend to compose ([`crate::metal::vulkan::backend::Backend::compose`]):
/// into the scanout buffer the display named `target` when it was
/// imported, `pitch` bytes a row, the rows and layers of `plan`, written
/// complete once the compose marker shows `ticket`. Guest RAM not yet
/// imported is imported only when `import` allows it: an import waits for
/// the host's kernel, which the caller may not wait for.
pub struct Job<'a> {
    pub target: u32,
    pub pitch: u32,
    pub plan: &'a Plan,
    pub ticket: u64,
    pub import: bool,
}

/// What the backend did with a [`Job`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Submitted {
    /// The GPU has the frame; it is complete once the marker shows its
    /// ticket.
    Queued,
    /// The scanout buffer named is not imported: nothing was submitted.
    NoTarget,
    /// A plane is not imported and the job allowed no import, or the
    /// previous frame is still being composed: nothing was submitted.
    NotReady,
}

/// Whether a compose marker reading `marker` shows `ticket` done: the
/// marker holds the low 32 bits of the latest ticket the GPU finished,
/// and tickets increase by one a frame.
pub fn shows(marker: u32, ticket: u64) -> bool {
    marker.wrapping_sub(ticket as u32) as i32 >= 0
}

#[cfg(test)]
mod tests;
