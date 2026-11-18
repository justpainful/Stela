//! A present composed outside the devices' lock.
//!
//! Composing a swap reads its surfaces' rows (to find the changed ones, then
//! to compose them) and writes the host scanout: milliseconds a frame, which
//! held every PE's device access while the step ran under the devices' lock.
//! On a board shared between PEs ([`super::Coprocessor::defer_compose`]) the
//! step instead resolves every DART page the latched layers can reach, under
//! the lock and without a fault's effects ([`crate::dart::Dart::resolve`]),
//! and leaves the compose to the PE that released the lock ([`Compose::run`])
//! with the scanout and its kept rows. The next transfer takes them back and
//! finishes the present as one in place would have at the same vblank: the
//! swaps it showed complete, in order, after the frame is on the scanout.
//!
//! The surfaces stay mapped through the compose: the AP takes a swap's
//! surfaces back only after its completion (D594), which follows the
//! compose. A layer with any page that does not resolve, and a read the
//! resolved pages do not cover, are presented in place under the lock
//! instead, where a DART fault has its effects as before. Calls that drop
//! queued swaps (an abort, a link shutdown, a sleep) first wait for a
//! compose in flight to end, so they act on the swaps as they were after it.
//!
//! A scanout whose host GPU composes ([`Scanout::try_present_layers`]) is
//! offered the frame first, its layers' planes as the guest physical runs
//! the resolved pages give ([`Pages::gpu_layers`]). It answers before the
//! GPU is done: the compose then ends at once, and the present finishes,
//! completing its swaps, only at the first look that finds the GPU done.
//! The kept rows are forgotten, as the CPU did not compose what the
//! scanout shows. A sink that declines or refuses has the frame composed
//! on the CPU, as without a GPU.
use super::super::scanout::{
    self, Composed, GpuFrame, GpuLayer, GpuPending, GpuPlane, Layer, PixelFormat, Scanout,
};
use crate::Result;
use crate::handoff::Handoff;
use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::ops::Range;
use stela_contracts::Refusal;
use stela_contracts::dma::{DmaError, Memory};
use stela_contracts::gate::MemoryRange;

/// The physical address of every page a present's layers can read, from
/// the DART of the display pipe, by I/O address.
pub(crate) struct Pages {
    page: u64,
    /// Each reachable span's first page's I/O address and every page's
    /// physical address, in order.
    spans: Vec<(u64, Vec<u64>)>,
    /// For each span, in the same order: its layer's index, its plane's
    /// (0 colour, 1 alpha), and the I/O address of its first byte and the
    /// plane's stride.
    planes: Vec<(usize, usize, u64, u32)>,
}

impl Pages {
    /// The pages `layers` can read, each resolved by `resolve` (the I/O
    /// address of a page to its physical address); `None` when one does not
    /// resolve or an address overflows, which leaves the present in place.
    pub(crate) fn resolve(
        layers: &[Layer],
        page: u64,
        mut resolve: impl FnMut(u64) -> core::result::Result<u64, DmaError>,
    ) -> Option<Self> {
        if !page.is_power_of_two() {
            return None;
        }
        let mut spans = Vec::new();
        let mut planes = Vec::new();
        for (index, layer) in layers.iter().enumerate() {
            for (plane, (span, stride)) in layer_spans(layer)?.into_iter().enumerate() {
                if span.is_empty() {
                    continue;
                }
                let first = span.start & !(page - 1);
                let mut pages = Vec::new();
                let mut at = first;
                while at < span.end {
                    pages.push(resolve(at).ok()?);
                    at = at.checked_add(page)?;
                }
                spans.push((first, pages));
                planes.push((index, plane, span.start, stride));
            }
        }
        Some(Self {
            page,
            spans,
            planes,
        })
    }

    /// `layers`, the ones these pages were resolved for, each plane at the
    /// guest physical runs its pages make, consecutive pages merged into
    /// one run: what a host GPU reads, which takes a run at a time rather
    /// than a page. `surface_ids` names each layer's surface where known.
    /// `None` when a layer lacks a plane's pages (an empty source), and
    /// the CPU composes the frame.
    pub(crate) fn gpu_layers(
        &self,
        layers: &[Layer],
        surface_ids: &[Option<u32>],
    ) -> Option<Vec<GpuLayer>> {
        let mut gpu: Vec<GpuLayer> = layers
            .iter()
            .enumerate()
            .map(|(index, layer)| GpuLayer {
                format: layer.surface.format,
                planes: Vec::new(),
                source: layer.source,
                destination: layer.destination,
                premultiplied: layer.premultiplied,
                surface_id: surface_ids.get(index).copied().flatten(),
            })
            .collect();
        for ((first, pages), &(index, plane, start, stride)) in self.spans.iter().zip(&self.planes)
        {
            let mut ranges: Vec<MemoryRange> = Vec::new();
            for &physical in pages {
                match ranges.last_mut() {
                    Some(run) if run.address.checked_add(run.bytes) == Some(physical) => {
                        run.bytes += self.page;
                    }
                    _ => ranges.push(MemoryRange {
                        address: physical,
                        bytes: self.page,
                    }),
                }
            }
            let layer = gpu.get_mut(index)?;
            if layer.planes.len() != plane {
                return None;
            }
            layer.planes.push(GpuPlane {
                ranges,
                offset: start - first,
                stride,
            });
        }
        let planes = |layer: &GpuLayer| match layer.format {
            PixelFormat::Xr10A8 { .. } => 2,
            _ => 1,
        };
        gpu.iter()
            .all(|layer| layer.planes.len() == planes(layer))
            .then_some(gpu)
    }

    /// The physical address of the page holding `iova`, if resolved.
    fn physical(&self, iova: u64) -> Option<u64> {
        self.spans.iter().find_map(|(first, pages)| {
            let index = iova.checked_sub(*first)? / self.page;
            pages
                .get(usize::try_from(index).ok()?)
                .map(|physical| physical + iova % self.page)
        })
    }

    /// Read `bytes` at `iova` from guest RAM `memory` through the resolved
    /// pages, a page at a time as a port does; false when a page is not
    /// among them or not RAM.
    fn read(&self, memory: &mut dyn Memory, iova: u64, bytes: &mut [u8]) -> Result<bool, Missed> {
        let mut done = 0;
        while done < bytes.len() {
            let Some(address) = iova.checked_add(done as u64) else {
                return Err(Missed);
            };
            let chunk = ((self.page - address % self.page) as usize).min(bytes.len() - done);
            let Some(physical) = self.physical(address) else {
                return Err(Missed);
            };
            if memory
                .read_progress(physical, &mut bytes[done..done + chunk])
                .is_err()
            {
                return Ok(false);
            }
            done += chunk;
        }
        Ok(true)
    }
}

/// A read past the resolved pages: the present is made again in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Missed;

/// The I/O address ranges `layer`'s source rows occupy in each of its
/// surface's planes, with the plane's stride: from its first row's first
/// pixel to its last row's last; `None` on overflow.
fn layer_spans(layer: &Layer) -> Option<Vec<(Range<u64>, u32)>> {
    let source = layer.source;
    if source.width == 0 || source.height == 0 {
        return Some(Vec::new());
    }
    let span = |base: u64, stride: u32, size: u64| {
        let first = u64::from(source.y)
            .checked_mul(stride.into())?
            .checked_add(u64::from(source.x).checked_mul(size)?)?
            .checked_add(base)?;
        let last_row = u64::from(source.y) + u64::from(source.height) - 1;
        let end = last_row
            .checked_mul(stride.into())?
            .checked_add((u64::from(source.x) + u64::from(source.width)).checked_mul(size)?)?
            .checked_add(base)?;
        Some((first..end, stride))
    };
    let surface = layer.surface;
    let mut spans = alloc::vec![span(surface.iova, surface.stride, 4)?];
    if let PixelFormat::Xr10A8 {
        alpha,
        alpha_stride,
    } = surface.format
    {
        spans.push(span(alpha, alpha_stride, 1)?);
    }
    Some(spans)
}

/// What a compose hands back: the scanout and its kept rows, what the
/// present came to, and whether a read missed the resolved pages; with the
/// host GPU's ticket when it took the frame, whose present finishes once
/// the GPU is done, and its refusal when it refused one the CPU then
/// composed.
pub(crate) struct Composition {
    pub sink: Box<dyn Scanout>,
    pub composed: Composed,
    pub result: Result<()>,
    pub missed: bool,
    pub gpu: Option<GpuPending>,
    pub gpu_refused: Option<Refusal>,
}

/// A present's compose, run outside the devices' lock ([`Self::run`]).
pub struct Compose {
    sink: Box<dyn Scanout>,
    composed: Composed,
    layers: Vec<Layer>,
    /// Each layer's surface ID, where the swaps the present shows name it.
    surface_ids: Vec<Option<u32>>,
    damage: Option<Range<usize>>,
    pages: Pages,
    done: Arc<Handoff<Composition>>,
}

impl Compose {
    pub(crate) fn new(
        sink: Box<dyn Scanout>,
        composed: Composed,
        layers: Vec<Layer>,
        surface_ids: Vec<Option<u32>>,
        damage: Option<Range<usize>>,
        pages: Pages,
    ) -> (Self, Arc<Handoff<Composition>>) {
        let done = Handoff::new();
        (
            Self {
                sink,
                composed,
                layers,
                surface_ids,
                damage,
                pages,
                done: Arc::clone(&done),
            },
            done,
        )
    }

    /// Compose the present onto the scanout, reading guest RAM `memory`:
    /// on the host GPU when the scanout takes the frame, else on the CPU.
    pub fn run(self, memory: &mut dyn Memory) {
        let Self {
            mut sink,
            mut composed,
            layers,
            surface_ids,
            damage,
            pages,
            done,
        } = self;
        let gpu_refused = match offer_gpu(sink.as_mut(), &layers, &surface_ids, &damage, &pages) {
            Ok(Some(pending)) => {
                composed.forget();
                done.put(Composition {
                    sink,
                    composed,
                    result: Ok(()),
                    missed: false,
                    gpu: Some(pending),
                    gpu_refused: None,
                });
                return;
            }
            Ok(None) => None,
            Err(reason) => Some(reason),
        };
        let mut missed = false;
        let read = |iova: u64, bytes: &mut [u8]| match pages.read(memory, iova, bytes) {
            Ok(reached) => reached,
            Err(Missed) => {
                missed = true;
                false
            }
        };
        let result = scanout::present(sink.as_mut(), &layers, damage, read, &mut composed);
        done.put(Composition {
            sink,
            composed,
            result,
            missed,
            gpu: None,
            gpu_refused,
        });
    }
}

/// Offer the frame of `layers` to `sink`'s host GPU, its rows of `damage`
/// within the mode: the GPU's ticket when it took the frame. A present
/// that changes no row is not offered, as the CPU leaves the scanout
/// untouched for it; nor is one whose planes the pages do not all give.
fn offer_gpu(
    sink: &mut dyn Scanout,
    layers: &[Layer],
    surface_ids: &[Option<u32>],
    damage: &Option<Range<usize>>,
    pages: &Pages,
) -> Result<Option<GpuPending>> {
    let mode = sink.mode();
    let height = mode.height as usize;
    let damage = damage
        .clone()
        .map(|rows| rows.start.min(height)..rows.end.min(height));
    if damage.as_ref().is_some_and(Range::is_empty) {
        return Ok(None);
    }
    let Some(gpu) = pages.gpu_layers(layers, surface_ids) else {
        return Ok(None);
    };
    sink.try_present_layers(&GpuFrame {
        mode,
        damage,
        layers: &gpu,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::scanout::{Rect, Surface};
    use super::*;

    const PAGE: u64 = 0x4000;

    /// A two-plane layer's planes reach the GPU as their pages' physical
    /// runs, consecutive pages merged and the others apart, each plane's
    /// offset at the source's top-left pixel and with its own stride.
    #[test]
    fn a_layers_planes_are_their_pages_merged_into_physical_runs() {
        // One page a colour row, four rows; the alpha rows in one page.
        let surface = Surface {
            width: 4096,
            height: 4,
            stride: 0x4000,
            format: PixelFormat::Xr10A8 {
                alpha: 0x4_0000,
                alpha_stride: 0x1000,
            },
            iova: 0x1_0000,
        };
        let source = Rect {
            x: 1,
            y: 0,
            width: 4095,
            height: 4,
        };
        let destination = Rect {
            x: 0,
            y: 0,
            width: 4095,
            height: 4,
        };
        let layer = Layer::new(surface, source, destination, false).unwrap();
        let physical = |iova: u64| -> core::result::Result<u64, DmaError> {
            match iova {
                0x1_0000 => Ok(0x10_0000),
                0x1_4000 => Ok(0x10_4000),
                0x1_8000 => Ok(0x20_0000),
                0x1_c000 => Ok(0x20_4000),
                0x4_0000 => Ok(0x30_0000),
                _ => Err(DmaError::Fault(1)),
            }
        };
        let pages = Pages::resolve(&[layer], PAGE, physical).expect("every page resolves");
        let gpu = pages.gpu_layers(&[layer], &[Some(5)]).unwrap();
        let run = |address: u64, bytes: u64| MemoryRange { address, bytes };
        assert_eq!(
            gpu,
            [GpuLayer {
                format: surface.format,
                planes: alloc::vec![
                    GpuPlane {
                        ranges: alloc::vec![run(0x10_0000, 0x8000), run(0x20_0000, 0x8000)],
                        offset: 4,
                        stride: 0x4000,
                    },
                    GpuPlane {
                        ranges: alloc::vec![run(0x30_0000, PAGE)],
                        offset: 1,
                        stride: 0x1000,
                    },
                ],
                source,
                destination,
                premultiplied: false,
                surface_id: Some(5),
            }]
        );
    }
}
