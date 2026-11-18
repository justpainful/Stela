use super::*;
use alloc::{vec, vec::Vec};

fn rect(x: u32, y: u32, width: u32, height: u32) -> Rect {
    Rect {
        x,
        y,
        width,
        height,
    }
}

/// A tight plane of `bytes`, from guest address 0.
fn plane(bytes: u64, stride: u32) -> Plane {
    Plane {
        ranges: vec![MemoryRange { address: 0, bytes }],
        offset: 0,
        stride,
    }
}

/// A layer of `format` over a tight plane of `width` by `height` pixels,
/// shown at its own size at `x`, `y`.
fn layer(format: Format, width: u32, height: u32, x: u32, y: u32) -> Layer {
    let alpha = (format == Format::Xr10A8).then(|| plane(u64::from(width * height), width));
    Layer {
        format,
        color: plane(u64::from(width * height * 4), width * 4),
        alpha,
        source: rect(0, 0, width, height),
        destination: rect(x, y, width, height),
        premultiplied: false,
    }
}

#[test]
fn a_frame_the_gpu_cannot_give_exactly_is_declined() {
    let base = layer(Format::Bgra8888, 4, 4, 0, 0);
    let five = vec![base.clone(); 5];
    assert_eq!(plan(4, 4, 0..4, &five), Err(Decline::Layers));
    assert_eq!(
        plan(4, 4, 0..5, core::slice::from_ref(&base)),
        Err(Decline::Rows)
    );
    assert_eq!(
        plan(4, 4, 2..2, core::slice::from_ref(&base)),
        Err(Decline::Rows)
    );
    let empty = Layer {
        source: rect(0, 0, 0, 4),
        ..base.clone()
    };
    assert_eq!(plan(4, 4, 0..4, &[empty]), Err(Decline::Empty));
    let huge = Layer {
        destination: rect(0, 0, MAX_SIDE + 1, 4),
        ..base.clone()
    };
    assert_eq!(plan(4, 4, 0..4, &[huge]), Err(Decline::Size));
    // The last row's last pixel lies one byte past the plane.
    let mut short = base.clone();
    short.color.ranges[0].bytes -= 1;
    assert_eq!(plan(4, 4, 0..4, &[short]), Err(Decline::Plane));
    let mut narrow = base.clone();
    narrow.color.stride = 12;
    assert_eq!(plan(4, 4, 0..4, &[narrow]), Err(Decline::Plane));
    let mut far = base.clone();
    far.color.offset = u64::MAX - 8;
    assert_eq!(plan(4, 4, 0..4, &[far]), Err(Decline::Plane));
    // A two-plane layer needs its alpha plane, whole.
    let mut alphaless = layer(Format::Xr10A8, 4, 4, 0, 0);
    alphaless.alpha = None;
    assert_eq!(plan(4, 4, 0..4, &[alphaless]), Err(Decline::Plane));
    let mut thin = layer(Format::Xr10A8, 4, 4, 0, 0);
    thin.alpha.as_mut().unwrap().offset = 1;
    assert_eq!(plan(4, 4, 0..4, &[thin]), Err(Decline::Plane));
}

#[test]
fn layers_are_clipped_to_the_screen_and_one_off_it_is_left_out() {
    let cursor = layer(Format::Bgra8888, 4, 4, 2, 3);
    let gone = layer(Format::Rgba8888, 2, 2, 4, 0);
    let plan = plan(4, 4, 1..3, &[cursor.clone(), gone]).unwrap();
    assert_eq!(plan.rows, 1..3);
    assert_eq!(
        plan.layers,
        [Placed {
            layer: cursor,
            columns: 2..4,
            rows: 3..4,
        }]
    );
}

/// What `fetch` reads for a layer from `planes`: its colour words, then
/// its alpha bytes.
fn from(planes: &[(Vec<u32>, Vec<u8>)]) -> impl FnMut(usize, usize, u64) -> u32 + '_ {
    |index, plane, at| match plane {
        0 => {
            assert_eq!(at % 4, 0);
            planes[index].0[(at / 4) as usize]
        }
        _ => u32::from(planes[index].1[at as usize]),
    }
}

#[test]
fn an_unscaled_layer_is_copied_opaque_in_the_scanouts_order() {
    let bgra = layer(Format::Bgra8888, 1, 1, 0, 0);
    let rgba = layer(Format::Rgba8888, 1, 1, 1, 0);
    let plan = plan(2, 1, 0..1, &[bgra, rgba]).unwrap();
    let planes = [(vec![0xff03_0201], vec![]), (vec![0xff03_0201], vec![])];
    assert_eq!(compose_pixel(&plan, 0, 0, from(&planes)), 0xff03_0201);
    assert_eq!(compose_pixel(&plan, 1, 0, from(&planes)), 0xff01_0203);
}

#[test]
fn layers_blend_source_over_black_as_the_cpu_compose_rounds() {
    let below = Layer {
        premultiplied: true,
        ..layer(Format::Bgra8888, 1, 1, 0, 0)
    };
    let above = layer(Format::Bgra8888, 1, 1, 0, 0);
    // Straight alpha over black: 200 * 128 / 255, rounded, is 100.
    let alone = plan(1, 1, 0..1, core::slice::from_ref(&above)).unwrap();
    let planes = [(vec![0x8000_00c8], vec![])];
    assert_eq!(compose_pixel(&alone, 0, 0, from(&planes)), 0xff00_0064);
    // A premultiplied layer shows its colour over black whatever its alpha.
    let alone = plan(1, 1, 0..1, core::slice::from_ref(&below)).unwrap();
    let planes = [(vec![0x0007_0809], vec![])];
    assert_eq!(compose_pixel(&alone, 0, 0, from(&planes)), 0xff07_0809);
    // Premultiplied 50 at alpha 128 over 200: 50 + 200 * 127 / 255 = 150;
    // and 250 at alpha 10 over 255 saturates.
    let over = Layer {
        premultiplied: true,
        ..above
    };
    let both = plan(1, 1, 0..1, &[below, over]).unwrap();
    let planes = [(vec![0xff00_ffc8], vec![]), (vec![0x8000_fa32], vec![])];
    assert_eq!(compose_pixel(&both, 0, 0, from(&planes)) & 0xffff, 0xff96);
}

#[test]
fn an_extended_range_layer_shows_its_codes_clamped_with_its_alpha_plane() {
    // Codes 383 (black), 385 (1) and 1000 (white), alpha 255; then a
    // straight layer at alpha 0 over it changes nothing.
    let word = 383 | 385 << 10 | 1000 << 20;
    let base = layer(Format::Xr10A8, 1, 1, 0, 0);
    let plan = plan(1, 1, 0..1, &[base.clone(), base]).unwrap();
    let planes = [(vec![word], vec![255]), (vec![word], vec![0])];
    assert_eq!(compose_pixel(&plan, 0, 0, from(&planes)), 0xffff_0100);
    assert_eq!(xr10(384), 0);
    assert_eq!(xr10(639), 128);
    assert_eq!(xr10(893), 255);
}

#[test]
fn a_scaled_layer_takes_the_source_pixel_under_each_centre() {
    // Two columns widened to four, then four narrowed to two.
    let wide = Layer {
        destination: rect(0, 0, 4, 1),
        ..layer(Format::Bgra8888, 2, 1, 0, 0)
    };
    let narrow = Layer {
        destination: rect(0, 0, 2, 1),
        ..layer(Format::Bgra8888, 4, 1, 0, 0)
    };
    let columns = |layer: Layer, width: u32| -> Vec<u8> {
        let plan = plan(width, 1, 0..1, &[layer]).unwrap();
        let planes = [((0..4).map(|x| 0xff00_0000 | x).collect::<Vec<_>>(), vec![])];
        (0..width)
            .map(|x| compose_pixel(&plan, x, 0, from(&planes)) as u8)
            .collect()
    };
    assert_eq!(columns(wide, 4), [0, 0, 1, 1]);
    assert_eq!(columns(narrow, 2), [1, 3]);
}

#[test]
fn a_source_row_starts_at_its_offset_and_stride() {
    // Row 1 of a source two pixels into a plane of three-pixel rows.
    let mut layer = layer(Format::Bgra8888, 1, 2, 0, 0);
    layer.color = Plane {
        offset: 8,
        ..plane(24, 12)
    };
    let plan = plan(1, 2, 0..2, &[layer]).unwrap();
    let planes = [((0..6).map(|i| 0xff00_0000 | i).collect::<Vec<_>>(), vec![])];
    assert_eq!(compose_pixel(&plan, 0, 1, from(&planes)) as u8, 5);
}

#[test]
fn the_arguments_match_the_kernels_layout() {
    // `STELA_METAL_COMPOSE_*`.
    assert_eq!(Arguments::BYTES, 288);
    let mut alpha = layer(Format::Xr10A8, 2, 2, 1, 0);
    alpha.premultiplied = true;
    alpha.color.offset = 16;
    alpha.color.ranges[0].bytes += 16;
    let plan = plan(4, 4, 1..3, &[layer(Format::Rgba8888, 4, 4, 0, 0), alpha]).unwrap();
    let bytes = Arguments::encode(&plan, 0x1000, 32, &[(0x2000, 0), (0x3000, 0x4000)]);
    let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
    assert_eq!(u64_at(0), 0x1000);
    assert_eq!(
        [u32_at(8), u32_at(12), u32_at(16), u32_at(20), u32_at(24)],
        [8, 32, 4, 1, 2]
    );
    assert_eq!(
        (u64_at(32), u32_at(40), u32_at(44)),
        (0x2000, 16, Arguments::RGBA)
    );
    let second = 32 + 64;
    assert_eq!(u64_at(second), 0x3010);
    assert_eq!(
        u32_at(second + 12),
        Arguments::XR10 | Arguments::PREMULTIPLIED
    );
    let fields: Vec<u32> = (16..48).step_by(4).map(|at| u32_at(second + at)).collect();
    assert_eq!(fields, [2, 2, 1, 0, 2, 2, 3, 2]);
    assert_eq!((u64_at(second + 48), u32_at(second + 56)), (0x4000, 2));
    assert_eq!(Arguments::groups(&plan), 1);
}

#[test]
fn the_marker_shows_a_ticket_once_it_reaches_it_across_a_wrap() {
    assert!(!shows(0, 1));
    assert!(shows(1, 1));
    assert!(shows(2, 1));
    assert!(shows(0, u64::from(u32::MAX)));
    assert!(!shows(u32::MAX, 1 << 32));
}
