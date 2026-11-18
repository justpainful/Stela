//! DeviceTree placement and validation for display topology.
use super::dma_routes::{native_dma_mapper_list, native_dma_phandle, native_dma_route};
use super::{
    ASC, DCP_EXPERT, DCP_EXPERT_HARDWARE_FEATURES, DISPLAY_CROSSBAR, DPTX_PHY, ExternalOutput,
    Iommu, Node, RTBUDDY, Vec, asc, dcp, external_output, format, rtbuddy_nub,
    second_display_output,
};
use crate::{Refusal, Result};
use stela_contracts::{refusal::Within, refuse};

/// The `dcpext` display coprocessor and the register apertures around it,
/// from the J473AP DeviceTree: the ASC wrapper `reg[0]` and four mailbox IRQs,
/// whether the nub is `quiesced`, each visible display aperture's primary
/// Complete register views with a stable model name, and the DART streams it masters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DcpController {
    pub phandle: Option<u32>,
    pub mapper: u32,
    pub surface_phandle: Option<u32>,
    /// Full original surface mapper list, primary first; additional PIODMA
    /// ownership is held without inventing execution for that producer.
    pub surface_routes: Vec<DcpMapperRoute>,
    pub role: &'static str,
    pub identity: dcp::ExternalIdentity,
    /// Explicit sole-TRAD selection, checked before this ordinal is held.
    pub instance: usize,
    /// OS mapper context, distinct from the incoming `stream` before remap.
    pub mapper_context: u32,
    pub surface_instance: Option<usize>,
    pub wrapper: (u64, u64),
    pub interrupts: [u32; 4],
    pub quiesced: bool,
    pub regions: Vec<(&'static str, (u64, u64))>,
    /// The [`super::Board::darts`] index and stream of its `iommu-parent`
    /// (`mapper-dcpext`, dart-dcpext stream 5 on J473AP).
    pub iommu: usize,
    pub stream: u32,
    /// That DART's `vm-base` when the node is `truncate-dvas`: the AP then
    /// hands the coprocessor only a DVA's low 32 bits
    /// (`AppleA7IOP::_restrictDVA`, kernel file 0x1bb1424).
    pub window: Option<u64>,
    /// The [`super::Board::darts`] index and stream the swap surfaces are read
    /// through: the visible display pipe's ([`super::DISPEXT`]) first
    /// `iommu-parent` (J473AP `mapper-dispext0`, dart-dispext0 stream 0, the
    /// framebuffer mapping Linux `plane.c` gives the display DART; the second,
    /// `mapper-dispext0-piodma`, is the pipe's programmed-I/O DMA).
    pub surface: Option<(usize, u32)>,
}

/// One original display-pipe mapper owner; metadata does not select traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DcpMapperRoute {
    pub mapper: u32,
    pub iommu: usize,
    pub instance: usize,
    pub mapper_context: u32,
    pub stream: u32,
}

/// One selected external coprocessor, its expert and pipe, with distinct
/// incoming DMA stream and OS mapper context. Refuse incomplete graphs and
/// ambiguous owners before keeping an instance ordinal.
pub(super) fn dcp_device(
    io: &Node<'_>,
    translate: &dyn Fn(u64) -> Option<u64>,
    darts: &[Iommu],
) -> Result<Option<DcpController>> {
    let children = io
        .children()
        .ok_or(refuse!("DCP arm-io children are malformed"))?;
    let mut matches = children.iter().filter_map(|n| {
        n.compatible(ASC)
            .then(|| n.property("role").and_then(external_output))
            .flatten()
            .map(|profile| (n, profile))
    });
    let Some((node, profile)) = matches.next() else {
        return Ok(None);
    };
    if matches.next().is_some() {
        return Err(refuse!(
            "one GOP resource cannot back two external pipelines"
        ));
    }
    place_output(io, node, profile, translate, darts).map(Some)
}

/// The second display pipeline ([`super::DCP_ROLE`]), placed as
/// [`dcp_device`] places the main one, when the DeviceTree presents it
/// admitted. A board template keeps the DCP-role coprocessor's `compatible`
/// too, so that alone admits nothing: the boot chain admits the pipeline
/// only for a boot that asks for it (`stela_loader::devices`, under
/// [`super::Admission::second_display`]), and marks its nub then, as every
/// modeled coprocessor's, a `running` native RTKit. A visible DCP-role
/// coprocessor without that mark is the template's, and none is placed.
pub(super) fn second_display_device(
    io: &Node<'_>,
    translate: &dyn Fn(u64) -> Option<u64>,
    darts: &[Iommu],
) -> Result<Option<DcpController>> {
    let children = io
        .children()
        .ok_or(refuse!("DCP arm-io children are malformed"))?;
    let mut matches = children.iter().filter_map(|n| {
        n.compatible(ASC)
            .then(|| n.property("role").and_then(second_display_output))
            .flatten()
            .map(|profile| (n, profile))
    });
    let Some((node, profile)) = matches.next() else {
        return Ok(None);
    };
    if matches.next().is_some() {
        return Err(refuse!(
            "two DCP-role coprocessors cannot back one second display"
        ));
    }
    let admitted = node.children().is_some_and(|nubs| {
        nubs.iter()
            .any(|nub| nub.compatible(RTBUDDY) && nub.property("running").is_some())
    });
    if !admitted {
        return Ok(None);
    }
    place_output(io, node, profile, translate, darts).map(Some)
}

/// Place the coprocessor `node` of `profile`'s pipeline: its nub, its
/// same-role expert, its pipe and their DMA routes, refusing an incomplete
/// or ambiguous graph.
fn place_output(
    io: &Node<'_>,
    node: &Node<'_>,
    profile: &ExternalOutput,
    translate: &dyn Fn(u64) -> Option<u64>,
    darts: &[Iommu],
) -> Result<DcpController> {
    let children = io
        .children()
        .ok_or(refuse!("DCP arm-io children are malformed"))?;
    let path = format!("/arm-io/{}", profile.controller);
    let err = |what: &'static str| refuse!("{path}", path).because(Refusal::new(what));
    let place = |offset: u64, bytes: u64| -> Result<(u64, u64)> {
        if bytes == 0 {
            return Err(err("aperture is empty"));
        }
        let physical = io
            .translate_range(offset, bytes)
            .ok_or_else(|| err("complete aperture is not owned by contiguous arm-io ranges"))?;
        if translate(offset) != Some(physical) {
            return Err(err(
                "aperture translation differs from its declared arm-io owner",
            ));
        }
        Ok((physical, bytes))
    };
    let nub = rtbuddy_nub(node, profile.role)?;
    if profile
        .external_index
        .is_some_and(|index| nub.property("external-index") != Some(index.to_le_bytes().as_slice()))
    {
        return Err(err(
            "RTBuddy external-index differs from its native output identity",
        ));
    }
    let mut experts = children
        .iter()
        .filter(|n| n.compatible(DCP_EXPERT) && n.property("role") == Some(profile.role_property));
    let expert = experts
        .next()
        .ok_or_else(|| err("required same-role DCP expert is absent"))?;
    if experts.next().is_some() {
        return Err(err("same-role DCP experts are ambiguous"));
    }
    if expert.property("reg").is_some()
        || expert.property("join-power-plane") != Some(1u32.to_le_bytes().as_slice())
        || DCP_EXPERT_HARDWARE_FEATURES
            .iter()
            .any(|flag| expert.property(flag).is_some())
    {
        return Err(err(
            "DCP expert requires an unmodeled hardware power operation",
        ));
    }
    let pipe = io
        .child(profile.pipe)
        .filter(|n| n.compatible(profile.compatible))
        .ok_or_else(|| err("matching external display pipe is absent"))?;
    if profile.external_index.is_some() && pipe.first_u32("external") != Some(1) {
        return Err(err(
            "selected external pipe lacks its original external marker",
        ));
    }
    let wrapper = node
        .pairs("reg")
        .unwrap_or_default()
        .first()
        .copied()
        .ok_or_else(|| err("ASC has no register range"))?;
    if wrapper.1 < asc::WRAPPER_BYTES {
        return Err(err("ASC aperture is incomplete"));
    }
    let interrupts: [u32; 4] = node
        .cells32("interrupts")
        .unwrap_or_default()
        .as_slice()
        .try_into()
        .ok()
        .filter(|_| {
            node.property("interrupts")
                .is_some_and(|bytes| bytes.len() == 16)
        })
        .ok_or_else(|| err("ASC requires exactly four mailbox interrupts"))?;
    if interrupts.iter().any(|&irq| u16::try_from(irq).is_err()) {
        return Err(err("mailbox interrupt exceeds the AIC encoding"));
    }
    let route = |parent: u32| -> Result<(usize, usize, u32, u32)> {
        let route = native_dma_route(darts, parent).within(|| refuse!("{path}", path))?;
        Ok((route.iommu, route.instance, route.context, route.stream))
    };
    let phandle = native_dma_phandle(node, "AAPL,phandle").within(|| refuse!("{path}", path))?;
    let mapper = native_dma_phandle(node, "iommu-parent")
        .within(|| refuse!("{path}", path))?
        .ok_or_else(|| err("controller has no iommu-parent"))?;
    let surface_phandle =
        native_dma_phandle(&pipe, "AAPL,phandle").within(|| refuse!("{path}", path))?;
    let surface_mappers =
        native_dma_mapper_list(&pipe, "iommu-parent").within(|| refuse!("{path}", path))?;
    let (iommu, instance, mapper_context, stream) = route(mapper)?;
    let surface_routes = surface_mappers
        .into_iter()
        .map(|mapper| {
            let (iommu, instance, mapper_context, stream) = route(mapper)?;
            Ok(DcpMapperRoute {
                mapper,
                iommu,
                instance,
                mapper_context,
                stream,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let primary_route = surface_routes
        .first()
        .ok_or_else(|| err("pipe has no surface iommu-parent"))?;
    let pipe_reg = pipe
        .property("reg")
        .ok_or_else(|| err("pipe has no register apertures"))?;
    if pipe_reg.is_empty() || !pipe_reg.len().is_multiple_of(16) {
        return Err(err("pipe register apertures are incomplete"));
    }
    let mut regions = pipe
        .pairs("reg")
        .unwrap_or_default()
        .into_iter()
        .map(|(base, bytes)| {
            if bytes == 0 {
                return Err(err("pipe register aperture is empty"));
            }
            let range = place(base, bytes)?;
            if range.0.checked_add(range.1).is_none() {
                return Err(err("pipe register aperture overflows"));
            }
            Ok((profile.pipe, range))
        })
        .collect::<Result<Vec<_>>>()?;
    // Preserve existing optional J473 aperture placement. These physical-link
    // services remain outside admission; they are not invented T6000 nodes.
    if profile.identity == dcp::ExternalIdentity::Unnumbered {
        for (name, compatible) in [
            ("dptx-phy", DPTX_PHY),
            ("display-crossbar0", DISPLAY_CROSSBAR),
        ] {
            if let Some(child) = io.child(name).filter(|n| n.compatible(compatible)) {
                let reg = child
                    .pairs("reg")
                    .unwrap_or_default()
                    .first()
                    .copied()
                    .ok_or_else(|| err("display aperture lacks reg"))?;
                regions.push((name, place(reg.0, reg.1)?));
            }
        }
    }
    Ok(DcpController {
        phandle,
        mapper,
        surface_phandle,
        surface_instance: Some(primary_route.instance),
        surface: Some((primary_route.iommu, primary_route.stream)),
        surface_routes,
        role: profile.role,
        identity: profile.identity,
        instance,
        mapper_context,
        wrapper: place(wrapper.0, wrapper.1)?,
        interrupts,
        quiesced: nub.property("quiesced").is_some(),
        regions,
        iommu,
        stream,
        window: node.property("truncate-dvas").map(|_| darts[iommu].vm_base),
    })
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::board::fixture_support::{self, tree_with_io};

    #[test]
    fn dcp_placement_reads_the_wrapper_irqs_nub_and_display_apertures_without_admitting_it() {
        use stela_contracts::adt::encode;
        const MODELED: [&str; 3] = [UART, WATCHDOG, "aic,2"];
        let z = |text: &str| [text.as_bytes(), &[0]].concat();
        let words = |cells: &[u32]| {
            cells
                .iter()
                .flat_map(|c| c.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let pairs = |cells: &[u64]| {
            cells
                .iter()
                .flat_map(|c| c.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let irqs = [0x242, 0x242, 0x245, 0x244];
        let build = |role: &[u8], quiesced: bool, with_regions: bool| {
            let mut nub_props: Vec<(&str, &[u8])> = if quiesced {
                vec![("name", b"iop-dcpext-nub\0"), ("quiesced", &[])]
            } else {
                vec![("name", b"iop-dcpext-nub\0")]
            };
            nub_props.push(("compatible", b"iop-nub,rtbuddy-v2\0"));
            let nub = encode(&nub_props, &[]);
            let mapper = encode(
                &[
                    ("name", &z("mapper-dcpext")),
                    ("compatible", &z(IOMMU_MAPPER)),
                    ("reg", &words(&[5])),
                    ("AAPL,phandle", &words(&[0xc8])),
                ],
                &[],
            );
            let dart = encode(
                &[
                    ("name", &z("dart-dcpext")),
                    ("compatible", &z(DART)),
                    ("reg", &pairs(&[0x7130_c000, 0x4000])),
                    ("sid-count", &words(&[16])),
                    ("instance", b"TRAD\0\0\0\0DARTASC\0"),
                    ("interrupts", &words(&[0x251])),
                    ("vm-base", &words(&[0, 8])),
                ],
                &[mapper],
            );
            let dcpext = encode(
                &[
                    ("name", &z("dcpext")),
                    ("compatible", &z(ASC)),
                    ("role", role),
                    ("interrupts", &words(&irqs)),
                    ("iommu-parent", &words(&[0xc8])),
                    ("truncate-dvas", &[]),
                    ("reg", &pairs(&[0x71c0_0000, 0x6c000, 0x7185_0000, 0x4000])),
                ],
                &[nub],
            );
            let surface_mappers = [
                encode(
                    &[
                        ("name", &z("mapper-dispext0")),
                        ("compatible", &z(IOMMU_MAPPER)),
                        ("reg", &words(&[0])),
                        ("AAPL,phandle", &words(&[0xca])),
                    ],
                    &[],
                ),
                encode(
                    &[
                        ("name", &z("mapper-dispext0-piodma")),
                        ("compatible", &z(IOMMU_MAPPER)),
                        ("reg", &words(&[4])),
                        ("AAPL,phandle", &words(&[0xcb])),
                    ],
                    &[],
                ),
            ];
            let pipe_dart = encode(
                &[
                    ("name", &z("dart-dispext0")),
                    ("compatible", &z(DART)),
                    ("reg", &pairs(&[0x7130_4000, 0x4000, 0x7130_0000, 0x4000])),
                    ("sid-count", &words(&[16])),
                    (
                        "instance",
                        b"TRAD\0\0\0\0DART\0\0\0\0UMMS\0\0\0\0SMMU\0\0\0\0",
                    ),
                    ("interrupts", &words(&[0x251])),
                ],
                &surface_mappers,
            );
            let expert = encode(
                &[
                    ("name", &z("dcpext-expert")),
                    ("compatible", &z(DCP_EXPERT)),
                    ("role", DCPEXT_ROLE),
                    ("join-power-plane", &words(&[1])),
                ],
                &[],
            );
            let mut children = alloc::vec![dart, dcpext, expert];
            if with_regions {
                children.push(pipe_dart);
                children.push(encode(
                    &[
                        ("name", &z("dispext0")),
                        ("compatible", &z(DISPEXT)),
                        ("reg", &pairs(&[0x7000_0000, 0x61c000])),
                        ("iommu-parent", &words(&[0xca, 0xcb])),
                    ],
                    &[],
                ));
                children.push(encode(
                    &[
                        ("name", &z("dptx-phy")),
                        ("compatible", &z(DPTX_PHY)),
                        ("reg", &pairs(&[0x3c54_0000, 0xc000])),
                    ],
                    &[],
                ));
                children.push(encode(
                    &[
                        ("name", &z("display-crossbar0")),
                        ("compatible", &z(DISPLAY_CROSSBAR)),
                        ("reg", &pairs(&[0x3c00_0000, 0x4000])),
                    ],
                    &[],
                ));
            }
            tree_with_io(0x8112, "Apple M2", MODELED, &[], &children)
        };
        let found = board(&build(DCPEXT_ROLE, true, true)).unwrap();
        let dcp = found.dcp.expect("dcpext is parsed");
        // /arm-io ranges translate the low window by +0x2_0000_0000.
        assert_eq!(dcp.wrapper, (0x2_71c0_0000, 0x6c000));
        assert_eq!(dcp.interrupts, irqs);
        assert!(dcp.quiesced);
        assert_eq!((dcp.iommu, dcp.stream), (0, 5));
        assert_eq!(found.darts[0].vm_base, 0x8_0000_0000);
        assert_eq!(dcp.window, Some(0x8_0000_0000));
        // The surfaces translate through dart-dispext0 stream 0; its SMMU
        // block is the second reg entry.
        assert_eq!(dcp.surface, Some((1, 0)));
        assert_eq!(dcp.mapper, 0xc8);
        assert_eq!(
            dcp.surface_routes
                .iter()
                .map(|r| (r.mapper, r.mapper_context, r.stream))
                .collect::<Vec<_>>(),
            [(0xca, 0, 0), (0xcb, 4, 4)]
        );
        assert_eq!(found.darts[1].instances[0].physical, 0x2_7130_4000);
        assert_eq!(found.darts[1].smmu, Some((0x2_7130_0000, 0x4000)));
        assert_eq!(found.darts[0].smmu, None);
        assert_eq!(
            dcp.regions,
            alloc::vec![
                ("dispext0", (0x2_7000_0000, 0x61c000)),
                ("dptx-phy", (0x2_3c54_0000, 0xc000)),
                ("display-crossbar0", (0x2_3c00_0000, 0x4000)),
            ]
        );
        // A native controller cannot be exposed without its pipe.
        assert!(board(&build(DCPEXT_ROLE, false, false)).is_err());
        // DCPEXT and its pipe are admitted with a native nub; the PHY and
        // crossbar stay hidden, and a wrong role is not a DCP.
        let role = |key: &str| (key == "role").then_some(DCPEXT_ROLE);
        assert!(models(ASC, role));
        assert_eq!(model_children(ASC, role)[0].compatible, RTBUDDY);
        assert!(models(DISPEXT, |_| None));
        let t6000_role = |key: &str| (key == "role").then_some(b"DCPEXT0\0".as_slice());
        assert_eq!(models(ASC, t6000_role), T6000_EXTERNAL_OUTPUT_ADMITTED);
        assert_eq!(
            models(DISPEXT_T6000, |_| None),
            T6000_EXTERNAL_OUTPUT_ADMITTED
        );
        assert!(external_output(b"DCPEXT1\0").is_none());

        assert!(!models(DPTX_PHY, |_| None) && !models(DISPLAY_CROSSBAR, |_| None));
        // Its power expert binds; the internal display's expert does not.
        assert!(models(DCP_EXPERT, role));
        assert!(model_children(DCP_EXPERT, role).is_empty());
        assert!(!models(DCP_EXPERT, |key| (key == "role")
            .then_some(b"DCP\0".as_slice())));
        assert!(board(&build(b"DCP\0", true, false)).unwrap().dcp.is_none());
    }

    #[test]
    fn second_display_is_placed_only_once_the_boot_chain_admits_it() {
        // A template's DCP-role coprocessor is not a second display: the
        // main pipeline alone is placed, as it always was.
        let template = board(&fixture_support::two_display_pipelines(false, true)).unwrap();
        assert_eq!(template.dcp.as_ref().map(|dcp| dcp.role), Some("DCPEXT"));
        assert!(template.second_display.is_none());
        assert_eq!(template.displays().count(), 1);

        let found = board(&fixture_support::two_display_pipelines(true, true)).unwrap();
        let main = found.dcp.as_ref().expect("dcpext stays the main display");
        let second = found.second_display.as_ref().expect("disp0 is placed");
        assert_eq!(
            found.displays().map(|dcp| dcp.role).collect::<Vec<_>>(),
            ["DCPEXT", "DCP"]
        );
        assert_eq!(main.identity, dcp::ExternalIdentity::Unnumbered);
        assert_eq!(second.identity, dcp::ExternalIdentity::Disp0);
        assert_eq!(second.wrapper, (0x2_31c0_0000, 0x6c000));
        assert_eq!(second.interrupts, [0x218, 0x217, 0x21a, 0x219]);
        assert!(second.quiesced);
        // Its own DARTs and streams: dart-dcp stream 5 for the link, with
        // no truncate-dvas window, and dart-disp0 stream 0 for surfaces.
        assert_eq!(second.mapper, 0xbd);
        assert_ne!(second.iommu, main.iommu);
        assert_eq!(
            found.darts[second.iommu].instances[0].physical,
            0x2_3130_c000
        );
        assert_eq!(second.stream, 5);
        assert_eq!(second.window, None);
        assert_eq!(main.window, Some(0x8_0000_0000));
        let (surface, stream) = second.surface.expect("disp0 surfaces");
        assert_ne!(Some((surface, stream)), main.surface);
        assert_eq!(found.darts[surface].instances[0].physical, 0x2_3130_4000);
        assert_eq!(stream, 0);
        assert_eq!(
            second
                .surface_routes
                .iter()
                .map(|r| (r.mapper, r.mapper_context, r.stream))
                .collect::<Vec<_>>(),
            [(0xbf, 0, 0), (0xc0, 4, 4)]
        );
        // Only the pipe's own aperture: the crossbar and DPTX PHY stay
        // `dcpext`'s optional views, and hidden.
        assert_eq!(
            second.regions,
            [
                ("disp0", (0x2_3000_0000, 0x61c000)),
                ("disp0", fixture_support::SHARED_DISPLAY_APERTURE),
            ]
        );
        assert_eq!(main.regions[1], ("dispext0", second.regions[1].1));

        // An admitted coprocessor without its power expert refuses.
        assert!(board(&fixture_support::two_display_pipelines(true, false)).is_err());
    }

    #[test]
    fn second_display_claims_hold_only_under_its_admission() {
        let dcp = |key: &str| (key == "role").then_some(DCP_ROLE);
        let second = Admission {
            second_display: true,
        };
        assert!(!models(ASC, dcp) && !models(DCP_EXPERT, dcp) && !models(DISP0, |_| None));
        assert!(model_children(ASC, dcp).is_empty());
        assert!(models_under(ASC, dcp, second));
        assert!(models_under(DCP_EXPERT, dcp, second));
        assert!(models_under(DISP0, |_| None, second));
        assert_eq!(
            model_children_under(ASC, dcp, second)[0].compatible,
            RTBUDDY
        );
        assert!(model_boot_properties_under(ASC, dcp, second).is_empty());
        assert!(
            model_children_under(ASC, dcp, second)[0]
                .boot_properties
                .iter()
                .any(|&(key, _)| key == "running")
        );
        // The rest of the board is admitted as it always is.
        let dcpext = |key: &str| (key == "role").then_some(DCPEXT_ROLE);
        assert!(models_under(ASC, dcpext, second) && models(ASC, dcpext));
        assert!(!models_under(DPTX_PHY, |_| None, second));
        assert!(!models_under(DISPLAY_CROSSBAR, |_| None, second));
        // The DCP role never stands in for the board's one external output.
        assert!(external_output(DCP_ROLE).is_none());
        let profile = second_display_output(DCP_ROLE).expect("disp0 profile");
        assert!(!profile.admitted_under(Admission::default()));
        assert!(profile.admitted_under(second));
        assert_eq!(
            (profile.controller, profile.expert, profile.pipe),
            ("dcp", "dcp0-expert", "disp0")
        );
        assert!(second_display_output(DCPEXT_ROLE).is_none());
    }
}
