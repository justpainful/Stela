//! Shared pure ADT fixtures for Data parsing and Runtime model integration.
//! These builders grant no mapping, executor or native effect permission.
use super::*;
use crate::Result;
pub const PMGR: &str = "pmgr1,t8112";
/// Existing unit fixtures explicitly declare an affine synthetic map.
/// Production callers supply Node::translate_range, never this adapter.
pub fn flat_fixture_extent(
    point: &dyn Fn(u64) -> Option<u64>,
) -> impl Fn(u64, u64) -> Option<u64> + '_ {
    move |address, bytes| {
        if bytes == 0 {
            return None;
        }
        address.checked_add(bytes)?;
        let physical = point(address)?;
        physical.checked_add(bytes)?;
        Some(physical)
    }
}

pub fn layout_of(layout: &[(u8, u32)]) -> Vec<Processor> {
    layout
        .iter()
        .map(|&(cluster_type, reg)| Processor {
            cluster_type,
            reg,
            l2_bytes: Some(if cluster_type == b'P' {
                16 << 20
            } else {
                4 << 20
            }),
        })
        .collect()
}

/// A J473AP-shaped DeviceTree: two PEs, `/arm-io` child 0 at PA
/// 0x2_0000_0000, and the UART, watchdog and AIC as `compatibles` names.
pub fn tree(chip_id: u32, soc: &str, compatibles: [&str; 3]) -> Vec<u8> {
    tree_with(chip_id, soc, compatibles, &[])
}

/// [`tree`] with more `/arm-io/aic` properties.
pub fn tree_with(
    chip_id: u32,
    soc: &str,
    compatibles: [&str; 3],
    aic_properties: &[(&str, u32)],
) -> Vec<u8> {
    tree_with_io(chip_id, soc, compatibles, aic_properties, &[])
}

/// [`tree_with`] with more `/arm-io` children.
pub fn tree_with_io(
    chip_id: u32,
    soc: &str,
    compatibles: [&str; 3],
    aic_properties: &[(&str, u32)],
    io_children: &[Vec<u8>],
) -> Vec<u8> {
    use stela_contracts::adt::encode;
    let z = |text: &str| [text.as_bytes(), &[0]].concat();
    let u32s = |cells: &[u32]| {
        cells
            .iter()
            .flat_map(|c| c.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let u64s = |cells: &[u64]| {
        cells
            .iter()
            .flat_map(|c| c.to_le_bytes())
            .collect::<Vec<_>>()
    };
    // The J473AP's blocks: cpu0's and cpu4's `cpu-impl-reg`, and their
    // clusters' `cpm-impl-reg`.
    let cpu = |name: &str, kind: &[u8], reg: u32| {
        let cluster = 0x100_0000 * u64::from(reg >> 8);
        encode(
            &[
                ("name", &z(name)),
                ("cluster-type", kind),
                ("reg", &u32s(&[reg])),
                ("l2-cache-size", &u32s(&[4 << 20 << (2 * (reg >> 8))])),
                ("cpu-impl-reg", &u64s(&[0x2_1005_0000 + cluster, 0x9010])),
                ("coresight-reg", &u64s(&[0x2_1001_0000 + cluster, 0x300c8])),
                ("cpm-impl-reg", &u64s(&[0x2_10e4_0000 + cluster, 0x9010])),
            ],
            &[],
        )
    };
    let device = |name: &str, compatible: &str, reg: u64, irq: u32| {
        encode(
            &[
                ("name", &z(name)),
                ("compatible", &z(compatible)),
                ("reg", &u64s(&[reg, 0x4000])),
                ("interrupt-parent", &u32s(&[0x63])),
                ("interrupts", &u32s(&[irq])),
            ],
            &[],
        )
    };
    let (name, compatible) = (z("aic"), z(compatibles[2]));
    let (reg, phandle) = (u64s(&[0x3b0c_0000, 0x2_0000]), u32s(&[0x63]));
    // An AICv1 states its 740 inputs (the UART's 739 is the highest) to
    // the word.
    let ipid_mask = vec![0xff; 24 * 4];
    let values: Vec<_> = aic_properties
        .iter()
        .map(|(_, value)| u32s(&[*value]))
        .collect();
    let properties: Vec<(&str, &[u8])> = [
        ("name", name.as_slice()),
        ("compatible", &compatible),
        ("reg", &reg),
        ("AAPL,phandle", &phandle),
    ]
    .into_iter()
    .chain((compatibles[2] == "aic,1").then_some(("ipid-mask", ipid_mask.as_slice())))
    .chain(
        aic_properties
            .iter()
            .zip(&values)
            .map(|((name, _), value)| (*name, value.as_slice())),
    )
    .collect();
    let aic = encode(&properties, &[]);
    let io = encode(
        &[
            ("name", &z("arm-io")),
            (
                "ranges",
                &u64s(&[
                    0,
                    0x2_0000_0000,
                    0x1_0000_0000,
                    0x3_8000_0000,
                    0x3_8000_0000,
                    0x1_0000_0000,
                ]),
            ),
        ],
        &[
            vec![
                device("uart0", compatibles[0], 0x3520_0000, 739),
                device("wdt", compatibles[1], 0x3d2b_0000, 379),
                aic,
            ],
            io_children.to_vec(),
        ]
        .concat(),
    );
    encode(
        &[("name", &z("device-tree"))],
        &[
            encode(
                &[("name", &z("chosen")), ("chip-id", &u32s(&[chip_id]))],
                &[],
            ),
            encode(
                &[("name", &z("product")), ("product-soc-name", &z(soc))],
                &[],
            ),
            encode(
                &[("name", &z("cpus"))],
                &[cpu("cpu0", b"E", 0), cpu("cpu1", b"P", 0x100)],
            ),
            io,
        ],
    )
}

/// `/arm-io` children shaped like J514cAP's: the always-on PTD
/// (phandle 0x127, one cell per notification) and a CLPC (its
/// `/arm-io/pmgr/clpc`) that names the AIC (0x63) for its first three
/// specifiers and the PTD for the four FourCCs after them.
pub fn clpc_with_ptd() -> Vec<Vec<u8>> {
    use stela_contracts::adt::encode;
    let z = |text: &str| [text.as_bytes(), &[0]].concat();
    let words = |cells: &[u32]| {
        cells
            .iter()
            .flat_map(|c| c.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let ptd = encode(
        &[
            ("name", &z("aon-ptd")),
            ("AAPL,phandle", &words(&[0x127])),
            ("#interrupt-cells", &words(&[1])),
            ("interrupt-controller", &z("master")),
        ],
        &[],
    );
    let clpc = encode(
        &[
            ("name", &z("clpc")),
            (
                "interrupt-parent",
                &words(&[0x63, 0x63, 0x63, 0x127, 0x127, 0x127, 0x127]),
            ),
            (
                "interrupts",
                &words(&[
                    0x5bf,
                    0,
                    0x5bd,
                    0x7030_5452,
                    0x7031_5452,
                    0x6130_5452,
                    0x6630_5452,
                ]),
            ),
        ],
        &[],
    );
    vec![ptd, clpc]
}
pub fn cache_fixture(bytes: u32, ways: u32) -> Cache {
    super::cache(bytes, ways)
}
pub fn spmi_controllers(
    io: &Node<'_>,
    translate: &dyn Fn(u64, u64) -> Option<u64>,
    aic: u32,
) -> Result<Vec<SpmiController>> {
    super::spmi_topology::spmi_controllers(io, translate, aic)
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImplicitSupplyFixture {
    pub physical: u64,
    pub bytes: u64,
    pub register: u64,
}
pub fn implicit_state_fixture() -> (PowerStates, [ImplicitSupplyFixture; 3]) {
    super::power_resources::implicit::fixture_support::complete_fixture()
}

pub fn parse_watchdog_fixture(
    node: &Node<'_>,
    translate: &dyn Fn(u64, u64) -> Option<u64>,
) -> Result<Option<WatchdogSide>> {
    super::watchdog_side(node, translate)
}
pub fn parse_iommu_fixture(
    node: &Node<'_>,
    translate: &dyn Fn(u64, u64) -> Option<u64>,
) -> Result<Iommu> {
    super::dart_topology::iommu(node, translate)
}
pub fn parse_dcp_fixture(
    io: &Node<'_>,
    translate: &dyn Fn(u64) -> Option<u64>,
    darts: &[Iommu],
) -> Result<Option<DcpController>> {
    super::display_topology::dcp_device(io, translate, darts)
}

pub const T6050_FIXTURE_LIMITS: KernelLimits = super::T6050_LIMITS;
pub fn parse_smc_fixture(
    root: &Node<'_>,
    io: &Node<'_>,
    translate: &dyn Fn(u64, u64) -> Option<u64>,
) -> Result<Option<SmcController>> {
    super::smc_topology::smc_controller(root, io, translate)
}

/// Explicit synthetic gate/resource metadata for Runtime model fixtures.
/// The Runtime owner still validates every lifecycle/supply binding.
pub fn power_states_fixture(
    blocks: Vec<(u64, u64)>,
    registers: Vec<u64>,
    domains: Vec<crate::pmgr::coverage::Domain>,
    resources: Vec<PowerResource>,
) -> PowerStates {
    PowerStates {
        blocks,
        registers,
        domains,
        resources,
        implicit: Vec::new(),
    }
}

/// The aperture both pipes of [`two_display_pipelines`] share, translated.
pub const SHARED_DISPLAY_APERTURE: (u64, u64) = (0x2_3b3d_0000, 0x4000);

/// The J473AP's two display pipelines as its DeviceTree gives
/// them (phandles, mapper streams, IRQs and apertures), the DCP-role
/// one's nub marked `running` as the boot chain marks an admitted one, and
/// its power expert present with `dcp_expert`. Both pipes end with one
/// shared SoC aperture ([`SHARED_DISPLAY_APERTURE`]), as J473AP's
/// `dispext0` and `disp0` end with the same three.
pub fn two_display_pipelines(dcp_running: bool, dcp_expert: bool) -> Vec<u8> {
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
    let mapper = |name: &str, stream: u32, phandle: u32| {
        encode(
            &[
                ("name", &z(name)),
                ("compatible", &z(IOMMU_MAPPER)),
                ("reg", &words(&[stream])),
                ("AAPL,phandle", &words(&[phandle])),
            ],
            &[],
        )
    };
    let dart = |name: &str, regs: &[u64], instance: &[u8], irq: u32, mappers: &[Vec<u8>]| {
        encode(
            &[
                ("name", &z(name)),
                ("compatible", &z(DART)),
                ("reg", &pairs(regs)),
                ("sid-count", &words(&[16])),
                ("instance", instance),
                ("interrupts", &words(&[irq])),
                ("vm-base", &words(&[0, 8])),
            ],
            mappers,
        )
    };
    let pipe_instances: &[u8] = b"TRAD\0\0\0\0DART\0\0\0\0UMMS\0\0\0\0SMMU\0\0\0\0";
    let nub = |name: &str, running: bool| {
        let mut props: Vec<(&str, &[u8])> = alloc::vec![
            ("name", name.as_bytes()),
            ("compatible", b"iop-nub,rtbuddy-v2\0"),
            ("quiesced", &[]),
        ];
        if running {
            props.extend([("pre-loaded", &[][..]), ("running", &[][..])]);
        }
        encode(&props, &[])
    };
    let expert = |name: &str, role: &[u8]| {
        encode(
            &[
                ("name", &z(name)),
                ("compatible", &z(DCP_EXPERT)),
                ("role", role),
                ("join-power-plane", &words(&[1])),
            ],
            &[],
        )
    };
    let mut children = alloc::vec![
        dart(
            "dart-dcpext",
            &[0x7130_c000, 0x4000],
            b"TRAD\0\0\0\0DARTASC\0",
            0x251,
            &[mapper("mapper-dcpext", 5, 0xc8)],
        ),
        encode(
            &[
                ("name", &z("dcpext")),
                ("compatible", &z(ASC)),
                ("role", DCPEXT_ROLE),
                ("interrupts", &words(&[0x242, 0x242, 0x245, 0x244])),
                ("iommu-parent", &words(&[0xc8])),
                ("truncate-dvas", &[]),
                ("reg", &pairs(&[0x71c0_0000, 0x6c000, 0x7185_0000, 0x4000])),
            ],
            &[nub("iop-dcpext-nub\0", true)],
        ),
        expert("dcpext-expert", DCPEXT_ROLE),
        dart(
            "dart-dispext0",
            &[0x7130_4000, 0x4000, 0x7130_0000, 0x4000],
            pipe_instances,
            0x251,
            &[
                mapper("mapper-dispext0", 0, 0xca),
                mapper("mapper-dispext0-piodma", 4, 0xcb),
            ],
        ),
        encode(
            &[
                ("name", &z("dispext0")),
                ("compatible", &z(DISPEXT)),
                ("reg", &pairs(&[0x7000_0000, 0x61c000, 0x3b3d_0000, 0x4000])),
                ("iommu-parent", &words(&[0xca, 0xcb])),
            ],
            &[],
        ),
        dart(
            "dart-dcp",
            &[0x3130_c000, 0x4000],
            b"TRAD\0\0\0\0DARTASC\0",
            0x229,
            &[mapper("mapper-dcp", 5, 0xbd)],
        ),
        encode(
            &[
                ("name", &z("dcp")),
                ("compatible", &z(ASC)),
                ("role", DCP_ROLE),
                ("interrupts", &words(&[0x218, 0x217, 0x21a, 0x219])),
                ("iommu-parent", &words(&[0xbd])),
                ("hdcp-channels", &words(&[5, 6])),
                ("dp-switch-ufp-endpoint", &words(&[0])),
                ("reg", &pairs(&[0x31c0_0000, 0x6c000, 0x3185_0000, 0x4000])),
            ],
            &[nub("iop-dcp-nub\0", dcp_running)],
        ),
        dart(
            "dart-disp0",
            &[0x3130_4000, 0x4000, 0x3130_0000, 0x4000],
            pipe_instances,
            0x229,
            &[
                mapper("mapper-disp0", 0, 0xbf),
                mapper("mapper-disp0-piodma", 4, 0xc0),
            ],
        ),
        encode(
            &[
                ("name", &z("disp0")),
                ("compatible", &z(DISP0)),
                ("external", &words(&[1])),
                ("reg", &pairs(&[0x3000_0000, 0x61c000, 0x3b3d_0000, 0x4000])),
                ("iommu-parent", &words(&[0xbf, 0xc0])),
            ],
            &[],
        ),
    ];
    if dcp_expert {
        children.push(expert("dcp0-expert", DCP_ROLE));
    }
    tree_with_io(0x8112, "Apple M2", MODELED, &[], &children)
}
