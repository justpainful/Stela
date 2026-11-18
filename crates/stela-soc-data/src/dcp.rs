//! Immutable external display-pipe interface identity.
use crate::Result;
use stela_contracts::refuse;
pub const AVIPC_SERVICE_NAME: &str = "dcpav-controller-epic";
/// Identity of one backed external output. The unnumbered DCPEXT and
/// numbered DCPEXT0/1 use the same original firmware protocol, but publish
/// distinct AFKEPV2 interface names. This does not model an internal panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExternalIdentity {
    #[default]
    Unnumbered,
    /// The external pipe ordinal, not the AFKEPV2 interface ID or unit.
    Numbered(u8),
    /// The J473AP DCP-role pipe `disp0` (`/arm-io/dcp`, role `DCP`), which
    /// the Mac mini wires to its USB-C DisplayPort and marks `external` = 1
    /// in its DeviceTree: an external output of its own beside `dcpext`'s,
    /// admitted only for a second display (`board::Admission`). The main
    /// display stays the unnumbered DCPEXT pipe's.
    Disp0,
}

impl ExternalIdentity {
    pub fn interface_name(self) -> Result<alloc::string::String> {
        // t600xdcp_restore 0xf9434..0xf944c formats %s%c:%s:%u. Its external
        // instance byte 0 yields E; 1..10 yield 0..9. Each model has one
        // controller (unit 0) and an independently allocated interface ID.
        let (prefix, suffix) = match self {
            Self::Unnumbered => ("dispext", 'E'),
            Self::Numbered(index @ 0..=9) => ("dispext", char::from(b'0' + index)),
            Self::Numbered(_) => {
                return Err(refuse!(
                    "DCP external pipe ordinal exceeds the native single-character identity"
                ));
            }
            // Inferred, not read from the DCP-role firmware: the same format
            // with the pipe's own DeviceTree name, `disp` and its unit 0.
            Self::Disp0 => ("disp", '0'),
        };
        Ok(alloc::format!("{prefix}{suffix}:{}:0", AVIPC_SERVICE_NAME))
    }

    /// Whether this is the display macOS starts as its main one: the
    /// external pipes the board has alone, never the second display.
    pub fn main_display(self) -> bool {
        self != Self::Disp0
    }

    /// The IOMFB framebuffer ID the pipe answers `get_framebuffer_id` with:
    /// 1 for the external pipes, as the model always has, and 0 for
    /// `disp0`, the pipe the DeviceTree numbers first.
    pub fn framebuffer_id(self) -> u32 {
        match self {
            Self::Disp0 => 0,
            Self::Unnumbered | Self::Numbered(_) => 1,
        }
    }

    /// Whether the monitor reaches the pipe's DisplayPort through the
    /// Mac's DP-to-HDMI converter (`dp2hdmi-gpio`, the J473AP HDMI port that
    /// `dcpext` drives) rather than directly, as `disp0`'s USB-C
    /// DisplayPort does.
    pub fn hdmi(self) -> bool {
        self != Self::Disp0
    }
}
