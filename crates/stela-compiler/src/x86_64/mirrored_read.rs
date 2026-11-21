//! MRS of a mirrored System register completed inside the system-mode block
//! that runs it.
//!
//! The engine keeps the plain reads of some System registers in cells
//! ([`super::register_read::Cell`], at [`Anchor::Cells`]): `readable` is
//! nonzero while the read at the current EL and state is the plain register
//! read, neither UNDEFINED nor trapped, and `value` is then what it reads.
//! An MRS of such a register otherwise ends its block, and a thunk the
//! engine installs over the System exit completes it from the cell
//! (`register_read::thunk`). Given the registers and their cells
//! ([`Reads`]), a block runs on past the MRS instead ([`emit`]): it does what
//! that thunk does without the block boundary, the thunk's install and the
//! transfer through it. While the cell is not readable the block leaves at
//! the MRS through a System exit ([`place`]), which retires the instructions
//! before it, and the engine completes the MRS as it completes the exit of a
//! block ending at it. Nothing else changes: the read touches no guest
//! memory, flags or other registers. RAX and host flags are clobbered.
//!
//! A register whose thunk reads more than the cell's value names where its
//! read comes from ([`Source`]): DAIF's, while native code writes the masks
//! itself, from the masks in the frame (`register_read::mask_read_thunk`).
use super::asm::{Alu, Code, Cond, Gpr, Mem, Patch, Rm, ShiftOp, Size};
use crate::{Anchor, GuestAddress, GuestState, Place, aarch64::Register, block::ExitKind};

/// Rt in an MRS word.
const RT: u32 = 31;
/// The cells' fields, as [`super::register_read`] thunks read them.
const READABLE: i32 = core::mem::offset_of!(super::register_read::Cell, readable) as i32;
const VALUE: i32 = core::mem::offset_of!(super::register_read::Cell, value) as i32;
/// PSTATE.{D,A,I,F} in the frame, and their place in DAIF: bits 9:6.
const MASKS: i32 = core::mem::offset_of!(GuestState, daif) as i32;
const MASK_SHIFT: u8 = 6;

/// What an open cell's read completes with.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum Source {
    /// The cell's `value` (`register_read::thunk`).
    #[default]
    Value,
    /// PSTATE's D, A, I and F masks in the frame ([`GuestState::daif`]),
    /// at DAIF's bits 9:6 (`register_read::mask_read_thunk`).
    Masks,
}

/// One register: the MRS of it (Rt is ignored), the offset of its cell
/// from [`Anchor::Cells`] and what its read completes with.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct Read {
    pub word: u32,
    pub cell: u32,
    #[serde(default)]
    pub source: Source,
}

/// At most this many registers complete inside blocks.
pub const READS: usize = 5;

/// The registers whose MRS a block completes inside itself.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct Reads {
    pub reads: [Option<Read>; READS],
}

impl Reads {
    /// Every MRS ends its block.
    pub const NONE: Self = Self {
        reads: [None; READS],
    };

    /// The register MRS `word` reads, if blocks complete it.
    pub fn read(&self, word: u32) -> Option<Read> {
        self.reads
            .iter()
            .flatten()
            .find(|read| read.word & !RT == word & !RT)
            .copied()
    }

    /// The offset of the cell of the register MRS `word` reads, if blocks
    /// complete it.
    pub fn cell(&self, word: u32) -> Option<u32> {
        self.read(word).map(|read| read.cell)
    }
}

/// The guest register MRS `word` writes: none for XZR.
pub(crate) fn destination(word: u32) -> Register {
    match (word & RT) as u8 {
        31 => Register::Zero,
        rt => Register::X(rt),
    }
}

/// What [`emit`] leaves to [`place`] out of line: the branch taken while the
/// cell is closed, to the System exit at `pc`.
pub(crate) struct Slow {
    pc: GuestAddress,
    closed: Patch,
}

/// The inline completion of MRS `word` at `pc` from the cell of `read`:
///
/// ```text
/// mov rax,[rdi+anchors.cells]; cmp qword [rax+cell],0; je closed
/// mov xt,[rax+cell+8]
/// ```
///
/// (through RAX when Xt is in the frame; no load for XZR, whose read can
/// still be UNDEFINED). The masks' read loads `mov eax,[rdi+daif]; shl
/// eax,6` instead, which leaves bits 63:10 and 5:0 clear.
pub(crate) fn emit(code: &mut Code, word: u32, pc: GuestAddress, read: Read) -> Slow {
    let cell = i32::try_from(read.cell).expect("a cell within reach of a displacement");
    let mut asm = code.asm();
    asm.place(Gpr::Rax, Place::new(Anchor::Cells, 0));
    asm.alu_imm8(
        Alu::Cmp,
        Size::Qword,
        Mem::base(Gpr::Rax, cell + READABLE),
        0,
    );
    let closed = asm.jcc(Cond::E);
    let xt = asm.guest(destination(word));
    match read.source {
        Source::Value => {
            let value = Mem::base(Gpr::Rax, cell + VALUE);
            match xt {
                None => {}
                Some(Rm::Reg(host)) => asm.mov_load(Size::Qword, host, value),
                Some(storage) => {
                    asm.mov_load(Size::Qword, Gpr::Rax, value);
                    asm.mov_store(Size::Qword, storage, Gpr::Rax);
                }
            }
        }
        Source::Masks => {
            // A 32-bit write clears the host register's upper half.
            let host = match xt {
                Some(Rm::Reg(host)) => host,
                _ => Gpr::Rax,
            };
            if xt.is_some() {
                asm.mov_load(Size::Dword, host, Mem::base_disp32(Gpr::Rdi, MASKS));
                asm.shift_imm(ShiftOp::Shl, Size::Dword, host, MASK_SHIFT);
            }
            if let Some(storage @ Rm::Mem(_)) = xt {
                asm.mov_store(Size::Qword, storage, Gpr::Rax);
            }
        }
    }
    Slow { pc, closed }
}

/// `slow`'s path: the System exit at the MRS.
pub(crate) fn place(code: &mut Code, slow: Slow) {
    let mut asm = code.asm();
    asm.bind(slow.closed);
    super::block::direct_exit_with(&mut asm, slow.pc, ExitKind::System);
}

#[cfg(test)]
mod tests;
