//! Which accesses of a system-mode block check their address against the
//! guest's native ranges ([`super::RangeCheck`]).
//!
//! An immediate-offset access moves at most 2^16 bytes from its base, so
//! once one through a base register has been made, the base is within
//! 2^16 of a range. While the base keeps that value or moves from it by
//! immediates (writeback, or ADD and SUB immediate, also into another
//! register), every later immediate-offset access through it is within
//! 2^35 of a range, as a block holds at most 1,024 instructions. With
//! five-level paging that is in a range, or in PML5 entry 1 or 510, which
//! no shadow root maps, where an x86 #PF takes the fault the guest's
//! translation gives; Stela's window, 63 entries from either range, stays
//! out of reach. With four-level paging it is in a range, above 2^47
//! (non-canonical, #GP), in the unmapped guard of at least 2^35 bytes below
//! the upper range (#PF), or across 0 or 2^64, wrapping into a range as
//! Arm's address arithmetic does; never in Stela's window below the guard.
//! Those accesses need no check of their own. Any other write of the
//! register, and any instruction this analysis does not know, forgets it.
//!
//! A base the block formed from its PC (ADR or ADRP, then ADD and SUB
//! immediate) holds a value known when the block compiles, at a distance
//! from the PC that stays when the engine moves the block to another page.
//! Where that distance is within 4 GiB but an access's reach and the range
//! check passes the PC's 4 GiB slice and those on either side of it
//! ([`super::RangeCheck::passes_around`]), the check the first access
//! through the base would make passes wherever such a block runs: that
//! access needs none either ([`Constants`]). The engine moves a stored block
//! only to pages whose slices pass alike (`Preparation::context`).
use super::AROUND_DISTANCE;
use crate::aarch64::{
    Instruction, MemoryTransfer, Operation, Register, Width, atomic, literal, non_temporal,
    ordered, simd,
};

/// A register's bit in a set of bases: X0-X30 and SP as bit 31.
fn bit(register: Register) -> u32 {
    match register {
        Register::X(index) => 1 << index,
        Register::Sp => 1 << 31,
        Register::Zero => 0,
    }
}

/// For each instruction of a block, whether its access checks the guest
/// range: all but the immediate-offset accesses through a base the block
/// has already made such an access through, unmodified since but by
/// immediates, and those through a base formed from the PC that `range`
/// passes wherever the block runs ([`Constants::passes`]).
pub(crate) fn checks(range: super::RangeCheck, instructions: &[Instruction], rows: &mut [bool]) {
    assert_eq!(rows.len(), instructions.len());
    let mut bases = 0u32;
    let mut constants = Constants::default();
    for (row, instruction) in rows.iter_mut().zip(instructions) {
        *row = immediate_base(&instruction.operation).is_none_or(|base| {
            bases & bit(base) == 0 && !constants.passes(range, instruction.pc, base)
        });
        bases = after(&instruction.operation, bases);
        constants.after(&instruction.operation);
    }
}

/// The registers whose value the block formed from its PC alone: ADR and
/// ADRP, then ADD and SUB immediate of 64 bits. Any other write forgets a
/// register, as does an access through it (its writeback may move it), and
/// an instruction [`after`] does not know forgets them all.
#[derive(Default)]
struct Constants {
    /// The registers whose value is known, as bits of a set of bases.
    known: u32,
    values: [u64; 32],
}

impl Constants {
    fn get(&self, register: Register) -> Option<u64> {
        let bit = bit(register);
        (self.known & bit != 0).then(|| self.values[bit.trailing_zeros() as usize])
    }

    fn set(&mut self, register: Register, value: Option<u64>) {
        let bit = bit(register);
        if bit == 0 {
            return;
        }
        match value {
            Some(value) => {
                self.known |= bit;
                self.values[bit.trailing_zeros() as usize] = value;
            }
            None => self.known &= !bit,
        }
    }

    /// Whether `base`, read at `pc`, holds a value within
    /// [`AROUND_DISTANCE`] of `pc`, whose slice `range` passes with its
    /// neighbours: every address an immediate-offset access through it
    /// reaches (2^16 bytes either way) passes there, and wherever else a
    /// block moved with the same answer runs, the value moving with its PC.
    fn passes(&self, range: super::RangeCheck, pc: crate::GuestAddress, base: Register) -> bool {
        self.get(base).is_some_and(|value| {
            value.wrapping_sub(pc.0).wrapping_add(AROUND_DISTANCE) <= 2 * AROUND_DISTANCE
                && range.passes_around(pc.0)
        })
    }

    /// The values known after `operation`.
    fn after(&mut self, operation: &Operation) {
        match *operation {
            Operation::PcRelative(operand) => {
                self.set(operand.destination, Some(operand.address.0));
            }
            Operation::AddSub {
                width,
                subtract,
                destination,
                source,
                immediate,
                ..
            } => {
                let value = self
                    .get(source)
                    .filter(|_| width == Width::X64)
                    .map(|value| {
                        if subtract {
                            value.wrapping_sub(u64::from(immediate))
                        } else {
                            value.wrapping_add(u64::from(immediate))
                        }
                    });
                self.set(destination, value);
            }
            _ if self.known == 0 => {}
            _ => {
                // What [`after`] keeps of every base is what the operation
                // leaves unwritten, but for an access's own base, which its
                // writeback may move.
                let accessed = immediate_base(operation).map_or(0, bit);
                self.known &= after(operation, u32::MAX) & !accessed;
            }
        }
    }
}

/// The base of an access whose address is the base plus an immediate.
fn immediate_base(operation: &Operation) -> Option<Register> {
    match operation {
        Operation::Memory { base, .. }
        | Operation::MemoryPair { base, .. }
        | Operation::Simd(
            simd::Operand::Memory { base, .. } | simd::Operand::MemoryPair { base, .. },
        ) => Some(*base),
        Operation::Simd(simd::Operand::StructureMemory(operand)) => Some(operand.base),
        Operation::MemoryUnprivileged(operand) => Some(operand.base),
        Operation::MemoryNonTemporal(operand) => Some(operand.base),
        Operation::Atomic(operand) => Some(operand.base),
        Operation::Ordered(ordered::Operand::Transfer { base, .. }) => Some(*base),
        _ => None,
    }
}

/// The bases known after `operation`, given `bases` before it.
fn after(operation: &Operation, bases: u32) -> u32 {
    let loaded = |transfer: &MemoryTransfer| matches!(transfer, MemoryTransfer::Load { .. });
    match operation {
        Operation::Memory {
            transfer,
            data,
            base,
            ..
        } => {
            let bases = bases | bit(*base);
            if loaded(transfer) {
                bases & !bit(*data)
            } else {
                bases
            }
        }
        Operation::MemoryPair {
            transfer,
            data,
            base,
            ..
        } => {
            let bases = bases | bit(*base);
            if loaded(transfer) {
                bases & !bit(data[0]) & !bit(data[1])
            } else {
                bases
            }
        }
        Operation::MemoryUnprivileged(operand) => {
            let bases = bases | bit(operand.base);
            if loaded(&operand.transfer) {
                bases & !bit(operand.data)
            } else {
                bases
            }
        }
        Operation::MemoryNonTemporal(operand) => {
            let bases = bases | bit(operand.base);
            match operand.data {
                non_temporal::Data::Integer(data) if operand.load => {
                    bases & !bit(data[0]) & !bit(data[1])
                }
                _ => bases,
            }
        }
        // Vector data; the base moves by an immediate if at all.
        Operation::Simd(
            simd::Operand::Memory { base, .. } | simd::Operand::MemoryPair { base, .. },
        ) => bases | bit(*base),
        Operation::Simd(simd::Operand::StructureMemory(operand)) => match operand.post_index {
            simd::structure_memory::PostIndex::Register(_) => bases & !bit(operand.base),
            _ => bases | bit(operand.base),
        },
        Operation::AddSub {
            width,
            destination,
            source,
            ..
        } => {
            if *width == Width::X64 && bases & bit(*source) != 0 {
                bases | bit(*destination)
            } else {
                bases & !bit(*destination)
            }
        }
        Operation::AddSubRegister { destination, .. }
        | Operation::AddSubCarry { destination, .. }
        | Operation::AddSubExtended { destination, .. }
        | Operation::LogicalRegister { destination, .. }
        | Operation::LogicalImmediate { destination, .. }
        | Operation::MoveWide { destination, .. } => bases & !bit(*destination),
        // CAS and CASP return the old value in the compared registers.
        Operation::Atomic(operand) => {
            let bases = bases | bit(operand.base);
            match operand.kind {
                atomic::Kind::CompareExchange { compare, .. } => bases & !bit(compare),
                atomic::Kind::CompareExchangePair { compare, .. } => {
                    bases & !bit(compare[0]) & !bit(compare[1])
                }
                atomic::Kind::ReadModifyWrite { destination, .. } => bases & !bit(destination),
            }
        }
        Operation::Ordered(ordered::Operand::Transfer {
            kind, data, base, ..
        }) => {
            let bases = bases | bit(*base);
            if *kind == ordered::Kind::Release {
                bases
            } else {
                bases & !bit(*data)
            }
        }
        // A register offset proves nothing of its base.
        Operation::MemoryRegisterOffset(operand) => {
            if loaded(&operand.transfer) {
                bases & !bit(operand.data)
            } else {
                bases
            }
        }
        Operation::Literal(operand) => match operand.transfer {
            literal::Transfer::Integer { destination, .. } => bases & !bit(destination),
            literal::Transfer::Simd { .. } => bases,
        },
        Operation::PcRelative(operand) => bases & !bit(operand.destination),
        Operation::MirroredRead { word } => bases & !bit(super::mirrored_read::destination(*word)),
        Operation::ConditionalSelect(operand) => bases & !bit(operand.destination),
        Operation::Bitfield(operand) => bases & !bit(operand.destination),
        Operation::IntegerMath(operand) => bases & !bit(operand.destination),
        Operation::IntegerUnary(operand) => bases & !bit(operand.destination),
        Operation::IntegerMinMax(operand) => bases & !bit(operand.destination),
        Operation::Crc(operand) => bases & !bit(operand.destination),
        Operation::Flags(crate::aarch64::flags::Operand::Read { destination }) => {
            bases & !bit(*destination)
        }
        // NZCV, barriers and hints only; the block a branch leaves is over.
        Operation::Flags(_)
        | Operation::BranchCondition { .. }
        | Operation::BranchZero { .. }
        | Operation::BranchBit { .. }
        | Operation::Synchronization(_)
        | Operation::Ordered(ordered::Operand::Barrier { .. })
        | Operation::Nop
        | Operation::Prefetch(_)
        // DC ZVA writes memory only, after its own check.
        | Operation::ZeroBlock { .. } => bases,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::super::RangeCheck;
    use super::*;
    use crate::{GuestAddress, Lifter, aarch64::Decoder};
    use alloc::{vec, vec::Vec};

    fn lift(words: &[u32]) -> Vec<Instruction> {
        lift_at(0x1000, words)
    }

    fn lift_at(pc: u64, words: &[u32]) -> Vec<Instruction> {
        words
            .iter()
            .enumerate()
            .map(|(index, &word)| {
                Decoder
                    .lift(GuestAddress(pc.wrapping_add(4 * index as u64)), word)
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn a_base_is_checked_once_until_it_changes_but_by_immediates() {
        // stp x29, x30, [sp, #-16]!; mov x29, sp; ldr x0, [x29, #8];
        // str x1, [sp, #8]; sub sp, sp, #32; ldp x2, x3, [sp], #16;
        // ldr x4, [x5]; ldr x5, [x5, #8]; ldr x6, [x5]; add w5, w6, #1;
        // ldr x7, [x5]; ldr x8, [x6, x9]; ldr x10, [x6]
        let words = [
            0xa9bf_7bfd,
            0x9100_03fd,
            0xf940_07a0,
            0xf900_07e1,
            0xd100_83ff,
            0xa8c1_0fe2,
            0xf940_00a4,
            0xf940_04a5,
            0xf940_00a6,
            0x1100_04c5,
            0xf940_00a7,
            0xf869_68c8,
            0xf940_00ca,
        ];
        let mut rows = [false; 13];
        checks(RangeCheck::FIVE_LEVEL, &lift(&words), &mut rows);
        assert_eq!(
            rows,
            [
                true,  // the first access through SP
                true,  // a move
                false, // X29 = SP
                false, // SP again
                true,  // SUB SP, SP, #imm
                false, // SP moved by an immediate
                true,  // the first through X5
                false, // X5 again, which the load replaces
                true,  // X5 was loaded
                true,  // an add
                true,  // a W result is no address near a range
                true,  // a register offset is checked at its sum
                true,  // X6 was loaded and has made no access
            ]
        );
    }

    #[test]
    fn unknown_instructions_and_register_post_indexes_forget_bases() {
        // ldr x0, [x1]; fmov x2, d0 (not analyzed); ldr x6, [x1];
        // ld1 {v0.16b}, [x1], x7; ldr x8, [x1]; ld1 {v0.16b}, [x1], #16;
        // ldr x9, [x1]
        let words = [
            0xf940_0020,
            0x9e66_0002,
            0xf940_0026,
            0x4cc7_7020,
            0xf940_0028,
            0x4cdf_7020,
            0xf940_0029,
        ];
        let mut rows = [false; 7];
        checks(RangeCheck::FIVE_LEVEL, &lift(&words), &mut rows);
        assert_eq!(rows, [true, true, true, false, true, false, false]);
    }

    const ADRP_X8: u32 = 0x9000_0008;
    const ADR_X8: u32 = 0x1000_0008;
    const LDR_X9_X8_16: u32 = 0xf940_0909;

    /// A kernel PC, in the upper range of every placement below, and a
    /// program's, a page past a Mach-O's 4 GiB base.
    const KERNEL: u64 = 0xffff_fe00_0712_3400;
    const PROGRAM: u64 = 0x1_0000_4000;
    const PLACEMENTS: [RangeCheck; 3] = [
        RangeCheck::FIVE_LEVEL,
        RangeCheck::four_level(2),
        RangeCheck::four_level(9),
    ];

    fn checks_at(range: RangeCheck, pc: u64, words: &[u32]) -> Vec<bool> {
        let mut rows = vec![false; words.len()];
        checks(range, &lift_at(pc, words), &mut rows);
        rows
    }

    #[test]
    fn a_base_from_the_pc_that_passes_needs_no_check() {
        // adrp x8, #0; add x10, x8, #0; sub x11, x8, #32; ldr x9, [x10];
        // ldr x9, [x11]; add x8, x8, #16; ldr x9, [x8, #16]
        let words = [
            ADRP_X8,
            0x9100_010a,
            0xd100_810b,
            0xf940_0149,
            0xf940_0169,
            0x9100_4108,
            LDR_X9_X8_16,
        ];
        let skipped = [true, true, true, false, false, true, false];
        for range in PLACEMENTS {
            for (form, pc, expected) in [
                (range, KERNEL, skipped),
                (range, PROGRAM, skipped),
                // EL0's check passes the lower range alone: each first
                // access through a base checks at a kernel PC.
                (range.one_sided(true), KERNEL, [true; 7]),
                (range.one_sided(true), PROGRAM, skipped),
            ] {
                assert_eq!(checks_at(form, pc, &words), expected, "{form:?} {pc:#x}");
            }
        }
    }

    #[test]
    fn other_writes_of_a_constant_base_forget_it() {
        let cases: [(&[u32], &[bool]); 5] = [
            // adrp x8; add w8, w8, #1; ldr x9, [x8]
            (&[ADRP_X8, 0x1100_0508, 0xf940_0109], &[true, true, true]),
            // adrp x8; mov x10, x8; ldr x9, [x10]
            (&[ADRP_X8, 0xaa08_03ea, 0xf940_0149], &[true, true, true]),
            // adrp x8; ldr x8, [x8]; ldr x9, [x8]
            (&[ADRP_X8, 0xf940_0108, 0xf940_0109], &[true, false, true]),
            // adrp x8; fmov x2, d0 (not analyzed); ldr x9, [x8]
            (&[ADRP_X8, 0x9e66_0002, 0xf940_0109], &[true, true, true]),
            // adrp x8; ldr x9, [x8, #8]!; ldr x9, [x8]; add x10, x8, #0;
            // ldr x9, [x10]: the writeback moves X8 off its value, and the
            // access made it a base as any access does.
            (
                &[ADRP_X8, 0xf840_8d09, 0xf940_0109, 0x9100_010a, 0xf940_0149],
                &[true, false, false, true, false],
            ),
        ];
        for range in PLACEMENTS {
            for (words, expected) in cases {
                assert_eq!(
                    checks_at(range, KERNEL, words),
                    expected,
                    "{range:?} {words:08x?}"
                );
            }
        }
    }

    /// The first address of the 4 GiB slice `slice` (signed).
    fn slice(slice: i64) -> u64 {
        (slice << 32) as u64
    }

    #[test]
    fn a_constant_base_keeps_its_check_where_a_neighbouring_slice_fails() {
        // adrp x8 1 GiB up and down; and 4 GiB, ADRP's farthest pages,
        // beyond a slice's reach.
        const NEAR: [u32; 2] = [0x9020_0008, 0x90e0_0008];
        const FAR: [u32; 2] = [0xf07f_ffe8, 0x9080_0008];
        for range in PLACEMENTS {
            for form in [range, range.one_sided(true)] {
                // The slices of each range's ends: the last of the lower
                // range and the first of the upper, each with a failing
                // neighbour, and the ones inside them.
                let lower_end = (form.lower_limit() >> 32) as i64;
                let mut cases = vec![(lower_end - 1, true), (lower_end - 2, false)];
                if !form.one_sided {
                    let upper_start = (-(1 << 17)..0).find(|&at| form.passes(slice(at)));
                    let upper_start = upper_start.unwrap();
                    // The ranges meet at 0.
                    cases.extend([(upper_start, true), (upper_start + 1, false), (-1, false)]);
                }
                // EL0's range starts at 0, whose slice has a failing one
                // below it.
                cases.extend([(0, form.one_sided), (1, false)]);
                for (at, checked) in cases {
                    let pcs = [
                        slice(at),
                        slice(at) + 0x1000,
                        slice(at + 1).wrapping_sub(0x1000),
                    ];
                    for pc in pcs {
                        for word in [ADR_X8].iter().chain(&NEAR) {
                            assert_eq!(
                                checks_at(form, pc, &[*word, LDR_X9_X8_16]),
                                [true, checked],
                                "{form:?} {pc:#x} {word:#x}"
                            );
                        }
                        for word in FAR {
                            assert_eq!(
                                checks_at(form, pc, &[word, LDR_X9_X8_16]),
                                [true, true],
                                "{form:?} {pc:#x} {word:#x}"
                            );
                        }
                    }
                }
            }
        }
    }
}
