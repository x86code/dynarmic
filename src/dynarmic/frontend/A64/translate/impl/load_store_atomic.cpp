/* This file is part of the dynarmic project.
 * Copyright (c) 2026 designgears (x86Code)
 * SPDX-License-Identifier: 0BSD
 *
 * ARMv8.1 LSE (LD<op>, SWP, CAS, CASP), ARMv8.3 RCPC (LDAPR) and ARMv8.4 RCPC2 (LDAPUR*, STLUR).
 * The atomics become A64AtomicRMW / A64AtomicCAS / A64AtomicCAS128, which the x64 backend emits as lock-prefixed
 * instructions on the fastmem address. x86 locked instructions are full barriers, so every ordering variant (A/L/AL)
 * maps to the same code; RCPC loads/stores are plain ORDERED accesses (TSO).
 */

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

static IR::U64 AtomicAddress(TranslatorVisitor& v, Reg Rn) {
    if (Rn == Reg::SP) {
        return v.SP(64);
    }
    return v.X(64, Rn);
}

static IR::U64 Zext64(TranslatorVisitor& v, size_t size, Reg r) {
    return size == 3 ? v.ir.GetX(r) : v.ir.ZeroExtendToLong(v.ir.GetW(r));
}

static void WriteOld(TranslatorVisitor& v, size_t size, Reg r, const IR::U64& old) {
    if (size == 3) {
        v.X(64, r, old);
    } else {
        v.X(32, r, v.ir.LeastSignificantWord(old));
    }
}

bool TranslatorVisitor::AtomicMemOp(Imm<2> sz, bool, bool, Reg Rs, bool o3, Imm<3> opc, Reg Rn, Reg Rt) {
    const size_t size = sz.ZeroExtend<size_t>();
    u8 op = opc.ZeroExtend<u8>();  // 0 ADD 1 CLR 2 EOR 3 SET 4 SMAX 5 SMIN 6 UMAX 7 UMIN
    if (o3) {
        if (op != 0) {
            return UnallocatedEncoding();  // LDAPR is matched by its own pattern
        }
        op = 8;  // SWP
    }
    const IR::U64 address = AtomicAddress(*this, Rn);
    const IR::U64 old = ir.AtomicRMW(address, Zext64(*this, size, Rs), op, size);
    WriteOld(*this, size, Rt, old);
    return true;
}

bool TranslatorVisitor::CAS(Imm<2> sz, bool, Reg Rs, bool, Reg Rn, Reg Rt) {
    const size_t size = sz.ZeroExtend<size_t>();
    const IR::U64 address = AtomicAddress(*this, Rn);
    const IR::U64 old = ir.AtomicCAS(address, Zext64(*this, size, Rs), Zext64(*this, size, Rt), size);
    WriteOld(*this, size, Rs, old);
    return true;
}

bool TranslatorVisitor::CASP(bool sz, bool, Reg Rs, bool, Reg Rn, Reg Rt) {
    if ((static_cast<size_t>(Rs) & 1) || (static_cast<size_t>(Rt) & 1)) {
        return UnallocatedEncoding();
    }
    const IR::U64 address = AtomicAddress(*this, Rn);
    if (!sz) {
        const IR::U64 expected = ir.Pack2x32To1x64(ir.GetW(Rs), ir.GetW(Rs + 1));
        const IR::U64 desired = ir.Pack2x32To1x64(ir.GetW(Rt), ir.GetW(Rt + 1));
        const IR::U64 old = ir.AtomicCAS(address, expected, desired, 3);
        X(32, Rs, ir.LeastSignificantWord(old));
        X(32, Rs + 1, ir.MostSignificantWord(old).result);
    } else {
        const IR::U128 expected = ir.Pack2x64To1x128(ir.GetX(Rs), ir.GetX(Rs + 1));
        const IR::U128 desired = ir.Pack2x64To1x128(ir.GetX(Rt), ir.GetX(Rt + 1));
        const IR::U128 old = ir.AtomicCAS128(address, expected, desired);
        X(64, Rs, ir.VectorGetElement(64, old, 0));
        X(64, Rs + 1, ir.VectorGetElement(64, old, 1));
    }
    return true;
}

bool TranslatorVisitor::LDAPR(Imm<2> sz, Reg Rn, Reg Rt) {
    const size_t size = sz.ZeroExtend<size_t>();
    const size_t regsize = size == 3 ? 64 : 32;
    const IR::UAny data = Mem(AtomicAddress(*this, Rn), size_t(1) << size, IR::AccType::ORDERED);
    X(regsize, Rt, ZeroExtend(data, regsize));
    return true;
}

bool TranslatorVisitor::STLUR_LDAPUR(Imm<2> sz, Imm<2> opc, Imm<9> imm9, Reg Rn, Reg Rt) {
    const size_t size = sz.ZeroExtend<size_t>();
    const size_t o = opc.ZeroExtend<size_t>();
    if ((size == 3 && o >= 2) || (size == 2 && o == 3)) {
        return UnallocatedEncoding();
    }
    const size_t datasize = 8 << size;
    const IR::U64 address = ir.Add(AtomicAddress(*this, Rn), ir.Imm64(imm9.SignExtend<u64>()));
    if (o == 0) {
        Mem(address, size_t(1) << size, IR::AccType::ORDERED, X(datasize, Rt));
        return true;
    }
    const IR::UAny data = Mem(address, size_t(1) << size, IR::AccType::ORDERED);
    if (o == 1) {
        const size_t regsize = size == 3 ? 64 : 32;
        X(regsize, Rt, ZeroExtend(data, regsize));
    } else {
        const size_t regsize = o == 2 ? 64 : 32;
        X(regsize, Rt, SignExtend(data, regsize));
    }
    return true;
}

}  // namespace Dynarmic::A64
