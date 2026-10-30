/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

bool TranslatorVisitor::BRK(Imm<16> /*imm16*/) {
    return RaiseException(Exception::Breakpoint);
}

bool TranslatorVisitor::SVC(Imm<16> imm16) {
    // x86Code: guest→native thunks are "svc #i; ret". With fuse_svc_ret, peek the next insn; if it is RET X30,
    // skip PushRSB and end in FastDispatchHint: the ret block runs straight from the dispatcher instead of
    // PopRSBHint, which (RSB off under shared translations) returns from Run() after every libsystem call.
    constexpr u32 kRetX30 = 0xD65F03C0u;
    const u64 svc_pc = ir.current_location->PC();
    const u64 next_pc = svc_pc + 4;
    bool fuse_ret = false;
    if (options.fuse_svc_ret && options.memory_read_code) {
        if (const auto next = options.memory_read_code(next_pc)) {
            fuse_ret = (*next == kRetX30);
        }
    }

    if (fuse_ret) {
        ir.SetPC(ir.Imm64(next_pc));
        ir.CallSupervisor(imm16.ZeroExtend());
        ir.SetTerm(IR::Term::CheckHalt{IR::Term::FastDispatchHint{}});
        return false;
    }

    ir.PushRSB(ir.current_location->AdvancePC(4));
    ir.SetPC(ir.Imm64(next_pc));
    ir.CallSupervisor(imm16.ZeroExtend());
    ir.SetTerm(IR::Term::CheckHalt{IR::Term::PopRSBHint{}});
    return false;
}

}  // namespace Dynarmic::A64
