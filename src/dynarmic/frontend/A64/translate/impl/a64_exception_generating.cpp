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
    // x86Code: our guest→native thunks are the 8-byte sequence "svc #i; ret". Without a fuse,
    // every libsystem call pays CallSVC and then a second block for the ret. Peek the next
    // instruction; when it is RET X30, leave PC at svc+4 as a sentinel, CallSVC, then let the
    // x64 emitter (fuse_svc_ret) replace an unchanged PC with X30 and FastDispatch/PopRSB to
    // the caller — skipping the ret block. Handlers that SetPC (redirect) keep their PC.
    constexpr u32 kRetX30 = 0xD65F03C0u;
    const u64 svc_pc = ir.current_location->PC();
    const u64 next_pc = svc_pc + 4;
    bool fuse_ret = false;
    if (options.memory_read_code) {
        if (const auto next = options.memory_read_code(next_pc)) {
            fuse_ret = (*next == kRetX30);
        }
    }

    if (fuse_ret) {
        // Sentinel: EmitA64CallSupervisor compares post-CallSVC PC to this value.
        ir.SetPC(ir.Imm64(next_pc));
        ir.CallSupervisor(imm16.ZeroExtend());
        ir.SetTerm(IR::Term::CheckHalt{IR::Term::PopRSBHint{}});
        return false;
    }

    ir.PushRSB(ir.current_location->AdvancePC(4));
    ir.SetPC(ir.Imm64(next_pc));
    ir.CallSupervisor(imm16.ZeroExtend());
    ir.SetTerm(IR::Term::CheckHalt{IR::Term::PopRSBHint{}});
    return false;
}

}  // namespace Dynarmic::A64
