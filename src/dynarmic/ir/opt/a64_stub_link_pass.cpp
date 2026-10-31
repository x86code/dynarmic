/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */

#include <boost/variant/get.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/frontend/A64/a64_ir_emitter.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/ir/opt/passes.h"

namespace Dynarmic::Optimization {

// x86Code: cross-library import stubs (adrp x16, got@PAGE; ldr x16, [x16, got@PAGEOFF]; br x16) end in an indirect
// jump, so every call through one is a FastDispatch lookup. After constant propagation the target is a 64-bit load
// from a constant address. When the embedder can read that slot now (UserCallbacks::MemoryReadConstant64: the
// shared cache's bound GOTs), the block keeps the load and the jump to it, and links directly to the value it read,
// guarded: check bit = (loaded == expected), CheckBit{LinkBlock{expected}, FastDispatchHint}. A slot that changes
// later (interposing, rebinding) takes the old path; nothing is assumed.
void A64StubLinkPass(IR::Block& block, A64::UserCallbacks* cb) {
    const IR::Terminal term = block.GetTerminal();
    if (!boost::get<IR::Term::FastDispatchHint>(&term))
        return;

    IR::Inst* set_pc = nullptr;
    for (auto& inst : block) {
        if (inst.GetOpcode() == IR::Opcode::A64SetPC)
            set_pc = &inst;
        else if (inst.GetOpcode() == IR::Opcode::A64SetCheckBit)
            return;
    }
    if (!set_pc || set_pc->GetArg(0).IsImmediate())
        return;
    IR::Inst* load = set_pc->GetArg(0).GetInst();
    if (load->GetOpcode() != IR::Opcode::A64ReadMemory64 || !load->GetArg(1).IsImmediate())
        return;

    const auto target = cb->MemoryReadConstant64(load->GetArg(1).GetU64());
    if (!target || (*target & 3) != 0)
        return;

    A64::IREmitter ir{block};
    ir.SetInsertionPointAfter(set_pc);
    const IR::U64 loaded{IR::Value{load}};
    ir.SetCheckBit(ir.IsZero(IR::U64{ir.Sub(loaded, ir.Imm64(*target))}));

    const auto here = A64::LocationDescriptor{block.Location()};
    block.ReplaceTerminal(IR::Term::CheckBit{IR::Term::LinkBlock{here.SetPC(*target)}, IR::Term::FastDispatchHint{}});
}

}  // namespace Dynarmic::Optimization
