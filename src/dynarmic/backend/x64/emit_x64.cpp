/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <sys/mman.h>
#include "dynarmic/backend/x64/emit_x64.h"

#include <algorithm>
#include <iterator>

#include <mcl/assert.hpp>
#include <mcl/bit/bit_field.hpp>
#include <mcl/scope_exit.hpp>
#include <mcl/stdint.hpp>
#include <tsl/robin_set.h>

#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/nzcv_util.h"
#include "dynarmic/backend/x64/perf_map.h"
#include "dynarmic/backend/x64/stack_layout.h"
#include "dynarmic/backend/x64/verbose_debugging_output.h"
#include "dynarmic/common/variant_util.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

// TODO: Have ARM flags in host flags and not have them use up GPR registers unless necessary.
// TODO: Actually implement that proper instruction selector you've always wanted to sweetheart.

namespace Dynarmic::Backend::X64 {

using namespace Xbyak::util;

EmitContext::EmitContext(RegAlloc& reg_alloc, IR::Block& block)
        : reg_alloc(reg_alloc), block(block) {}

EmitContext::~EmitContext() = default;

void EmitContext::EraseInstruction(IR::Inst* inst) {
    block.Instructions().erase(inst);
    inst->ClearArgs();
}

EmitX64::EmitX64(BlockOfCode& code)
        : code(code) {
    exception_handler.Register(code);
}

EmitX64::~EmitX64() = default;

EmitX64::LockFreeBlockMap::Table* EmitX64::LockFreeBlockMap::Alloc(size_t n) {
    void* p = mmap(nullptr, 16 + n * sizeof(Entry), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    ASSERT(p != MAP_FAILED);
    Table* t = static_cast<Table*>(p);  // zero pages == empty entries
    t->mask = n - 1;
    return t;
}

EmitX64::LockFreeBlockMap::LockFreeBlockMap() {
    table.store(Alloc(size_t{1} << 14), std::memory_order_release);
}

EmitX64::LockFreeBlockMap::~LockFreeBlockMap() {
    Table* t = table.load();
    munmap(t, 16 + (t->mask + 1) * sizeof(Entry));
}

void EmitX64::LockFreeBlockMap::Insert(Table* t, u64 k, CodePtr c) {
    for (size_t i = Hash(k) & t->mask;; i = (i + 1) & t->mask) {
        const u64 e = t->e[i].key.load(std::memory_order_relaxed);
        if (e == k) {
            t->e[i].code.store(c, std::memory_order_release);
            return;
        }
        if (e == 0) {
            t->e[i].code.store(c, std::memory_order_relaxed);
            t->e[i].key.store(k, std::memory_order_release);
            return;
        }
    }
}

void EmitX64::LockFreeBlockMap::Set(u64 k, CodePtr c) {
    if (!k) return;
    Table* t = table.load(std::memory_order_relaxed);
    if (!c) {
        // Removal: clear the code pointer, keep the key (probe chains of concurrent readers stay intact).
        for (size_t i = Hash(k) & t->mask;; i = (i + 1) & t->mask) {
            const u64 e = t->e[i].key.load(std::memory_order_relaxed);
            if (e == k) t->e[i].code.store(nullptr, std::memory_order_release);
            if (e == k || e == 0) return;
        }
    }
    if ((used + 1) * 2 > t->mask + 1) {
        const size_t n = (t->mask + 1) * 2;
        Table* nt = Alloc(n);
        used = 0;
        for (size_t i = 0; i <= t->mask; i++) {
            const u64 e = t->e[i].key.load(std::memory_order_relaxed);
            CodePtr ec = t->e[i].code.load(std::memory_order_relaxed);
            if (e && ec) {
                Insert(nt, e, ec);
                used++;
            }
        }
        table.store(nt, std::memory_order_release);
        // Readers still in t see zeros (miss) from here on; the mapping is never unmapped.
        madvise(t, 16 + (t->mask + 1) * sizeof(Entry), MADV_DONTNEED);
        t = nt;
    }
    for (size_t i = Hash(k) & t->mask;; i = (i + 1) & t->mask) {
        const u64 e = t->e[i].key.load(std::memory_order_relaxed);
        if (e == k) break;
        if (e == 0) {
            used++;
            break;
        }
    }
    Insert(t, k, c);
}

void EmitX64::LockFreeBlockMap::Clear() {
    Table* t = table.load(std::memory_order_relaxed);
    for (size_t i = 0; i <= t->mask; i++) {
        if (t->e[i].key.load(std::memory_order_relaxed)) t->e[i].code.store(nullptr, std::memory_order_release);
    }
}

std::optional<EmitX64::BlockDescriptor> EmitX64::GetBasicBlock(IR::LocationDescriptor descriptor) {
    const auto iter = block_descriptors.find(descriptor);
    if (iter != block_descriptors.end()) {
        return iter->second;
    }
    if (static_image.active) {
        const size_t i = static_image.Index(descriptor.Value());
        if (i < static_image.n_blocks && !static_image.Dead(i)) {
            const auto& b = static_image.blocks[i];
            const BlockDescriptor bd{static_image.code_begin + b.off, b.size, 1};
            if (!static_image.Live(i)) {
                static_image.SetLive(i, true);
                lf_map.Set(descriptor.Value(), bd.entrypoint);
                static_image.registered++;
            }
            return bd;
        }
    }
    return std::nullopt;
}

size_t EmitX64::StaticImage::Index(u64 desc) const {
    const Block key{desc, 0, 0};
    const Block* it = std::lower_bound(blocks, blocks + n_blocks, key, BlockLess);
    return it != blocks + n_blocks && it->desc == desc ? static_cast<size_t>(it - blocks) : n_blocks;
}

std::pair<const EmitX64::StaticImage::Link*, const EmitX64::StaticImage::Link*> EmitX64::StaticImage::Sites(u64 target) const {
    const Link* lo = std::lower_bound(links, links + n_links, target, [](const Link& l, u64 t) { return l.target < t; });
    const Link* hi = lo;
    while (hi != links + n_links && hi->target == target) {
        hi++;
    }
    return {lo, hi};
}

void EmitX64::StaticImage::Reset() {
    if (map) {
        munmap(map, map_len);
    }
    *this = StaticImage{};
}

void EmitX64::PatchSite(PatchInformation::Kind kind, CodePtr location, const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    code.SetCodePtr(location);
    switch (kind) {
    case PatchInformation::JG:
        EmitPatchJg(target_desc, target_code_ptr);
        break;
    case PatchInformation::JZ:
        EmitPatchJz(target_desc, target_code_ptr);
        break;
    case PatchInformation::JMP:
        EmitPatchJmp(target_desc, target_code_ptr);
        break;
    case PatchInformation::MOV_RCX:
        EmitPatchMovRcx(target_code_ptr);
        break;
    }
}

void EmitX64::PatchStaticSites(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const auto [lo, hi] = static_image.Sites(target_desc.Value());
    for (const auto* l = lo; l != hi; l++) {
        PatchSite(static_cast<PatchInformation::Kind>(l->kind), static_image.code_begin + l->site_off, target_desc, target_code_ptr);
    }
}

void EmitX64::EmitVoid(EmitContext&, IR::Inst*) {
}

void EmitX64::EmitIdentity(EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    if (!args[0].IsImmediate()) {
        ctx.reg_alloc.DefineValue(inst, args[0]);
    }
}

void EmitX64::EmitBreakpoint(EmitContext&, IR::Inst*) {
    code.int3();
}

void EmitX64::EmitCallHostFunction(EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(nullptr, args[1], args[2], args[3]);
    code.mov(rax, args[0].GetImmediateU64());
    code.call(rax);
}

void EmitX64::PushRSBHelper(Xbyak::Reg64 loc_desc_reg, Xbyak::Reg64 index_reg, IR::LocationDescriptor target) {
    using namespace Xbyak::util;

    const auto block = GetBasicBlock(target);
    CodePtr target_code_ptr = block ? block->entrypoint : code.GetReturnFromRunCodeAddress();

    code.mov(index_reg.cvt32(), dword[r15 + code.GetJitStateInfo().offsetof_rsb_ptr]);

    code.mov(loc_desc_reg, target.Value());

    if (atomic_patch_sites) {
        // the imm64 of `mov rcx, imm64` (bytes 2-9) must be 8-byte aligned
        const size_t misalign = (reinterpret_cast<uintptr_t>(code.getCurr()) + 2) & 7;
        if (misalign) {
            code.nop(8 - misalign);
        }
    }
    patch_information.Add(target, PatchInformation::MOV_RCX, code.getCurr());
    EmitPatchMovRcx(target_code_ptr);

    code.mov(qword[r15 + index_reg * 8 + code.GetJitStateInfo().offsetof_rsb_location_descriptors], loc_desc_reg);
    code.mov(qword[r15 + index_reg * 8 + code.GetJitStateInfo().offsetof_rsb_codeptrs], rcx);

    code.add(index_reg.cvt32(), 1);
    code.and_(index_reg.cvt32(), u32(code.GetJitStateInfo().rsb_ptr_mask));
    code.mov(dword[r15 + code.GetJitStateInfo().offsetof_rsb_ptr], index_reg.cvt32());
}

void EmitX64::EmitVerboseDebuggingOutput(RegAlloc& reg_alloc) {
    code.sub(rsp, sizeof(RegisterData));
    code.stmxcsr(dword[rsp + offsetof(RegisterData, mxcsr)]);
    for (int i = 0; i < 16; i++) {
        if (rsp.getIdx() == i) {
            continue;
        }
        code.mov(qword[rsp + offsetof(RegisterData, gprs) + sizeof(u64) * i], Xbyak::Reg64{i});
    }
    for (int i = 0; i < 16; i++) {
        code.movaps(xword[rsp + offsetof(RegisterData, xmms) + 2 * sizeof(u64) * i], Xbyak::Xmm{i});
    }
    code.lea(rax, ptr[rsp + sizeof(RegisterData) + offsetof(StackLayout, spill)]);
    code.mov(xword[rsp + offsetof(RegisterData, spill)], rax);

    reg_alloc.EmitVerboseDebuggingOutput();

    for (int i = 0; i < 16; i++) {
        if (rsp.getIdx() == i) {
            continue;
        }
        code.mov(Xbyak::Reg64{i}, qword[rsp + offsetof(RegisterData, gprs) + sizeof(u64) * i]);
    }
    for (int i = 0; i < 16; i++) {
        code.movaps(Xbyak::Xmm{i}, xword[rsp + offsetof(RegisterData, xmms) + 2 * sizeof(u64) * i]);
    }
    code.ldmxcsr(dword[rsp + offsetof(RegisterData, mxcsr)]);
    code.add(rsp, sizeof(RegisterData));
}

void EmitX64::EmitPushRSB(EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[0].IsImmediate());
    const u64 unique_hash_of_target = args[0].GetImmediateU64();

    ctx.reg_alloc.ScratchGpr(HostLoc::RCX);
    const Xbyak::Reg64 loc_desc_reg = ctx.reg_alloc.ScratchGpr();
    const Xbyak::Reg64 index_reg = ctx.reg_alloc.ScratchGpr();

    PushRSBHelper(loc_desc_reg, index_reg, IR::LocationDescriptor{unique_hash_of_target});
}

void EmitX64::EmitGetCarryFromOp(EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.RegisterPseudoOperation(inst);
}

void EmitX64::EmitGetOverflowFromOp(EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.RegisterPseudoOperation(inst);
}

void EmitX64::EmitGetGEFromOp(EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.RegisterPseudoOperation(inst);
}

void EmitX64::EmitGetUpperFromOp(EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.RegisterPseudoOperation(inst);
}

void EmitX64::EmitGetLowerFromOp(EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.RegisterPseudoOperation(inst);
}

void EmitX64::EmitGetNZFromOp(EmitContext& ctx, IR::Inst* inst) {
    if (ctx.reg_alloc.IsValueLive(inst)) {
        ctx.reg_alloc.RegisterPseudoOperation(inst);
        return;
    }

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    const int bitsize = [&] {
        switch (args[0].GetType()) {
        case IR::Type::U8:
            return 8;
        case IR::Type::U16:
            return 16;
        case IR::Type::U32:
            return 32;
        case IR::Type::U64:
            return 64;
        default:
            UNREACHABLE();
        }
    }();

    const Xbyak::Reg64 nz = ctx.reg_alloc.ScratchGpr(HostLoc::RAX);
    const Xbyak::Reg value = ctx.reg_alloc.UseGpr(args[0]).changeBit(bitsize);
    code.test(value, value);
    code.lahf();
    code.movzx(eax, ah);
    ctx.reg_alloc.DefineValue(inst, nz);
}

void EmitX64::EmitGetNZCVFromOp(EmitContext& ctx, IR::Inst* inst) {
    if (ctx.reg_alloc.IsValueLive(inst)) {
        ctx.reg_alloc.RegisterPseudoOperation(inst);
        return;
    }

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    const int bitsize = [&] {
        switch (args[0].GetType()) {
        case IR::Type::U8:
            return 8;
        case IR::Type::U16:
            return 16;
        case IR::Type::U32:
            return 32;
        case IR::Type::U64:
            return 64;
        default:
            UNREACHABLE();
        }
    }();

    const Xbyak::Reg64 nzcv = ctx.reg_alloc.ScratchGpr(HostLoc::RAX);
    const Xbyak::Reg value = ctx.reg_alloc.UseGpr(args[0]).changeBit(bitsize);
    code.test(value, value);
    code.lahf();
    code.mov(al, 0);
    ctx.reg_alloc.DefineValue(inst, nzcv);
}

void EmitX64::EmitGetCFlagFromNZCV(EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    if (args[0].IsImmediate()) {
        const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
        const u32 value = (args[0].GetImmediateU32() >> 8) & 1;
        code.mov(result, value);
        ctx.reg_alloc.DefineValue(inst, result);
    } else {
        const Xbyak::Reg32 result = ctx.reg_alloc.UseScratchGpr(args[0]).cvt32();
        code.shr(result, 8);
        code.and_(result, 1);
        ctx.reg_alloc.DefineValue(inst, result);
    }
}

void EmitX64::EmitNZCVFromPackedFlags(EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    if (args[0].IsImmediate()) {
        const Xbyak::Reg32 nzcv = ctx.reg_alloc.ScratchGpr().cvt32();
        u32 value = 0;
        value |= mcl::bit::get_bit<31>(args[0].GetImmediateU32()) ? (1 << 15) : 0;
        value |= mcl::bit::get_bit<30>(args[0].GetImmediateU32()) ? (1 << 14) : 0;
        value |= mcl::bit::get_bit<29>(args[0].GetImmediateU32()) ? (1 << 8) : 0;
        value |= mcl::bit::get_bit<28>(args[0].GetImmediateU32()) ? (1 << 0) : 0;
        code.mov(nzcv, value);
        ctx.reg_alloc.DefineValue(inst, nzcv);
    } else if (code.HasHostFeature(HostFeature::FastBMI2)) {
        const Xbyak::Reg32 nzcv = ctx.reg_alloc.UseScratchGpr(args[0]).cvt32();
        const Xbyak::Reg32 tmp = ctx.reg_alloc.ScratchGpr().cvt32();

        code.shr(nzcv, 28);
        code.mov(tmp, NZCV::x64_mask);
        code.pdep(nzcv, nzcv, tmp);

        ctx.reg_alloc.DefineValue(inst, nzcv);
    } else {
        const Xbyak::Reg32 nzcv = ctx.reg_alloc.UseScratchGpr(args[0]).cvt32();

        code.shr(nzcv, 28);
        code.imul(nzcv, nzcv, NZCV::to_x64_multiplier);
        code.and_(nzcv, NZCV::x64_mask);

        ctx.reg_alloc.DefineValue(inst, nzcv);
    }
}

void EmitX64::EmitAddCycles(size_t cycles) {
    ASSERT(cycles < std::numeric_limits<s32>::max());
    code.sub(qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], static_cast<u32>(cycles));
}

Xbyak::Label EmitX64::EmitCond(IR::Cond cond) {
    Xbyak::Label pass;

    code.mov(eax, dword[r15 + code.GetJitStateInfo().offsetof_cpsr_nzcv]);

    code.LoadRequiredFlagsForCondFromRax(cond);

    switch (cond) {
    case IR::Cond::EQ:
        code.jz(pass);
        break;
    case IR::Cond::NE:
        code.jnz(pass);
        break;
    case IR::Cond::CS:
        code.jc(pass);
        break;
    case IR::Cond::CC:
        code.jnc(pass);
        break;
    case IR::Cond::MI:
        code.js(pass);
        break;
    case IR::Cond::PL:
        code.jns(pass);
        break;
    case IR::Cond::VS:
        code.jo(pass);
        break;
    case IR::Cond::VC:
        code.jno(pass);
        break;
    case IR::Cond::HI:
        code.ja(pass);
        break;
    case IR::Cond::LS:
        code.jna(pass);
        break;
    case IR::Cond::GE:
        code.jge(pass);
        break;
    case IR::Cond::LT:
        code.jl(pass);
        break;
    case IR::Cond::GT:
        code.jg(pass);
        break;
    case IR::Cond::LE:
        code.jle(pass);
        break;
    default:
        ASSERT_MSG(false, "Unknown cond {}", static_cast<size_t>(cond));
        break;
    }

    return pass;
}

EmitX64::BlockDescriptor EmitX64::RegisterBlock(const IR::LocationDescriptor& descriptor, CodePtr entrypoint, size_t size) {
    PerfMapRegister(entrypoint, code.getCurr(), LocationDescriptorToFriendlyName(descriptor));
    Patch(descriptor, entrypoint);

    BlockDescriptor block_desc{entrypoint, static_cast<u32>(size)};
    block_descriptors.insert({IR::LocationDescriptor{descriptor.Value()}, block_desc});
    lf_map.Set(descriptor.Value(), entrypoint);
    return block_desc;
}

void EmitX64::EmitTerminal(IR::Terminal terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    Common::VisitVariant<void>(terminal, [this, initial_location, is_single_step](auto x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (!std::is_same_v<T, IR::Term::Invalid>) {
            this->EmitTerminalImpl(x, initial_location, is_single_step);
        } else {
            ASSERT_MSG(false, "Invalid terminal");
        }
    });
}

void EmitX64::Patch(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const CodePtr save_code_ptr = code.getCurr();
    patching = true;

    patch_information.ForEachSite(target_desc, [&](PatchInformation::Kind kind, CodePtr location) {
        PatchSite(kind, location, target_desc, target_code_ptr);
    });
    if (static_image.active) {
        PatchStaticSites(target_desc, target_code_ptr);
    }

    patching = false;
    code.SetCodePtr(save_code_ptr);
}

void EmitX64::Unpatch(const IR::LocationDescriptor& target_desc) {
    if (patch_information.count(target_desc) || (static_image.active && static_image.Sites(target_desc.Value()).first != static_image.Sites(target_desc.Value()).second)) {
        Patch(target_desc, nullptr);
    }
}

void EmitX64::ClearCache() {
    block_descriptors.clear();
    lf_map.Clear();
    patch_information.clear();
    static_image.Reset();

    PerfMapClear();
}

void EmitX64::InvalidateBasicBlocks(const tsl::robin_set<IR::LocationDescriptor>& locations) {
    code.EnableWriting();
    SCOPE_EXIT {
        code.DisableWriting();
    };

    for (const auto& descriptor : locations) {
        const auto it = block_descriptors.find(descriptor);
        bool known = it != block_descriptors.end();
        if (static_image.active) {
            const size_t i = static_image.Index(descriptor.Value());
            if (i < static_image.n_blocks && !static_image.Dead(i)) {
                static_image.Kill(i);
                if (static_image.Live(i)) {
                    static_image.SetLive(i, false);
                    lf_map.Set(descriptor.Value(), nullptr);
                }
                known = true;
            }
        }
        if (!known) {
            continue;
        }

        Unpatch(descriptor);

        if (it != block_descriptors.end()) {
            lf_map.Set(descriptor.Value(), nullptr);
            block_descriptors.erase(it);
        }
    }
}

}  // namespace Dynarmic::Backend::X64
