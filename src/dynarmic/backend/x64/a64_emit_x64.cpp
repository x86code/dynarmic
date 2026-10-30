/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/x64/a64_emit_x64.h"
#include <immintrin.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <mcl/assert.hpp>
#include <mcl/scope_exit.hpp>
#include <mcl/stdint.hpp>
#include <mcl/type_traits/integer_of_size.hpp>

#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/abi.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/emit_x64.h"
#include "dynarmic/backend/x64/nzcv_util.h"
#include "dynarmic/backend/x64/perf_map.h"
#include "dynarmic/backend/x64/stack_layout.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/a64_types.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/cond.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

// TODO: Have ARM flags in host flags and not have them use up GPR registers unless necessary.
// TODO: Actually implement that proper instruction selector you've always wanted to sweetheart.

namespace Dynarmic::Backend::X64 {

using namespace Xbyak::util;

A64EmitContext::A64EmitContext(const A64::UserConfig& conf, RegAlloc& reg_alloc, IR::Block& block)
        : EmitContext(reg_alloc, block), conf(conf) {
    // Any HostCall in this block marks it non-persistable (absolute host fn/this embeds).
    reg_alloc.SetHostCallFlag(&has_host_call);
}

A64::LocationDescriptor A64EmitContext::Location() const {
    return A64::LocationDescriptor{block.Location()};
}

bool A64EmitContext::IsSingleStep() const {
    return Location().SingleStepping();
}

FP::FPCR A64EmitContext::FPCR(bool fpcr_controlled) const {
    return fpcr_controlled ? Location().FPCR() : Location().FPCR().ASIMDStandardValue();
}

A64EmitX64::A64EmitX64(BlockOfCode& code, A64::UserConfig conf, A64::Jit* jit_interface)
        : EmitX64(code), conf(conf), jit_interface{jit_interface} {
    atomic_patch_sites = conf.shared_translations;
    GenMemory128Accessors();
    GenFastmemFallbacks();
    GenTerminalHandlers();
    code.PreludeComplete();
    ClearFastDispatchTable();

    exception_handler.SetFastmemCallback([this](u64 rip_) {
        std::unique_lock lock{shared_mutex, std::defer_lock};
        if (this->conf.shared_translations) {
            lock.lock();
        }
        return FastmemCallback(rip_);
    });
}

A64EmitX64::~A64EmitX64() = default;

A64EmitX64::BlockDescriptor A64EmitX64::Emit(IR::Block& block) {
    if (conf.very_verbose_debugging_output) {
        std::puts(IR::DumpBlock(block).c_str());
    }

    code.EnableWriting();
    SCOPE_EXIT {
        code.DisableWriting();
    };

    const std::vector<HostLoc> gpr_order = [this] {
        std::vector<HostLoc> gprs{any_gpr};
        if (conf.page_table) {
            gprs.erase(std::find(gprs.begin(), gprs.end(), HostLoc::R14));
        }
        if (conf.fastmem_pointer) {
            gprs.erase(std::find(gprs.begin(), gprs.end(), HostLoc::R13));
        }
        return gprs;
    }();

    RegAlloc reg_alloc{code, gpr_order, any_xmm};
    A64EmitContext ctx{conf, reg_alloc, block};
    code.SetCallFlag(&ctx.has_host_call);
    SCOPE_EXIT {
        code.SetCallFlag(nullptr);
    };

    // Start emitting.
    code.align();
    const u8* const entrypoint = code.getCurr();

    ASSERT(block.GetCondition() == IR::Cond::AL);

    for (auto iter = block.begin(); iter != block.end(); ++iter) {
        IR::Inst* inst = &*iter;

        // Call the relevant Emit* member function.
        switch (inst->GetOpcode()) {
#define OPCODE(name, type, ...)            \
    case IR::Opcode::name:                 \
        A64EmitX64::Emit##name(ctx, inst); \
        break;
#define A32OPC(...)
#define A64OPC(name, type, ...)               \
    case IR::Opcode::A64##name:               \
        A64EmitX64::EmitA64##name(ctx, inst); \
        break;
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC

        default:
            ASSERT_MSG(false, "Invalid opcode: {}", inst->GetOpcode());
            break;
        }

        ctx.reg_alloc.EndOfAllocScope();

        if (conf.very_verbose_debugging_output) {
            EmitVerboseDebuggingOutput(reg_alloc);
        }
    }

    reg_alloc.AssertNoMoreUses();

    if (conf.enable_cycle_counting) {
        EmitAddCycles(block.CycleCount());
    }
    EmitX64::EmitTerminal(block.GetTerminal(), ctx.Location().SetSingleStepping(false), ctx.IsSingleStep());
    code.int3();

    for (auto& deferred_emit : ctx.deferred_emits) {
        deferred_emit();
    }
    code.int3();

    const size_t size = static_cast<size_t>(code.getCurr() - entrypoint);

    const A64::LocationDescriptor descriptor{block.Location()};
    const A64::LocationDescriptor end_location{block.EndLocation()};

    const auto range = boost::icl::discrete_interval<u64>::closed(descriptor.PC(), end_location.PC() - 1);
    block_ranges.AddRange(range, descriptor);

    auto bd = RegisterBlock(descriptor, entrypoint, size);
    // Track persistable blocks (no host calls; PC filter applied at save time).
    if (!ctx.has_host_call && !ctx.IsSingleStep()) {
        persistable_blocks.push_back({descriptor.UniqueHash(), entrypoint, size});
    }
    return bd;
}

// x86Code: hand-written objc_msgSend cache probe matching iOS 27.0 sim libobjcMsgSend
// (isa mask 0x7ffffffffffff8, cache at class+0x10, HIGH_16_BIG_ADDRS, bucket {imp^class, sel},
// post-index ldp #-16). One x86 block; hit → FastDispatch to IMP; miss → uncached; nil → LR.
A64EmitX64::BlockDescriptor A64EmitX64::EmitInlineObjCMsgSend(IR::LocationDescriptor location) {
    ASSERT(conf.inline_objc_msgsend != 0);
    ASSERT(conf.inline_objc_uncached != 0);

    code.EnableWriting();
    SCOPE_EXIT {
        code.DisableWriting();
    };

    code.align();
    const u8* const entrypoint = code.getCurr();

    constexpr u64 kIsaMask = 0x7ffffffffffff8ull;
    constexpr u64 kBucketsMask = 0xffffffffffffull;
    constexpr u64 kTbi = 0x00ffffffffffffffull;
    constexpr u64 kCacheOffset = 0x10;

    const auto reg_off = [](size_t i) {
        return offsetof(A64JitState, reg) + sizeof(u64) * i;
    };
    const auto vec_off = [](size_t i) {
        // Each V register is 2×u64 in A64JitState::vec.
        return offsetof(A64JitState, vec) + sizeof(u64) * 2 * i;
    };

    Xbyak::Label nil_path, tagged_path, miss_path, hit_path, wrap_setup, wrap_loop, probe_loop;
    constexpr auto Near = Xbyak::CodeGenerator::T_NEAR;

    // rax = obj (x0), rcx = sel (x1)
    code.mov(rax, qword[r15 + reg_off(0)]);
    code.mov(rcx, qword[r15 + reg_off(1)]);

    code.test(rax, rax);
    code.jz(nil_path, Near);
    code.test(rax, rax);
    code.js(tagged_path, Near);  // MSB set ⇒ tagged

    // cls = *obj & isa_mask (TBI on the object pointer). Masks >32-bit need a register.
    code.mov(rdx, rax);
    code.mov(rbx, kTbi);
    code.and_(rdx, rbx);
    code.mov(rdx, qword[rdx]);  // raw isa
    code.mov(rbx, kIsaMask);
    code.and_(rdx, rbx);        // class
    code.mov(qword[r15 + reg_off(15)], rdx);  // x15 = class
    code.mov(qword[r15 + reg_off(16)], rdx);  // x16 = class

    // cache = *(class+0x10); buckets = cache & mask48; mask = cache >> 48
    code.mov(r8, qword[rdx + kCacheOffset]);
    code.mov(r9, r8);
    code.mov(rbx, kBucketsMask);
    code.and_(r9, rbx);  // buckets
    code.shr(r8, 48);    // mask in r8d

    // idx = sel & mask; cur = buckets + idx*16; first = cur
    code.mov(r10d, ecx);
    code.and_(r10d, r8d);
    code.shl(r10, 4);
    code.add(r10, r9);  // cur
    code.mov(r11, r10); // first

    code.L(probe_loop);
    code.mov(rbx, qword[r10 + 8]);  // bsel first: see the LDP note in load_store_register_pair.cpp
    code.mov(rax, qword[r10]);      // stored_imp
    code.sub(r10, 16);
    code.cmp(rbx, rcx);
    code.je(hit_path, Near);
    code.test(rbx, rbx);
    code.jz(miss_path, Near);
    code.cmp(r10, r9);
    code.jae(probe_loop);

    // Wrap to last bucket
    code.L(wrap_setup);
    code.mov(r10, r8);
    code.shl(r10, 4);
    code.add(r10, r9);

    code.L(wrap_loop);
    code.mov(rbx, qword[r10 + 8]);  // bsel first: see the LDP note in load_store_register_pair.cpp
    code.mov(rax, qword[r10]);      // stored_imp
    code.sub(r10, 16);
    code.cmp(rbx, rcx);
    code.je(hit_path, Near);
    code.test(rbx, rbx);
    code.jz(miss_path, Near);
    code.cmp(r10, r11);
    code.ja(wrap_loop);
    code.jmp(miss_path, Near);

    code.L(hit_path);
    code.xor_(rax, rdx);  // imp ^= class
    // Optional counters (conf.inline_objc_hit_counter non-null).
    if (conf.inline_objc_hit_counter) {
        EmitMovAbs64(rbx, reinterpret_cast<u64>(conf.inline_objc_hit_counter), RelocKind::ObjcHitCounter);
        code.lock();
        code.inc(qword[rbx]);
    }
    code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
    code.mov(code.byte[r15 + offsetof(A64JitState, exclusive_state)], u8(0));
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.jmp(terminal_handler_fast_dispatch_hint);
    } else {
        code.ReturnFromRunCode();
    }

    code.L(miss_path);
    if (conf.inline_objc_miss_counter) {
        EmitMovAbs64(rbx, reinterpret_cast<u64>(conf.inline_objc_miss_counter), RelocKind::ObjcMissCounter);
        code.lock();
        code.inc(qword[rbx]);
    }
    // x15/x16 already set; go to __objc_msgSend_uncached
    code.mov(rax, conf.inline_objc_uncached);
    code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
    code.mov(code.byte[r15 + offsetof(A64JitState, exclusive_state)], u8(0));
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.jmp(terminal_handler_fast_dispatch_hint);
    } else {
        code.ReturnFromRunCode();
    }

    code.L(tagged_path);
    if (conf.inline_objc_miss_counter) {
        EmitMovAbs64(rbx, reinterpret_cast<u64>(conf.inline_objc_miss_counter), RelocKind::ObjcMissCounter);
        code.lock();
        code.inc(qword[rbx]);
    }
    code.mov(rax, conf.inline_objc_tagged ? conf.inline_objc_tagged : conf.inline_objc_uncached);
    code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
    code.mov(code.byte[r15 + offsetof(A64JitState, exclusive_state)], u8(0));
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.jmp(terminal_handler_fast_dispatch_hint);
    } else {
        code.ReturnFromRunCode();
    }

    code.L(nil_path);
    // Match Apple nil msgSend: x0=0, x1=0, d0–d3=0, return to LR
    code.mov(qword[r15 + reg_off(0)], u64(0));
    code.mov(qword[r15 + reg_off(1)], u64(0));
    for (size_t i = 0; i < 4; i++) {
        code.mov(qword[r15 + vec_off(i)], u64(0));
        code.mov(qword[r15 + vec_off(i) + 8], u64(0));
    }
    code.mov(rax, qword[r15 + reg_off(30)]);
    code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
    code.mov(code.byte[r15 + offsetof(A64JitState, exclusive_state)], u8(0));
    code.jmp(terminal_handler_pop_rsb_hint);

    code.int3();

    const size_t size = static_cast<size_t>(code.getCurr() - entrypoint);
    const A64::LocationDescriptor descriptor{location};
    // Cover a small window so IC IVAU near the entry still finds us.
    const auto range = boost::icl::discrete_interval<u64>::closed(descriptor.PC(), descriptor.PC() + 7);
    block_ranges.AddRange(range, descriptor);
    PerfMapRegister(entrypoint, code.getCurr(), "a64_inline_objc_msgSend");
    auto bd = RegisterBlock(descriptor, entrypoint, size);
    persistable_blocks.push_back({descriptor.UniqueHash(), entrypoint, size});
    return bd;
}

namespace {

// A64JC04: reloc table + fastmem sites with callback rebound by fallback key (not absolute prelude addr).
// A64JC05: also persist the x86 constant pool (RIP targets from movi/FP immediates).
// A64JC06: block link sites (target, site offset, kind) so loaded blocks are linked, not dispatched (Speed 5).
constexpr char kPersistMagic[8] = {'A', '6', '4', 'J', 'C', '0', '6', '\0'};

struct PersistHeader {
    char magic[8];
    u8 uuid[16];
    u64 build_id;
    u64 code_size;
    u64 n_blocks;
    u64 prelude_size;  // GetCodeBegin() - allocation base; must match on load
    u64 n_relocs;
    u64 n_fastmem;
    u64 old_bases[static_cast<size_t>(A64EmitX64::RelocKind::Count)];
    u64 const_pool_used;  // bytes of ConstantPool to restore after the code slab
    u64 n_links;          // LinkPersist records after the fastmem records
};

struct LinkPersist {
    u64 target;  // LocationDescriptor of the linked-to block
    u32 site_off;
    u8 kind;     // PatchInformation::Kind (JG/JZ/JMP; MOV_RCX sites are not persisted)
    u8 pad[3]{};
};

// fault/resume are code_begin-relative; callback is rebound on load via the live fallback table
// (prelude layout can differ across ASLR near/far call choices even when prelude_size matches).
struct FastmemPersist {
    s64 fault_off;
    s64 resume_off;
    u8 kind;       // 0=read 1=write 2=exclusive_write
    u8 ordered;    // 0/1
    u16 bitsize;   // 8/16/32/64/128
    u8 vaddr_idx;
    u8 value_idx;
    u8 pad[2]{};
};

}  // namespace

void A64EmitX64::EmitMovAbs64(const Xbyak::Reg64& reg, u64 value, RelocKind kind) {
    const u8* begin = static_cast<const u8*>(code.GetCodeBegin());
    const u8* at = code.getCurr<const u8*>();
    // mov r64, imm64 is 10 bytes: REX.W + B8+r + imm64. Imm starts at +2.
    // Skip recording while patching an existing site (Unpatch/Patch); the original emit already has an entry.
    if (!patching && begin && at >= begin) {
        const u32 imm_off = static_cast<u32>(at - begin) + 2;
        reloc_entries.push_back({imm_off, kind});
    }
    code.mov(reg, value);
}

size_t A64EmitX64::SavePersistentCache(const std::string& path, const u8 cache_uuid[16], u64 build_id,
                                       u64 cache_lo, u64 cache_hi) {
    // Speed 5: snapshot under the shared lock, write the file after it (periodic saves from a live process must
    // not stall guest threads for the file write nor unlink their code).
    std::unique_lock lock{shared_mutex};
    // Every link site goes to the file in its unlinked form. Linked jumps may target HostCall blocks (excluded from
    // the descriptor list but still present in the slab); those bake ASLR-sensitive absolute addresses and crash on
    // reload. With atomic patch sites (shared translations) the unlinked heads are written into the copy below and
    // the live code stays linked; otherwise every block is unlinked in place first (old behaviour).
    if (!atomic_patch_sites) {
        code.EnableWriting();
        for (const auto& [desc, _] : block_descriptors) {
            Unpatch(desc);
        }
        code.DisableWriting();
    }

    const u8* begin = static_cast<const u8*>(code.GetCodeBegin());
    const u8* end = code.getCurr<const u8*>();
    if (!begin || end < begin) {
        return 0;
    }
    const u64 code_size = static_cast<u64>(end - begin);

    const u64 begin_u = reinterpret_cast<u64>(begin);
    const u64 end_u = begin_u + code_size;
    const u64 alloc_u = reinterpret_cast<u64>(code.getCode());

    // Persist every non-HostCall block in the dyld-cache range, including those with fastmem
    // sites. HostCall blocks stay out of the descriptor list (absolute CallFunction embeds) but
    // their bytes remain in the slab — scrub those gaps to int3 below so a stray rel32 cannot
    // execute stale absolute Imm64s (warm MAPERR root cause).
    std::vector<PersistEntry> entries;
    entries.reserve(persistable_blocks.size());
    std::vector<bool> in_persisted(code_size, false);
    for (const auto& e : persistable_blocks) {
        const A64::LocationDescriptor desc{IR::LocationDescriptor{e.location_descriptor}};
        const u64 pc = desc.PC();
        if (pc < cache_lo || pc >= cache_hi) {
            continue;
        }
        const u8* ep = static_cast<const u8*>(e.entrypoint);
        if (ep < begin || ep >= end) {
            continue;
        }
        const u64 eoff = reinterpret_cast<u64>(ep) - begin_u;
        if (eoff >= code_size) {
            continue;
        }
        const u64 eend = std::min(eoff + static_cast<u64>(e.size), code_size);
        for (u64 i = eoff; i < eend; i++) {
            in_persisted[i] = true;
        }
        entries.push_back(e);
    }
    if (entries.empty()) {
        return 0;
    }

    // Drop reloc entries whose Imm64 sits outside the saved slab (prelude / truncated).
    std::vector<RelocEntry> relocs;
    relocs.reserve(reloc_entries.size());
    for (const auto& r : reloc_entries) {
        if (static_cast<u64>(r.offset) + 8 <= code_size) {
            relocs.push_back(r);
        }
    }

    // Fastmem fault sites that sit inside a persisted block. Rebind callbacks by fallback-table
    // key on load (not absolute prelude addr). Sites in HostCall gaps are dropped with the scrub.
    std::unordered_map<u64, FastmemPersist> cb_to_key;
    auto index_fb = [&](u8 kind, const auto& map) {
        for (const auto& [tup, fn] : map) {
            FastmemPersist k{};
            k.kind = kind;
            k.ordered = static_cast<u8>(std::get<0>(tup));
            k.bitsize = static_cast<u16>(std::get<1>(tup));
            k.vaddr_idx = static_cast<u8>(std::get<2>(tup));
            k.value_idx = static_cast<u8>(std::get<3>(tup));
            cb_to_key.emplace(reinterpret_cast<u64>(fn), k);
        }
    };
    index_fb(0, read_fallbacks);
    index_fb(1, write_fallbacks);
    index_fb(2, exclusive_write_fallbacks);
    std::vector<FastmemPersist> fm_entries;
    fm_entries.reserve(fastmem_patch_info.size());
    for (const auto& [fault_rip, info] : fastmem_patch_info) {
        if (fault_rip < begin_u || fault_rip >= end_u) {
            continue;
        }
        const u64 fault_off = fault_rip - begin_u;
        if (!in_persisted[fault_off]) {
            continue;
        }
        if (info.resume_rip < begin_u || info.resume_rip > end_u + 64) {
            continue;
        }
        const auto it = cb_to_key.find(info.callback);
        if (it == cb_to_key.end()) {
            continue;  // unknown trampoline (e.g. exclusive-read path); skip site
        }
        FastmemPersist e = it->second;
        e.fault_off = static_cast<s64>(fault_off);
        e.resume_off = static_cast<s64>(info.resume_rip - begin_u);
        fm_entries.push_back(e);
    }

    // Scrub HostCall / unlisted gaps so a mid-block rel32 into them cannot run ASLR-stale code.
    // Deferred abort stubs live inside block.size (Emit runs deferred_emits before sizing).
    std::vector<u8> slab(code_size, 0xCC);
    for (const auto& e : entries) {
        const u64 off = static_cast<u64>(static_cast<const u8*>(e.entrypoint) - begin);
        const u64 sz = std::min(static_cast<u64>(e.size), code_size - off);
        std::memcpy(slab.data() + off, begin + off, static_cast<size_t>(sz));
    }
    if (atomic_patch_sites) {
        for (const auto& [site_addr, head] : unlinked_heads) {
            if (site_addr >= begin_u && site_addr + 8 <= end_u && in_persisted[site_addr - begin_u]) {
                std::memcpy(slab.data() + (site_addr - begin_u), &head, 8);
            }
        }
    }
    std::vector<u8> pool(static_cast<size_t>(code.GetConstantPoolUsedBytes()));
    if (!pool.empty() && pool.size() <= code.GetConstantPoolTotalBytes()) {
        std::memcpy(pool.data(), code.GetConstantPoolData(), pool.size());
    }

    std::vector<LinkPersist> links;
    for (const auto& [target, info] : patch_information) {
        for (const auto& [kind, site] : info.sites) {
            const u64 sa = reinterpret_cast<u64>(site);
            if (kind == PatchInformation::MOV_RCX || sa < begin_u || sa + 8 > end_u || !in_persisted[sa - begin_u]) {
                continue;
            }
            LinkPersist l{};
            l.target = target.Value();
            l.site_off = static_cast<u32>(sa - begin_u);
            l.kind = static_cast<u8>(kind);
            links.push_back(l);
        }
    }

    // header bases (live addresses) are taken under the lock too
    PersistHeader hdr{};
    hdr.n_links = links.size();
    std::memcpy(hdr.magic, kPersistMagic, 8);
    std::memcpy(hdr.uuid, cache_uuid, 16);
    hdr.build_id = build_id;
    hdr.code_size = code_size;
    hdr.n_blocks = entries.size();
    hdr.prelude_size = begin_u - alloc_u;
    hdr.n_relocs = relocs.size();
    hdr.n_fastmem = fm_entries.size();
    hdr.old_bases[static_cast<size_t>(RelocKind::CodeBegin)] = begin_u;
    hdr.old_bases[static_cast<size_t>(RelocKind::ReturnFromRun)] =
        reinterpret_cast<u64>(code.GetReturnFromRunCodeAddress());
    hdr.old_bases[static_cast<size_t>(RelocKind::ForceReturn)] =
        reinterpret_cast<u64>(code.GetForceReturnFromRunCodeAddress());
    hdr.old_bases[static_cast<size_t>(RelocKind::TermFastDispatch)] =
        reinterpret_cast<u64>(terminal_handler_fast_dispatch_hint);
    hdr.old_bases[static_cast<size_t>(RelocKind::TermPopRSB)] =
        reinterpret_cast<u64>(terminal_handler_pop_rsb_hint);
    hdr.old_bases[static_cast<size_t>(RelocKind::FastDispatchTable)] =
        reinterpret_cast<u64>(fast_dispatch_table.data());
    hdr.old_bases[static_cast<size_t>(RelocKind::ObjcHitCounter)] =
        reinterpret_cast<u64>(conf.inline_objc_hit_counter);
    hdr.old_bases[static_cast<size_t>(RelocKind::ObjcMissCounter)] =
        reinterpret_cast<u64>(conf.inline_objc_miss_counter);
    hdr.old_bases[static_cast<size_t>(RelocKind::CallbackThis)] =
        reinterpret_cast<u64>(conf.callbacks);
    // Emitted blocks RIP-load FP/vector immediates from BlockOfCode's ConstantPool (prelude).
    // A warm process only populates pool slots touched by GenRunCode; Emit slots stay zero and
    // movi-derived stores (e.g. dyld hash-table mask=1) write 0 → classic warm MAPERR.
    hdr.const_pool_used = pool.size();
    lock.unlock();

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        return 0;
    }
    if (std::fwrite(&hdr, sizeof hdr, 1, f) != 1) {
        std::fclose(f);
        return 0;
    }
    for (const auto& e : entries) {
        const u64 off = static_cast<u64>(static_cast<const u8*>(e.entrypoint) - begin);
        const u64 sz = e.size;
        const u64 desc = e.location_descriptor;
        if (std::fwrite(&desc, 8, 1, f) != 1 || std::fwrite(&off, 8, 1, f) != 1 || std::fwrite(&sz, 8, 1, f) != 1) {
            std::fclose(f);
            return 0;
        }
    }
    for (const auto& r : relocs) {
        const u32 off = r.offset;
        const u8 kind = static_cast<u8>(r.kind);
        if (std::fwrite(&off, 4, 1, f) != 1 || std::fwrite(&kind, 1, 1, f) != 1 ||
            std::fwrite("\0\0\0", 1, 3, f) != 3) {
            std::fclose(f);
            return 0;
        }
    }
    for (const auto& fm : fm_entries) {
        if (std::fwrite(&fm, sizeof fm, 1, f) != 1) {
            std::fclose(f);
            return 0;
        }
    }
    if (!links.empty() && std::fwrite(links.data(), sizeof(LinkPersist), links.size(), f) != links.size()) {
        std::fclose(f);
        return 0;
    }
    if (std::fwrite(slab.data(), 1, code_size, f) != code_size) {
        std::fclose(f);
        return 0;
    }
    if (hdr.const_pool_used) {
        if (std::fwrite(pool.data(), 1, pool.size(), f) != pool.size()) {
            std::fclose(f);
            return 0;
        }
    }
    std::fclose(f);
    if (std::getenv("X86CODE_A64_JITCACHE_DEBUG")) {
        size_t gap = 0;
        for (bool p : in_persisted) {
            if (!p) gap++;
        }
        std::fprintf(stderr,
                     "a64jitcache: save blocks=%zu fm=%zu relocs=%zu scrubbed_gap=%zu/%llu const_pool=%llu\n",
                     entries.size(), fm_entries.size(), relocs.size(), gap,
                     static_cast<unsigned long long>(code_size),
                     static_cast<unsigned long long>(hdr.const_pool_used));
    }
    return static_cast<size_t>(sizeof(PersistHeader) + entries.size() * 24 + relocs.size() * 8 +
                               fm_entries.size() * sizeof(FastmemPersist) + links.size() * sizeof(LinkPersist) + code_size +
                               hdr.const_pool_used);
}

std::vector<u64> A64EmitX64::ReadPersistentDescriptors(const std::string& path, const u8 cache_uuid[16], u64 build_id) {
    std::vector<u64> out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return out;
    }
    PersistHeader hdr{};
    // Hot sets outlive the format: A64JC05 files have the same layout up to n_links (8 bytes shorter header).
    const bool ok = std::fread(&hdr, sizeof hdr, 1, f) == 1;
    const bool v5 = ok && std::memcmp(hdr.magic, "A64JC05", 8) == 0;
    if (v5) {
        std::fseek(f, static_cast<long>(sizeof hdr - sizeof hdr.n_links), SEEK_SET);
    }
    if (ok && (v5 || std::memcmp(hdr.magic, kPersistMagic, 8) == 0) &&
        (build_id == 0 || hdr.build_id == build_id) && std::memcmp(hdr.uuid, cache_uuid, 16) == 0) {
        out.reserve(hdr.n_blocks);
        for (u64 i = 0; i < hdr.n_blocks; i++) {
            u64 e[3];
            if (std::fread(e, 8, 3, f) != 3) {
                out.clear();
                break;
            }
            out.push_back(e[0]);
        }
    }
    std::fclose(f);
    return out;
}

size_t A64EmitX64::LoadPersistentCache(const std::string& path, const u8 cache_uuid[16], u64 build_id,
                                       u64 cache_lo, u64 cache_hi) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return 0;
    }
    PersistHeader hdr{};
    if (std::fread(&hdr, sizeof hdr, 1, f) != 1) {
        std::fclose(f);
        return 0;
    }
    if (std::memcmp(hdr.magic, kPersistMagic, 8) != 0 || hdr.build_id != build_id ||
        std::memcmp(hdr.uuid, cache_uuid, 16) != 0) {
        std::fclose(f);
        return 0;
    }
    u8* begin = const_cast<u8*>(static_cast<const u8*>(code.GetCodeBegin()));
    const u64 prelude_size = reinterpret_cast<u64>(begin) - reinterpret_cast<u64>(code.getCode());
    if (hdr.prelude_size != prelude_size) {
        std::fprintf(stderr,
                     "a64jitcache: prelude_size mismatch saved=%llu live=%llu (near/far call layout?)\n",
                     static_cast<unsigned long long>(hdr.prelude_size),
                     static_cast<unsigned long long>(prelude_size));
        std::fclose(f);
        return 0;
    }
    if (std::getenv("X86CODE_A64_JITCACHE_DEBUG")) {
        std::fprintf(stderr,
                     "a64jitcache: load begin old=%#llx new=%#llx ret old=%#llx new=%#llx prelude=%llu relocs=%llu fastmem=%llu\n",
                     static_cast<unsigned long long>(hdr.old_bases[static_cast<size_t>(RelocKind::CodeBegin)]),
                     static_cast<unsigned long long>(reinterpret_cast<u64>(begin)),
                     static_cast<unsigned long long>(hdr.old_bases[static_cast<size_t>(RelocKind::ReturnFromRun)]),
                     static_cast<unsigned long long>(reinterpret_cast<u64>(code.GetReturnFromRunCodeAddress())),
                     static_cast<unsigned long long>(prelude_size),
                     static_cast<unsigned long long>(hdr.n_relocs),
                     static_cast<unsigned long long>(hdr.n_fastmem));
    }
    if (hdr.code_size == 0 || hdr.code_size > code.SpaceRemaining()) {
        std::fclose(f);
        return 0;
    }

    std::vector<std::tuple<u64, u64, u64>> entries;
    entries.resize(hdr.n_blocks);
    for (u64 i = 0; i < hdr.n_blocks; i++) {
        u64 desc = 0, off = 0, sz = 0;
        if (std::fread(&desc, 8, 1, f) != 1 || std::fread(&off, 8, 1, f) != 1 || std::fread(&sz, 8, 1, f) != 1) {
            std::fclose(f);
            return 0;
        }
        entries[i] = {desc, off, sz};
    }

    std::vector<RelocEntry> relocs;
    relocs.resize(hdr.n_relocs);
    for (u64 i = 0; i < hdr.n_relocs; i++) {
        u32 off = 0;
        u8 kind = 0;
        u8 pad[3];
        if (std::fread(&off, 4, 1, f) != 1 || std::fread(&kind, 1, 1, f) != 1 || std::fread(pad, 1, 3, f) != 3) {
            std::fclose(f);
            return 0;
        }
        if (kind >= static_cast<u8>(RelocKind::Count) || static_cast<u64>(off) + 8 > hdr.code_size) {
            std::fclose(f);
            return 0;
        }
        relocs[i] = {off, static_cast<RelocKind>(kind)};
    }

    std::vector<FastmemPersist> fm_entries;
    fm_entries.resize(hdr.n_fastmem);
    for (u64 i = 0; i < hdr.n_fastmem; i++) {
        if (std::fread(&fm_entries[i], sizeof(FastmemPersist), 1, f) != 1) {
            std::fclose(f);
            return 0;
        }
    }

    std::vector<LinkPersist> links(hdr.n_links);
    if (hdr.n_links && std::fread(links.data(), sizeof(LinkPersist), hdr.n_links, f) != hdr.n_links) {
        std::fclose(f);
        return 0;
    }

    u64 new_bases[static_cast<size_t>(RelocKind::Count)]{};
    new_bases[static_cast<size_t>(RelocKind::CodeBegin)] = reinterpret_cast<u64>(begin);
    new_bases[static_cast<size_t>(RelocKind::ReturnFromRun)] =
        reinterpret_cast<u64>(code.GetReturnFromRunCodeAddress());
    new_bases[static_cast<size_t>(RelocKind::ForceReturn)] =
        reinterpret_cast<u64>(code.GetForceReturnFromRunCodeAddress());
    new_bases[static_cast<size_t>(RelocKind::TermFastDispatch)] =
        reinterpret_cast<u64>(terminal_handler_fast_dispatch_hint);
    new_bases[static_cast<size_t>(RelocKind::TermPopRSB)] =
        reinterpret_cast<u64>(terminal_handler_pop_rsb_hint);
    new_bases[static_cast<size_t>(RelocKind::FastDispatchTable)] =
        reinterpret_cast<u64>(fast_dispatch_table.data());
    new_bases[static_cast<size_t>(RelocKind::ObjcHitCounter)] =
        reinterpret_cast<u64>(conf.inline_objc_hit_counter);
    new_bases[static_cast<size_t>(RelocKind::ObjcMissCounter)] =
        reinterpret_cast<u64>(conf.inline_objc_miss_counter);
    new_bases[static_cast<size_t>(RelocKind::CallbackThis)] =
        reinterpret_cast<u64>(conf.callbacks);

    // Refuse kinds we cannot relocate (old base was set, new base is null, or vice versa for counters/callback).
    for (size_t k = 0; k < static_cast<size_t>(RelocKind::Count); k++) {
        const bool old_set = hdr.old_bases[k] != 0;
        const bool new_set = new_bases[k] != 0;
        if (old_set != new_set) {
            // Terminal handlers / FD table / code begin must always be present when used.
            if (k <= static_cast<size_t>(RelocKind::FastDispatchTable) ||
                k == static_cast<size_t>(RelocKind::CallbackThis)) {
                std::fclose(f);
                return 0;
            }
            // Objc counters: allow both-null; if only one side has them, refuse.
            std::fclose(f);
            return 0;
        }
    }

    code.EnableWriting();
    if (std::fread(begin, 1, hdr.code_size, f) != hdr.code_size) {
        code.DisableWriting();
        std::fclose(f);
        return 0;
    }
    // Restore ConstantPool bytes that persisted blocks RIP-reference (see save comment).
    if (hdr.const_pool_used) {
        if (hdr.const_pool_used > code.GetConstantPoolTotalBytes()) {
            code.DisableWriting();
            std::fclose(f);
            return 0;
        }
        std::vector<u8> pool_bytes(static_cast<size_t>(hdr.const_pool_used));
        if (std::fread(pool_bytes.data(), 1, hdr.const_pool_used, f) != hdr.const_pool_used) {
            code.DisableWriting();
            std::fclose(f);
            return 0;
        }
        // Pool lives in the writable prelude (allocated before PreludeComplete); safe to fill now.
        code.RestoreConstantPool(pool_bytes.data(), static_cast<size_t>(hdr.const_pool_used));
        if (std::getenv("X86CODE_A64_JITCACHE_DEBUG")) {
            std::fprintf(stderr, "a64jitcache: restored const_pool %llu bytes\n",
                         static_cast<unsigned long long>(hdr.const_pool_used));
        }
    }
    // Apply emitter-recorded relocations against the new bases.
    for (const auto& r : relocs) {
        const size_t ki = static_cast<size_t>(r.kind);
        const u64 old_base = hdr.old_bases[ki];
        const u64 new_base = new_bases[ki];
        if (old_base == 0 && new_base == 0) {
            continue;
        }
        u64* slot = reinterpret_cast<u64*>(begin + r.offset);
        const u64 old_val = *slot;
        *slot = old_val - old_base + new_base;
    }
    // Restore fastmem fault→fallback map; rebind callbacks to this process's prelude trampolines.
    // Also rewrite the deferred abort stub's `call rel32` (E8) so the TBI/high-bits slow path
    // hits the live trampoline even if prelude near/far layout drifted within the same size.
    const u64 new_begin = reinterpret_cast<u64>(begin);
    fastmem_patch_info.clear();
    size_t fm_bound = 0;
    size_t fm_call_patched = 0;
    for (const auto& fm : fm_entries) {
        if (fm.fault_off < 0 || static_cast<u64>(fm.fault_off) >= hdr.code_size) {
            continue;
        }
        const u64 fault = static_cast<u64>(static_cast<s64>(new_begin) + fm.fault_off);
        const u64 resume = static_cast<u64>(static_cast<s64>(new_begin) + fm.resume_off);
        const auto key = std::make_tuple(fm.ordered != 0, static_cast<size_t>(fm.bitsize),
                                         static_cast<int>(fm.vaddr_idx), static_cast<int>(fm.value_idx));
        void (*fn)() = nullptr;
        if (fm.kind == 0) {
            const auto it = read_fallbacks.find(key);
            if (it != read_fallbacks.end()) fn = it->second;
        } else if (fm.kind == 1) {
            const auto it = write_fallbacks.find(key);
            if (it != write_fallbacks.end()) fn = it->second;
        } else if (fm.kind == 2) {
            const auto it = exclusive_write_fallbacks.find(key);
            if (it != exclusive_write_fallbacks.end()) fn = it->second;
        }
        if (!fn) {
            continue;
        }
        fastmem_patch_info.emplace(fault, FastmemPatchInfo{resume, reinterpret_cast<u64>(fn)});
        fm_bound++;
        // Abort stub shape: call wrapped_fn; <resume>. Patch E8 rel32 immediately before resume.
        if (fm.resume_off >= 5) {
            u8* call_site = begin + static_cast<size_t>(fm.resume_off) - 5;
            if (call_site[0] == 0xE8) {
                const u64 next = reinterpret_cast<u64>(call_site) + 5;
                const s32 rel = static_cast<s32>(static_cast<s64>(reinterpret_cast<u64>(fn) - next));
                std::memcpy(call_site + 1, &rel, sizeof(rel));
                fm_call_patched++;
            }
        }
    }
    if (std::getenv("X86CODE_A64_JITCACHE_DEBUG")) {
        std::fprintf(stderr, "a64jitcache: rebound %zu/%zu fastmem sites (abort-call patched %zu)\n",
                     fm_bound, fm_entries.size(), fm_call_patched);
    }
    code.SetCodePtr(begin + hdr.code_size);
    code.DisableWriting();
    std::fclose(f);

    size_t loaded = 0;
    persistable_blocks.clear();
    reloc_entries = std::move(relocs);  // keep for a later re-save under the new bases
    // Optional diagnostic: X86CODE_A64_JITCACHE_SKIP_RANGE=lo-hi skips registering blocks in
    // [lo, hi) so they retranslate fresh (e.g. to bisect a bad translation).
    u64 skip_lo = 0, skip_hi = 0;
    if (const char* sk = std::getenv("X86CODE_A64_JITCACHE_SKIP_RANGE")) {
        std::sscanf(sk, "%llx-%llx", &skip_lo, &skip_hi);
    }
    // block_ranges is for IC invalidation only; cover a single guest insn (4 bytes). Using host
    // size as a guest PC span (the old bug) made ranges hundreds of bytes wide and could
    // invalidate / confuse overlapping blocks after IVAU.
    for (const auto& [desc_v, off, sz] : entries) {
        const A64::LocationDescriptor desc{IR::LocationDescriptor{desc_v}};
        const u64 pc = desc.PC();
        if (pc < cache_lo || pc >= cache_hi) {
            continue;
        }
        if (skip_hi && pc >= skip_lo && pc < skip_hi) {
            continue;
        }
        if (off + sz > hdr.code_size) {
            continue;
        }
        CodePtr ep = begin + off;
        BlockDescriptor bd{ep, static_cast<size_t>(sz)};
        block_descriptors.insert({IR::LocationDescriptor{desc_v}, bd});
        const auto range = boost::icl::discrete_interval<u64>::closed(pc, pc + 3);
        block_ranges.AddRange(range, desc);
        persistable_blocks.push_back({desc_v, ep, static_cast<size_t>(sz)});
        loaded++;
    }
    // Rebuild link sites (the slab holds every site in its unlinked form) and link callers to loaded blocks.
    patch_information.clear();
    unlinked_heads.clear();
    for (const auto& l : links) {
        if (static_cast<u64>(l.site_off) + 8 > hdr.code_size || l.kind > PatchInformation::JMP) {
            continue;
        }
        CodePtr site = begin + l.site_off;
        patch_information[IR::LocationDescriptor{l.target}].add(static_cast<PatchInformation::Kind>(l.kind), site);
        if (atomic_patch_sites) {
            u64 head;
            std::memcpy(&head, site, 8);
            unlinked_heads[reinterpret_cast<u64>(site)] = head;
        }
    }
    size_t linked = 0;
    if (!std::getenv("X86CODE_A64_JITCACHE_NOLINK")) {
        code.EnableWriting();
        for (const auto& [desc, bd] : block_descriptors) {
            if (patch_information.count(desc)) {
                Patch(desc, bd.entrypoint);
                linked++;
            }
        }
        code.DisableWriting();
    }
    if (std::getenv("X86CODE_A64_JITCACHE_DEBUG")) {
        std::fprintf(stderr, "a64jitcache: %zu link sites, %zu target blocks linked\n", links.size(), linked);
    }
    ClearFastDispatchTable();
    return loaded;
}

void A64EmitX64::ClearCache() {
    EmitX64::ClearCache();
    block_ranges.ClearCache();
    ClearFastDispatchTable();
    fastmem_patch_info.clear();
    fastmem_recompile_markers.clear();
    unlinked_heads.clear();
    persistable_blocks.clear();
    reloc_entries.clear();
}

void A64EmitX64::InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges) {
    InvalidateBasicBlocks(block_ranges.InvalidateRanges(ranges));
}

void A64EmitX64::ClearFastDispatchTable() {
    if (!conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        return;
    }
    if (conf.shared_translations) {
        // Concurrent FD lookups use aligned 16-byte loads; clear each entry the same way Unpatch does.
        for (auto& entry : fast_dispatch_table) {
            _mm_store_si128(reinterpret_cast<__m128i*>(&entry), _mm_set_epi64x(0, -1));
        }
    } else {
        fast_dispatch_table.fill({});
    }
}

void A64EmitX64::GenTerminalHandlers() {
    // PC ends up in rbp, location_descriptor ends up in rbx
    const auto calculate_location_descriptor = [this] {
        // This calculation has to match up with A64::LocationDescriptor::UniqueHash
        // TODO: Optimization is available here based on known state of fpcr.
        code.mov(rbp, qword[r15 + offsetof(A64JitState, pc)]);
        code.mov(rcx, A64::LocationDescriptor::pc_mask);
        code.and_(rcx, rbp);
        code.mov(ebx, dword[r15 + offsetof(A64JitState, fpcr)]);
        code.and_(ebx, A64::LocationDescriptor::fpcr_mask);
        code.shl(rbx, A64::LocationDescriptor::fpcr_shift);
        code.or_(rbx, rcx);
    };

    Xbyak::Label fast_dispatch_cache_miss, rsb_cache_miss;

    code.align();
    terminal_handler_pop_rsb_hint = code.getCurr<const void*>();
    calculate_location_descriptor();
    code.mov(eax, dword[r15 + offsetof(A64JitState, rsb_ptr)]);
    code.sub(eax, 1);
    code.and_(eax, u32(A64JitState::RSBPtrMask));
    code.mov(dword[r15 + offsetof(A64JitState, rsb_ptr)], eax);
    code.cmp(rbx, qword[r15 + offsetof(A64JitState, rsb_location_descriptors) + rax * sizeof(u64)]);
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.jne(rsb_cache_miss);
    } else {
        code.jne(code.GetReturnFromRunCodeAddress());
    }
    code.mov(rax, qword[r15 + offsetof(A64JitState, rsb_codeptrs) + rax * sizeof(u64)]);
    code.jmp(rax);
    PerfMapRegister(terminal_handler_pop_rsb_hint, code.getCurr(), "a64_terminal_handler_pop_rsb_hint");

    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.align();
        terminal_handler_fast_dispatch_hint = code.getCurr<const void*>();
        calculate_location_descriptor();
        code.L(rsb_cache_miss);
        // Prelude (before GetCodeBegin): not in the persistent slab; no reloc entry needed.
        code.mov(r12, reinterpret_cast<u64>(fast_dispatch_table.data()));
        code.mov(rbp, rbx);
        if (code.HasHostFeature(HostFeature::SSE42)) {
            code.crc32(rbp, r12);
        }
        code.and_(ebp, fast_dispatch_table_mask);
        code.lea(rbp, ptr[r12 + rbp]);
        if (conf.shared_translations) {
            // Entries are read and written as one aligned 16-byte access (atomic on AVX-capable x86): another
            // thread may be replacing the entry. xmm0/xmm1 are dead between blocks.
            Xbyak::Label shared_miss;
            code.movdqa(xmm0, xword[rbp]);
            code.movq(rax, xmm0);
            code.cmp(rbx, rax);
            code.jne(shared_miss);
            code.pextrq(rax, xmm0, 1);
            code.jmp(rax);
            code.L(shared_miss);
            code.LookupBlock();
            code.movq(xmm0, rbx);
            code.movq(xmm1, rax);
            code.punpcklqdq(xmm0, xmm1);
            code.movdqa(xword[rbp], xmm0);
            code.jmp(rax);
        }
        code.cmp(rbx, qword[rbp + offsetof(FastDispatchEntry, location_descriptor)]);
        code.jne(fast_dispatch_cache_miss);
        code.jmp(ptr[rbp + offsetof(FastDispatchEntry, code_ptr)]);
        code.L(fast_dispatch_cache_miss);
        code.mov(qword[rbp + offsetof(FastDispatchEntry, location_descriptor)], rbx);
        code.LookupBlock();
        code.mov(ptr[rbp + offsetof(FastDispatchEntry, code_ptr)], rax);
        code.jmp(rax);
        PerfMapRegister(terminal_handler_fast_dispatch_hint, code.getCurr(), "a64_terminal_handler_fast_dispatch_hint");

        code.align();
        fast_dispatch_table_lookup = code.getCurr<FastDispatchEntry& (*)(u64)>();
        code.mov(code.ABI_PARAM2, reinterpret_cast<u64>(fast_dispatch_table.data()));
        if (code.HasHostFeature(HostFeature::SSE42)) {
            code.crc32(code.ABI_PARAM1, code.ABI_PARAM2);
        }
        code.and_(code.ABI_PARAM1.cvt32(), fast_dispatch_table_mask);
        code.lea(code.ABI_RETURN, code.ptr[code.ABI_PARAM2 + code.ABI_PARAM1]);
        code.ret();
        PerfMapRegister(fast_dispatch_table_lookup, code.getCurr(), "a64_fast_dispatch_table_lookup");
    }
}

void A64EmitX64::EmitPushRSB(EmitContext& ctx, IR::Inst* inst) {
    if (!conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
        return;
    }

    EmitX64::EmitPushRSB(ctx, inst);
}

void A64EmitX64::EmitA64SetCheckBit(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg8 to_store = ctx.reg_alloc.UseGpr(args[0]).cvt8();
    code.mov(code.byte[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, check_bit)], to_store);
}

void A64EmitX64::EmitA64GetCFlag(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, dword[r15 + offsetof(A64JitState, cpsr_nzcv)]);
    code.shr(result, NZCV::x64_c_flag_bit);
    code.and_(result, 1);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetNZCVRaw(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 nzcv_raw = ctx.reg_alloc.ScratchGpr().cvt32();

    code.mov(nzcv_raw, dword[r15 + offsetof(A64JitState, cpsr_nzcv)]);

    if (code.HasHostFeature(HostFeature::FastBMI2)) {
        const Xbyak::Reg32 tmp = ctx.reg_alloc.ScratchGpr().cvt32();
        code.mov(tmp, NZCV::x64_mask);
        code.pext(nzcv_raw, nzcv_raw, tmp);
        code.shl(nzcv_raw, 28);
    } else {
        code.and_(nzcv_raw, NZCV::x64_mask);
        code.imul(nzcv_raw, nzcv_raw, NZCV::from_x64_multiplier);
        code.and_(nzcv_raw, NZCV::arm_mask);
    }

    ctx.reg_alloc.DefineValue(inst, nzcv_raw);
}

void A64EmitX64::EmitA64SetNZCVRaw(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg32 nzcv_raw = ctx.reg_alloc.UseScratchGpr(args[0]).cvt32();

    code.shr(nzcv_raw, 28);
    if (code.HasHostFeature(HostFeature::FastBMI2)) {
        const Xbyak::Reg32 tmp = ctx.reg_alloc.ScratchGpr().cvt32();
        code.mov(tmp, NZCV::x64_mask);
        code.pdep(nzcv_raw, nzcv_raw, tmp);
    } else {
        code.imul(nzcv_raw, nzcv_raw, NZCV::to_x64_multiplier);
        code.and_(nzcv_raw, NZCV::x64_mask);
    }
    code.mov(dword[r15 + offsetof(A64JitState, cpsr_nzcv)], nzcv_raw);
}

void A64EmitX64::EmitA64SetNZCV(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg32 to_store = ctx.reg_alloc.UseScratchGpr(args[0]).cvt32();
    code.mov(dword[r15 + offsetof(A64JitState, cpsr_nzcv)], to_store);
}

void A64EmitX64::EmitA64GetW(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();

    code.mov(result, dword[r15 + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)]);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetX(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr();

    code.mov(result, qword[r15 + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)]);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetS(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = qword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
    code.movd(result, addr);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetD(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = qword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
    code.movq(result, addr);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetQ(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
    code.movaps(result, addr);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetSP(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr();
    code.mov(result, qword[r15 + offsetof(A64JitState, sp)]);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetFPCR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, dword[r15 + offsetof(A64JitState, fpcr)]);
    ctx.reg_alloc.DefineValue(inst, result);
}

static u32 GetFPSRImpl(A64JitState* jit_state) {
    return jit_state->GetFpsr();
}

void A64EmitX64::EmitA64GetFPSR(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.has_host_call = true;
    ctx.reg_alloc.HostCall(inst);
    code.mov(code.ABI_PARAM1, code.r15);
    code.stmxcsr(code.dword[code.r15 + offsetof(A64JitState, guest_MXCSR)]);
    code.CallFunction(GetFPSRImpl);
}

void A64EmitX64::EmitA64SetW(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const auto addr = qword[r15 + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)];
    if (args[1].FitsInImmediateS32()) {
        code.mov(addr, args[1].GetImmediateS32());
    } else {
        // TODO: zext tracking, xmm variant
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseScratchGpr(args[1]);
        code.mov(to_store.cvt32(), to_store.cvt32());
        code.mov(addr, to_store);
    }
}

void A64EmitX64::EmitA64SetX(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const auto addr = qword[r15 + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)];
    if (args[1].FitsInImmediateS32()) {
        code.mov(addr, args[1].GetImmediateS32());
    } else if (args[1].IsInXmm()) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[1]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(args[1]);
        code.mov(addr, to_store);
    }
}

void A64EmitX64::EmitA64SetS(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[1]);
    const Xbyak::Xmm tmp = ctx.reg_alloc.ScratchXmm();
    // TODO: Optimize
    code.pxor(tmp, tmp);
    code.movss(tmp, to_store);
    code.movaps(addr, tmp);
}

void A64EmitX64::EmitA64SetD(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseScratchXmm(args[1]);
    code.movq(to_store, to_store);  // TODO: Remove when able
    code.movaps(addr, to_store);
}

void A64EmitX64::EmitA64SetQ(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[1]);
    code.movaps(addr, to_store);
}

void A64EmitX64::EmitA64SetSP(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto addr = qword[r15 + offsetof(A64JitState, sp)];
    if (args[0].FitsInImmediateS32()) {
        code.mov(addr, args[0].GetImmediateS32());
    } else if (args[0].IsInXmm()) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[0]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(args[0]);
        code.mov(addr, to_store);
    }
}

static void SetFPCRImpl(A64JitState* jit_state, u32 value) {
    jit_state->SetFpcr(value);
}

void A64EmitX64::EmitA64SetFPCR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.has_host_call = true;
    ctx.reg_alloc.HostCall(nullptr, {}, args[0]);
    code.mov(code.ABI_PARAM1, code.r15);
    code.CallFunction(SetFPCRImpl);
    code.ldmxcsr(code.dword[code.r15 + offsetof(A64JitState, guest_MXCSR)]);
}

static void SetFPSRImpl(A64JitState* jit_state, u32 value) {
    jit_state->SetFpsr(value);
}

void A64EmitX64::EmitA64SetFPSR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.has_host_call = true;
    ctx.reg_alloc.HostCall(nullptr, {}, args[0]);
    code.mov(code.ABI_PARAM1, code.r15);
    code.CallFunction(SetFPSRImpl);
    code.ldmxcsr(code.dword[code.r15 + offsetof(A64JitState, guest_MXCSR)]);
}

void A64EmitX64::EmitA64SetPC(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto addr = qword[r15 + offsetof(A64JitState, pc)];
    if (args[0].FitsInImmediateS32()) {
        code.mov(addr, args[0].GetImmediateS32());
    } else if (args[0].IsInXmm()) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[0]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(args[0]);
        code.mov(addr, to_store);
    }
}

void A64EmitX64::EmitA64CallSupervisor(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.has_host_call = true;
    ctx.reg_alloc.HostCall(nullptr);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[0].IsImmediate());
    const u32 imm = args[0].GetImmediateU32();
    Devirtualize<&A64::UserCallbacks::CallSVC>(conf.callbacks).EmitCall(code, [&](RegList param) {
        code.mov(param[0], imm);
    });
    // The kernel would have to execute ERET to get here, which would clear exclusive state.
    code.mov(code.byte[r15 + offsetof(A64JitState, exclusive_state)], u8(0));
    // x86Code: no PC=X30 rewrite here. The old one fired after every svc, including svcs not followed by a ret
    // (it skipped their code); that broke boot. Optional skip-ret lives in GuestThread::CallSVC.
}

void A64EmitX64::EmitA64ExceptionRaised(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.has_host_call = true;
    ctx.reg_alloc.HostCall(nullptr);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[0].IsImmediate() && args[1].IsImmediate());
    const u64 pc = args[0].GetImmediateU64();
    const u64 exception = args[1].GetImmediateU64();
    Devirtualize<&A64::UserCallbacks::ExceptionRaised>(conf.callbacks).EmitCall(code, [&](RegList param) {
        code.mov(param[0], pc);
        code.mov(param[1], exception);
    });
}

void A64EmitX64::EmitA64DataCacheOperationRaised(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.has_host_call = true;
    ctx.reg_alloc.HostCall(nullptr, {}, args[1], args[2]);
    Devirtualize<&A64::UserCallbacks::DataCacheOperationRaised>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64InstructionCacheOperationRaised(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.has_host_call = true;
    ctx.reg_alloc.HostCall(nullptr, {}, args[0], args[1]);
    Devirtualize<&A64::UserCallbacks::InstructionCacheOperationRaised>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64DataSynchronizationBarrier(A64EmitContext&, IR::Inst*) {
    code.mfence();
    code.lfence();
}

void A64EmitX64::EmitA64DataMemoryBarrier(A64EmitContext&, IR::Inst*) {
    code.mfence();
}

void A64EmitX64::EmitA64InstructionSynchronizationBarrier(A64EmitContext& ctx, IR::Inst*) {
    if (!conf.hook_isb) {
        return;
    }

    ctx.has_host_call = true;
    ctx.reg_alloc.HostCall(nullptr);
    Devirtualize<&A64::UserCallbacks::InstructionSynchronizationBarrierRaised>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64GetCNTFRQ(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, conf.cntfrq_el0);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetCNTPCT(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.has_host_call = true;
    ctx.reg_alloc.HostCall(inst);
    if (!conf.wall_clock_cntpct) {
        code.UpdateTicks();
    }
    Devirtualize<&A64::UserCallbacks::GetCNTPCT>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64GetCTR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, conf.ctr_el0);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetDCZID(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, conf.dczid_el0);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetTPIDR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr();
    if (conf.shared_translations) {
        code.mov(result, qword[r15 + offsetof(A64JitState, tpidr_el0_ptr)]);
        code.mov(result, qword[result]);
    } else if (conf.tpidr_el0) {
        code.mov(result, u64(conf.tpidr_el0));
        code.mov(result, qword[result]);
    } else {
        code.xor_(result.cvt32(), result.cvt32());
    }
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetTPIDRRO(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr();
    if (conf.shared_translations) {
        code.mov(result, qword[r15 + offsetof(A64JitState, tpidrro_el0_ptr)]);
        code.mov(result, qword[result]);
    } else if (conf.tpidrro_el0) {
        code.mov(result, u64(conf.tpidrro_el0));
        code.mov(result, qword[result]);
    } else {
        code.xor_(result.cvt32(), result.cvt32());
    }
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64SetTPIDR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg64 value = ctx.reg_alloc.UseGpr(args[0]);
    const Xbyak::Reg64 addr = ctx.reg_alloc.ScratchGpr();
    if (conf.shared_translations) {
        code.mov(addr, qword[r15 + offsetof(A64JitState, tpidr_el0_ptr)]);
        code.mov(qword[addr], value);
    } else if (conf.tpidr_el0) {
        code.mov(addr, u64(conf.tpidr_el0));
        code.mov(qword[addr], value);
    }
}

std::string A64EmitX64::LocationDescriptorToFriendlyName(const IR::LocationDescriptor& ir_descriptor) const {
    const A64::LocationDescriptor descriptor{ir_descriptor};
    return fmt::format("a64_{:016X}_fpcr{:08X}",
                       descriptor.PC(),
                       descriptor.FPCR().Value());
}

void A64EmitX64::EmitTerminalImpl(IR::Term::Interpret terminal, IR::LocationDescriptor, bool) {
    code.SwitchMxcsrOnExit();
    Devirtualize<&A64::UserCallbacks::InterpreterFallback>(conf.callbacks).EmitCall(code, [&](RegList param) {
        code.mov(param[0], A64::LocationDescriptor{terminal.next}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], param[0]);
        code.mov(param[1].cvt32(), terminal.num_instructions);
    });
    code.ReturnFromRunCode(true);  // TODO: Check cycles
}

void A64EmitX64::EmitTerminalImpl(IR::Term::ReturnToDispatch, IR::LocationDescriptor, bool) {
    code.ReturnFromRunCode();
}

void A64EmitX64::EmitTerminalImpl(IR::Term::LinkBlock terminal, IR::LocationDescriptor, bool is_single_step) {
    if (!conf.HasOptimization(OptimizationFlag::BlockLinking) || is_single_step) {
        code.mov(rax, A64::LocationDescriptor{terminal.next}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.ReturnFromRunCode();
        return;
    }

    if (conf.enable_cycle_counting) {
        code.cmp(qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], 0);

        if (conf.shared_translations) {
            code.align(8);
        }
        patch_information[terminal.next].add(PatchInformation::JG, code.getCurr());
        if (const auto next_bb = GetBasicBlock(terminal.next)) {
            EmitPatchJg(terminal.next, next_bb->entrypoint);
        } else {
            EmitPatchJg(terminal.next);
        }
    } else {
        code.cmp(dword[r15 + offsetof(A64JitState, halt_reason)], 0);

        if (conf.shared_translations) {
            code.align(8);
        }
        patch_information[terminal.next].add(PatchInformation::JZ, code.getCurr());
        if (const auto next_bb = GetBasicBlock(terminal.next)) {
            EmitPatchJz(terminal.next, next_bb->entrypoint);
        } else {
            EmitPatchJz(terminal.next);
        }
    }

    code.mov(rax, A64::LocationDescriptor{terminal.next}.PC());
    code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
    code.ForceReturnFromRunCode();
}

void A64EmitX64::EmitTerminalImpl(IR::Term::LinkBlockFast terminal, IR::LocationDescriptor, bool is_single_step) {
    if (!conf.HasOptimization(OptimizationFlag::BlockLinking) || is_single_step) {
        code.mov(rax, A64::LocationDescriptor{terminal.next}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.ReturnFromRunCode();
        return;
    }

    if (conf.shared_translations) {
        code.align(8);
    }
    patch_information[terminal.next].add(PatchInformation::JMP, code.getCurr());
    if (auto next_bb = GetBasicBlock(terminal.next)) {
        EmitPatchJmp(terminal.next, next_bb->entrypoint);
    } else {
        EmitPatchJmp(terminal.next);
    }
}

void A64EmitX64::EmitTerminalImpl(IR::Term::PopRSBHint, IR::LocationDescriptor, bool is_single_step) {
    if (!conf.HasOptimization(OptimizationFlag::ReturnStackBuffer) || is_single_step) {
        code.ReturnFromRunCode();
        return;
    }

    code.jmp(terminal_handler_pop_rsb_hint);
}

void A64EmitX64::EmitTerminalImpl(IR::Term::FastDispatchHint, IR::LocationDescriptor, bool is_single_step) {
    if (!conf.HasOptimization(OptimizationFlag::FastDispatch) || is_single_step) {
        code.ReturnFromRunCode();
        return;
    }

    code.jmp(terminal_handler_fast_dispatch_hint);
}

void A64EmitX64::EmitTerminalImpl(IR::Term::If terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    switch (terminal.if_) {
    case IR::Cond::AL:
    case IR::Cond::NV:
        EmitTerminal(terminal.then_, initial_location, is_single_step);
        break;
    default:
        Xbyak::Label pass = EmitCond(terminal.if_);
        EmitTerminal(terminal.else_, initial_location, is_single_step);
        code.L(pass);
        EmitTerminal(terminal.then_, initial_location, is_single_step);
        break;
    }
}

void A64EmitX64::EmitTerminalImpl(IR::Term::CheckBit terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    Xbyak::Label fail;
    code.cmp(code.byte[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, check_bit)], u8(0));
    code.jz(fail);
    EmitTerminal(terminal.then_, initial_location, is_single_step);
    code.L(fail);
    EmitTerminal(terminal.else_, initial_location, is_single_step);
}

void A64EmitX64::EmitTerminalImpl(IR::Term::CheckHalt terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    code.cmp(dword[r15 + offsetof(A64JitState, halt_reason)], 0);
    code.jne(code.GetForceReturnFromRunCodeAddress());
    EmitTerminal(terminal.else_, initial_location, is_single_step);
}

// x86Code shared translations. A jg/jz/jmp site always holds its unlinked form (mov rax, pc; mov [r15+pc], rax;
// jcc/jmp dispatcher; 23/22 bytes) and starts 8-byte aligned. Linking replaces only the first 8 bytes, atomically:
// `jcc rel32; jmp short <site end>` or `jmp rel32`. A thread inside the site sees either head, and the tail it may
// be executing never changes. kind: 0 jg, 1 jz, 2 jmp.
void A64EmitX64::WriteSharedLinkHead(int kind, CodePtr site, CodePtr target_code_ptr) {
    const u64 site_addr = reinterpret_cast<u64>(site);
    ASSERT((site_addr & 7) == 0);
    u64 head;
    if (!target_code_ptr) {
        head = unlinked_heads.at(site_addr);
    } else {
        u8 bytes[8];
        std::memcpy(bytes, site, 8);
        const size_t insn_len = kind == 2 ? 5 : 6;
        const s64 rel = static_cast<s64>(reinterpret_cast<u64>(target_code_ptr)) - static_cast<s64>(site_addr + insn_len);
        ASSERT(rel == static_cast<s32>(rel));
        const s32 rel32 = static_cast<s32>(rel);
        if (kind == 2) {
            bytes[0] = 0xE9;
            std::memcpy(bytes + 1, &rel32, 4);
        } else {
            bytes[0] = 0x0F;
            bytes[1] = kind == 0 ? 0x8F : 0x84;
            std::memcpy(bytes + 2, &rel32, 4);
            bytes[6] = 0xEB;  // jmp short over the unlinked tail to the end of the site
            bytes[7] = 23 - 8;
        }
        std::memcpy(&head, bytes, 8);
    }
    __atomic_store_n(reinterpret_cast<u64*>(site_addr), head, __ATOMIC_RELEASE);
}

void A64EmitX64::EmitSharedPatchSite(int kind, const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const CodePtr site = code.getCurr();
    if (patching) {
        WriteSharedLinkHead(kind, site, target_code_ptr);
        return;
    }
    // always the 10-byte `mov rax, imm64` (48 B8 imm64): no instruction may start inside the 8-byte head
    code.db(0x48);
    code.db(0xB8);
    code.dq(A64::LocationDescriptor{target_desc}.PC());
    code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
    if (kind == 0) {
        code.jg(code.GetReturnFromRunCodeAddress());
    } else if (kind == 1) {
        code.jz(code.GetReturnFromRunCodeAddress());
    } else {
        code.jmp(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(site, kind == 2 ? 22 : 23);
    u64 head;
    std::memcpy(&head, site, 8);
    unlinked_heads[reinterpret_cast<u64>(site)] = head;
    if (target_code_ptr) {
        WriteSharedLinkHead(kind, site, target_code_ptr);
    }
}

void A64EmitX64::EmitPatchJg(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    if (conf.shared_translations) {
        return EmitSharedPatchSite(0, target_desc, target_code_ptr);
    }
    const CodePtr patch_location = code.getCurr();
    if (target_code_ptr) {
        code.jg(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.jg(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 23);
}

void A64EmitX64::EmitPatchJz(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    if (conf.shared_translations) {
        return EmitSharedPatchSite(1, target_desc, target_code_ptr);
    }
    const CodePtr patch_location = code.getCurr();
    if (target_code_ptr) {
        code.jz(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.jz(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 23);
}

void A64EmitX64::EmitPatchJmp(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    if (conf.shared_translations) {
        return EmitSharedPatchSite(2, target_desc, target_code_ptr);
    }
    const CodePtr patch_location = code.getCurr();
    if (target_code_ptr) {
        code.jmp(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.jmp(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 22);
}

void A64EmitX64::EmitPatchMovRcx(CodePtr target_code_ptr) {
    if (!target_code_ptr) {
        target_code_ptr = code.GetReturnFromRunCodeAddress();
    }
    const CodePtr patch_location = code.getCurr();
    if (conf.shared_translations && patching) {
        // the imm64 is 8-byte aligned (PushRSBHelper): one atomic store
        __atomic_store_n(reinterpret_cast<u64*>(reinterpret_cast<u64>(patch_location) + 2), reinterpret_cast<u64>(target_code_ptr), __ATOMIC_RELEASE);
        return;
    }
    // Absolute code pointer: block entry (CodeBegin-relative) or ReturnFromRunCode.
    const RelocKind kind = (target_code_ptr == code.GetReturnFromRunCodeAddress())
                               ? RelocKind::ReturnFromRun
                               : (target_code_ptr == code.GetForceReturnFromRunCodeAddress())
                                     ? RelocKind::ForceReturn
                                     : RelocKind::CodeBegin;
    EmitMovAbs64(code.rcx, reinterpret_cast<u64>(target_code_ptr), kind);
    code.EnsurePatchLocationSize(patch_location, 10);
    ASSERT(!conf.shared_translations || (code.getCurr<const u8*>() - static_cast<const u8*>(patch_location) == 10 && ((reinterpret_cast<u64>(patch_location) + 2) & 7) == 0));
}

void A64EmitX64::Unpatch(const IR::LocationDescriptor& location) {
    EmitX64::Unpatch(location);
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.DisableWriting();
        auto& entry = (*fast_dispatch_table_lookup)(location.Value());
        if (conf.shared_translations) {
            _mm_store_si128(reinterpret_cast<__m128i*>(&entry), _mm_set_epi64x(0, -1));
        } else {
            entry = {};
        }
        code.EnableWriting();
    }
}

}  // namespace Dynarmic::Backend::X64
