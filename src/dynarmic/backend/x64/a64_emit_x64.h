/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>

#include "dynarmic/backend/block_range_information.h"
#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/emit_x64.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/ir/terminal.h"

namespace Dynarmic::Backend::X64 {

class RegAlloc;

struct A64EmitContext final : public EmitContext {
    A64EmitContext(const A64::UserConfig& conf, RegAlloc& reg_alloc, IR::Block& block);

    A64::LocationDescriptor Location() const;
    bool IsSingleStep() const;
    FP::FPCR FPCR(bool fpcr_controlled = true) const override;

    bool HasOptimization(OptimizationFlag flag) const override {
        return conf.HasOptimization(flag);
    }

    const A64::UserConfig& conf;
    /// x86Code: set when the block emits a HostCall (CallSVC, etc.); those blocks are not persisted.
    bool has_host_call = false;
};

class A64EmitX64 final : public EmitX64 {
public:
    A64EmitX64(BlockOfCode& code, A64::UserConfig conf, A64::Jit* jit_interface);
    ~A64EmitX64() override;

    /**
     * Emit host machine code for a basic block with intermediate representation `block`.
     * @note block is modified.
     */
    BlockDescriptor Emit(IR::Block& block);

    /// x86Code: emit a hand-written x86 objc_msgSend cache probe for conf.inline_objc_*.
    BlockDescriptor EmitInlineObjCMsgSend(IR::LocationDescriptor descriptor);

    /// x86Code: persistent translation cache (dyld shared-cache blocks without host calls).
    size_t SavePersistentCache(const std::string& path, const u8 cache_uuid[16], u64 build_id,
                               u64 cache_lo, u64 cache_hi);
    size_t LoadPersistentCache(const std::string& path, const u8 cache_uuid[16], u64 build_id,
                               u64 cache_lo, u64 cache_hi);
    static std::vector<u64> ReadPersistentDescriptors(const std::string& path, const u8 cache_uuid[16], u64 build_id);

    void ClearCache() override;

    void InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges);

    // x86Code: with shared_translations, guards the block map, patch/fastmem info and emission.
    std::recursive_mutex shared_mutex;

    struct PersistEntry {
        u64 location_descriptor;
        CodePtr entrypoint;
        size_t size;
    };
    std::vector<PersistEntry> persistable_blocks;

    /// x86Code: emitter-recorded relocation for absolute host Imm64 embeds (AOT stage 4 precursor).
    /// On load: *(u64*)(code_begin + offset) += new_base[kind] - old_base[kind].
    enum class RelocKind : u8 {
        CodeBegin = 0,          // JIT translated-code slab (GetCodeBegin)
        ReturnFromRun = 1,      // dispatcher leave
        ForceReturn = 2,
        TermFastDispatch = 3,   // terminal_handler_fast_dispatch_hint
        TermPopRSB = 4,
        FastDispatchTable = 5,  // fast_dispatch_table.data()
        ObjcHitCounter = 6,
        ObjcMissCounter = 7,
        CallbackThis = 8,       // conf.callbacks
        Count = 9,
    };
    struct RelocEntry {
        u32 offset;  // byte offset from GetCodeBegin()
        RelocKind kind;
    };
    std::vector<RelocEntry> reloc_entries;

    /// Emit `mov reg, Imm64(value)` and record a reloc when value is relative to `kind`.
    void EmitMovAbs64(const Xbyak::Reg64& reg, u64 value, RelocKind kind);

protected:
    const A64::UserConfig conf;
    A64::Jit* jit_interface;
    BlockRangeInformation<u64> block_ranges;

    struct FastDispatchEntry {
        u64 location_descriptor = 0xFFFF'FFFF'FFFF'FFFFull;
        const void* code_ptr = nullptr;
    };
    static_assert(sizeof(FastDispatchEntry) == 0x10);
    // x86Code: 64K entries (1 MB) instead of 1M (16 MB, all touched when cleared): there is one Jit per guest thread
    static constexpr u64 fast_dispatch_table_mask = 0xFFFF0;
    static constexpr size_t fast_dispatch_table_size = 0x10000;
    alignas(16) std::array<FastDispatchEntry, fast_dispatch_table_size> fast_dispatch_table;  // 16-byte atomic entries when shared
    // x86Code shared translations: original first 8 bytes of each unlinked jg/jz/jmp patch site
    tsl::robin_map<u64, u64> unlinked_heads;
    void EmitSharedPatchSite(int kind, const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr);
    void WriteSharedLinkHead(int kind, CodePtr site, CodePtr target_code_ptr);
    void ClearFastDispatchTable();

    void (*memory_read_128)();
    void (*memory_write_128)();
    void (*memory_exclusive_write_128)();
    void GenMemory128Accessors();

    std::map<std::tuple<bool, size_t, int, int>, void (*)()> read_fallbacks;
    std::map<std::tuple<bool, size_t, int, int>, void (*)()> write_fallbacks;
    std::map<std::tuple<bool, size_t, int, int>, void (*)()> exclusive_write_fallbacks;
    void GenFastmemFallbacks();

    const void* terminal_handler_pop_rsb_hint;
    const void* terminal_handler_fast_dispatch_hint = nullptr;
    FastDispatchEntry& (*fast_dispatch_table_lookup)(u64) = nullptr;
    void GenTerminalHandlers();

    // Microinstruction emitters
    void EmitPushRSB(EmitContext& ctx, IR::Inst* inst);
#define OPCODE(...)
#define A32OPC(...)
#define A64OPC(name, type, ...) void EmitA64##name(A64EmitContext& ctx, IR::Inst* inst);
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC

    // Helpers
    std::string LocationDescriptorToFriendlyName(const IR::LocationDescriptor&) const override;

    // Fastmem information
    using DoNotFastmemMarker = std::tuple<IR::LocationDescriptor, unsigned>;
    // x86Code: keep the fault→fallback map to two u64s. The marker/recompile flag live in a
    // side map that stays empty when recompile_on_fastmem_failure is false (our default),
    // shrinking ~1.5M-entry bucket storage from ~224 MB (56 B/entry) to ~128 MB (32 B/entry).
    struct FastmemPatchInfo {
        u64 resume_rip;
        u64 callback;
    };
    tsl::robin_map<u64, FastmemPatchInfo> fastmem_patch_info;
    tsl::robin_map<u64, DoNotFastmemMarker> fastmem_recompile_markers;
    std::set<DoNotFastmemMarker> do_not_fastmem;
    std::optional<DoNotFastmemMarker> ShouldFastmem(A64EmitContext& ctx, IR::Inst* inst) const;
    FakeCall FastmemCallback(u64 rip);

    // Memory access helpers
    void EmitCheckMemoryAbort(A64EmitContext& ctx, IR::Inst* inst, Xbyak::Label* end = nullptr);
    template<std::size_t bitsize, auto callback>
    void EmitMemoryRead(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitMemoryWrite(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveReadMemory(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveWriteMemory(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveReadMemoryInline(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveWriteMemoryInline(A64EmitContext& ctx, IR::Inst* inst);

    // Terminal instruction emitters
    void EmitTerminalImpl(IR::Term::Interpret terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::ReturnToDispatch terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::LinkBlock terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::LinkBlockFast terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::PopRSBHint terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::FastDispatchHint terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::If terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::CheckBit terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::CheckHalt terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;

    // Patching
    void Unpatch(const IR::LocationDescriptor& target_desc) override;
    void EmitPatchJg(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) override;
    void EmitPatchJz(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) override;
    void EmitPatchJmp(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) override;
    void EmitPatchMovRcx(CodePtr target_code_ptr = nullptr) override;
};

}  // namespace Dynarmic::Backend::X64
