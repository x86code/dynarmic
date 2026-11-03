/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <atomic>

#include <array>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>

#include "dynarmic/backend/block_range_information.h"
#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/emit_x64.h"
#include "dynarmic/backend/x64/fastmem_table.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/ir/terminal.h"

namespace Dynarmic::Backend::X64 {

// x86Code RAM 3: see A64::UserConfig::fastmem_unchecked
inline bool FastmemUnchecked(const A64::UserConfig& conf) {
    return conf.fastmem_unchecked && conf.fastmem_pointer && *conf.fastmem_pointer == 0 && !conf.silently_mirror_fastmem
        && conf.fastmem_address_space_bits == 47;
}
// The fastmem address of vaddr for code that does not check the range itself (LSE atomics, CAS).
inline Xbyak::RegExp FastmemBase(const A64::UserConfig& conf, Xbyak::Reg64 vaddr) {
    if (FastmemUnchecked(conf)) {
        return vaddr;
    }
    return Xbyak::util::r13 + vaddr;
}


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
    /// x86Code RAM 3: unchecked fastmem, but this block's accesses faulted often: emit the inline check here.
    bool checked_memory = false;
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
    /// shared_image: map the code from a relocated, linked copy of the file (`<path>.<layout hash>.map`, written by the
    /// first process with this layout when ASLR is off) so processes share its pages (Stage 4 step 2).
    size_t LoadPersistentCache(const std::string& path, const u8 cache_uuid[16], u64 build_id,
                               u64 cache_lo, u64 cache_hi, bool shared_image = false);
    std::vector<u64> DynamicDescriptors(u64 lo, u64 hi);  // Speed 31 (Jit::GetDynamicDescriptors)
    std::vector<u64> UsedImageDescriptors(u64 lo, u64 hi);  // RAM 2 (Jit::GetUsedImageDescriptors)
    static std::vector<u64> ReadPersistentDescriptors(const std::string& path, const u8 cache_uuid[16], u64 build_id);
    /// x86Code Speed 18 (chunked AOT image): when set, Emit appends the block's direct successors (link targets, call
    /// return addresses) here, so a static builder can follow them.
    std::vector<u64>* aot_successors = nullptr;
    /// Save only the code from code_off on (one chunk's slot) as an "A64CK01" chunk file; offsets stay slab-absolute.
    size_t SaveChunk(const std::string& path, const u8 cache_uuid[16], u64 build_id, u64 cache_lo, u64 cache_hi, u64 code_off);
    /// Start a chunk: drop every registered block and emit from code_off on.
    void BeginChunk(u64 code_off);
    /// Constant pool bytes of a chunk or image file (empty on failure).
    static std::vector<u8> ReadPersistentPool(const std::string& path);
    /// Restore the union pool and put new constants from byte offset region_off on (parallel builders: disjoint regions).
    void SetConstantPoolRegion(const std::vector<u8>& pool, size_t region_off);

    void ClearCache() override;

    // x86code: true when any translated (or static-image) block was found in the ranges
    bool InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges);

    /// x86Code Speed 8 (shared translations): GetBlock's FastDispatch probe (one atomic 16-byte read of the entry the
    /// dispatcher would use) and fill (only with the shared mutex held).
    CodePtr FastDispatchProbe(u64 location_descriptor) const;
    void FastDispatchFill(u64 location_descriptor, CodePtr code_ptr);
    /// Default on (Speed 9); X86CODE_A64_GETBLOCK_FD=0: the old way (the dispatcher's miss path fills).
    static bool GetBlockFillsFastDispatch();

    // x86Code: with shared_translations, guards the block map, patch/fastmem info and emission.
    std::recursive_mutex shared_mutex;

    struct PersistEntry {
        u64 location_descriptor;
        CodePtr entrypoint;
        size_t size;
    };
    /// x86Code: approximate host bytes held by the emitter's maps, one "name MB" list for memlog.
    std::string MemoryReport() const;

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
    std::vector<RelocEntry, MetaAlloc<RelocEntry, kMetaRelocs>> reloc_entries;

    /// Emit `mov reg, Imm64(value)` and record a reloc when value is relative to `kind`.
    void EmitMovAbs64(const Xbyak::Reg64& reg, u64 value, RelocKind kind);

protected:
    const A64::UserConfig conf;
    A64::Jit* jit_interface;
    BlockRangeInformation<u64> block_ranges;

    struct FastDispatchEntry {
        u64 location_descriptor = 0;  // x86Code: stored inverted (~desc), so all-zero memory is an empty entry
        const void* code_ptr = nullptr;
    };
    static_assert(sizeof(FastDispatchEntry) == 0x10);
    // x86Code: 64K entries (1 MB) instead of 1M (16 MB, all touched when cleared): there is one Jit per guest thread
    // x86Code: 256K entries (4 MB); 64K thrashed with 175K-750K-block hot sets (SpringBoard startup: ~35% in GetBlock).
    // x86Code Speed 7: set per process before the emitter exists (SetFastDispatchEntries): 64K for daemons.
    const size_t fast_dispatch_table_size = NextFastDispatchEntries();
    const u64 fast_dispatch_table_mask = (fast_dispatch_table_size - 1) << 4;
    static size_t NextFastDispatchEntries();
    // x86Code: mapped right after the code cache (a fixed address relative to it, so code that embeds the table's
    // address is identical in processes with the same code cache address: shared AOT image). 16-byte atomic entries.
    struct FastDispatchStorage {
        explicit FastDispatchStorage(BlockOfCode& code);
        ~FastDispatchStorage();
        FastDispatchEntry* data() const { return table; }
        FastDispatchEntry* begin() const { return table; }
        FastDispatchEntry* end() const { return table + size; }
        size_t size;
        void fill(const FastDispatchEntry& v) { std::fill(begin(), end(), v); }
        FastDispatchEntry* table;
    } fast_dispatch_table{code};
    // x86Code shared translations: original first 8 bytes of each unlinked jg/jz/jmp patch site
    void EmitSharedPatchSite(int kind, const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr);
    void WriteSharedLinkHead(int kind, CodePtr site, const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr);
    // The first 8 bytes of an unlinked site: `mov rax, imm64` (48 B8) and the low 6 bytes of the target PC
    // (x86Code Speed 6: computed instead of kept in a site -> head map, 48 MB for SpringBoard).
    static u64 UnlinkedHead(const IR::LocationDescriptor& target_desc) {
        return 0xB848u | (A64::LocationDescriptor{target_desc}.PC() << 16);
    }
    void ClearFastDispatchTable();
    FastDispatchEntry* FastDispatchSlot(u64 location_descriptor) const;

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
    FastmemTable fastmem_patch_info;
    // x86Code RAM 3 (under the fastmem callback's lock): fastmem faults; fault rip -> count; blocks retranslated with
    // the inline address check after one of their sites faulted kCheckAfterFaults times (unchecked fastmem only).
    u64 fastmem_faults = 0;
    std::map<u64, u64> fault_sites;
    tsl::robin_set<u64> checked_blocks;
    size_t checked_retranslations = 0;
    static constexpr u64 kCheckAfterFaults = 8;
    void OnFastmemFault(u64 rip);
public:
    u64* shared_invalidate_gen = nullptr;  // SharedCode::invalidate_gen (bumped when a block is retranslated checked)
protected:
    std::vector<FastmemTable::StaticEntry> static_fm_own;  // static fastmem sites when the index file is not mapped
    void PublishStaticIndex(const std::string& idx_path, std::vector<FastmemTable::StaticEntry> fm, std::vector<u64> cbs, bool complete);
    bool MapStaticIndex(const std::string& idx_path, u8* begin);
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
