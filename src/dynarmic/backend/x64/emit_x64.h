/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <mcl/bitsizeof.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>

#include "dynarmic/backend/exception_handler.h"
#include "dynarmic/backend/x64/reg_alloc.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/ir/location_descriptor.h"
#include "dynarmic/ir/terminal.h"

namespace Dynarmic::IR {
class Block;
class Inst;
}  // namespace Dynarmic::IR

namespace Dynarmic {
enum class OptimizationFlag : u32;
}  // namespace Dynarmic

namespace Dynarmic::Backend::X64 {

class BlockOfCode;

using A64FullVectorWidth = std::integral_constant<size_t, 128>;

// Array alias that always sizes itself according to the given type T
// relative to the size of a vector register. e.g. T = u32 would result
// in a std::array<u32, 4>.
template<typename T>
using VectorArray = std::array<T, A64FullVectorWidth::value / mcl::bitsizeof<T>>;

template<typename T>
using HalfVectorArray = std::array<T, A64FullVectorWidth::value / mcl::bitsizeof<T> / 2>;

struct EmitContext {
    EmitContext(RegAlloc& reg_alloc, IR::Block& block);
    virtual ~EmitContext();

    void EraseInstruction(IR::Inst* inst);

    virtual FP::FPCR FPCR(bool fpcr_controlled = true) const = 0;

    virtual bool HasOptimization(OptimizationFlag flag) const = 0;

    RegAlloc& reg_alloc;
    IR::Block& block;

    std::vector<std::function<void()>> deferred_emits;
};

using SharedLabel = std::shared_ptr<Xbyak::Label>;

inline SharedLabel GenSharedLabel() {
    return std::make_shared<Xbyak::Label>();
}

class EmitX64 {
public:
    struct BlockDescriptor {
        CodePtr entrypoint;  // Entrypoint of emitted code
        u32 size;            // Length in bytes of emitted code
        u32 persistable = 0; // x86Code: no host calls, may go to the persistent cache (was a separate vector)
    };

    explicit EmitX64(BlockOfCode& code);
    virtual ~EmitX64();

    /// Looks up an emitted host block in the cache.
    /// x86Code: registers a static AOT image block on its first lookup (callers hold the emitter lock).
    std::optional<BlockDescriptor> GetBasicBlock(IR::LocationDescriptor descriptor);

    /// x86Code: lock-free GetBasicBlock for the dispatcher hit path (shared translations: GetBasicBlock needs the
    /// shared mutex, this does not). nullptr = not known here; the caller falls back to the locked path.
    CodePtr LookupEntrypoint(IR::LocationDescriptor descriptor) const {
        const u64 k = descriptor.Value();
        if (const u32 i = sl_map.Lookup(k, static_image.blocks); i != StaticSlotMap::kNone) {
            return static_image.code_begin + static_image.blocks[i].off;
        }
        return lf_map.Lookup(k);
    }
    /// x86Code debug (GETBLOCK_FD_VERIFY): descriptor whose block starts at ep (block_descriptors, then the static
    /// image); 0 if none. Linear; call only on a mismatch, with the shared mutex held.
    u64 OwnerOf(CodePtr ep) const;

    /// Empties the entire cache.
    virtual void ClearCache();

    /// Invalidates a selection of basic blocks.
    void InvalidateBasicBlocks(const tsl::robin_set<IR::LocationDescriptor>& locations);

protected:
    // Microinstruction emitters
#define OPCODE(name, type, ...) void Emit##name(EmitContext& ctx, IR::Inst* inst);
#define A32OPC(...)
#define A64OPC(...)
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC

    // Helpers
    virtual std::string LocationDescriptorToFriendlyName(const IR::LocationDescriptor&) const = 0;
    void EmitAddCycles(size_t cycles);
    Xbyak::Label EmitCond(IR::Cond cond);
    BlockDescriptor RegisterBlock(const IR::LocationDescriptor& location_descriptor, CodePtr entrypoint, size_t size);
    void PushRSBHelper(Xbyak::Reg64 loc_desc_reg, Xbyak::Reg64 index_reg, IR::LocationDescriptor target);

    void EmitVerboseDebuggingOutput(RegAlloc& reg_alloc);

    // Terminal instruction emitters
    void EmitTerminal(IR::Terminal terminal, IR::LocationDescriptor initial_location, bool is_single_step);
    virtual void EmitTerminalImpl(IR::Term::Interpret terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::ReturnToDispatch terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::LinkBlock terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::LinkBlockFast terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::PopRSBHint terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::FastDispatchHint terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::If terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::CheckBit terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::CheckHalt terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;

    // Patching
    // x86Code: one tagged vector instead of four std::vectors (96 B/value → 24 B).
    // With ~1M pending link targets that cut the robin_map bucket array ~224 MB → ~80 MB.
    struct PatchInformation {
        enum Kind : u8 { JG = 0, JZ = 1, JMP = 2, MOV_RCX = 3 };
    };
    // x86Code (Speed 6): link sites per target as singly linked lists in one pool: a robin_map target -> head (index+1)
    // and 12 bytes per site (site pointer tagged with the kind in its top bits, next index). Was a robin_map of
    // std::vector (40 B buckets for every block ever registered, since Patch() used operator[]): SpringBoard 195 MB.
    class PatchTable {
    public:
        // x86Code (Speed 11): robin hood probing stays short at 0.9; the default 0.5 left half of the 24 B buckets empty
        PatchTable() { heads.max_load_factor(0.9f); }
        void Add(const IR::LocationDescriptor& target, PatchInformation::Kind kind, CodePtr site) {
            u32& head = heads[target.Value()];
            sites.push_back(reinterpret_cast<u64>(site) | (static_cast<u64>(kind) << 62));
            next.push_back(head);
            head = static_cast<u32>(sites.size());
        }
        bool count(const IR::LocationDescriptor& target) const { return heads.count(target.Value()) != 0; }
        template<typename F>
        void ForEachSite(const IR::LocationDescriptor& target, F&& f) const {
            const auto it = heads.find(target.Value());
            for (u32 i = it == heads.end() ? 0 : it->second; i; i = next[i - 1]) {
                f(static_cast<PatchInformation::Kind>(sites[i - 1] >> 62), reinterpret_cast<CodePtr>(sites[i - 1] & ~(u64(3) << 62)));
            }
        }
        template<typename F>
        void ForEach(F&& f) const {
            for (const auto& [target, head] : heads) {
                for (u32 i = head; i; i = next[i - 1]) {
                    f(target, static_cast<PatchInformation::Kind>(sites[i - 1] >> 62), reinterpret_cast<CodePtr>(sites[i - 1] & ~(u64(3) << 62)));
                }
            }
        }
        void clear() {
            heads.clear();
            sites.clear();
            next.clear();
        }
        size_t size() const { return heads.size(); }
        size_t MemoryBytes() const { return heads.bucket_count() * 24 + sites.capacity() * 8 + next.capacity() * 4; }

    private:
        tsl::robin_map<u64, u32> heads;
        std::vector<u64> sites;
        std::vector<u32> next;
    };
    void Patch(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr);
    // x86Code: shared translations. Patch sites are aligned so a link/unlink is one atomic 8-byte store (other threads
    // may be executing the site); `patching` tells EmitPatch* whether it rewrites an existing site.
    bool atomic_patch_sites = false;
    bool patching = false;
    virtual void Unpatch(const IR::LocationDescriptor& target_desc);
    virtual void EmitPatchJg(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) = 0;
    virtual void EmitPatchJz(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) = 0;
    virtual void EmitPatchJmp(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) = 0;
    virtual void EmitPatchMovRcx(CodePtr target_code_ptr = nullptr) = 0;

    // State
    BlockOfCode& code;
    ExceptionHandler exception_handler;
    tsl::robin_map<IR::LocationDescriptor, BlockDescriptor> block_descriptors;
    // x86Code: insert-only open-addressing mirror of block_descriptors (key -> entrypoint), written under the shared
    // mutex (single writer), read without it. Removal clears the code pointer and keeps the key, so probe chains stay
    // intact. A reader may get an entrypoint that is invalidated right after; the locked path had the same window.
    struct LockFreeBlockMap {
        struct Entry {
            std::atomic<u64> key{0};
            std::atomic<CodePtr> code{nullptr};
        };
        struct Table {
            u64 mask;
            u64 pad;
            Entry e[1];
        };
        // Grows by doubling at 75% load; the old table is published away, then zeroed with MADV_DONTNEED (kept
        // mapped), so a reader still probing it sees empty keys = miss.
        std::atomic<Table*> table{nullptr};
        size_t used = 0;
        LockFreeBlockMap();
        ~LockFreeBlockMap();
        static size_t Hash(u64 k) { return (k * 0x9E3779B97F4A7C15ull) >> 24; }
        CodePtr Lookup(u64 k) const {
            const Table* t = table.load(std::memory_order_acquire);
            for (size_t i = Hash(k) & t->mask;; i = (i + 1) & t->mask) {
                const u64 e = t->e[i].key.load(std::memory_order_acquire);
                if (e == k) return k ? t->e[i].code.load(std::memory_order_acquire) : nullptr;
                if (e == 0) return nullptr;
            }
        }
        void Set(u64 k, CodePtr c);
        void Clear();
        size_t MemoryBytes() const { return (table.load()->mask + 1) * sizeof(Entry); }
        static Table* Alloc(size_t n);
        static void Insert(Table* t, u64 k, CodePtr c);
    } lf_map;
    // x86Code (Speed 15): registered AOT image blocks, in 8-byte slots (32-bit key fingerprint, image block index + 1,
    // bit 31 = killed) instead of lf_map's 16-byte key/code entries: the full key is checked in the image's (shared,
    // file-backed) block table. SpringBoard / Maps / Preferences register 430-620K image blocks (lf_map was 16 MB of
    // anon in each). Same publication, growth and removal rules as lf_map; a killed slot keeps its chain.
    struct StaticSlotMap {
        static constexpr u32 kNone = ~u32{0};
        struct Table {
            u64 mask;
            u64 pad;
            std::atomic<u64> e[1];
        };
        std::atomic<Table*> table{nullptr};
        size_t used = 0;
        StaticSlotMap();
        ~StaticSlotMap();
        static size_t Hash(u64 k) { return (k * 0x9E3779B97F4A7C15ull) >> 24; }
        static u64 Fp(u64 k) { return ((k * 0xC2B2AE3D27D4EB4Full) >> 32) << 32; }
        template<typename B>
        u32 Lookup(u64 k, const B* blocks) const {
            const Table* t = table.load(std::memory_order_acquire);
            const u64 fp = Fp(k);
            for (size_t i = Hash(k) & t->mask;; i = (i + 1) & t->mask) {
                const u64 e = t->e[i].load(std::memory_order_acquire);
                if (e == 0) return kNone;
                if ((e & ~u64{0xffffffff}) == fp && !(e & 0x80000000u)) {
                    const u32 idx = static_cast<u32>(e & 0x7fffffff) - 1;
                    if (blocks[idx].desc == k) return idx;
                }
            }
        }
        // idx < 2^31 - 1; descs: the image's block table as u64 words, a block's key at descs[idx * stride] (for growth)
        void Set(u64 k, u32 idx, const u64* descs, size_t stride);
        void Kill(u64 k, u32 idx);  // marks the slot of (k, idx) killed
        void Clear();
        size_t MemoryBytes() const { return (table.load()->mask + 1) * sizeof(u64); }
        static Table* Alloc(size_t n);
    } sl_map;
    PatchTable patch_information;

    // x86Code (Speed 7): the AOT image's metadata as sorted arrays, in a file mapped shared by every process with the
    // same layout (or private arrays). Was copied into block_descriptors/lf_map/patch_information/block_ranges/
    // fastmem for all ~1M blocks in every AOT process (~300 MB each). A block now goes into block_descriptors and
    // lf_map on its first lookup; link sites and PC ranges are queried here. Image-internal sites are linked in the
    // mapped code already. Invalidated blocks get a dead bit and are never registered from here again.
    struct StaticImage {
        struct Block {
            u64 desc;
            u32 off;   // entrypoint - code begin
            u32 size;
        };
        struct Link {
            u64 target;
            u32 site_off;
            u32 kind;  // PatchInformation::Kind (JG/JZ/JMP)
        };
        bool active = false;
        u8* code_begin = nullptr;
        const Block* blocks = nullptr;  // sorted by (Pc(desc), desc)
        size_t n_blocks = 0;
        const Link* links = nullptr;  // sorted by target
        size_t n_links = 0;
        std::vector<u64> dead;
        std::vector<u64> live;  // registered in lf_map (static blocks never go into block_descriptors)
        size_t registered = 0;
        void* map = nullptr;
        size_t map_len = 0;
        std::vector<Block> own_blocks;
        std::vector<Link> own_links;

        static u64 Pc(u64 desc) { return desc & ((u64(1) << 56) - 1); }
        static bool BlockLess(const Block& a, const Block& b) { return Pc(a.desc) != Pc(b.desc) ? Pc(a.desc) < Pc(b.desc) : a.desc < b.desc; }
        // index of desc's block or n_blocks
        size_t Index(u64 desc) const;
        bool Dead(size_t i) const { return (dead[i >> 6] >> (i & 63)) & 1; }
        void Kill(size_t i) { dead[i >> 6] |= u64(1) << (i & 63); }
        bool Live(size_t i) const { return (live[i >> 6] >> (i & 63)) & 1; }
        void SetLive(size_t i, bool v) { v ? live[i >> 6] |= u64(1) << (i & 63) : live[i >> 6] &= ~(u64(1) << (i & 63)); }
        std::pair<const Link*, const Link*> Sites(u64 target) const;
        void Reset();
    } static_image;
    // static sites of target_desc, relinked to target_code_ptr (nullptr: unlinked)
    void PatchStaticSites(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr);
    void PatchSite(PatchInformation::Kind kind, CodePtr location, const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr);
};

}  // namespace Dynarmic::Backend::X64
