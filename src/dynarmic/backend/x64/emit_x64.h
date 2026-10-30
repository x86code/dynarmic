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
    std::optional<BlockDescriptor> GetBasicBlock(IR::LocationDescriptor descriptor) const;

    /// x86Code: lock-free GetBasicBlock for the dispatcher hit path (shared translations: GetBasicBlock needs the
    /// shared mutex, this does not). nullptr = not known here; the caller falls back to the locked path.
    CodePtr LookupEntrypoint(IR::LocationDescriptor descriptor) const { return lf_map.Lookup(descriptor.Value()); }

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
        // Grows by doubling at 50% load; the old table is published away, then zeroed with MADV_DONTNEED (kept
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
    PatchTable patch_information;
};

}  // namespace Dynarmic::Backend::X64
