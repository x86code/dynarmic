/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/halt_reason.h"

namespace Dynarmic {
namespace A64 {

class Jit final {
public:
    explicit Jit(UserConfig conf);

    /// x86Code: one translation cache (code, block map, fast dispatch table) shared by several Jits, e.g. one per
    /// host thread of a process. conf.shared_translations must be set; conf.callbacks must be thread-agnostic.
    struct SharedCode;
    static std::shared_ptr<SharedCode> MakeSharedCode(UserConfig conf);
    /// A Jit with its own guest state over a shared translation cache. Of conf only processor_id, tpidr_el0 and
    /// tpidrro_el0 are used (they are per-thread); everything else comes from the shared cache's config.
    Jit(UserConfig conf, std::shared_ptr<SharedCode> shared);
    /// x86Code: translation statistics of a shared cache (profiling).
    struct SharedStats {
        std::uint64_t blocks = 0, guest_instructions = 0, code_bytes = 0, translate_ns = 0, lookups = 0;
    };
    static SharedStats GetSharedStats(const SharedCode& shared);
    /// x86Code RAM 4: runs fn(arg) with the shared emitter lock held (every write to translated code, including the
    /// AOT image's link sites, happens under it): libarm64jit drops page-table entries of clean image pages.
    static void WithSharedLock(SharedCode& shared, void (*fn)(void*), void* arg);
    /// x86Code Speed 31: descriptors of the blocks this process translated itself (not the image's), persistable ones
    /// with a guest PC in [lo, hi) (training feedback from processes that run on the AOT image).
    static std::vector<std::uint64_t> GetDynamicDescriptors(SharedCode& shared, std::uint64_t lo, std::uint64_t hi);
    /// x86Code RAM 2 (JIT): descriptors of the AOT image's blocks this process has run (registered on lookup), guest PC
    /// in [lo, hi): with GetDynamicDescriptors, everything the process used (the .live training file's aging).
    static std::vector<std::uint64_t> GetUsedImageDescriptors(SharedCode& shared, std::uint64_t lo, std::uint64_t hi);
    /// x86Code: host memory held by the shared emitter's maps (memlog; takes the shared lock).
    static std::string GetMemoryReport(SharedCode& shared);
    /// x86Code: dump translated blocks whose guest PC is in [cache_lo, cache_hi) to path.
    /// Keyed by cache_uuid (16 bytes) + build_id. Returns bytes written, or 0 on failure.
    static std::size_t SavePersistentCache(SharedCode& shared, const std::string& path,
                                           const std::uint8_t cache_uuid[16], std::uint64_t build_id,
                                           std::uint64_t cache_lo, std::uint64_t cache_hi);
    /// Load a previously saved cache. Returns blocks restored, or 0 on mismatch/failure.
    static std::size_t LoadPersistentCache(SharedCode& shared, const std::string& path,
                                           const std::uint8_t cache_uuid[16], std::uint64_t build_id,
                                           std::uint64_t cache_lo, std::uint64_t cache_hi, bool shared_image = false);
    /// x86Code: place the next code cache at this address if free (shared AOT image: same layout in every process).
    static void SetCodeCacheAddress(std::uint64_t address);
    /// x86Code RAM 13: page of host-call stubs (RWX, 16 bytes per stub) every CallFunction goes through.
    static void SetHostCallStubs(void* base, std::size_t bytes);
    /// x86Code: FastDispatch table entries (power of two, 4K-1M) for emitters created after this call.
    static void SetFastDispatchEntries(std::size_t n);
    /// x86Code: AOT. Location descriptors of the blocks in a saved cache (empty on mismatch or failure).
    /// build_id 0 accepts any build (the hot set outlives codegen changes; only the image UUID must match).
    static std::vector<std::uint64_t> ReadPersistentDescriptors(const std::string& path,
                                                                const std::uint8_t cache_uuid[16],
                                                                std::uint64_t build_id);
    /// x86Code: AOT. Translate the block at a location descriptor without running it.
    void Pretranslate(std::uint64_t location_descriptor);

    /// x86Code Speed 18 (chunked AOT image). Translate a block and append its direct successors (link targets, call
    /// return addresses); false if it was already translated.
    bool PretranslateCollect(std::uint64_t location_descriptor, std::vector<std::uint64_t>& successors);
    /// Drop every registered block and emit from this offset (from the code begin) on.
    static void BeginChunk(SharedCode& shared, std::uint64_t code_off);
    /// Current emit offset from the code begin; total slab capacity.
    static std::uint64_t ChunkCodeEnd(SharedCode& shared);
    static std::uint64_t CodeCapacity(SharedCode& shared);
    /// Release the pages of [from_off, to_off) (a saved chunk).
    static void DiscardCode(SharedCode& shared, std::uint64_t from_off, std::uint64_t to_off);
    /// Save the blocks emitted from code_off on as a chunk file. Returns bytes written.
    static std::size_t SaveChunk(SharedCode& shared, const std::string& path, const std::uint8_t cache_uuid[16],
                                 std::uint64_t build_id, std::uint64_t cache_lo, std::uint64_t cache_hi, std::uint64_t code_off);
    /// Restore a merged constant pool; new constants go from byte offset region_off on.
    static void SetConstantPoolRegion(SharedCode& shared, const std::vector<std::uint8_t>& pool, std::size_t region_off);
    static std::size_t ConstantPoolUsed(SharedCode& shared);
    static std::vector<std::uint8_t> ReadPersistentPool(const std::string& path);
    ~Jit();

    /**
     * Runs the emulated CPU.
     * Cannot be recursively called.
     */
    HaltReason Run();

    /**
     * Step the emulated CPU for one instruction.
     * Cannot be recursively called.
     */
    HaltReason Step();

    /**
     * Clears the code cache of all compiled code.
     * Can be called at any time. Halts execution if called within a callback.
     */
    void ClearCache();

    /**
     * Invalidate the code cache at a range of addresses.
     * @param start_address The starting address of the range to invalidate.
     * @param length The length (in bytes) of the range to invalidate.
     */
    void InvalidateCacheRange(std::uint64_t start_address, std::size_t length);

    /**
     * Reset CPU state to state at startup. Does not clear code cache.
     * Cannot be called from a callback.
     */
    void Reset();

    /**
     * Stops execution in Jit::Run.
     */
    void HaltExecution(HaltReason hr = HaltReason::UserDefined1);

    /**
     * Clears a halt reason from flags.
     * Warning: Only use this if you're sure this won't introduce races.
     */
    void ClearHalt(HaltReason hr = HaltReason::UserDefined1);

    /// Read Stack Pointer
    std::uint64_t GetSP() const;
    /// Modify Stack Pointer
    void SetSP(std::uint64_t value);

    /// Read Program Counter
    std::uint64_t GetPC() const;
    /// Modify Program Counter
    void SetPC(std::uint64_t value);

    /// Read general-purpose register.
    std::uint64_t GetRegister(std::size_t index) const;
    /// Modify general-purpose register.
    void SetRegister(size_t index, std::uint64_t value);

    /// Read all general-purpose registers.
    std::array<std::uint64_t, 31> GetRegisters() const;
    /// Modify all general-purpose registers.
    void SetRegisters(const std::array<std::uint64_t, 31>& value);

    /// Read floating point and SIMD register.
    Vector GetVector(std::size_t index) const;
    /// Modify floating point and SIMD register.
    void SetVector(std::size_t index, Vector value);

    /// Read all floating point and SIMD registers.
    std::array<Vector, 32> GetVectors() const;
    /// Modify all floating point and SIMD registers.
    void SetVectors(const std::array<Vector, 32>& value);

    /// View FPCR.
    std::uint32_t GetFpcr() const;
    /// Modify FPCR.
    void SetFpcr(std::uint32_t value);

    /// View FPSR.
    std::uint32_t GetFpsr() const;
    /// Modify FPSR.
    void SetFpsr(std::uint32_t value);

    /// View PSTATE
    std::uint32_t GetPstate() const;
    /// Modify PSTATE
    void SetPstate(std::uint32_t value);

    /// Clears exclusive state for this core.
    void ClearExclusiveState();

    /**
     * Returns true if Jit::Run was called but hasn't returned yet.
     * i.e.: We're in a callback.
     */
    bool IsExecuting() const;

    /// Debugging: Dump a disassembly all of compiled code to the console.
    void DumpDisassembly() const;

    /*
     * Disassemble the instructions following the current pc and return
     * the resulting instructions as a vector of their string representations.
     */
    std::vector<std::string> Disassemble() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace A64
}  // namespace Dynarmic
