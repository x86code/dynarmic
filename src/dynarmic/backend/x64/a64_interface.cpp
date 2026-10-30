/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>

#include <boost/icl/interval_set.hpp>
#include <mcl/assert.hpp>
#include <mcl/bit_cast.hpp>
#include <mcl/scope_exit.hpp>

#include "dynarmic/backend/x64/a64_emit_x64.h"
#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/exclusive_monitor_friend.h"
#include "dynarmic/backend/x64/jitstate_info.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/common/x64_disassemble.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opt/passes.h"

namespace Dynarmic::A64 {

using namespace Backend::X64;

static RunCodeCallbacks GenRunCodeCallbacks(A64::UserCallbacks* cb, CodePtr (*LookupBlock)(void* lookup_block_arg), void* arg, const A64::UserConfig& conf) {
    return RunCodeCallbacks{
        std::make_unique<ArgCallback>(LookupBlock, reinterpret_cast<u64>(arg)),
        std::make_unique<ArgCallback>(Devirtualize<&A64::UserCallbacks::AddTicks>(cb)),
        std::make_unique<ArgCallback>(Devirtualize<&A64::UserCallbacks::GetTicksRemaining>(cb)),
        conf.enable_cycle_counting,
    };
}

static std::function<void(BlockOfCode&)> GenRCP(const A64::UserConfig& conf) {
    return [conf](BlockOfCode& code) {
        if (conf.page_table) {
            code.mov(code.r14, mcl::bit_cast<u64>(conf.page_table));
        }
        if (conf.fastmem_pointer) {
            code.mov(code.r13, *conf.fastmem_pointer);
        }
    };
}

static Optimization::PolyfillOptions GenPolyfillOptions(const BlockOfCode& code) {
    return Optimization::PolyfillOptions{
        .sha256 = !code.HasHostFeature(HostFeature::SHA),
        .vector_multiply_widen = true,
    };
}

// x86Code: the translation side of a Jit (code, block map, emitter, fast dispatch table). A plain Jit owns one; with
// UserConfig::shared_translations several Jits (threads) share one. The dispatcher's block lookup finds the running
// Jit through a thread_local (t_running), so the emitted code does not depend on which Jit runs it.
struct Jit::SharedCode final {
    explicit SharedCode(UserConfig conf_)
            : conf(conf_)
            , block_of_code(GenRunCodeCallbacks(conf.callbacks, &LookupBlockThunk, nullptr, conf), JitStateInfo{A64JitState{}}, conf.code_cache_size, GenRCP(conf))
            , emitter(block_of_code, conf, nullptr)
            , polyfill_options(GenPolyfillOptions(block_of_code)) {}

    static CodePtr LookupBlockThunk(void*);

    const UserConfig conf;
    BlockOfCode block_of_code;
    A64EmitX64 emitter;
    Optimization::PolyfillOptions polyfill_options;
    std::recursive_mutex& mutex() { return emitter.shared_mutex; }
    Jit::SharedStats stats;  // under mutex()
    // Bumped under mutex() on every shared Invalidate/Clear so a thread that translated IR
    // without the lock can detect a concurrent invalidation and discard that IR.
    u64 invalidate_gen = 0;
};

Jit::SharedStats Jit::GetSharedStats(const SharedCode& shared) {
    return shared.stats;
}

void Jit::SetCodeCacheAddress(std::uint64_t address) {
    Backend::X64::SetNextCodeCacheAddress(address);
}

std::string Jit::GetMemoryReport(SharedCode& shared) {
    std::unique_lock lock{shared.mutex()};
    return shared.emitter.MemoryReport();
}

std::shared_ptr<Jit::SharedCode> Jit::MakeSharedCode(UserConfig conf) {
    ASSERT(conf.shared_translations);
    return std::make_shared<SharedCode>(conf);
}

std::size_t Jit::SavePersistentCache(SharedCode& shared, const std::string& path,
                                     const std::uint8_t cache_uuid[16], std::uint64_t build_id,
                                     std::uint64_t cache_lo, std::uint64_t cache_hi) {
    return shared.emitter.SavePersistentCache(path, cache_uuid, build_id, cache_lo, cache_hi);  // locks itself
}

std::size_t Jit::LoadPersistentCache(SharedCode& shared, const std::string& path,
                                     const std::uint8_t cache_uuid[16], std::uint64_t build_id,
                                     std::uint64_t cache_lo, std::uint64_t cache_hi, bool shared_image) {
    std::unique_lock lock{shared.mutex()};
    const size_t n = shared.emitter.LoadPersistentCache(path, cache_uuid, build_id, cache_lo, cache_hi, shared_image);
    if (n) {
        shared.stats.blocks += n;
    }
    return n;
}

struct Jit::Impl final {
public:
    Impl(Jit*, UserConfig thread_conf, std::shared_ptr<SharedCode> shared_)
            : shared(shared_ ? std::move(shared_) : std::make_shared<SharedCode>(thread_conf))
            , conf(shared->conf)
            , block_of_code(shared->block_of_code)
            , emitter(shared->emitter)
            , polyfill_options(shared->polyfill_options) {
        ASSERT(conf.page_table_address_space_bits >= 12 && conf.page_table_address_space_bits <= 64);
        jit_state.tpidr_el0_ptr = thread_conf.tpidr_el0;
        jit_state.tpidrro_el0_ptr = thread_conf.tpidrro_el0;
        if (conf.global_monitor) {
            jit_state.excl_address_ptr = GetExclusiveMonitorAddressPointer(conf.global_monitor, thread_conf.processor_id);
            jit_state.excl_value_ptr = GetExclusiveMonitorValuePointer(conf.global_monitor, thread_conf.processor_id);
            jit_state.excl_lock_ptr =
                reinterpret_cast<void*>(const_cast<int*>(GetExclusiveMonitorLockPointer(conf.global_monitor)));
            jit_state.excl_addresses_base = GetExclusiveMonitorAddressPointer(conf.global_monitor, 0);
        }
    }

    ~Impl() = default;

    void Pretranslate(u64 location_descriptor) {
        GetBlock(IR::LocationDescriptor{location_descriptor});
    }

    HaltReason Run() {
        // x86code: Run may be re-entered from a callback (a native function calling back into guest code on the same
        // thread). The caller saves and restores the guest state around the nested run. Code may only be freed at the
        // outermost level: the outer run is still inside a block. A nested run passes cache invalidation requests on.
        const bool nested = run_depth > 0;
        if (!nested) {
            PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));
        }
        // Shared IC invalidate unlinks blocks but does not touch per-thread RSB. Drop stale return
        // predictions whenever we observe a newer invalidate_gen (monotonic, lock-free read).
        if (conf.shared_translations && shared) {
            const u64 gen = shared->invalidate_gen;
            if (gen != rsb_invalidate_gen) {
                rsb_invalidate_gen = gen;
                jit_state.ResetRSB();
            }
        }

        is_executing = true;
        ++run_depth;
        Impl* const outer_running = t_running;
        t_running = this;
        SCOPE_EXIT {
            t_running = outer_running;
            if (--this->run_depth == 0) {
                this->is_executing = false;
            }
        };

        // TODO: Check code alignment

        const CodePtr current_code_ptr = [this] {
            // RSB optimization
            const u32 new_rsb_ptr = (jit_state.rsb_ptr - 1) & A64JitState::RSBPtrMask;
            if (jit_state.GetUniqueHash() == jit_state.rsb_location_descriptors[new_rsb_ptr]) {
                jit_state.rsb_ptr = new_rsb_ptr;
                return reinterpret_cast<CodePtr>(jit_state.rsb_codeptrs[new_rsb_ptr]);
            }

            return GetCurrentBlock();
        }();

        const HaltReason hr = block_of_code.RunCode(&jit_state, current_code_ptr);

        if (!nested) {
            PerformRequestedCacheInvalidation(hr);
        } else if (Has(hr, HaltReason::CacheInvalidation)) {
            Atomic::Or(&jit_state.halt_reason, static_cast<u32>(HaltReason::CacheInvalidation));
        }

        return hr;
    }

    HaltReason Step() {
        ASSERT(!is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));

        is_executing = true;
        Impl* const outer_running = t_running;
        t_running = this;
        SCOPE_EXIT {
            t_running = outer_running;
            this->is_executing = false;
        };

        const HaltReason hr = block_of_code.StepCode(&jit_state, GetCurrentSingleStep());

        PerformRequestedCacheInvalidation(hr);

        return hr;
    }

    void ClearCache() {
        if (conf.shared_translations) {
            // Must use emitter.ClearCache() (O(map size) clear of descriptors/FD/block_ranges), not a
            // full-range InvalidateCacheRanges. Invalidate leaves block_ranges populated (dynarmic TODO)
            // and walks every block under the shared mutex — after minutes of SpringBoard that stalls
            // every thread for seconds and surfaces as BKS/XPC assert failures.
            std::unique_lock lock{shared->mutex()};
            ++shared->invalidate_gen;
            emitter.ClearCache();
            return;
        }
        std::unique_lock lock{invalidation_mutex};
        invalidate_entire_cache = true;
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void InvalidateCacheRange(u64 start_address, size_t length) {
        const auto end_address = static_cast<u64>(start_address + length - 1);
        const auto range = boost::icl::discrete_interval<u64>::closed(start_address, end_address);
        if (conf.shared_translations) {
            std::unique_lock lock{shared->mutex()};
            ++shared->invalidate_gen;
            emitter.InvalidateCacheRanges(boost::icl::interval_set<u64>{range});
            return;
        }
        std::unique_lock lock{invalidation_mutex};
        invalid_cache_ranges.add(range);
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void Reset() {
        ASSERT(!is_executing);
        A64JitState fresh{};
        fresh.tpidr_el0_ptr = jit_state.tpidr_el0_ptr;
        fresh.tpidrro_el0_ptr = jit_state.tpidrro_el0_ptr;
        fresh.excl_address_ptr = jit_state.excl_address_ptr;
        fresh.excl_value_ptr = jit_state.excl_value_ptr;
        fresh.excl_lock_ptr = jit_state.excl_lock_ptr;
        fresh.excl_addresses_base = jit_state.excl_addresses_base;
        jit_state = fresh;
    }

    void HaltExecution(HaltReason hr) {
        Atomic::Or(&jit_state.halt_reason, static_cast<u32>(hr));
    }

    void ClearHalt(HaltReason hr) {
        Atomic::And(&jit_state.halt_reason, ~static_cast<u32>(hr));
    }

    u64 GetSP() const {
        return jit_state.sp;
    }

    void SetSP(u64 value) {
        jit_state.sp = value;
    }

    u64 GetPC() const {
        return jit_state.pc;
    }

    void SetPC(u64 value) {
        jit_state.pc = value;
    }

    u64 GetRegister(size_t index) const {
        if (index == 31)
            return GetSP();
        return jit_state.reg.at(index);
    }

    void SetRegister(size_t index, u64 value) {
        if (index == 31)
            return SetSP(value);
        jit_state.reg.at(index) = value;
    }

    std::array<u64, 31> GetRegisters() const {
        return jit_state.reg;
    }

    void SetRegisters(const std::array<u64, 31>& value) {
        jit_state.reg = value;
    }

    Vector GetVector(size_t index) const {
        return {jit_state.vec.at(index * 2), jit_state.vec.at(index * 2 + 1)};
    }

    void SetVector(size_t index, Vector value) {
        jit_state.vec.at(index * 2) = value[0];
        jit_state.vec.at(index * 2 + 1) = value[1];
    }

    std::array<Vector, 32> GetVectors() const {
        std::array<Vector, 32> ret;
        static_assert(sizeof(ret) == sizeof(jit_state.vec));
        std::memcpy(ret.data(), jit_state.vec.data(), sizeof(jit_state.vec));
        return ret;
    }

    void SetVectors(const std::array<Vector, 32>& value) {
        static_assert(sizeof(value) == sizeof(jit_state.vec));
        std::memcpy(jit_state.vec.data(), value.data(), sizeof(jit_state.vec));
    }

    u32 GetFpcr() const {
        return jit_state.GetFpcr();
    }

    void SetFpcr(u32 value) {
        jit_state.SetFpcr(value);
    }

    u32 GetFpsr() const {
        return jit_state.GetFpsr();
    }

    void SetFpsr(u32 value) {
        jit_state.SetFpsr(value);
    }

    u32 GetPstate() const {
        return jit_state.GetPstate();
    }

    void SetPstate(u32 value) {
        jit_state.SetPstate(value);
    }

    void ClearExclusiveState() {
        jit_state.exclusive_state = 0;
    }

    bool IsExecuting() const {
        return is_executing;
    }

    void DumpDisassembly() const {
        const size_t size = reinterpret_cast<const char*>(block_of_code.getCurr()) - reinterpret_cast<const char*>(block_of_code.GetCodeBegin());
        Common::DumpDisassembledX64(block_of_code.GetCodeBegin(), size);
    }

    std::vector<std::string> Disassemble() const {
        const size_t size = reinterpret_cast<const char*>(block_of_code.getCurr()) - reinterpret_cast<const char*>(block_of_code.GetCodeBegin());
        return Common::DisassembleX64(block_of_code.GetCodeBegin(), size);
    }

    static thread_local Impl* t_running;
    CodePtr GetCurrentBlock() {
        return GetBlock(GetCurrentLocation());
    }

private:

    IR::LocationDescriptor GetCurrentLocation() const {
        return IR::LocationDescriptor{jit_state.GetUniqueHash()};
    }

    CodePtr GetCurrentSingleStep() {
        return GetBlock(A64::LocationDescriptor{GetCurrentLocation()}.SetSingleStepping(true));
    }

    CodePtr GetBlock(IR::LocationDescriptor current_location) {
        // Shared mode: hold the mutex only for map lookup and Emit. IR Translate/opts run unlocked so
        // other threads can still LookupBlock/Emit. A concurrent Invalidate bumps invalidate_gen; if it
        // changed while we translated, discard the IR and retry (the map may already have a fresh block).
        std::unique_lock lock{shared->mutex(), std::defer_lock};
        if (conf.shared_translations) {
            if (const CodePtr ep = emitter.LookupEntrypoint(current_location)) {
                return ep;
            }
        }
        for (;;) {
            if (conf.shared_translations) {
                lock.lock();
            }
            shared->stats.lookups++;
            if (auto block = emitter.GetBasicBlock(current_location))
                return block->entrypoint;
            const u64 gen_at_miss = conf.shared_translations ? shared->invalidate_gen : 0;
            if (conf.shared_translations) {
                lock.unlock();
            }

            timespec t0;
            clock_gettime(CLOCK_MONOTONIC, &t0);

            // JIT Compile (unlocked when shared): MemoryReadCode must not need the shared mutex.
            const auto get_code = [this](u64 vaddr) { return conf.callbacks->MemoryReadCode(vaddr); };
            // x86Code: hand-written objc_msgSend probe (no IR translate / no CallSVC).
            if (conf.inline_objc_msgsend != 0) {
                const A64::LocationDescriptor want{current_location};
                if (want.PC() == conf.inline_objc_msgsend && !want.SingleStepping()) {
                    if (conf.shared_translations) {
                        lock.lock();
                        if (auto block = emitter.GetBasicBlock(current_location))
                            return block->entrypoint;
                        if (shared->invalidate_gen != gen_at_miss) {
                            lock.unlock();
                            continue;
                        }
                    }
                    const auto descriptor = emitter.EmitInlineObjCMsgSend(current_location);
                    if (conf.shared_translations) {
                        shared->stats.blocks++;
                        shared->stats.guest_instructions += 8;
                        shared->stats.code_bytes += descriptor.size;
                    }
                    return descriptor.entrypoint;
                }
            }
            A64::TranslationOptions topts;
            topts.define_unpredictable_behaviour = conf.define_unpredictable_behaviour;
            topts.wall_clock_cntpct = conf.wall_clock_cntpct;
            topts.memory_read_code = get_code;
            topts.fuse_svc_ret = conf.fuse_svc_ret;
            IR::Block ir_block = A64::Translate(A64::LocationDescriptor{current_location}, get_code, topts);
            Optimization::PolyfillPass(ir_block, polyfill_options);
            Optimization::A64CallbackConfigPass(ir_block, conf);
            Optimization::NamingPass(ir_block);
            if (conf.HasOptimization(OptimizationFlag::GetSetElimination) && !conf.check_halt_on_memory_access) {
                Optimization::A64GetSetElimination(ir_block);
                Optimization::DeadCodeElimination(ir_block);
            }
            if (conf.HasOptimization(OptimizationFlag::ConstProp)) {
                Optimization::ConstantPropagation(ir_block);
                Optimization::DeadCodeElimination(ir_block);
            }
            if (conf.HasOptimization(OptimizationFlag::MiscIROpt)) {
                Optimization::A64MergeInterpretBlocksPass(ir_block, conf.callbacks);
            }
            Optimization::IdentityRemovalPass(ir_block);
            Optimization::VerificationPass(ir_block);
            const size_t guest_instructions = ir_block.CycleCount();

            if (conf.shared_translations) {
                lock.lock();
                if (shared->invalidate_gen != gen_at_miss) {
                    // Memory may have changed; another thread may already have a valid block.
                    lock.unlock();
                    continue;
                }
                if (auto block = emitter.GetBasicBlock(current_location))
                    return block->entrypoint;
            }

            constexpr size_t MINIMUM_REMAINING_CODESIZE = 1 * 1024 * 1024;
            if (block_of_code.SpaceRemaining() < MINIMUM_REMAINING_CODESIZE) {
                // Shared code is never freed: other threads may be inside it (or return into it from native calls).
                ASSERT_MSG(!conf.shared_translations, "dynarmic: shared code cache full ({} bytes)", conf.code_cache_size);
                // Immediately evacuate cache
                invalidate_entire_cache = true;
                PerformRequestedCacheInvalidation(HaltReason::CacheInvalidation);
            }
            block_of_code.EnsureMemoryCommitted(MINIMUM_REMAINING_CODESIZE);

            const auto start_code = block_of_code.getCurr<const u8*>();
            // Emit may re-enter GetBlock via LookupBlock (block linking); recursive_mutex allows that.
            const CodePtr entrypoint = emitter.Emit(ir_block).entrypoint;
            timespec t1;
            clock_gettime(CLOCK_MONOTONIC, &t1);
            shared->stats.blocks++;
            shared->stats.guest_instructions += guest_instructions;
            shared->stats.code_bytes += block_of_code.getCurr<const u8*>() - start_code;
            shared->stats.translate_ns += (t1.tv_sec - t0.tv_sec) * 1000000000ull + t1.tv_nsec - t0.tv_nsec;
            return entrypoint;
        }
    }

    void PerformRequestedCacheInvalidation(HaltReason hr) {
        if (Has(hr, HaltReason::CacheInvalidation)) {
            std::unique_lock lock{invalidation_mutex};

            ClearHalt(HaltReason::CacheInvalidation);

            if (!invalidate_entire_cache && invalid_cache_ranges.empty()) {
                return;
            }

            jit_state.ResetRSB();
            if (invalidate_entire_cache) {
                block_of_code.ClearCache();
                emitter.ClearCache();
            } else {
                emitter.InvalidateCacheRanges(invalid_cache_ranges);
            }
            invalid_cache_ranges.clear();
            invalidate_entire_cache = false;
        }
    }

    bool is_executing = false;
    size_t run_depth = 0;
    u64 rsb_invalidate_gen = 0;  // last shared->invalidate_gen we ResetRSB'd for

    std::shared_ptr<SharedCode> shared;
    const UserConfig& conf;
    A64JitState jit_state;
    BlockOfCode& block_of_code;
    A64EmitX64& emitter;
    const Optimization::PolyfillOptions& polyfill_options;

    bool invalidate_entire_cache = false;
    boost::icl::interval_set<u64> invalid_cache_ranges;
    std::mutex invalidation_mutex;
};

thread_local Jit::Impl* Jit::Impl::t_running = nullptr;

CodePtr Jit::SharedCode::LookupBlockThunk(void*) {
    return Jit::Impl::t_running->GetCurrentBlock();
}

Jit::Jit(UserConfig conf)
        : impl(std::make_unique<Jit::Impl>(this, conf, nullptr)) {}

Jit::Jit(UserConfig conf, std::shared_ptr<SharedCode> shared)
        : impl(std::make_unique<Jit::Impl>(this, conf, std::move(shared))) {}

Jit::~Jit() = default;

void Jit::Pretranslate(std::uint64_t location_descriptor) {
    impl->Pretranslate(location_descriptor);
}

std::vector<std::uint64_t> Jit::ReadPersistentDescriptors(const std::string& path, const std::uint8_t cache_uuid[16],
                                                          std::uint64_t build_id) {
    return A64EmitX64::ReadPersistentDescriptors(path, cache_uuid, build_id);
}

HaltReason Jit::Run() {
    return impl->Run();
}

HaltReason Jit::Step() {
    return impl->Step();
}

void Jit::ClearCache() {
    impl->ClearCache();
}

void Jit::InvalidateCacheRange(u64 start_address, size_t length) {
    impl->InvalidateCacheRange(start_address, length);
}

void Jit::Reset() {
    impl->Reset();
}

void Jit::HaltExecution(HaltReason hr) {
    impl->HaltExecution(hr);
}

void Jit::ClearHalt(HaltReason hr) {
    impl->ClearHalt(hr);
}

u64 Jit::GetSP() const {
    return impl->GetSP();
}

void Jit::SetSP(u64 value) {
    impl->SetSP(value);
}

u64 Jit::GetPC() const {
    return impl->GetPC();
}

void Jit::SetPC(u64 value) {
    impl->SetPC(value);
}

u64 Jit::GetRegister(size_t index) const {
    return impl->GetRegister(index);
}

void Jit::SetRegister(size_t index, u64 value) {
    impl->SetRegister(index, value);
}

std::array<u64, 31> Jit::GetRegisters() const {
    return impl->GetRegisters();
}

void Jit::SetRegisters(const std::array<u64, 31>& value) {
    impl->SetRegisters(value);
}

Vector Jit::GetVector(size_t index) const {
    return impl->GetVector(index);
}

void Jit::SetVector(size_t index, Vector value) {
    impl->SetVector(index, value);
}

std::array<Vector, 32> Jit::GetVectors() const {
    return impl->GetVectors();
}

void Jit::SetVectors(const std::array<Vector, 32>& value) {
    impl->SetVectors(value);
}

u32 Jit::GetFpcr() const {
    return impl->GetFpcr();
}

void Jit::SetFpcr(u32 value) {
    impl->SetFpcr(value);
}

u32 Jit::GetFpsr() const {
    return impl->GetFpsr();
}

void Jit::SetFpsr(u32 value) {
    impl->SetFpsr(value);
}

u32 Jit::GetPstate() const {
    return impl->GetPstate();
}

void Jit::SetPstate(u32 value) {
    impl->SetPstate(value);
}

void Jit::ClearExclusiveState() {
    impl->ClearExclusiveState();
}

bool Jit::IsExecuting() const {
    return impl->IsExecuting();
}

void Jit::DumpDisassembly() const {
    return impl->DumpDisassembly();
}

std::vector<std::string> Jit::Disassemble() const {
    return impl->Disassemble();
}

}  // namespace Dynarmic::A64
