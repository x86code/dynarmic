/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <array>
#include <initializer_list>
#include <tuple>
#include <utility>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <mcl/type_traits/integer_of_size.hpp>
#include <xbyak/xbyak.h>

#include "dynarmic/backend/x64/a64_emit_x64.h"
#include "dynarmic/backend/x64/abi.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/emit_x64_memory.h"
#include "dynarmic/backend/x64/exclusive_monitor_friend.h"
#include "dynarmic/backend/x64/perf_map.h"
#include "dynarmic/common/spin_lock_x64.h"
#include "dynarmic/common/x64_disassemble.h"
#include "dynarmic/interface/exclusive_monitor.h"

namespace Dynarmic::Backend::X64 {

using namespace Xbyak::util;

void A64EmitX64::GenMemory128Accessors() {
    code.align();
    memory_read_128 = code.getCurr<void (*)()>();
#ifdef _WIN32
    Devirtualize<&A64::UserCallbacks::MemoryRead128>(conf.callbacks).EmitCallWithReturnPointer(code, [&](Xbyak::Reg64 return_value_ptr, [[maybe_unused]] RegList args) {
        code.mov(code.ABI_PARAM3, code.ABI_PARAM2);
        code.sub(rsp, 8 + 16 + ABI_SHADOW_SPACE);
        code.lea(return_value_ptr, ptr[rsp + ABI_SHADOW_SPACE]);
    });
    code.movups(xmm1, xword[code.ABI_RETURN]);
    code.add(rsp, 8 + 16 + ABI_SHADOW_SPACE);
#else
    code.sub(rsp, 8);
    Devirtualize<&A64::UserCallbacks::MemoryRead128>(conf.callbacks).EmitCall(code);
    if (code.HasHostFeature(HostFeature::SSE41)) {
        code.movq(xmm1, code.ABI_RETURN);
        code.pinsrq(xmm1, code.ABI_RETURN2, 1);
    } else {
        code.movq(xmm1, code.ABI_RETURN);
        code.movq(xmm2, code.ABI_RETURN2);
        code.punpcklqdq(xmm1, xmm2);
    }
    code.add(rsp, 8);
#endif
    code.ret();
    PerfMapRegister(memory_read_128, code.getCurr(), "a64_memory_read_128");

    code.align();
    memory_write_128 = code.getCurr<void (*)()>();
#ifdef _WIN32
    code.sub(rsp, 8 + 16 + ABI_SHADOW_SPACE);
    code.lea(code.ABI_PARAM3, ptr[rsp + ABI_SHADOW_SPACE]);
    code.movaps(xword[code.ABI_PARAM3], xmm1);
    Devirtualize<&A64::UserCallbacks::MemoryWrite128>(conf.callbacks).EmitCall(code);
    code.add(rsp, 8 + 16 + ABI_SHADOW_SPACE);
#else
    code.sub(rsp, 8);
    if (code.HasHostFeature(HostFeature::SSE41)) {
        code.movq(code.ABI_PARAM3, xmm1);
        code.pextrq(code.ABI_PARAM4, xmm1, 1);
    } else {
        code.movq(code.ABI_PARAM3, xmm1);
        code.punpckhqdq(xmm1, xmm1);
        code.movq(code.ABI_PARAM4, xmm1);
    }
    Devirtualize<&A64::UserCallbacks::MemoryWrite128>(conf.callbacks).EmitCall(code);
    code.add(rsp, 8);
#endif
    code.ret();
    PerfMapRegister(memory_write_128, code.getCurr(), "a64_memory_write_128");

    code.align();
    memory_exclusive_write_128 = code.getCurr<void (*)()>();
#ifdef _WIN32
    code.sub(rsp, 8 + 32 + ABI_SHADOW_SPACE);
    code.lea(code.ABI_PARAM3, ptr[rsp + ABI_SHADOW_SPACE]);
    code.lea(code.ABI_PARAM4, ptr[rsp + ABI_SHADOW_SPACE + 16]);
    code.movaps(xword[code.ABI_PARAM3], xmm1);
    code.movaps(xword[code.ABI_PARAM4], xmm2);
    Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive128>(conf.callbacks).EmitCall(code);
    code.add(rsp, 8 + 32 + ABI_SHADOW_SPACE);
#else
    code.sub(rsp, 8);
    if (code.HasHostFeature(HostFeature::SSE41)) {
        code.movq(code.ABI_PARAM3, xmm1);
        code.pextrq(code.ABI_PARAM4, xmm1, 1);
        code.movq(code.ABI_PARAM5, xmm2);
        code.pextrq(code.ABI_PARAM6, xmm2, 1);
    } else {
        code.movq(code.ABI_PARAM3, xmm1);
        code.punpckhqdq(xmm1, xmm1);
        code.movq(code.ABI_PARAM4, xmm1);
        code.movq(code.ABI_PARAM5, xmm2);
        code.punpckhqdq(xmm2, xmm2);
        code.movq(code.ABI_PARAM6, xmm2);
    }
    Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive128>(conf.callbacks).EmitCall(code);
    code.add(rsp, 8);
#endif
    code.ret();
    PerfMapRegister(memory_exclusive_write_128, code.getCurr(), "a64_memory_exclusive_write_128");
}

void A64EmitX64::GenFastmemFallbacks() {
    const std::initializer_list<int> idxes{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const std::array<std::pair<size_t, ArgCallback>, 4> read_callbacks{{
        {8, Devirtualize<&A64::UserCallbacks::MemoryRead8>(conf.callbacks)},
        {16, Devirtualize<&A64::UserCallbacks::MemoryRead16>(conf.callbacks)},
        {32, Devirtualize<&A64::UserCallbacks::MemoryRead32>(conf.callbacks)},
        {64, Devirtualize<&A64::UserCallbacks::MemoryRead64>(conf.callbacks)},
    }};
    const std::array<std::pair<size_t, ArgCallback>, 4> write_callbacks{{
        {8, Devirtualize<&A64::UserCallbacks::MemoryWrite8>(conf.callbacks)},
        {16, Devirtualize<&A64::UserCallbacks::MemoryWrite16>(conf.callbacks)},
        {32, Devirtualize<&A64::UserCallbacks::MemoryWrite32>(conf.callbacks)},
        {64, Devirtualize<&A64::UserCallbacks::MemoryWrite64>(conf.callbacks)},
    }};
    const std::array<std::pair<size_t, ArgCallback>, 4> exclusive_write_callbacks{{
        {8, Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive8>(conf.callbacks)},
        {16, Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive16>(conf.callbacks)},
        {32, Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive32>(conf.callbacks)},
        {64, Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive64>(conf.callbacks)},
    }};

    for (bool ordered : {false, true}) {
        for (int vaddr_idx : idxes) {
            if (vaddr_idx == 4 || vaddr_idx == 15) {
                continue;
            }

            for (int value_idx : idxes) {
                code.align();
                read_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(value_idx));
                if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                    code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                }
                if (ordered) {
                    code.mfence();
                }
                code.call(memory_read_128);
                if (value_idx != 1) {
                    code.movaps(Xbyak::Xmm{value_idx}, xmm1);
                }
                ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(value_idx));
                code.ret();
                PerfMapRegister(read_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)], code.getCurr(), "a64_read_fallback_128");

                code.align();
                write_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                ABI_PushCallerSaveRegistersAndAdjustStack(code);
                if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                    code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                }
                if (value_idx != 1) {
                    code.movaps(xmm1, Xbyak::Xmm{value_idx});
                }
                code.call(memory_write_128);
                if (ordered) {
                    code.mfence();
                }
                ABI_PopCallerSaveRegistersAndAdjustStack(code);
                code.ret();
                PerfMapRegister(write_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)], code.getCurr(), "a64_write_fallback_128");

                code.align();
                exclusive_write_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLoc::RAX);
                if (value_idx != 1) {
                    code.movaps(xmm1, Xbyak::Xmm{value_idx});
                }
                if (code.HasHostFeature(HostFeature::SSE41)) {
                    code.movq(xmm2, rax);
                    code.pinsrq(xmm2, rdx, 1);
                } else {
                    code.movq(xmm2, rax);
                    code.movq(xmm0, rdx);
                    code.punpcklqdq(xmm2, xmm0);
                }
                if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                    code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                }
                code.call(memory_exclusive_write_128);
                ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLoc::RAX);
                code.ret();
                PerfMapRegister(exclusive_write_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)], code.getCurr(), "a64_exclusive_write_fallback_128");

                if (value_idx == 4 || value_idx == 15) {
                    continue;
                }

                for (const auto& [bitsize, callback] : read_callbacks) {
                    code.align();
                    read_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                    ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLocRegIdx(value_idx));
                    if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                        code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                    }
                    if (ordered) {
                        code.mfence();
                    }
                    callback.EmitCall(code);
                    if (value_idx != code.ABI_RETURN.getIdx()) {
                        code.mov(Xbyak::Reg64{value_idx}, code.ABI_RETURN);
                    }
                    ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocRegIdx(value_idx));
                    code.ZeroExtendFrom(bitsize, Xbyak::Reg64{value_idx});
                    code.ret();
                    PerfMapRegister(read_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)], code.getCurr(), fmt::format("a64_read_fallback_{}", bitsize));
                }

                for (const auto& [bitsize, callback] : write_callbacks) {
                    code.align();
                    write_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                    ABI_PushCallerSaveRegistersAndAdjustStack(code);
                    if (vaddr_idx == code.ABI_PARAM3.getIdx() && value_idx == code.ABI_PARAM2.getIdx()) {
                        code.xchg(code.ABI_PARAM2, code.ABI_PARAM3);
                    } else if (vaddr_idx == code.ABI_PARAM3.getIdx()) {
                        code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                        if (value_idx != code.ABI_PARAM3.getIdx()) {
                            code.mov(code.ABI_PARAM3, Xbyak::Reg64{value_idx});
                        }
                    } else {
                        if (value_idx != code.ABI_PARAM3.getIdx()) {
                            code.mov(code.ABI_PARAM3, Xbyak::Reg64{value_idx});
                        }
                        if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                            code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                        }
                    }
                    code.ZeroExtendFrom(bitsize, code.ABI_PARAM3);
                    callback.EmitCall(code);
                    if (ordered) {
                        code.mfence();
                    }
                    ABI_PopCallerSaveRegistersAndAdjustStack(code);
                    code.ret();
                    PerfMapRegister(write_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)], code.getCurr(), fmt::format("a64_write_fallback_{}", bitsize));
                }

                for (const auto& [bitsize, callback] : exclusive_write_callbacks) {
                    code.align();
                    exclusive_write_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                    ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLoc::RAX);
                    if (vaddr_idx == code.ABI_PARAM3.getIdx() && value_idx == code.ABI_PARAM2.getIdx()) {
                        code.xchg(code.ABI_PARAM2, code.ABI_PARAM3);
                    } else if (vaddr_idx == code.ABI_PARAM3.getIdx()) {
                        code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                        if (value_idx != code.ABI_PARAM3.getIdx()) {
                            code.mov(code.ABI_PARAM3, Xbyak::Reg64{value_idx});
                        }
                    } else {
                        if (value_idx != code.ABI_PARAM3.getIdx()) {
                            code.mov(code.ABI_PARAM3, Xbyak::Reg64{value_idx});
                        }
                        if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                            code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                        }
                    }
                    code.ZeroExtendFrom(bitsize, code.ABI_PARAM3);
                    code.mov(code.ABI_PARAM4, rax);
                    code.ZeroExtendFrom(bitsize, code.ABI_PARAM4);
                    callback.EmitCall(code);
                    ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLoc::RAX);
                    code.ret();
                    PerfMapRegister(exclusive_write_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)], code.getCurr(), fmt::format("a64_exclusive_write_fallback_{}", bitsize));
                }
            }
        }
    }
}

#define Axx A64
#include "dynarmic/backend/x64/emit_x64_memory.cpp.inc"
#undef Axx

void A64EmitX64::EmitA64ReadMemory8(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<8, &A64::UserCallbacks::MemoryRead8>(ctx, inst);
}

void A64EmitX64::EmitA64ReadMemory16(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<16, &A64::UserCallbacks::MemoryRead16>(ctx, inst);
}

void A64EmitX64::EmitA64ReadMemory32(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<32, &A64::UserCallbacks::MemoryRead32>(ctx, inst);
}

void A64EmitX64::EmitA64ReadMemory64(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<64, &A64::UserCallbacks::MemoryRead64>(ctx, inst);
}

void A64EmitX64::EmitA64ReadMemory128(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<128, &A64::UserCallbacks::MemoryRead128>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory8(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<8, &A64::UserCallbacks::MemoryWrite8>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory16(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<16, &A64::UserCallbacks::MemoryWrite16>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory32(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<32, &A64::UserCallbacks::MemoryWrite32>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory64(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<64, &A64::UserCallbacks::MemoryWrite64>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory128(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<128, &A64::UserCallbacks::MemoryWrite64>(ctx, inst);
}

void A64EmitX64::EmitA64ClearExclusive(A64EmitContext&, IR::Inst*) {
    code.mov(code.byte[r15 + offsetof(A64JitState, exclusive_state)], u8(0));
}

void A64EmitX64::EmitA64ExclusiveReadMemory8(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<8, &A64::UserCallbacks::MemoryRead8>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<8, &A64::UserCallbacks::MemoryRead8>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveReadMemory16(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<16, &A64::UserCallbacks::MemoryRead16>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<16, &A64::UserCallbacks::MemoryRead16>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveReadMemory32(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<32, &A64::UserCallbacks::MemoryRead32>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<32, &A64::UserCallbacks::MemoryRead32>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveReadMemory64(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<64, &A64::UserCallbacks::MemoryRead64>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<64, &A64::UserCallbacks::MemoryRead64>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveReadMemory128(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<128, &A64::UserCallbacks::MemoryRead128>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<128, &A64::UserCallbacks::MemoryRead128>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory8(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<8, &A64::UserCallbacks::MemoryWriteExclusive8>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<8, &A64::UserCallbacks::MemoryWriteExclusive8>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory16(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<16, &A64::UserCallbacks::MemoryWriteExclusive16>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<16, &A64::UserCallbacks::MemoryWriteExclusive16>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory32(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<32, &A64::UserCallbacks::MemoryWriteExclusive32>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<32, &A64::UserCallbacks::MemoryWriteExclusive32>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory64(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<64, &A64::UserCallbacks::MemoryWriteExclusive64>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<64, &A64::UserCallbacks::MemoryWriteExclusive64>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory128(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<128, &A64::UserCallbacks::MemoryWriteExclusive128>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<128, &A64::UserCallbacks::MemoryWriteExclusive128>(ctx, inst);
    }
}

void A64EmitX64::EmitCheckMemoryAbort(A64EmitContext&, IR::Inst* inst, Xbyak::Label* end) {
    if (!conf.check_halt_on_memory_access) {
        return;
    }

    Xbyak::Label skip;

    const A64::LocationDescriptor current_location{IR::LocationDescriptor{inst->GetArg(0).GetU64()}};

    code.test(dword[r15 + offsetof(A64JitState, halt_reason)], static_cast<u32>(HaltReason::MemoryAbort));
    if (end) {
        code.jz(*end, code.T_NEAR);
    } else {
        code.jz(skip, code.T_NEAR);
    }
    code.mov(rax, current_location.PC());
    code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
    code.ForceReturnFromRunCode();
    code.L(skip);
}

}  // namespace Dynarmic::Backend::X64

// ---- LSE atomics (designgears/x86code) --------------------------------------------------------------------------
// Inline lock-prefixed x86 on the fastmem address (r13 + vaddr). x86 locked RMWs are full barriers, so the
// acquire/release variants need nothing extra. Without fastmem, a non-atomic callback fallback is used.

namespace Dynarmic::Backend::X64 {

namespace {

u64 SizedRead(A64::UserCallbacks* cb, u64 vaddr, size_t log2) {
    switch (log2) {
    case 0: return cb->MemoryRead8(vaddr);
    case 1: return cb->MemoryRead16(vaddr);
    case 2: return cb->MemoryRead32(vaddr);
    default: return cb->MemoryRead64(vaddr);
    }
}

void SizedWrite(A64::UserCallbacks* cb, u64 vaddr, size_t log2, u64 value) {
    switch (log2) {
    case 0: cb->MemoryWrite8(vaddr, static_cast<u8>(value)); break;
    case 1: cb->MemoryWrite16(vaddr, static_cast<u16>(value)); break;
    case 2: cb->MemoryWrite32(vaddr, static_cast<u32>(value)); break;
    default: cb->MemoryWrite64(vaddr, value); break;
    }
}

s64 SignExtendSized(u64 v, size_t log2) {
    const int shift = 64 - (8 << log2);
    return static_cast<s64>(v << shift) >> shift;
}

u64 AtomicRMWFallback(A64::UserCallbacks* cb, u64 vaddr, u64 operand, u64 imm) {
    const size_t log2 = imm & 0xf;
    const u64 mask = log2 == 3 ? ~u64(0) : (u64(1) << (8 << log2)) - 1;
    const u64 old = SizedRead(cb, vaddr, log2) & mask;
    operand &= mask;
    u64 nv = 0;
    switch (imm >> 4) {
    case 0: nv = old + operand; break;
    case 1: nv = old & ~operand; break;
    case 2: nv = old ^ operand; break;
    case 3: nv = old | operand; break;
    case 4: nv = SignExtendSized(old, log2) > SignExtendSized(operand, log2) ? old : operand; break;
    case 5: nv = SignExtendSized(old, log2) < SignExtendSized(operand, log2) ? old : operand; break;
    case 6: nv = old > operand ? old : operand; break;
    case 7: nv = old < operand ? old : operand; break;
    default: nv = operand; break;
    }
    SizedWrite(cb, vaddr, log2, nv);
    return old;
}

u64 AtomicCASFallback(A64::UserCallbacks* cb, u64 vaddr, u64 expected, u64 desired, u64 log2) {
    const u64 mask = log2 == 3 ? ~u64(0) : (u64(1) << (8 << log2)) - 1;
    const u64 old = SizedRead(cb, vaddr, log2) & mask;
    if (old == (expected & mask)) {
        SizedWrite(cb, vaddr, log2, desired);
    }
    return old;
}

Xbyak::Address SizedPtr(BlockOfCode& code, size_t log2, const Xbyak::RegExp& e) {
    switch (log2) {
    case 0: return code.byte[e];
    case 1: return code.word[e];
    case 2: return code.dword[e];
    default: return code.qword[e];
    }
}

Xbyak::Reg SizedReg(const Xbyak::Reg64& r, size_t log2) {
    switch (log2) {
    case 0: return r.cvt8();
    case 1: return r.cvt16();
    case 2: return r.cvt32();
    default: return r;
    }
}

// dst = zero- or sign-extension of the low (8 << log2) bits of src, as a 64-bit value.
void Extend(BlockOfCode& code, const Xbyak::Reg64& dst, const Xbyak::Reg64& src, size_t log2, bool sign) {
    switch (log2) {
    case 0: sign ? code.movsx(dst, src.cvt8()) : code.movzx(dst.cvt32(), src.cvt8()); break;
    case 1: sign ? code.movsx(dst, src.cvt16()) : code.movzx(dst.cvt32(), src.cvt16()); break;
    case 2: sign ? code.movsxd(dst, src.cvt32()) : code.mov(dst.cvt32(), src.cvt32()); break;
    default: if (dst != src) code.mov(dst, src); break;
    }
}

}  // namespace

void A64EmitX64::EmitA64AtomicRMW(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const u8 imm = args[2].GetImmediateU8();
    const size_t log2 = imm & 0xf;
    const unsigned op = imm >> 4;

    if (!conf.fastmem_pointer) {
        ctx.reg_alloc.HostCall(inst, {}, args[0], args[1], args[2]);
        code.mov(code.ABI_PARAM1, mcl::bit_cast<u64>(conf.callbacks));
        code.CallFunction(&AtomicRMWFallback);
        return;
    }

    if (op == 0 || op == 8) {
        const Xbyak::Reg64 value = ctx.reg_alloc.UseScratchGpr(args[1]);
        const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(args[0]);
        const auto mem = SizedPtr(code, log2, FastmemBase(conf, vaddr));
        if (op == 0) {
            code.lock();
            code.xadd(mem, SizedReg(value, log2));
        } else {
            code.xchg(mem, SizedReg(value, log2));  // xchg with memory is implicitly locked
        }
        Extend(code, value, value, log2, false);
        ctx.reg_alloc.DefineValue(inst, value);
        return;
    }

    ctx.reg_alloc.ScratchGpr(HostLoc::RAX);
    const Xbyak::Reg64 operand = ctx.reg_alloc.UseScratchGpr(args[1]);
    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(args[0]);
    const Xbyak::Reg64 tmp = ctx.reg_alloc.ScratchGpr();
    const Xbyak::Reg64 cur = ctx.reg_alloc.ScratchGpr();
    const bool is_signed = op == 4 || op == 5;
    const bool is_minmax = op >= 4;
    if (is_minmax) {
        Extend(code, operand, operand, log2, is_signed);
    } else if (op == 1) {
        code.not_(operand);
    }

    const auto mem = SizedPtr(code, log2, FastmemBase(conf, vaddr));
    switch (log2) {
    case 0: code.movzx(eax, mem); break;
    case 1: code.movzx(eax, mem); break;
    case 2: code.mov(eax, mem); break;
    default: code.mov(rax, mem); break;
    }

    Xbyak::Label loop;
    code.L(loop);
    if (is_minmax) {
        Extend(code, cur, rax, log2, is_signed);
        code.mov(tmp, cur);
        code.cmp(cur, operand);
        switch (op) {
        case 4: code.cmovl(tmp, operand); break;   // SMAX: cur < operand -> operand
        case 5: code.cmovg(tmp, operand); break;   // SMIN
        case 6: code.cmovb(tmp, operand); break;   // UMAX
        case 7: code.cmova(tmp, operand); break;   // UMIN
        }
    } else {
        code.mov(tmp, rax);
        switch (op) {
        case 1: code.and_(tmp, operand); break;  // CLR (operand already inverted)
        case 2: code.xor_(tmp, operand); break;
        case 3: code.or_(tmp, operand); break;
        }
    }
    code.lock();
    code.cmpxchg(mem, SizedReg(tmp, log2));
    code.jnz(loop);

    Extend(code, rax, rax, log2, false);
    ctx.reg_alloc.DefineValue(inst, rax);
}

void A64EmitX64::EmitA64AtomicCAS(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const size_t log2 = args[3].GetImmediateU8();

    if (!conf.fastmem_pointer) {
        ctx.reg_alloc.HostCall(inst, {}, args[0], args[1], args[2]);
        code.mov(code.ABI_PARAM1, mcl::bit_cast<u64>(conf.callbacks));
        code.mov(code.ABI_PARAM5, log2);
        code.CallFunction(&AtomicCASFallback);
        return;
    }

    ctx.reg_alloc.UseScratch(args[1], HostLoc::RAX);
    const Xbyak::Reg64 desired = ctx.reg_alloc.UseGpr(args[2]);
    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(args[0]);
    code.lock();
    code.cmpxchg(SizedPtr(code, log2, FastmemBase(conf, vaddr)), SizedReg(desired, log2));
    Extend(code, rax, rax, log2, false);
    ctx.reg_alloc.DefineValue(inst, rax);
}

void A64EmitX64::EmitA64AtomicCAS128(A64EmitContext& ctx, IR::Inst* inst) {
    ASSERT_MSG(conf.fastmem_pointer, "CASP (128-bit) needs fastmem");
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.ScratchGpr(HostLoc::RAX);
    ctx.reg_alloc.ScratchGpr(HostLoc::RBX);
    ctx.reg_alloc.ScratchGpr(HostLoc::RCX);
    ctx.reg_alloc.ScratchGpr(HostLoc::RDX);
    const Xbyak::Xmm expected = ctx.reg_alloc.UseXmm(args[1]);
    const Xbyak::Xmm desired = ctx.reg_alloc.UseXmm(args[2]);
    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(args[0]);
    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();

    code.movq(rax, expected);
    code.movq(rbx, desired);
    if (code.HasHostFeature(HostFeature::SSE41)) {
        code.pextrq(rdx, expected, 1);
        code.pextrq(rcx, desired, 1);
    } else {
        code.movaps(result, expected);
        code.punpckhqdq(result, result);
        code.movq(rdx, result);
        code.movaps(result, desired);
        code.punpckhqdq(result, result);
        code.movq(rcx, result);
    }
    code.lock();
    code.cmpxchg16b(ptr[FastmemBase(conf, vaddr)]);
    code.movq(result, rax);
    if (code.HasHostFeature(HostFeature::SSE41)) {
        code.pinsrq(result, rdx, 1);
    } else {
        const Xbyak::Xmm hi = ctx.reg_alloc.ScratchXmm();
        code.movq(hi, rdx);
        code.punpcklqdq(result, hi);
    }
    ctx.reg_alloc.DefineValue(inst, result);
}

}  // namespace Dynarmic::Backend::X64
