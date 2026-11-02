/* x86Code Speed 34: inline FRINTA / FRINTX for the x64 backend (scalar and vector, 32/64-bit).
 * SPDX-License-Identifier: 0BSD
 */
#pragma once

#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/emit_x64.h"
#include "dynarmic/common/fp/rounding_mode.h"

namespace Dynarmic::Backend::X64 {

// ROUNDSS/SD/PS/PD immediate for an ARM rounding mode; ties-away is built on truncation. Bit 3 suppresses the
// precision exception (MXCSR.PE -> FPSR.IXC): set for the non-exact forms only, so FRINTX raises IXC when inexact
// and FRINTN/P/M/Z/A never do.
inline int FrintImmediate(FP::RoundingMode mode, bool exact) {
    int imm = 0;
    switch (mode) {
    case FP::RoundingMode::ToNearest_TieEven: imm = 0b00; break;
    case FP::RoundingMode::TowardsPlusInfinity: imm = 0b10; break;
    case FP::RoundingMode::TowardsMinusInfinity: imm = 0b01; break;
    case FP::RoundingMode::TowardsZero: imm = 0b11; break;
    case FP::RoundingMode::ToNearest_TieAwayFromZero: imm = 0b11; break;
    default: return -1;
    }
    return imm | (exact ? 0 : 0b1000);
}

// Round `result` (holding x, all lanes for vector=true) in place. Uses two scratch xmms plus xmm0.
// Ties-away: t = trunc(x); d = x - t (exact); result = t + (|d| >= 0.5 ? copysign(1, x) : copysign(0, x)).
// Lanes with |x| >= 2^52 (2^23) - already integral, inf, NaN - are masked to d = 0 with an integer compare
// before the subtraction, so no lane raises a spurious invalid / inexact flag, -0 stays -0 in every rounding mode and NaN
// lanes pass through quiet.
template<size_t fsize, bool vector>
void EmitFrint(BlockOfCode& code, Xbyak::Xmm result, Xbyak::Xmm a, Xbyak::Xmm b, FP::RoundingMode mode, bool exact) {
    using namespace Xbyak::util;
    const int imm = FrintImmediate(mode, exact);
    const bool tie_away = mode == FP::RoundingMode::ToNearest_TieAwayFromZero;
    auto rep = [](u64 v) -> u64 { return fsize == 32 ? (v | (v << 32)) : v; };
    const u64 sign = rep(fsize == 64 ? 0x8000000000000000 : 0x80000000);
    const u64 abs = rep(fsize == 64 ? 0x7FFFFFFFFFFFFFFF : 0x7FFFFFFF);
    const u64 thr = rep(fsize == 64 ? 0x432FFFFFFFFFFFFF : 0x4AFFFFFF);  // largest value below 2^52 / 2^23
    const u64 half = rep(fsize == 64 ? 0x3FE0000000000000 : 0x3F000000);
    const u64 one = rep(fsize == 64 ? 0x3FF0000000000000 : 0x3F800000);

    // packed ops for vector lanes; scalar ops for scalars (upper lanes may hold anything, e.g. SNaN)
    auto sub = [&](Xbyak::Xmm x, Xbyak::Xmm y) {
        if constexpr (vector && fsize == 64) code.subpd(x, y);
        else if constexpr (vector) code.subps(x, y);
        else if constexpr (fsize == 64) code.subsd(x, y);
        else code.subss(x, y);
    };
    auto add = [&](Xbyak::Xmm x, Xbyak::Xmm y) {
        if constexpr (vector && fsize == 64) code.addpd(x, y);
        else if constexpr (vector) code.addps(x, y);
        else if constexpr (fsize == 64) code.addsd(x, y);
        else code.addss(x, y);
    };
    auto cmple = [&](Xbyak::Xmm x, Xbyak::Xmm y) {
        if constexpr (vector && fsize == 64) code.cmplepd(x, y);
        else if constexpr (vector) code.cmpleps(x, y);
        else if constexpr (fsize == 64) code.cmplesd(x, y);
        else code.cmpless(x, y);
    };
    if (tie_away) {
        code.movaps(a, result);  // x
    }
    if constexpr (vector) {
        if constexpr (fsize == 64) {
            code.roundpd(result, result, imm);
        } else {
            code.roundps(result, result, imm);
        }
    } else {
        if constexpr (fsize == 64) {
            code.roundsd(result, result, imm);
        } else {
            code.roundss(result, result, imm);
        }
    }
    if (!tie_away) {
        return;
    }
    code.movaps(b, code.Const(xword, abs, abs));
    code.andps(b, a);  // |x|
    if constexpr (fsize == 64) {
        code.pcmpgtq(b, code.Const(xword, thr, thr));
    } else {
        code.pcmpgtd(b, code.Const(xword, thr, thr));
    }
    code.movaps(xmm0, b);  // big
    code.andnps(b, a);  // x & ~big
    code.andnps(xmm0, result);  // t & ~big
    sub(b, xmm0);
    code.movaps(xmm0, code.Const(xword, abs, abs));
    code.andps(xmm0, b);  // |d|
    code.movaps(a, code.Const(xword, half, half));
    cmple(a, xmm0);  // 0.5 <= |d|
    code.andps(a, code.Const(xword, one, one));  // mask ? 1 : +0
    code.movaps(b, code.Const(xword, sign, sign));
    code.andps(b, result);  // sign(t) = sign(x)
    code.orps(b, a);  // mask ? copysign(1, x) : copysign(0, x)
    add(result, b);  // exact; x + same-signed zero = x in every rounding mode
}

}  // namespace Dynarmic::Backend::X64
