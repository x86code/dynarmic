/* LSE / RCPC / RCPC2 tests (designgears/x86code). Each instruction runs on the fastmem path (inline lock-prefixed
 * x86) and on the callback fallback, and is checked against a reference model.
 * SPDX-License-Identifier: 0BSD
 */

#include <array>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "./testenv.h"

using namespace Dynarmic;

namespace {

constexpr u64 kData = 0x1000;

struct Result {
    std::array<u64, 31> x;
    std::array<u8, 32> mem;
};

// Runs `insn` once with the given registers and 32 bytes of memory at kData. fastmem selects the inline path.
Result Run(bool fastmem, u32 insn, const std::array<u64, 31>& regs, const std::array<u8, 32>& mem) {
    Result r{};
    if (fastmem) {
        alignas(4096) static char backing[0x2000];
        std::memset(backing, 0, sizeof(backing));
        A64FastmemTestEnv env{backing};
        A64::UserConfig config{&env};
        config.fastmem_pointer = reinterpret_cast<uintptr_t>(backing);
        config.fastmem_address_space_bits = 64;
        config.recompile_on_fastmem_failure = false;
        A64::Jit jit{config};
        env.write<u32>(0, insn);
        env.write<u32>(4, 0x14000000);  // B .
        std::memcpy(backing + kData, mem.data(), mem.size());
        for (size_t i = 0; i < 31; i++) jit.SetRegister(i, regs[i]);
        jit.SetPC(0);
        env.ticks_left = 2;
        jit.Run();
        for (size_t i = 0; i < 31; i++) r.x[i] = jit.GetRegister(i);
        std::memcpy(r.mem.data(), backing + kData, mem.size());
        REQUIRE(jit.GetPC() == 4);
    } else {
        A64TestEnv env;
        A64::Jit jit{A64::UserConfig{&env}};
        env.code_mem.emplace_back(insn);
        env.code_mem.emplace_back(0x14000000);
        for (size_t i = 0; i < mem.size(); i++) env.MemoryWrite8(kData + i, mem[i]);
        for (size_t i = 0; i < 31; i++) jit.SetRegister(i, regs[i]);
        jit.SetPC(0);
        env.ticks_left = 2;
        jit.Run();
        for (size_t i = 0; i < 31; i++) r.x[i] = jit.GetRegister(i);
        for (size_t i = 0; i < mem.size(); i++) r.mem[i] = env.MemoryRead8(kData + i);
        REQUIRE(jit.GetPC() == 4);
    }
    return r;
}

u32 EncLdOp(u32 size, u32 o3, u32 opc, u32 A, u32 R, u32 Rs, u32 Rn, u32 Rt) {
    return (size << 30) | (0b111000u << 24) | (A << 23) | (R << 22) | (1u << 21) | (Rs << 16) | (o3 << 15) | (opc << 12) | (Rn << 5) | Rt;
}
u32 EncCas(u32 size, u32 L, u32 o0, u32 Rs, u32 Rn, u32 Rt) {
    return (size << 30) | (0b0010001u << 23) | (L << 22) | (1u << 21) | (Rs << 16) | (o0 << 15) | (0x1fu << 10) | (Rn << 5) | Rt;
}
u32 EncCasp(u32 sz, u32 L, u32 o0, u32 Rs, u32 Rn, u32 Rt) {
    return (sz << 30) | (0b0010000u << 23) | (L << 22) | (1u << 21) | (Rs << 16) | (o0 << 15) | (0x1fu << 10) | (Rn << 5) | Rt;
}

u64 Mask(u32 size) { return size == 3 ? ~u64(0) : (u64(1) << (8 << size)) - 1; }
s64 Sext(u64 v, u32 size) { const int sh = 64 - (8 << size); return static_cast<s64>(v << sh) >> sh; }

u64 RefOp(u32 op, u64 old, u64 operand, u32 size) {
    const u64 m = Mask(size);
    old &= m;
    operand &= m;
    switch (op) {
    case 0: return (old + operand) & m;
    case 1: return old & ~operand;
    case 2: return old ^ operand;
    case 3: return old | operand;
    case 4: return Sext(old, size) > Sext(operand, size) ? old : operand;
    case 5: return Sext(old, size) < Sext(operand, size) ? old : operand;
    case 6: return old > operand ? old : operand;
    case 7: return old < operand ? old : operand;
    default: return operand;  // SWP
    }
}

u64 ReadMem(const std::array<u8, 32>& m, u32 size) { u64 v = 0; std::memcpy(&v, m.data(), size_t(1) << size); return v; }

const u64 kPatterns[][2] = {
    {0x0123456789abcdefULL, 0x1111111111111111ULL},
    {0xfedcba9876543210ULL, 0x8000000000000001ULL},
    {0x0000000000000080ULL, 0x000000000000007fULL},
    {0x7fffffff7fff7f7fULL, 0xffffffffffffffffULL},
};

void CheckLdOp(u32 op) {
    const char* names[] = {"LDADD", "LDCLR", "LDEOR", "LDSET", "LDSMAX", "LDSMIN", "LDUMAX", "LDUMIN", "SWP"};
    for (bool fastmem : {true, false}) {
        for (u32 size = 0; size < 4; size++) {
            for (u32 AR = 0; AR < 4; AR++) {
                for (const auto& p : kPatterns) {
                    std::array<u64, 31> regs{};
                    std::array<u8, 32> mem{};
                    std::memcpy(mem.data(), &p[0], 8);
                    std::memset(mem.data() + 8, 0xa5, 24);
                    regs[1] = kData;
                    regs[2] = p[1] | (size < 3 ? 0xdead000000000000ULL : 0);  // upper garbage must be ignored
                    const u32 o3 = op == 8;
                    const u32 opc = op == 8 ? 0 : op;
                    const u32 insn = EncLdOp(size, o3, opc, AR >> 1, AR & 1, 2, 1, 3);
                    INFO(names[op] << " size=" << size << " AR=" << AR << " fastmem=" << fastmem << " insn=" << std::hex << insn);
                    const Result r = Run(fastmem, insn, regs, mem);
                    const u64 old = p[0] & Mask(size);
                    const u64 expect = RefOp(op, p[0], p[1], size);
                    CHECK(r.x[3] == old);
                    CHECK(ReadMem(r.mem, 3) == ((p[0] & ~Mask(size)) | expect));
                    CHECK(r.mem[8] == 0xa5);
                    CHECK(r.x[2] == regs[2]);
                }
            }
        }
        // Rt = XZR (STADD etc.): memory updated, no register written.
        std::array<u64, 31> regs{};
        std::array<u8, 32> mem{};
        regs[1] = kData;
        regs[2] = 5;
        mem[0] = 9;
        const Result r = Run(fastmem, EncLdOp(3, op == 8, op == 8 ? 0 : op, 0, 0, 2, 1, 31), regs, mem);
        CHECK(ReadMem(r.mem, 3) == RefOp(op, 9, 5, 3));
    }
}

}  // namespace

TEST_CASE("A64: LDADD", "[a64][lse]") { CheckLdOp(0); }
TEST_CASE("A64: LDCLR", "[a64][lse]") { CheckLdOp(1); }
TEST_CASE("A64: LDEOR", "[a64][lse]") { CheckLdOp(2); }
TEST_CASE("A64: LDSET", "[a64][lse]") { CheckLdOp(3); }
TEST_CASE("A64: LDSMAX", "[a64][lse]") { CheckLdOp(4); }
TEST_CASE("A64: LDSMIN", "[a64][lse]") { CheckLdOp(5); }
TEST_CASE("A64: LDUMAX", "[a64][lse]") { CheckLdOp(6); }
TEST_CASE("A64: LDUMIN", "[a64][lse]") { CheckLdOp(7); }
TEST_CASE("A64: SWP", "[a64][lse]") { CheckLdOp(8); }

TEST_CASE("A64: CAS", "[a64][lse]") {
    for (bool fastmem : {true, false}) {
        for (u32 size = 0; size < 4; size++) {
            for (u32 ord = 0; ord < 4; ord++) {
                for (bool match : {true, false}) {
                    const u64 cur = 0x8899aabbccddeeffULL;
                    std::array<u64, 31> regs{};
                    std::array<u8, 32> mem{};
                    std::memcpy(mem.data(), &cur, 8);
                    regs[1] = kData;
                    regs[4] = (match ? cur : cur ^ 1) | (size < 3 ? 0x5500000000000000ULL : 0);
                    regs[5] = 0x0123456789abcdefULL;
                    INFO("size=" << size << " ord=" << ord << " match=" << match << " fastmem=" << fastmem);
                    const Result r = Run(fastmem, EncCas(size, ord >> 1, ord & 1, 4, 1, 5), regs, mem);
                    CHECK(r.x[4] == (cur & Mask(size)));
                    const u64 now = ReadMem(r.mem, 3);
                    CHECK(now == (match ? (cur & ~Mask(size)) | (regs[5] & Mask(size)) : cur));
                }
            }
        }
    }
}

TEST_CASE("A64: CASP", "[a64][lse]") {
    for (bool fastmem : {true}) {  // 128-bit CASP needs fastmem; the 32-bit pair also runs on the fallback below
        for (u32 sz = 0; sz < 2; sz++) {
            for (bool match : {true, false}) {
                const u64 lo = 0x1111222233334444ULL, hi = 0x5555666677778888ULL;
                std::array<u64, 31> regs{};
                std::array<u8, 32> mem{};
                std::memcpy(mem.data(), &lo, 8);
                std::memcpy(mem.data() + 8, &hi, 8);
                regs[1] = kData;
                if (sz) {
                    regs[4] = match ? lo : lo + 1;
                    regs[5] = hi;
                } else {
                    regs[4] = (lo & 0xffffffff) | 0xab00000000000000ULL;
                    regs[5] = match ? (lo >> 32) : 0;
                }
                regs[6] = 0xaaaaaaaabbbbbbbbULL;
                regs[7] = 0xccccccccddddddddULL;
                INFO("sz=" << sz << " match=" << match);
                const Result r = Run(fastmem, EncCasp(sz, 1, 1, 4, 1, 6), regs, mem);
                u64 nlo, nhi;
                std::memcpy(&nlo, r.mem.data(), 8);
                std::memcpy(&nhi, r.mem.data() + 8, 8);
                if (sz) {
                    CHECK(r.x[4] == lo);
                    CHECK(r.x[5] == hi);
                    CHECK(nlo == (match ? regs[6] : lo));
                    CHECK(nhi == (match ? regs[7] : hi));
                } else {
                    CHECK(r.x[4] == (lo & 0xffffffff));
                    CHECK(r.x[5] == (lo >> 32));
                    CHECK(nlo == (match ? ((regs[7] << 32) | (regs[6] & 0xffffffff)) : lo));
                    CHECK(nhi == hi);
                }
            }
        }
    }
    // 32-bit pair on the callback path
    std::array<u64, 31> regs{};
    std::array<u8, 32> mem{};
    mem[0] = 1;
    mem[4] = 2;
    regs[1] = kData;
    regs[4] = 1;
    regs[5] = 2;
    regs[6] = 7;
    regs[7] = 8;
    const Result r = Run(false, EncCasp(0, 0, 0, 4, 1, 6), regs, mem);
    CHECK(r.x[4] == 1);
    CHECK(r.x[5] == 2);
    CHECK(r.mem[0] == 7);
    CHECK(r.mem[4] == 8);
}

TEST_CASE("A64: LDAPR", "[a64][rcpc]") {
    for (bool fastmem : {true, false}) {
        for (u32 size = 0; size < 4; size++) {
            const u64 v = 0xf1e2d3c4b5a69788ULL;
            std::array<u64, 31> regs{};
            std::array<u8, 32> mem{};
            std::memcpy(mem.data(), &v, 8);
            regs[1] = kData;
            regs[3] = ~u64(0);
            const Result r = Run(fastmem, (size << 30) | 0x38BFC000u | (1 << 5) | 3, regs, mem);
            CHECK(r.x[3] == (v & Mask(size)));
        }
    }
}

TEST_CASE("A64: LDAPUR, LDAPURS*, STLUR", "[a64][rcpc2]") {
    const u64 v = 0xf1e2d3c4b5a69788ULL;
    for (bool fastmem : {true, false}) {
        for (u32 size = 0; size < 4; size++) {
            for (u32 opc = 0; opc < 4; opc++) {
                if ((size == 3 && opc >= 2) || (size == 2 && opc == 3)) continue;
                std::array<u64, 31> regs{};
                std::array<u8, 32> mem{};
                std::memcpy(mem.data() + 8, &v, 8);
                regs[1] = kData + 16;               // imm9 = -8
                regs[3] = 0x0102030405060708ULL;
                const u32 imm9 = (-8) & 0x1ff;
                const u32 insn = (size << 30) | (0b011001u << 24) | (opc << 22) | (imm9 << 12) | (1 << 5) | 3;
                INFO("size=" << size << " opc=" << opc << " fastmem=" << fastmem);
                const Result r = Run(fastmem, insn, regs, mem);
                u64 now;
                std::memcpy(&now, r.mem.data() + 8, 8);
                if (opc == 0) {
                    CHECK(now == ((v & ~Mask(size)) | (regs[3] & Mask(size))));
                    CHECK(r.x[3] == regs[3]);
                } else if (opc == 1) {
                    CHECK(r.x[3] == (v & Mask(size)));
                } else if (opc == 2) {
                    CHECK(r.x[3] == static_cast<u64>(Sext(v, size)));
                } else {
                    CHECK(r.x[3] == (static_cast<u64>(Sext(v, size)) & 0xffffffff));
                }
            }
        }
    }
}
