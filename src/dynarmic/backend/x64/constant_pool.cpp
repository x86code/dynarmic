/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/x64/constant_pool.h"

#include <cstring>

#include <mcl/assert.hpp>

#include "dynarmic/backend/x64/block_of_code.h"

namespace Dynarmic::Backend::X64 {

ConstantPool::ConstantPool(BlockOfCode& code, size_t size)
        : code(code), insertion_point(0) {
    code.EnsureMemoryCommitted(align_size + size);
    code.int3();
    code.align(align_size);
    pool = std::span<ConstantT>(
        // x86code: the pool is carved from the freshly mapped code space in BlockOfCode's constructor, so it is already
        // zero; a memset touched all 2 MB in every process (only the used part, tens of KB, is ever written).
        reinterpret_cast<ConstantT*>(code.AllocateFromCodeSpace(size, false)), size / align_size);
}

Xbyak::Address ConstantPool::GetConstant(const Xbyak::AddressFrame& frame, u64 lower, u64 upper) {
    const auto constant = ConstantT(lower, upper);
    auto iter = constant_info.find(constant);
    if (iter == constant_info.end()) {
        ASSERT(insertion_point < pool.size());
        ConstantT& target_constant = pool[insertion_point];
        target_constant = constant;
        iter = constant_info.insert({constant, &target_constant}).first;
        ++insertion_point;
    }
    return frame[code.rip + iter->second];
}

void ConstantPool::Restore(const u8* data, size_t used_bytes) {
    ASSERT(used_bytes <= pool.size_bytes());
    ASSERT(used_bytes % align_size == 0);
    if (used_bytes == 0) {
        return;
    }
    std::memcpy(pool.data(), data, used_bytes);
    const size_t new_point = used_bytes / align_size;
    // Never move insertion_point backwards — GenRunCode may already own low slots.
    if (new_point > insertion_point) {
        insertion_point = new_point;
    }
    // Rebuild the lookup map from restored slots so later GetConstant reuses them.
    constant_info.clear();
    // x86Code Speed 18: all-zero slots may be gaps of a merged pool (another builder's region): not reused.
    for (size_t i = 0; i < insertion_point; i++) {
        if (pool[i].first == 0 && pool[i].second == 0) continue;
        constant_info.insert({pool[i], &pool[i]});
    }
}

}  // namespace Dynarmic::Backend::X64
