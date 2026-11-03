/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <vector>

#include <boost/icl/interval_set.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include "dynarmic/common/vma_name.h"
#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
class BlockRangeInformation {
public:
    void AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location);
    void ClearCache();
    tsl::robin_set<IR::LocationDescriptor> InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges);
    // x86Code: approximate heap bytes (bucket arrays + per-page vectors), for memlog.
    std::size_t MemoryBytes() const {
        std::size_t n = page_blocks.bucket_count() * (sizeof(ProgramCounterType) + sizeof(std::vector<u64>) + 8);
        for (const auto& [_, v] : page_blocks) n += v.capacity() * sizeof(u64) + 16;
        return n;
    }

private:
    static constexpr ProgramCounterType kPageShift = 12;
    static constexpr ProgramCounterType PageOf(ProgramCounterType addr) { return addr >> kPageShift; }

    // page index -> blocks that touch that page. x86Code: a plain vector (8 B per block-page) instead of a robin_set
    // per page plus a block -> pages map (~150 MB for SpringBoard's 1M blocks). Invalidating a page drops its whole
    // vector; a multi-page block keeps a stale entry on its other pages, which at worst invalidates a newer block of
    // the same location early (it is retranslated).
    tsl::robin_map<ProgramCounterType, std::vector<u64>, std::hash<ProgramCounterType>, std::equal_to<ProgramCounterType>,
                   MetaAlloc<std::pair<ProgramCounterType, std::vector<u64>>, kMetaRanges>>
        page_blocks;
};

}  // namespace Dynarmic::Backend
