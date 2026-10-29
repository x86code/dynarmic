/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/block_range_information.h"

#include <boost/icl/interval_set.hpp>
#include <mcl/stdint.hpp>
#include <tsl/robin_set.h>

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location) {
    const ProgramCounterType first = boost::icl::first(range);
    const ProgramCounterType last = boost::icl::last(range);
    if (last < first) {
        return;
    }
    auto& pages = block_pages[location];
    const ProgramCounterType page_lo = PageOf(first);
    const ProgramCounterType page_hi = PageOf(last);
    // Cap per-call growth; a single block rarely spans more than a few pages.
    const std::size_t span = static_cast<std::size_t>(page_hi - page_lo) + 1;
    pages.reserve(pages.size() + span);
    for (ProgramCounterType page = page_lo;; ++page) {
        auto& set = page_blocks[page];
        const auto [it, inserted] = set.insert(location);
        (void)it;
        if (inserted) {
            pages.push_back(page);
        }
        if (page == page_hi) {
            break;
        }
    }
}

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::ClearCache() {
    page_blocks.clear();
    block_pages.clear();
}

template<typename ProgramCounterType>
tsl::robin_set<IR::LocationDescriptor> BlockRangeInformation<ProgramCounterType>::InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges) {
    tsl::robin_set<IR::LocationDescriptor> erase_locations;

    // Never walk page indices across a huge span (IALLU / wide IVAU used to pass [0,~0] and
    // would spin forever with a page loop). For large coverage, scan the live page map instead.
    std::uint64_t cover = 0;
    for (auto invalidate_interval : ranges) {
        const ProgramCounterType first = boost::icl::first(invalidate_interval);
        const ProgramCounterType last = boost::icl::last(invalidate_interval);
        if (last < first) {
            continue;
        }
        cover += static_cast<std::uint64_t>(PageOf(last) - PageOf(first) + 1);
    }
    constexpr std::uint64_t kMaxPageWalk = 4096;  // 16 MB of 4K pages

    if (cover > kMaxPageWalk) {
        for (auto it = page_blocks.begin(); it != page_blocks.end(); ++it) {
            const ProgramCounterType page = it->first;
            const ProgramCounterType lo = page << kPageShift;
            // last byte of page; avoid overflow on the final page index
            const ProgramCounterType hi = lo + (static_cast<ProgramCounterType>(1) << kPageShift) - 1;
            const auto page_iv = boost::icl::discrete_interval<ProgramCounterType>::closed(lo, hi);
            if (!boost::icl::intersects(ranges, page_iv)) {
                continue;
            }
            for (const auto& descriptor : it.value()) {
                erase_locations.insert(descriptor);
            }
        }
    } else {
        for (auto invalidate_interval : ranges) {
            const ProgramCounterType first = boost::icl::first(invalidate_interval);
            const ProgramCounterType last = boost::icl::last(invalidate_interval);
            if (last < first) {
                continue;
            }
            const ProgramCounterType page_lo = PageOf(first);
            const ProgramCounterType page_hi = PageOf(last);
            for (ProgramCounterType page = page_lo;; ++page) {
                auto it = page_blocks.find(page);
                if (it != page_blocks.end()) {
                    for (const auto& descriptor : it->second) {
                        erase_locations.insert(descriptor);
                    }
                }
                if (page == page_hi) {
                    break;
                }
            }
        }
    }

    for (const auto& descriptor : erase_locations) {
        auto bp = block_pages.find(descriptor);
        if (bp == block_pages.end()) {
            continue;
        }
        for (ProgramCounterType page : bp.value()) {
            auto pb = page_blocks.find(page);
            if (pb == page_blocks.end()) {
                continue;
            }
            pb.value().erase(descriptor);
            if (pb.value().empty()) {
                page_blocks.erase(pb);
            }
        }
        block_pages.erase(bp);
    }
    return erase_locations;
}

template class BlockRangeInformation<u32>;
template class BlockRangeInformation<u64>;

}  // namespace Dynarmic::Backend
