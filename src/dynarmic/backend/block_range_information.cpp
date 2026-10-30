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
    const ProgramCounterType page_lo = PageOf(first);
    const ProgramCounterType page_hi = PageOf(last);
    for (ProgramCounterType page = page_lo;; ++page) {
        auto& v = page_blocks[page];
        if (v.empty() || v.back() != location.Value()) {
            v.push_back(location.Value());
        }
        if (page == page_hi) {
            break;
        }
    }
}

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::ClearCache() {
    page_blocks.clear();
}

template<typename ProgramCounterType>
tsl::robin_set<IR::LocationDescriptor> BlockRangeInformation<ProgramCounterType>::InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges) {
    tsl::robin_set<IR::LocationDescriptor> erase_locations;
    std::vector<ProgramCounterType> erase_pages;
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
            if (boost::icl::intersects(ranges, page_iv)) {
                erase_pages.push_back(page);
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
                if (page_blocks.count(page)) {
                    erase_pages.push_back(page);
                }
                if (page == page_hi) {
                    break;
                }
            }
        }
    }
    for (const ProgramCounterType page : erase_pages) {
        auto it = page_blocks.find(page);
        if (it == page_blocks.end()) {
            continue;
        }
        for (const u64 d : it->second) {
            erase_locations.insert(IR::LocationDescriptor{d});
        }
        page_blocks.erase(it);
    }
    return erase_locations;
}

template class BlockRangeInformation<u32>;
template class BlockRangeInformation<u64>;

}  // namespace Dynarmic::Backend
