/* x86Code: compact fault-rip -> fastmem fallback table (was tsl::robin_map<u64, {u64, u64}>, 32 B/bucket at 50% load:
 * 128 MB for SpringBoard's 1.7M sites). Entries are 8 bytes: the fault rip as an offset from the first rip seen, the
 * distance to the resume rip, and an index into a small table of distinct callbacks. Appends arrive almost in order
 * (code is emitted linearly); lookups (fastmem faults, rare) binary-search the sorted prefix and scan the short tail,
 * sorting first when the tail is long. Not thread-safe; callers hold the emitter lock as before.
 * SPDX-License-Identifier: 0BSD
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <mcl/stdint.hpp>
#include <tsl/robin_map.h>

namespace Dynarmic::Backend::X64 {

struct FastmemPatchInfo {
    u64 resume_rip;
    u64 callback;
};

class FastmemTable {
public:
    void emplace(u64 fault, FastmemPatchInfo info) {
        if (entries.empty() && overflow.empty()) {
            base = fault > 0x80000000u ? fault - 0x80000000u : 0;  // +-2 GB around the code cache
        }
        const u64 off = fault - base;
        const u64 delta = info.resume_rip - fault;
        const u32 cb = CallbackIndex(info.callback);
        if (fault < base || off > 0xFFFFFFFFu || delta > 0xFFFFu || cb > 0xFFFFu) {
            overflow.insert_or_assign(fault, info);
            return;
        }
        const Entry e{static_cast<u32>(off), static_cast<u16>(delta), static_cast<u16>(cb)};
        if (!entries.empty() && e.off < entries.back().off && sorted == entries.size()) {
            // out of order: everything from here on is the unsorted tail
        } else if (sorted == entries.size()) {
            sorted++;
        }
        entries.push_back(e);
    }

    // x86Code (Speed 7): sites of a static AOT image (sorted by off from static_base; cb indexes static_callbacks),
    // usually in a file mapped shared by every process with the same layout. Searched after the dynamic entries.
    struct StaticEntry {
        u32 off;
        u16 delta;
        u16 cb;
    };
    static_assert(sizeof(StaticEntry) == 8);
    void SetStatic(u64 sbase, const StaticEntry* e, size_t n, std::vector<u64> cbs) {
        static_base = sbase;
        static_entries = e;
        static_n = n;
        static_callbacks = std::move(cbs);
    }
    template<typename F>
    void ForEachStatic(F&& f) const {
        for (size_t i = 0; i < static_n; i++) {
            const u64 fault = static_base + static_entries[i].off;
            f(fault, FastmemPatchInfo{fault + static_entries[i].delta, static_callbacks[static_entries[i].cb]});
        }
    }

    const FastmemPatchInfo* Lookup(u64 fault) {
        if (const auto it = overflow.find(fault); it != overflow.end()) {
            found = it->second;
            return &found;
        }
        if (static_n && fault >= static_base && fault - static_base <= 0xFFFFFFFFu) {
            const u32 off = static_cast<u32>(fault - static_base);
            const StaticEntry* it = std::lower_bound(static_entries, static_entries + static_n, off, [](const StaticEntry& e, u32 o) { return e.off < o; });
            if (it != static_entries + static_n && it->off == off) {
                found = FastmemPatchInfo{fault + it->delta, static_callbacks[it->cb]};
                return &found;
            }
        }
        if (fault < base || fault - base > 0xFFFFFFFFu) {
            return nullptr;
        }
        if (entries.size() - sorted > 4096) {
            Sort();
        }
        const u32 off = static_cast<u32>(fault - base);
        auto end = entries.begin() + static_cast<std::ptrdiff_t>(sorted);
        auto it = std::lower_bound(entries.begin(), end, off, [](const Entry& e, u32 o) { return e.off < o; });
        if (it == end || it->off != off) {
            it = std::find_if(end, entries.end(), [off](const Entry& e) { return e.off == off; });
            if (it == entries.end()) {
                return nullptr;
            }
        }
        found = FastmemPatchInfo{fault + it->delta, callbacks[it->cb]};
        return &found;
    }

    template<typename F>
    void ForEach(F&& f) const {
        for (const auto& e : entries) {
            f(base + e.off, FastmemPatchInfo{base + e.off + e.delta, callbacks[e.cb]});
        }
        for (const auto& [k, v] : overflow) {
            f(k, v);
        }
    }

    void clear() {
        entries.clear();
        entries.shrink_to_fit();
        overflow.clear();
        sorted = 0;
        static_n = 0;
        static_entries = nullptr;
    }

    size_t size() const { return entries.size() + overflow.size(); }
    size_t static_size() const { return static_n; }
    size_t MemoryBytes() const { return entries.capacity() * sizeof(Entry) + overflow.bucket_count() * 32 + callbacks.capacity() * 8; }

private:
    struct Entry {
        u32 off;
        u16 delta;
        u16 cb;
    };
    static_assert(sizeof(Entry) == 8);

    u32 CallbackIndex(u64 cb) {
        const auto [it, inserted] = callback_index.try_emplace(cb, static_cast<u32>(callbacks.size()));
        if (inserted) {
            callbacks.push_back(cb);
        }
        return it->second;
    }

    void Sort() {
        std::stable_sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.off < b.off; });
        sorted = entries.size();
    }

    u64 base = 0;
    u64 static_base = 0;
    const StaticEntry* static_entries = nullptr;
    size_t static_n = 0;
    std::vector<u64> static_callbacks;
    size_t sorted = 0;
    std::vector<Entry> entries;
    tsl::robin_map<u64, FastmemPatchInfo> overflow;
    std::vector<u64> callbacks;
    tsl::robin_map<u64, u32> callback_index;
    FastmemPatchInfo found{};
};

}  // namespace Dynarmic::Backend::X64
