// x86Code (RAM 1): names anonymous mappings in /proc/<pid>/smaps ("[anon:a64:...]") for memory attribution
// (tools/debug/memcat.py). RAM 2 (JIT): on by default (X86CODE_A64_VMANAME=0 off); was opt-in. Named mappings merge
// only with neighbours of the same name, and only the JIT's own few large mappings are named.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>
#include <sys/mman.h>
#include <sys/prctl.h>
namespace Dynarmic {
inline void NameVma(void* p, std::size_t len, const char* name) {
    static const bool on = [] { const char* e = std::getenv("X86CODE_A64_VMANAME"); return !(e && std::atoi(e) == 0); }();
    if (on && p) {
        prctl(0x53564d41 /* PR_SET_VMA */, 0 /* PR_SET_VMA_ANON_NAME */, reinterpret_cast<unsigned long>(p), len, name);
    }
}

// RAM 2 (JIT): allocator for the emitter's per-block metadata containers (block descriptors, link tables, fastmem
// sites, relocations). Arrays of 64 KB and up get their own anonymous mapping named "a64:meta:<what>" so smaps
// attributes them (they were anonymous glibc chunks mixed with the guest heap), and a freed array (a table that
// doubled) goes back to the kernel at once instead of staying in glibc's free lists. Smaller ones use malloc.
template<typename T, const char* Name>
struct MetaAlloc {
    using value_type = T;
    static constexpr std::size_t kMin = 64 << 10;
    MetaAlloc() noexcept = default;
    template<typename U>
    MetaAlloc(const MetaAlloc<U, Name>&) noexcept {}
    template<typename U>
    struct rebind {
        using other = MetaAlloc<U, Name>;
    };
    static std::size_t Len(std::size_t n) { return (n * sizeof(T) + 4095) & ~std::size_t{4095}; }
    T* allocate(std::size_t n) {
        if (n * sizeof(T) < kMin) {
            void* p = std::malloc(n * sizeof(T));
            if (!p) throw std::bad_alloc();
            return static_cast<T*>(p);
        }
        void* p = mmap(nullptr, Len(n), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) throw std::bad_alloc();
        NameVma(p, Len(n), Name);
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t n) noexcept {
        if (n * sizeof(T) < kMin) {
            std::free(p);
        } else {
            munmap(p, Len(n));
        }
    }
    template<typename U>
    bool operator==(const MetaAlloc<U, Name>&) const noexcept { return true; }
    template<typename U>
    bool operator!=(const MetaAlloc<U, Name>&) const noexcept { return false; }
};
inline constexpr char kMetaBlocks[] = "a64:meta:blocks";
inline constexpr char kMetaLinks[] = "a64:meta:links";
inline constexpr char kMetaFastmem[] = "a64:meta:fastmem";
inline constexpr char kMetaRelocs[] = "a64:meta:relocs";
inline constexpr char kMetaRanges[] = "a64:meta:ranges";
}  // namespace Dynarmic
