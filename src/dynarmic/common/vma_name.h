// x86Code (RAM 1): names anonymous mappings in /proc/<pid>/smaps ("[anon:a64:...]") for memory attribution, when
// X86CODE_A64_VMANAME is set (diagnostic only; naming keeps the kernel from merging neighbouring VMAs).
#pragma once
#include <cstddef>
#include <cstdlib>
#include <sys/prctl.h>
namespace Dynarmic {
inline void NameVma(void* p, std::size_t len, const char* name) {
    static const bool on = std::getenv("X86CODE_A64_VMANAME") != nullptr;
    if (on && p) {
        prctl(0x53564d41 /* PR_SET_VMA */, 0 /* PR_SET_VMA_ANON_NAME */, reinterpret_cast<unsigned long>(p), len, name);
    }
}
}  // namespace Dynarmic
