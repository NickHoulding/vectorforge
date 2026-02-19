#ifndef VECTORFORGE_CPU_FEATURES_H
#define VECTORFORGE_CPU_FEATURES_H

#include <cpuid.h>
#include <stdbool.h>

inline bool has_avx512f() {
    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid_max(0, nullptr) < 7) {
        return false;
    }

    __cpuid_count(7, 0, eax, ebx, ecx, edx);

    return (ebx & (1 << 16)) != 0;
}

inline bool has_avx2() {
    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid_max(0, nullptr) < 7) {
        return false;
    }

    __cpuid_count(7, 0, eax, ebx, ecx, edx);

    return (ebx & (1 << 5)) != 0;
}

inline bool os_supports_avx512() {
    unsigned int eax, ebx, ecx, edx;

    __cpuid_count(1, 0, eax, ebx, ecx, edx);

    if (!(ecx & (1 << 27))) {
        return false;
    }

    unsigned int xcr0_lo, xcr0_hi;
    __asm__("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));

    return (xcr0_lo & 0xE6) ==0xE6;
}

inline bool cpu_supports_avx_512() {
    return has_avx512f() && os_supports_avx512();
}

#endif
