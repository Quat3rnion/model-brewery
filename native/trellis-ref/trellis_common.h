#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define TR_HD __host__ __device__
#else
#define TR_HD
#endif

namespace trellis_ref {

// Exact binary16 expansion; 'bits' is the host-order value, not a byte pointer.
TR_HD inline float half_to_float(uint16_t bits) {
    const int e = (bits >> 10) & 31;
    const int f = bits & 1023;
    const float mag = e == 31 ? (f ? NAN : INFINITY)
                              : ::ldexpf(static_cast<float>(e ? 1024 + f : f),
                                         e ? e - 25 : -24);
    return bits & 0x8000 ? -mag : mag;
}

// Normalized Sylvester Hadamard, as used by both bounded reference layouts.
TR_HD inline void fwht(float* x, int n) {
    for (int h = 1; h < n; h *= 2) {
        for (int base = 0; base < n; base += 2 * h) {
            for (int j = base; j < base + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float divisor = ::sqrtf(static_cast<float>(n));
    for (int i = 0; i < n; ++i) x[i] /= divisor;
}

// GLQ uses descending power-of-two blocks without padding.
TR_HD inline void block_fwht(float* x, int n) {
    while (n > 0) {
        int block = 1;
        while (block <= n / 2) block *= 2;
        fwht(x, block);
        n -= block;
        x += block;
    }
}

TR_HD inline uint16_t le16(const uint8_t* bytes) {
    return uint16_t(bytes[0]) | (uint16_t(bytes[1]) << 8);
}

}  // namespace trellis_ref
