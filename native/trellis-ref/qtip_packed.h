#pragma once

#include "qtip.h"
#include "trellis_common.h"

namespace qtip::detail {

// Cornell-RelaxML/qtip e90c6688c8dfae326a3a81b5eb032db7c6680ec0:
// bitshift.py packs cyclic MSB-first windows in little-endian I16 words.
// finetune.py and ldlq.py define the separate HYB byte and element swizzles.
TR_HD inline uint8_t packed_byte(const Packed& w, int tr, int tc, int j) {
    int k = w.shape.K;
    int tile_cols = w.shape.in_features / 16;
    if (w.shape.layout == Layout::reference) {
        size_t tile = size_t(tr) * tile_cols + tc;
        return w.trellis[tile * (32 * k) + (j ^ 1)];
    }
    size_t block = size_t(tr / 2) * (tile_cols / 2) + tc / 2;
    size_t group = ((block * 32 + j / k) * 2 + tc % 2) * 2 + tr % 2;
    return w.trellis[group * k + (k - 1 - j % k)];
}

TR_HD inline uint32_t state(const Packed& w, int r, int c) {
    int ri = r % 16, ci = c % 16;
    int index = (ri * 16 + ci) / w.shape.V;
    if (w.shape.layout == Layout::cuda_swizzled) {
        // Invert the 16x16 tile permutation of upstream LDLQ.
        index = (((ri % 8) * 4 + (ci / 2) % 4) * 2 + ci / 8) * 2 + ri / 8;
    }
    int bit = index * w.shape.K * w.shape.V;
    int length_bytes = 32 * w.shape.K;
    int j = bit / 8;
    uint32_t bytes = uint32_t(packed_byte(w, r / 16, c / 16, j)) << 16;
    bytes |= uint32_t(packed_byte(w, r / 16, c / 16, (j + 1) % length_bytes)) << 8;
    if (bit % 8 + w.shape.L > 16)
        bytes |= packed_byte(w, r / 16, c / 16, (j + 2) % length_bytes);
    return (bytes >> (24 - bit % 8 - w.shape.L)) & ((1u << w.shape.L) - 1);
}

// Upstream 3inst adds two FP16 values and rounds before promoting to float.
TR_HD inline float round_half_sum(float x) {
    if (x == 0.f) return x;
    int exp;
    const float mag = ::fabsf(x);
    ::frexpf(mag, &exp);
    const int shift = exp < -13 ? 24 : 11 - exp;
    const float scaled = ::ldexpf(mag, shift);
    const float lo = ::floorf(scaled);
    const float frac = scaled - lo;
    const bool up = frac > .5f || (frac == .5f && (static_cast<int>(lo) & 1));
    return ::copysignf(::ldexpf(lo + float(up), -shift), x);
}

TR_HD inline float decode(const Packed& w, int r, int c) {
    using trellis_ref::half_to_float;
    const uint32_t s = state(w, r, c);
    const int lane = c % w.shape.V;
    const uint32_t mask = (1u << w.shape.Q) - 1;
    switch (w.shape.mode) {
    case Mode::lut:
        return half_to_float(w.tlut[s * w.shape.V + lane]);
    case Mode::quantlut: {
        const uint32_t h = s * (s + 1u);
        const uint32_t id = (h >> (16 - w.shape.Q)) & mask;
        return half_to_float(w.tlut[id * w.shape.V + lane]);
    }
    case Mode::quantlut_sym: {
        const uint32_t h = s * (s + 1u);
        const uint32_t id = (h >> (15 - w.shape.Q)) & mask;
        const float f = half_to_float(w.tlut[id * w.shape.V + lane]);
        return lane == 0 && (h & 0x8000) ? -f : f;
    }
    case Mode::inst3: {
        const uint32_t h = ((s * 89226354u + 64248484u) & 0x8fff8fffu) ^ 0x3b603b60u;
        return round_half_sum(half_to_float(uint16_t(h >> 16)) + half_to_float(uint16_t(h)));
    }
    case Mode::mad1:
    case Mode::mad2: {
        uint32_t h = s * 34038481u + 76625530u;
        if (w.shape.mode == Mode::mad2) {
            h = s * 264435761u + 1013904223u;
            h += uint32_t((uint64_t(h) * 1664525u) >> 32);
        }
        const int sum = int((h & 255) + ((h >> 8) & 255) + ((h >> 16) & 255) + (h >> 24));
        return float(sum - 510) / 147.800537109375f;
    }
    }
    return 0;
}

inline bool valid_io(const Packed& w, const float* x, float* y, float* work, size_t count) {
    return validate(w) == Status::ok && x && y && work &&
           count >= size_t(w.shape.in_features + w.shape.out_features);
}

}  // namespace qtip::detail
