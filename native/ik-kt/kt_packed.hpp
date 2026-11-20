#pragma once

#include "kt_matvec.hpp"
#include <cstring>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#define KT_HD __host__ __device__
#else
#define KT_HD
#endif

namespace ik_kt::detail {

// Layout from ggml_row_size and dequantize_row_iq{1,2,3,4}_kt at the pin.
template<Kind K>
constexpr unsigned block_bytes = K == Kind::iq1 ? 56 : K == Kind::iq2 ? 68 : K == Kind::iq3 ? 100 : 128;

template<Kind K>
constexpr unsigned group_size = K == Kind::iq4 ? 4 : 8;

template<Kind K>
inline size_t stride(size_t width) {
    const unsigned n = (width % 256) / 32;
    unsigned tail = 0;
    if constexpr (K == Kind::iq1) tail = n ? 8 + 4 * n + (n < 4 ? 4 * n : 16) : 0;
    if constexpr (K == Kind::iq2) tail = n ? 4 + 8 * n : 0;
    if constexpr (K == Kind::iq3) tail = (n + 1) / 2 + 12 * n;
    if constexpr (K == Kind::iq4) tail = 16 * n;
    // Even at SIZE_MAX width the byte product is <= width/2, plus 4+tail+padding.
    return (4 + width / 256 * block_bytes<K> + tail + 3) & ~size_t(3);
}

inline size_t valid_stride(Kind kind, size_t width) {
    if (!width || width % 32) return 0;
    switch (kind) {
        case Kind::iq1: return stride<Kind::iq1>(width);
        case Kind::iq2: return stride<Kind::iq2>(width);
        case Kind::iq3: return stride<Kind::iq3>(width);
        case Kind::iq4: return stride<Kind::iq4>(width);
    }
    return 0;
}

inline size_t validate(Kind kind, const void *packed, size_t bytes, size_t width,
                       size_t rows, const float *x, size_t x_count, const float *y, size_t y_count) {
    const auto stride = valid_stride(kind, width);
    constexpr auto max_floats = std::numeric_limits<size_t>::max() / sizeof(float);
    if (!stride || !rows || rows > bytes / stride || width > x_count || rows > y_count ||
        width > max_floats || rows > max_floats || !packed || !x || !y) return 0;
    return stride;
}

KT_HD inline uint32_t le16(const uint8_t *p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8;
}

KT_HD inline uint32_t le32(const uint8_t *p) {
    return le16(p) | le16(p + 2) << 16;
}

KT_HD inline float row_scale(const uint8_t *row) {
    const uint32_t u = le32(row);
    static_assert(sizeof(float) == sizeof u && std::numeric_limits<float>::is_iec559);
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_bit_cast(float, u);
#elif defined(__CUDA_ARCH__)
    return __uint_as_float(u);
#else
    float d;
    std::memcpy(&d, &u, sizeof d);
    return d;
#endif
}

KT_HD inline unsigned nibble(unsigned byte, unsigned i) {
    return (byte >> (4 * i)) & 15;
}

KT_HD inline float scale4(unsigned i) {
    constexpr int values[] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return values[i];
}

// Dot one codebook group without building a decoded row. A 256-value block has
// its metadata transposed across 32-value subblocks; a short tail uses its own layout.
template<Kind K, bool Device>
KT_HD inline auto group_dot(const uint8_t *payload, float d, size_t full, unsigned tail,
                           size_t group, const float *x) {
    using Sum = typename std::conditional<Device, float, double>::type;
    constexpr unsigned gs = group_size<K>;
    constexpr unsigned groups = 256 / gs;
    constexpr unsigned per_sub = 32 / gs;
    const size_t block = group / groups;
    const unsigned j = group % groups;
    const unsigned sub = j / per_sub;
    const unsigned g = j % per_sub;
    const bool short_block = block == full;
    const uint8_t *p = payload + block * block_bytes<K>;
    unsigned seed;
    unsigned signs = 0;
    unsigned offset = 4096;
    float scale;
    if constexpr (K == Kind::iq1) {
        const unsigned sh = p[sub];
        const unsigned h = p[(short_block ? 8 + tail * 4 : 40) + j % 16];
        seed = p[8 + j] | (nibble(h, j / 16) << 8) | ((sh << (8 - g)) & 0x1000);
        scale = d * scale4(sh & 15);
    } else if constexpr (K == Kind::iq2) {
        seed = le16(p + 4 + j * 2);
        scale = d * scale4(nibble(p[sub % 4], sub / 4));
    } else if constexpr (K == Kind::iq3) {
        seed = le16(p + (short_block ? 0 : 4) + j * 2);
        if (short_block) {
            signs = p[8 * tail + j];
            scale = (d * 1.01f) * nibble(p[12 * tail + sub / 2], sub % 2);
        } else {
            for (unsigned k = 0; k < gs; ++k) signs |= ((p[68 + g * gs + k] >> sub) & 1) << k;
            scale = (d * 1.01f) * nibble(p[sub % 4], sub / 4);
        }
    } else {
        const uint32_t sh = le32(p + (short_block ? sub * 16 : sub * 4));
        const uint8_t *ql = p + (short_block ? sub * 16 + 4 : 32);
        const unsigned h = short_block ? nibble(ql[8 + g / 2], g % 2)
                                       : nibble(ql[64 + j % 32], j / 32);
        seed = ql[short_block ? g : j] | (h << 8) | (((sh >> (8 + 3 * g)) & 7) << 12);
        scale = d * (int((sh & 255) >> 1) - 64);
        offset += (sh & 1) * 32768;
    }
    uint32_t state = seed + offset;
    Sum sum = 0;
    for (unsigned k = 0; k < gs; ++k) {
        state *= UINT32_C(0xCBAC1FED);
        int code = int((state & 63) + ((state >> 8) & 63) +
                       ((state >> 16) & 63) + ((state >> 24) & 63)) - 126;
        if constexpr (K == Kind::iq3) {
            if (code < 0) code = -code;
            if ((signs >> k) & 1) code = -code;
        }
        const float weight = scale * code;
        sum += Sum(weight) * x[k];
    }
    return sum;
}

} // namespace ik_kt::detail

#undef KT_HD
