#ifndef BREWERY_PACKED_HGN_DETAIL_H
#define BREWERY_PACKED_HGN_DETAIL_H

#include "hgn.h"

#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define HGN_INLINE __host__ __device__ inline
#else
#define HGN_INLINE inline
#endif

namespace hgn::detail {
enum class Store : unsigned { bf16, f32, f16, q4cp, q8g64, ht, i4r, q6g64, fp8g };

// Points only to the selected expert's code/scale/vector slices, plus a
// shared codebook. GPU dispatch copies only these bounded slices.
struct Packed {
    Store store;
    unsigned rows, cols, stride, bits;
    bool paired;
    Bytes code, scale, book, su, sv;
};

HGN_INLINE uint32_t u16(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8);
}
HGN_INLINE uint32_t u32(const uint8_t* p) {
    return u16(p) | (u16(p + 2) << 16);
}
HGN_INLINE double binary(unsigned sign, unsigned exponent, unsigned fraction,
                         int mantissa, int bias, unsigned max_exp) {
    if (exponent == max_exp) {
        if (fraction) return NAN;
        return sign ? -INFINITY : INFINITY;
    }
    unsigned magnitude = fraction + (exponent ? (1u << mantissa) : 0);
    double value = ldexp(double(magnitude), int(exponent ? exponent : 1) - bias - mantissa);
    return sign ? -value : value;
}
HGN_INLINE double half(unsigned u) {
    return binary(u >> 15, u >> 10 & 31, u & 1023, 10, 15, 31);
}
HGN_INLINE double f32(unsigned u) {
    return binary(u >> 31, u >> 23 & 255, u & 0x7fffff, 23, 127, 255);
}
HGN_INLINE double fp8(unsigned u) {
    unsigned exp = u >> 3 & 15;
    return binary(u >> 7, exp, u & 7, 3, 7, 16);
}
HGN_INLINE unsigned nibble(const uint8_t* p, unsigned c) {
    return (p[c / 2] >> (4 * (c % 2))) & 15;
}

HGN_INLINE double ht_weight(const Packed& p, unsigned r, unsigned c) {
    unsigned j = r % 128 / 16, band = r / 128, per_col = 8;
    if (p.paired) {
        unsigned half_rows = p.rows / 2;
        j += 8 * (r / half_rows);
        band = (r % half_rows) / 128;
        per_col = 16;
    }
    unsigned tile = (band * (p.cols / 16) + c / 16) * per_col + j;
    unsigned i = (c % 16 / 8) * 128 + r % 16 * 8 + c % 8;
    const uint8_t* words = p.code.data + size_t(tile) * 32 * p.bits;
    unsigned end = (i + 1) * p.bits;
    unsigned at = (end + 256 * p.bits - 16) % (256 * p.bits);
    unsigned a = u32(words + (at / 32) * 4), shift = at % 32;
    unsigned state = (a << shift) >> 16;
    if (shift > 16) {
        unsigned next = (at / 32 + 1) % (8 * p.bits);
        state |= u32(words + next * 4) >> (48 - shift);
    }
    unsigned h = state * 0x83dcd12du;
    int sum = (h & 255) + (h >> 8 & 255) + (h >> 16 & 255) + (h >> 24);
    // The caller installs a table of exact binary16-rounded codebook values.
    return half(u16(p.book.data + 2 * sum));
}

HGN_INLINE double weight(const Packed& p, unsigned r, unsigned c) {
    if (p.store == Store::ht) return ht_weight(p, r, c);
    const uint8_t* row = p.code.data + size_t(r) * p.stride;
    switch (p.store) {
    case Store::bf16: return f32(u16(row + 2 * c) << 16);
    case Store::f16: return half(u16(row + 2 * c));
    case Store::f32: return f32(u32(row + 4 * c));
    case Store::q8g64:
    case Store::q6g64: {
        bool six = p.store == Store::q6g64;
        unsigned code = six ? nibble(row, c) |
            (((row[p.cols / 2 + c / 4] >> (2 * (c % 4))) & 3) << 4) : row[c];
        const auto* pair = row + (six ? 3 * p.cols / 4 : p.cols) + 4 * (c / 64);
        return code * half(u16(pair)) + half(u16(pair + 2));
    }
    case Store::q4cp:
        return f32(u32(p.book.data + 4 * nibble(row, c))) *
               half(u16(p.scale.data + size_t(r) * ((p.cols / 16 + 15) / 16 * 16) + 2 * (c / 32)));
    case Store::i4r:
        return (int(nibble(row, c)) - 8) *
               half(u16(p.scale.data + size_t(r) * (p.cols / 64) + 2 * (c / 128)));
    default: return NAN;
    }
}

HGN_INLINE bool neg_walsh(unsigned v) {
    v ^= v >> 4;
    v ^= v >> 2;
    v ^= v >> 1;
    return v & 1;
}
HGN_INLINE double rotated(const double* src, const uint8_t* scales, unsigned i, bool input) {
    unsigned base = i & ~127u;
    double sum = 0;
    for (unsigned j = base; j < base + 128; ++j) {
        double x = src[j] * (input ? half(u16(scales + 2 * j)) : 1.0);
        sum += neg_walsh((i & j) & 127) ? -x : x;
    }
    return sum * (1.0 / 11.313708498984761) * (input ? 1.0 : half(u16(scales + 2 * i)));
}

#ifdef HGN_DEVICE_ENABLED
std::vector<double> device_matvec(const Packed& p, const std::vector<double>& x);
std::vector<double> device_gather(const Packed& p);
std::string device_description();
#endif
}

#endif
