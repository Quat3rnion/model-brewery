#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GYRO_HD __host__ __device__
#else
#define GYRO_HD
#endif

namespace gyro::detail {

struct Layout {
    unsigned k, steps, bytes;
};

GYRO_HD inline constexpr Layout layout(int type) {
    switch (type) {
        case 144: return {8, 32, 34};  // TQ2_T
        case 145: return {6, 32, 26};  // TQK6
        case 146: return {7, 32, 30};  // TQK7
        case 147: return {5, 64, 42};  // TQP5
        case 148: return {6, 64, 50};  // TQP6
        case 149: return {7, 64, 58};  // TQP7
        case 150: return {5, 32, 22};  // TQK5
        case 151: return {4, 32, 18};  // TQK4
        default:  return {};
    }
}

GYRO_HD inline float half(std::uint16_t bits) {
    const unsigned exponent = (bits >> 10) & 31;
    const unsigned mantissa = bits & 1023;
    const float value = exponent == 31 ? (mantissa ? NAN : INFINITY) :
        ::ldexpf(float(mantissa + (exponent ? 1024 : 0)), exponent ? int(exponent) - 25 : -24);
    return bits & 0x8000u ? -value : value;
}

// Packed block of 128 weights. Read the scale as little-endian FP16.
GYRO_HD inline float weight(const std::uint8_t* block, Layout format, unsigned col,
                            const std::uint16_t* codebook) {
    const unsigned stride = 128 / format.steps;
    const unsigned step = col / stride;
    const std::uint8_t* stream = block + 2;
    std::uint32_t state;
    if (format.k == 8) {
        state = (std::uint32_t(stream[(step + 31) & 31]) << 8) | stream[step];
    } else {
        const unsigned offset = (format.steps - 1 - step) * format.k;
        const unsigned bytes = format.bytes - 2;
        const unsigned i = offset >> 3;
        state = (std::uint32_t(stream[i]) |
                 (std::uint32_t(stream[(i + 1) % bytes]) << 8) |
                 (std::uint32_t(stream[(i + 2) % bytes]) << 16)) >> (offset & 7);
    }
    const std::uint32_t hash = (state & 0x7fffu) * 0x9e3779b1u;
    const unsigned point = (col % stride) < 2 ? hash >> 21 : (hash >> 10) & 2047u;
    return half(std::uint16_t(block[0] | (block[1] << 8))) * half(codebook[2 * point + (col & 1)]);
}

}  // namespace gyro::detail

#undef GYRO_HD
