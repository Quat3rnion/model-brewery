#pragma once

#include "glq.h"
#include "trellis_common.h"

namespace glq::detail {

TR_HD inline float decode_3inst(uint16_t state) {
    const uint32_t pattern = ((state * 89226354u + 64248484u) & 0x8fff8fffu) ^ 0x3b603b60u;
    const float sum = trellis_ref::half_to_float(static_cast<uint16_t>(pattern))
                    + trellis_ref::half_to_float(static_cast<uint16_t>(pattern >> 16));
    // Each addend's exponent is in [13,15] in binary16. Their exact sum fits in
    // FP32, and nonzero sums are normal FP16. Round to nearest, ties to even.
    if (sum == 0) return 0;
    int exponent;
    const float mantissa = ::frexpf(::fabsf(sum), &exponent) * 2048.0f;
    int rounded = static_cast<int>(mantissa);
    const float fraction = mantissa - static_cast<float>(rounded);
    if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1))) ++rounded;
    return ::copysignf(::ldexpf(static_cast<float>(rounded), exponent - 11), sum);
}

TR_HD inline int primary(int bpw) { return bpw > 4 ? 4 : bpw; }

TR_HD inline uint32_t chunk(const uint8_t* group, int lane, int slot, int rate) {
    const uint8_t* p = group + (lane * 4 + slot) * rate;
    uint32_t word = 0;
    for (int byte = 0; byte < rate; ++byte) word |= uint32_t(p[byte]) << (8 * byte);
    return word;
}

// Reverse the GLQ kernel-layout scatter at (r,c): select a 32x32 tile
// group, one of its four 16x16 slots, and an eight-state lane window.
TR_HD inline float weight(const uint8_t* bytes, int rate, int cols, int r, int c) {
    const int group = (r / 32) * (cols / 32) + c / 32;
    const int slot = (r / 16 & 1) + 2 * (c / 16 & 1);
    const int lane = (r & 7) * 4 + (c & 7) / 2;
    const int step = (c & 1) + ((r & 8) >> 2) + ((c & 8) >> 1);
    const uint8_t* packed = bytes + group * 128 * rate;
    uint32_t next = chunk(packed, (lane + 1) & 31, slot, rate);
    if (rate == 1) next = (next << 8) | chunk(packed, (lane + 2) & 31, slot, rate);
    else next >>= 8 * rate - 16;
    const uint64_t window = (uint64_t(chunk(packed, lane, slot, rate)) << 16) | next;
    return decode_3inst(static_cast<uint16_t>(window >> (rate * (8 - step))));
}

TR_HD inline float dot(const Layer& l, const float* x, int row) {
    float total = 0;
    for (int c = 0; c < l.cols; ++c) {
        float w = weight(l.trellis_packed.data, primary(l.bpw), l.cols, row, c);
        if (l.bpw > 4)
            w += l.inv_resid_scale2 * weight(l.trellis_packed2.data, l.bpw - 4, l.cols, row, c);
        total += w * x[c];
    }
    return total * l.wscale;
}

TR_HD inline void transform_input(const Layer& l, const float* x, float* rotated) {
    for (int c = 0; c < l.cols; ++c)
        rotated[c] = x[c] * trellis_ref::half_to_float(trellis_ref::le16(l.sv.data + 2 * c));
    trellis_ref::block_fwht(rotated, l.cols);
}

TR_HD inline void transform_output(const Layer& l, float* out) {
    trellis_ref::block_fwht(out, l.rows);
    for (int r = 0; r < l.rows; ++r)
        out[r] *= trellis_ref::half_to_float(trellis_ref::le16(l.su.data + 2 * r));
}

inline bool valid_io(const Layer& l, int batch, const float* x, std::size_t nx,
                     const float* y, std::size_t ny) {
    return batch > 0 && batch <= max_batch && x && y && x != y
        && nx == std::size_t(batch) * l.cols && ny == std::size_t(batch) * l.rows;
}

}  // namespace glq::detail
