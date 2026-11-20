#pragma once

#include <cstddef>
#include <cstdint>

namespace glq {

enum class Layout { unknown, kernel_3inst_block_diagonal };

struct Bytes {
    const uint8_t* data = nullptr;
    std::size_t size = 0;
};

// GLQ v0.8.38 checkpoint components after parsing quantization_config.
// All tensor pointers reference original little-endian native bytes.
struct Layer {
    Layout layout = Layout::unknown;
    int rows = 0, cols = 0, bpw = 0;
    Bytes trellis_packed, trellis_packed2, su, sv;
    float wscale = 0, inv_resid_scale2 = 0;
};

// Host validation includes SU/SV values. The device call checks metadata/sizes
// only; call validate on the host view before uploading the same tensor bytes.
bool validate_metadata(const Layer& layer);
bool validate(const Layer& layer);

// Contiguous original-basis FP32 [batch, cols] -> [batch, rows], no bias.
// Buffers must not alias. No dense weights are allocated by this path.
bool linear(const Layer& layer, int batch, const float* x, std::size_t x_count,
            float* y, std::size_t y_count);

inline constexpr int max_axis = 16384, max_batch = 8;

}  // namespace glq
