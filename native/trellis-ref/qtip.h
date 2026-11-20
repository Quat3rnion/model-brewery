#pragma once

#include <cstddef>
#include <cstdint>

namespace qtip {

enum class Mode : uint8_t { lut, mad1, mad2, inst3, quantlut, quantlut_sym };
enum class Layout : uint8_t { reference, cuda_swizzled };

struct Shape {
    int out_features, in_features;
    int L, K, V, Q;
    Mode mode;
    Layout layout;
};

// Cornell raw layer, rcp=0 only, no bias. Each axis is a power of two
// in [16, 32768], in 16x16 tiles. Trellis uses LE I16 words; SU, SV and
// optional TLUT use binary16 bits. Do not pass HF SV with scale folded in
// unless scale is 1. This is an FP32 matvec, not a CUDA FP16 parity API.
struct Packed {
    Shape shape;
    const uint8_t* trellis;
    const uint16_t* su;
    const uint16_t* sv;
    const uint16_t* tlut;
    size_t trellis_bytes, su_elements, sv_elements, tlut_elements;
    float scale;
};

enum class Status { ok, invalid, launch_failed };

// Structural validation only: buffers must contain finite coefficients and
// be accessible on the appropriate device for the duration of the call.
Status validate(const Packed& weights);

// Buffers must not overlap, with in_features input floats, out_features
// output floats, and in_features + out_features scratch floats.
Status linear(const Packed& weights, const float* input, float* output,
              float* scratch, size_t scratch_elements);

// Same contract with device pointers. Enqueues on the caller's CUDA/HIP
// stream (nullptr for default); caller must check asynchronous errors.
Status linear_gpu(const Packed& weights, const float* input, float* output,
                  float* scratch, size_t scratch_elements, void* stream);

}  // namespace qtip
