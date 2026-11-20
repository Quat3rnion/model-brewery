#pragma once

#include "glq.h"
#include "trellis_gpu.h"

namespace glq {

// Same operation as glq::linear. All tensor pointers, x/y and scratch are
// device pointers, disjoint and live until caller stream synchronization.
// Scratch contains exactly batch*cols FP32 entries. Signs must have been
// host-validated before upload. Launch errors are returned, not async errors.
trellis_ref::Error linear_gpu(const Layer& device_layer, int batch,
    const float* x, std::size_t x_count, float* y, std::size_t y_count,
    float* scratch, std::size_t scratch_count, trellis_ref::Stream stream);

}  // namespace glq
