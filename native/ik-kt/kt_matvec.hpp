#pragma once

#include <cstddef>
#include <cstdint>

namespace ik_kt {

// ggml_type tensor IDs in ik_llama.cpp 7c00f8788e303bfba2886138e288effb0fd5fbbf.
enum class Kind : uint32_t { iq1 = 158, iq2 = 153, iq3 = 154, iq4 = 155 };
enum class Status { ok, invalid_argument, launch_failed };

// Width must be a nonzero multiple of 32. Zero means unsupported input.
size_t row_bytes(Kind kind, size_t width);

// Contiguous packed [rows, width] times contiguous FP32 [width] -> FP32 [rows].
// Lengths are capacities, in bytes for packed and elements for x/y.
// Buffers must not overlap. Packed rows are little-endian; activations and
// row scales must be finite for a finite numeric result.
Status matvec_cpu(Kind kind, const void *packed, size_t bytes, size_t width,
                  size_t rows, const float *x, size_t x_count, float *y, size_t y_count);

// Same contract, with all three buffers on the current GPU. This enqueues work
// on the given cudaStream_t or hipStream_t (nullptr for the default stream).
// Only immediate launch errors are returned; the caller owns synchronization.
// Link kt_gpu.cu for either CUDA or HIP, not both. No CPU fallback or transfers.
Status matvec_gpu(Kind kind, const void *packed, size_t bytes, size_t width,
                  size_t rows, const float *x, size_t x_count, float *y, size_t y_count,
                  void *stream = nullptr);

} // namespace ik_kt
