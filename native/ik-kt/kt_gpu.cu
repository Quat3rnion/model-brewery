#if defined(__HIPCC__) || defined(__HIP__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "kt_packed.hpp"

namespace ik_kt {
namespace {
constexpr unsigned threads = 128;
constexpr unsigned max_grid = 65535;

template<Kind K>
__global__ void packed_matvec(const uint8_t *packed, size_t stride, size_t width,
                              size_t rows, const float *x, float *y) {
    constexpr auto gs = detail::group_size<K>;
    const auto full = width / 256;
    const unsigned tail = (width % 256) / 32;
    __shared__ float partial[threads];
    const unsigned t = threadIdx.x;
    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const auto *p = packed + row * stride;
        const float d = detail::row_scale(p);
        float sum = 0;
        for (size_t g = t; g < width / gs; g += threads) {
            sum += detail::group_dot<K, true>(p + 4, d, full, tail, g, x + gs * g);
        }
        partial[t] = sum;
        __syncthreads();
        // No wave/warp width assumption: the same reduction runs on HIP and CUDA.
        for (unsigned step = threads / 2; step; step >>= 1) {
            if (t < step) partial[t] += partial[t + step];
            __syncthreads();
        }
        if (t == 0) y[row] = partial[0];
        __syncthreads();
    }
}
} // namespace

Status matvec_gpu(Kind kind, const void *packed, size_t bytes, size_t width,
                  size_t rows, const float *x, size_t x_count, float *y, size_t y_count,
                  void *stream) {
    const auto stride = detail::validate(kind, packed, bytes, width, rows, x, x_count, y, y_count);
    if (!stride) return Status::invalid_argument;
    const auto *p = static_cast<const uint8_t *>(packed);
    const unsigned blocks = rows < max_grid ? unsigned(rows) : max_grid;
#if defined(__HIPCC__) || defined(__HIP__)
    auto s = static_cast<hipStream_t>(stream);
#else
    auto s = static_cast<cudaStream_t>(stream);
#endif
    switch (kind) {
        case Kind::iq1: packed_matvec<Kind::iq1><<<blocks, threads, 0, s>>>(p, stride, width, rows, x, y); break;
        case Kind::iq2: packed_matvec<Kind::iq2><<<blocks, threads, 0, s>>>(p, stride, width, rows, x, y); break;
        case Kind::iq3: packed_matvec<Kind::iq3><<<blocks, threads, 0, s>>>(p, stride, width, rows, x, y); break;
        case Kind::iq4: packed_matvec<Kind::iq4><<<blocks, threads, 0, s>>>(p, stride, width, rows, x, y); break;
    }
#if defined(__HIPCC__) || defined(__HIP__)
    return hipGetLastError() == hipSuccess ? Status::ok : Status::launch_failed;
#else
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::launch_failed;
#endif
}

} // namespace ik_kt
