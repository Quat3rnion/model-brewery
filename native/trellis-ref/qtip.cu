#include "trellis_gpu.h"
#include "qtip_packed.h"

namespace qtip {
namespace {

constexpr int threads = 128;

// One block per transform, with all butterflies at each level independent.
// Use caller scratch in global memory so the same kernel accepts either axis.
__device__ void fwht_block(float* x, int n) {
    for (int h = 1; h < n; h *= 2) {
        for (int i = threadIdx.x; i < n / 2; i += blockDim.x) {
            const int j = (i / h) * 2 * h + i % h;
            const float a = x[j], b = x[j + h];
            x[j] = a + b;
            x[j + h] = a - b;
        }
        __syncthreads();
    }
}

__global__ void project_input(Packed packed, const float* x, float* xr) {
    const int n = packed.shape.in_features;
    for (int c = threadIdx.x; c < n; c += blockDim.x)
        xr[c] = x[c] * trellis_ref::half_to_float(packed.su[c]);
    __syncthreads();
    fwht_block(xr, n);
    const float norm = ::sqrtf(float(n));
    for (int c = threadIdx.x; c < n; c += blockDim.x)
        xr[c] /= norm;
}

__global__ void packed_matvec(Packed packed, const float* xr, float* yr) {
    __shared__ float partial[threads];
    const int r = blockIdx.x;
    float sum = 0.f;
    for (int c = threadIdx.x; c < packed.shape.in_features; c += blockDim.x)
        sum += detail::decode(packed, r, c) * xr[c];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int d = threads / 2; d > 0; d /= 2) {
        if (threadIdx.x < unsigned(d)) partial[threadIdx.x] += partial[threadIdx.x + d];
        __syncthreads();
    }
    if (threadIdx.x == 0) yr[r] = partial[0];
}

__global__ void project_output(Packed packed, float* yr, float* y) {
    const int m = packed.shape.out_features;
    fwht_block(yr, m);
    const float norm = ::sqrtf(float(m));
    for (int r = threadIdx.x; r < m; r += blockDim.x)
        y[r] = (yr[r] / norm) * trellis_ref::half_to_float(packed.sv[r]) * packed.scale;
}

}  // namespace

Status linear_gpu(const Packed& packed, const float* x, float* y,
                  float* scratch, size_t capacity, void* stream) {
    if (!detail::valid_io(packed, x, y, scratch, capacity)) return Status::invalid;
    const auto s = static_cast<trellis_ref::Stream>(stream);
    float* xr = scratch;
    float* yr = scratch + packed.shape.in_features;
    project_input<<<1, threads, 0, s>>>(packed, x, xr);
    if (trellis_ref::launch_status() != trellis_ref::success) return Status::launch_failed;
    packed_matvec<<<packed.shape.out_features, threads, 0, s>>>(packed, xr, yr);
    if (trellis_ref::launch_status() != trellis_ref::success) return Status::launch_failed;
    project_output<<<1, threads, 0, s>>>(packed, yr, y);
    return trellis_ref::launch_status() == trellis_ref::success ? Status::ok : Status::launch_failed;
}

}  // namespace qtip
