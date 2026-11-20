#include "exl3.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

#include "gpu_forward.cuh"

extern "C" int exl3_cuda_forward(const uint8_t* packed, size_t bytes, int twice_bits, int book,
                                 const float* su, const float* sv, const float* x, float* y) {
    if (!packed || !su || !sv || !x || !y || twice_bits < 2 || twice_bits > 16 ||
        book < EXL3_3INST || book > EXL3_MUL1 ||
        ((twice_bits & 1) && (book != EXL3_MUL1 || twice_bits > 7)) ||
        bytes != 1024u * twice_bits) return -1;
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(x[i]) || !std::isfinite(su[i]) || !std::isfinite(sv[i])) return -1;

    // Single per-call allocation. No retained state and no expanded weights.
    uint8_t* data = nullptr;
    cudaError_t rc = cudaMalloc(&data, bytes + 4 * vec_bytes);
    if (rc != cudaSuccess) return static_cast<int>(rc);
    float* du = reinterpret_cast<float*>(data + bytes);
    float* dv = du + n;
    float* dx = dv + n;
    float* dy = dx + n;
    rc = cudaMemcpy(data, packed, bytes, cudaMemcpyHostToDevice);
    if (rc == cudaSuccess) rc = cudaMemcpy(du, su, vec_bytes, cudaMemcpyHostToDevice);
    if (rc == cudaSuccess) rc = cudaMemcpy(dv, sv, vec_bytes, cudaMemcpyHostToDevice);
    if (rc == cudaSuccess) rc = cudaMemcpy(dx, x, vec_bytes, cudaMemcpyHostToDevice);
    if (rc == cudaSuccess) {
        forward_block<<<1,n>>>(reinterpret_cast<const uint32_t*>(data), twice_bits, book, du, dv, dx, dy);
        rc = cudaGetLastError();
    }
    if (rc == cudaSuccess) rc = cudaMemcpy(y, dy, vec_bytes, cudaMemcpyDeviceToHost);
    cudaError_t cleanup = cudaFree(data);
    return static_cast<int>(rc == cudaSuccess ? cleanup : rc);
}
