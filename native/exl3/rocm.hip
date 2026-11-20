#include "exl3.h"
#include <hip/hip_runtime.h>
#if !defined(__HIP_PLATFORM_AMD__)
#error "exl3_rocm_forward requires the AMD HIP backend"
#endif
#include <hip/hip_fp16.h>
#include <cmath>
#include "gpu_forward.cuh"

extern "C" int exl3_rocm_forward(const uint8_t* packed, size_t bytes, int rate, int book,
                                 const float* su, const float* sv, const float* x, float* y) {
    if (!packed || !su || !sv || !x || !y || rate < 2 || rate > 16 ||
        book < EXL3_3INST || book > EXL3_MUL1 ||
        ((rate & 1) && (book != EXL3_MUL1 || rate > 7)) ||
        bytes != size_t(1024 * rate)) return -1;
    for (int i = 0; i < 128; ++i)
        if (!std::isfinite(x[i]) || !std::isfinite(su[i]) || !std::isfinite(sv[i])) return -1;

    constexpr size_t vb = 128 * sizeof(float);
    uint8_t* data = nullptr;
    hipError_t rc = hipMalloc(&data, bytes + 4 * vb);
    if (rc != hipSuccess) return static_cast<int>(rc);
    float* du = reinterpret_cast<float*>(data + bytes);
    float* dv = du + 128;
    float* dx = dv + 128;
    float* dy = dx + 128;
    rc = hipMemcpy(data, packed, bytes, hipMemcpyHostToDevice);
    if (rc == hipSuccess) rc = hipMemcpy(du, su, vb, hipMemcpyHostToDevice);
    if (rc == hipSuccess) rc = hipMemcpy(dv, sv, vb, hipMemcpyHostToDevice);
    if (rc == hipSuccess) rc = hipMemcpy(dx, x, vb, hipMemcpyHostToDevice);
    if (rc == hipSuccess) {
        forward_block<<<1, 128>>>(reinterpret_cast<const uint32_t*>(data), rate, book,
                                   du, dv, dx, dy);
        rc = hipGetLastError();
    }
    if (rc == hipSuccess) rc = hipMemcpy(y, dy, vb, hipMemcpyDeviceToHost);
    hipError_t cleanup = hipFree(data);
    return static_cast<int>(rc == hipSuccess ? cleanup : rc);
}
