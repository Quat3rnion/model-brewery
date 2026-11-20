#pragma once

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
namespace trellis_ref {
using Stream = hipStream_t;
using Error = hipError_t;
inline constexpr Error success = hipSuccess;
inline constexpr Error invalid = hipErrorInvalidValue;
inline Error launch_status() { return hipGetLastError(); }
}
#else
#include <cuda_runtime.h>
namespace trellis_ref {
using Stream = cudaStream_t;
using Error = cudaError_t;
inline constexpr Error success = cudaSuccess;
inline constexpr Error invalid = cudaErrorInvalidValue;
inline Error launch_status() { return cudaGetLastError(); }
}
#endif
