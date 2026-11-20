#pragma once

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define GYRO_API(name) hip##name
#define GYRO_BACKEND "rocm"
using GyroDeviceProp = hipDeviceProp_t;
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#define GYRO_API(name) cuda##name
#define GYRO_BACKEND "cuda"
using GyroDeviceProp = cudaDeviceProp;
#else
#error "Use hipcc or nvcc for the Gyro device code"
#endif
