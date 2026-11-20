#include "gemv.h"
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define TQ1 brewery_hip_tq1_0_gemv
#define TQ2 brewery_hip_tq2_0_gemv
#define LAUNCH_ERROR hipGetLastError
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define TQ1 brewery_cuda_tq1_0_gemv
#define TQ2 brewery_cuda_tq2_0_gemv
#define LAUNCH_ERROR cudaGetLastError
#endif

// The field order and trit positions match block_tq1_0/block_tq2_0 in
// stock ggml-common.h and dequantize_row_tq*_0 in ggml-quants.c.
template<int Bytes>
__global__ void packed_gemv(BREWERY_TQ_PARAMS) {
    const unsigned lane = threadIdx.x;
    const int blocks = k/256;
    const unsigned char *b = w + size_t(blockIdx.x)*blocks*Bytes;
    float sum = 0;
    for (int i = 0; i < blocks; ++i, b += Bytes) {
        unsigned short bits = b[Bytes-2] | (b[Bytes-1] << 8);
        float scale = __half2float(__ushort_as_half(bits));
        for (unsigned j = lane; j < 256; j += 64) {
            int q;
            if constexpr (Bytes == 54) {
                // 160 entries in five planes of 32, 80 in five of 16,
                // then 16 entries in four planes of 4 (the qh field).
                unsigned base = j < 160 ? 0 : (j < 240 ? 32 : 48);
                unsigned width = j < 160 ? 32 : (j < 240 ? 16 : 4);
                unsigned pos = j < 160 ? j : (j < 240 ? j-160 : j-240);
                constexpr unsigned pow3[] = {1, 3, 9, 27, 81};
                unsigned trit = (b[base + pos%width] * pow3[pos/width]) & 255;
                q = int((3*trit) >> 8) - 1;
            } else {
                unsigned byte = b[(j/128)*32 + j%32];
                q = int((byte >> (2*((j%128)/32))) & 3) - 1;
            }
            sum += (q*scale) * x[i*256+j];
        }
    }
    __shared__ float partial[64];
    partial[lane] = sum;
    for (unsigned n = 32; n; n >>= 1) {
        __syncthreads();
        if (lane < n) partial[lane] += partial[lane+n];
    }
    if (lane == 0) y[blockIdx.x] = partial[0];
}

template<int Bytes> static int launch(BREWERY_TQ_PARAMS) {
    if (!w || !x || !y || m < 1 || k < 256 || k%256) return -1;
    packed_gemv<Bytes><<<m,64>>>(w, x, y, m, k);
    return int(LAUNCH_ERROR());
}
extern "C" int TQ1(BREWERY_TQ_PARAMS) { return launch<54>(w, x, y, m, k); }
extern "C" int TQ2(BREWERY_TQ_PARAMS) { return launch<66>(w, x, y, m, k); }
