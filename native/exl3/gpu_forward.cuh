#pragma once

namespace {
constexpr int n = 128;
constexpr size_t vec_bytes = n * sizeof(float);

// Inverse of the 16x16 fragment permutation in v1.6.7 trellis.py.
// Each state ends floor((position+1)*bits) bits into its circular tile.
__device__ uint32_t tile_state(const uint32_t* tile, int rate, int r, int c) {
    int lane = 4 * (c & 7) + ((r & 7) >> 1);
    int element = (r & 1) + ((r & 8) >> 2) + ((c & 8) >> 1);
    int start = ((8 * lane + element + 1) * rate >> 1) - 16;
    int words = 4 * rate;
    if (start < 0) start += 32 * words;
    int w = start >> 5;
    int next = w + 1 == words ? 0 : w + 1;
    // Packed words are little-endian, but bits within each run MSB first.
    uint64_t pair = (static_cast<uint64_t>(tile[w]) << 32) | tile[next];
    return static_cast<uint32_t>((pair >> (48 - (start & 31))) & 0xffffu);
}

__device__ float decode(uint32_t state, int book) {
    if (book == EXL3_MUL1) {
        uint32_t h = state * 0x83dcd12du;
        int s = 1024 + (h & 255) + ((h >> 8) & 255) + ((h >> 16) & 255) + (h >> 24);
        return __half2float(__hfma(__int2half_rn(s), __ushort_as_half(0x1eee),
                                   __ushort_as_half(0xc931)));
    }
    uint32_t h = (book == EXL3_MCG) ? state * 0xcbac1fedu : state * 89226354u + 64248484u;
    h = (h & 0x8fff8fffu) ^ 0x3b603b60u;
    return __half2float(__hadd(__ushort_as_half(h & 0xffff), __ushort_as_half(h >> 16)));
}

// One thread per butterfly pair. The barrier also publishes v before return.
__device__ void hadamard(float* v, int tid) {
    __syncthreads();
    for (int step = 1; step < n; step *= 2) {
        if (tid < n / 2) {
            int lo = ((tid & ~(step - 1)) << 1) + (tid & (step - 1));
            float a = v[lo], b = v[lo + step];
            v[lo] = a + b;
            v[lo + step] = a - b;
        }
        __syncthreads();
    }
}

__global__ void forward_block(const uint32_t* packed, int rate, int book,
                              const float* su, const float* sv, const float* x, float* y) {
    int c = threadIdx.x;
    __shared__ float v[n];
    v[c] = x[c] * su[c];
    hadamard(v, c);
    float sum = 0;
    for (int r = 0; r < n; ++r) {
        int tile = (r / 16) * 8 + c / 16;
        uint32_t state = tile_state(packed + tile * (4 * rate), rate, r & 15, c & 15);
        sum = fmaf(v[r], decode(state, book), sum);
    }
    __syncthreads();  // All columns must finish reading the input transform.
    v[c] = sum;
    hadamard(v, c);
    y[c] = (v[c] / n) * sv[c];  // Both normalizing 1/sqrt(n) factors.
}
} // namespace
