#include "gpu_backend.h"
#include "gyro.h"
#include "layout.h"

namespace gyro {
namespace {

__constant__ const std::uint16_t codebook[] = {
#include "gyro_codebook.inc"
};
static_assert(sizeof(codebook) / sizeof(codebook[0]) == 4096);

constexpr unsigned threads = 128;
constexpr unsigned max_grid = 4096;

unsigned grid(std::size_t items, unsigned per_group = 1) {
    const auto n = 1 + (items - 1) / per_group;
    return n > max_grid ? max_grid : unsigned(n);
}

__global__ void copy_with_signs(const float* input, const std::int8_t* signs,
                                std::size_t n, float* x) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x)
        x[i] = signs && signs[i] == -1 ? -input[i] : input[i];
}

// One thread owns both members of a butterfly. Consecutive launches on
// the caller's stream ensure all writes are visible to the next stage.
__global__ void butterfly(float* x, std::size_t pairs, std::size_t span) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < pairs; i += blockDim.x * gridDim.x) {
        const std::size_t lo = 2 * i - (i & (span - 1));
        const float a = x[lo], b = x[lo + span];
        x[lo] = a + b;
        x[lo + span] = a - b;
    }
}

template<int type>
__global__ void packed_forward(const std::uint8_t* packed, const float* x, float scale,
                               std::size_t rows, std::size_t blocks, float* y) {
    constexpr auto f = detail::layout(type);
    __shared__ float sums[threads];
    const unsigned lane = threadIdx.x;
    for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        float sum = 0.f;
        for (std::size_t b = 0; b < blocks; ++b) {
            const auto* block = packed + (row * blocks + b) * f.bytes;
            sum += detail::weight(block, f, lane, codebook) * (x[128 * b + lane] * scale);
        }
        sums[lane] = sum;
        __syncthreads();
        for (unsigned span = threads / 2; span; span /= 2) {
            if (lane < span) sums[lane] += sums[lane + span];
            __syncthreads();
        }
        if (!lane) y[row] = sums[0];
        __syncthreads();
    }
}

}  // namespace

int forward_gpu(const Matrix& m, const std::uint8_t* packed, const float* input,
                Basis basis, const std::int8_t* signs, float* output,
                float* workspace, void* opaque_stream) noexcept {
    if (!valid(m, basis) || !packed || !input || !output || (signs && !m.rotation_block))
        return GYRO_API(ErrorInvalidValue);
    const auto stream = static_cast<GYRO_API(Stream_t)>(opaque_stream);
    float scale = 1.f;
    if (m.rotation_block && basis == Basis::original) {
        if (!workspace) return GYRO_API(ErrorInvalidValue);
        copy_with_signs<<<grid(m.columns, threads), threads, 0, stream>>>(input, signs, m.columns, workspace);
        auto status = GYRO_API(GetLastError)();
        if (status != GYRO_API(Success)) return status;
        for (std::size_t span = 1; span < m.rotation_block; span *= 2) {
            butterfly<<<grid(m.columns / 2, threads), threads, 0, stream>>>(workspace, m.columns / 2, span);
            status = GYRO_API(GetLastError)();
            if (status != GYRO_API(Success)) return status;
        }
        input = workspace;
        scale = 1.f / std::sqrt(float(m.rotation_block));
    }
    const unsigned groups = grid(m.rows);
    const auto blocks = m.columns / 128;
#define GYRO_LAUNCH(type) \
    case type: packed_forward<type><<<groups, threads, 0, stream>>>(packed, input, scale, m.rows, blocks, output); break
    switch (m.type) {
        GYRO_LAUNCH(144);
        GYRO_LAUNCH(145);
        GYRO_LAUNCH(146);
        GYRO_LAUNCH(147);
        GYRO_LAUNCH(148);
        GYRO_LAUNCH(149);
        GYRO_LAUNCH(150);
        GYRO_LAUNCH(151);
        default: return GYRO_API(ErrorInvalidValue);
    }
#undef GYRO_LAUNCH
    return GYRO_API(GetLastError)();
}

}  // namespace gyro
