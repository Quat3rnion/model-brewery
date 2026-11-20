// Standalone numerical comparison for one packed block. No CPU fallback.
#include "exl3.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    cudaError_t status = cudaGetDevice(&device);
    if (status == cudaSuccess) status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA device: %s\n", cudaGetErrorString(status));
        return 1;
    }
    std::printf("device=%s sm=%d.%d\n", prop.name, prop.major, prop.minor);

    std::array<float,128> x{}, su{}, sv{}, cpu{}, gpu{};
    std::mt19937 rng(0xE313);
    for (size_t i = 0; i < x.size(); ++i) {
        // Distinct signed input and output scales, not just rotation signs.
        x[i] = (static_cast<int>(rng() % 2049) - 1024) / 1024.0f;
        su[i] = ((i % 3) ? -1 : 1) * (9 + static_cast<int>(i % 17)) / 16.0f;
        sv[i] = ((i % 5) ? 1 : -1) * (7 + static_cast<int>(i % 19)) / 16.0f;
    }
    int cases = 0;
    float worst_abs = 0, worst_rel = 0;
    for (int rate = 2; rate <= 16; ++rate) {
        for (int book = 0; book < 3; ++book) {
            if ((rate & 1) && (rate > 7 || book != 2)) continue;
            std::vector<uint8_t> packed(1024 * rate);
            for (auto& b : packed) b = static_cast<uint8_t>(rng());
            gpu.fill(std::numeric_limits<float>::quiet_NaN());
            int ref = exl3_forward(packed.data(), packed.size(), rate, book,
                                   su.data(), sv.data(), x.data(), cpu.data());
            int rc = exl3_cuda_forward(packed.data(), packed.size(), rate, book,
                                       su.data(), sv.data(), x.data(), gpu.data());
            if (ref || rc) {
                std::fprintf(stderr, "rate=%d/2 book=%d cpu=%d cuda=%d\n", rate, book, ref, rc);
                return 1;
            }
            float max_ref = 0, max_err = 0;
            for (size_t i = 0; i < cpu.size(); ++i) {
                float err = std::abs(cpu[i] - gpu[i]);
                if (!std::isfinite(err) || err > 2e-5f + 2e-6f * std::abs(cpu[i])) {
                    std::fprintf(stderr, "rate=%d/2 book=%d index=%zu cpu=%g cuda=%g\n",
                                 rate, book, i, cpu[i], gpu[i]);
                    return 1;
                }
                max_err = std::max(max_err, err);
                max_ref = std::max(max_ref, std::abs(cpu[i]));
            }
            worst_abs = std::max(worst_abs, max_err);
            worst_rel = std::max(worst_rel, max_err / max_ref);
            ++cases;
        }
    }
    std::printf("passed=%d block=128x128 worst_abs=%g worst_block_maxerr_over_maxref=%g\n",
                cases, worst_abs, worst_rel);
    return cases == 27 ? 0 : 1;
}
