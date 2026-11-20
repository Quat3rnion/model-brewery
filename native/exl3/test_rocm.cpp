#include "exl3.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

// All these failures must precede a device launch.
static bool rejects_invalid() {
    std::array<uint8_t, 18432> p{};
    std::array<float, 128> su{}, sv{}, x{}, y{};
    auto call = [&](size_t bytes, int rate, int book) {
        return exl3_rocm_forward(p.data(), bytes, rate, book,
                                 su.data(), sv.data(), x.data(), y.data());
    };
    if (!call(2047, 2, 0) || !call(2049, 2, 0) || !call(2048, 2, -1) ||
        !call(2048, 2, 3) || !call(3072, 3, 0) || !call(3072, 3, 1) ||
        !call(9216, 9, 2) || !call(1024, 1, 2) || !call(18432, 18, 0)) return false;
    for (int i = 0; i < 5; ++i) {
        if (!exl3_rocm_forward(i == 0 ? nullptr : p.data(), 2048, 2, 0,
                               i == 1 ? nullptr : su.data(), i == 2 ? nullptr : sv.data(),
                               i == 3 ? nullptr : x.data(), i == 4 ? nullptr : y.data()))
            return false;
    }
    for (auto v : {su.data(), sv.data(), x.data()}) {
        v[127] = std::numeric_limits<float>::infinity();
        if (!call(2048, 2, 0)) return false;
        v[127] = std::numeric_limits<float>::quiet_NaN();
        if (!call(2048, 2, 0)) return false;
        v[127] = 0;
    }
    return true;
}

int main() {
    if (!rejects_invalid()) {
        std::fprintf(stderr, "HIP forward accepted an invalid argument\n");
        return 1;
    }
    std::array<float, 128> su{}, sv{}, x{}, y{}, expected{};
    std::mt19937 gen(0xE313);
    for (int i = 0; i < 128; ++i) {
        x[i] = (int(gen() % 2049) - 1024) / 1024.0f;
        su[i] = (i % 3 ? -1 : 1) * (9 + i % 17) / 16.0f;
        sv[i] = (i % 5 ? 1 : -1) * (7 + i % 19) / 16.0f;
    }
    int cases = 0;
    float worst = 0;
    for (int rate = 2; rate <= 16; ++rate) {
        for (int book = 0; book < 3; ++book) {
            if ((rate & 1) && (rate > 7 || book != EXL3_MUL1)) continue;
            size_t bytes = 1024 * rate;
            // The ABI requires no special alignment for the packed host buffer.
            std::vector<uint8_t> packed(bytes + 1);
            auto p = packed.data() + 1;
            for (size_t k = 0; k < bytes; ++k) p[k] = static_cast<uint8_t>(gen());
            y.fill(std::numeric_limits<float>::quiet_NaN());
            int ref = exl3_forward(p, bytes, rate, book, su.data(), sv.data(), x.data(), expected.data());
            int rc = exl3_rocm_forward(p, bytes, rate, book, su.data(), sv.data(), x.data(), y.data());
            if (ref || rc) {
                std::fprintf(stderr, "rate=%d/2 book=%d cpu=%d hip=%d\n", rate, book, ref, rc);
                return 1;
            }
            for (int i = 0; i < 128; ++i) {
                float err = std::abs(y[i] - expected[i]);
                if (!std::isfinite(err) || err > 2e-5f + 2e-6f * std::abs(expected[i])) {
                    std::fprintf(stderr, "rate=%d/2 book=%d out=%d cpu=%g hip=%g\n",
                                 rate, book, i, expected[i], y[i]);
                    return 1;
                }
                worst = std::max(worst, err);
            }
            ++cases;
        }
    }
    std::printf("HIP passed=%d block=128x128 worst_abs=%g\n", cases, worst);
    return cases == 27 ? 0 : 1;
}
