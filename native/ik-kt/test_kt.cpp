#include "kt_matvec.hpp"
#include "ggml.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(KT_TEST_GPU)
void begin_device();
void check_device(ik_kt::Kind, const std::vector<uint8_t> &, size_t, size_t,
                  const std::vector<float> &, const std::vector<double> &,
                  const std::vector<double> &);
#endif

static void require(bool ok, const char *msg) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", msg);
        std::exit(1);
    }
}

static void check(ik_kt::Kind kind, size_t width, size_t rows, bool encoded, std::mt19937 &rng) {
    const auto type = static_cast<ggml_type>(kind);
    const auto stride = ggml_row_size(type, width);
    require(ik_kt::row_bytes(kind, width) == stride, "packed row size differs from pin");
    // malloc-backed storage is aligned for the fork oracle's native block loads.
    std::vector<uint8_t> packed(rows * stride);
    std::vector<float> x(width), out(rows), decoded(width), source(width * rows);
    std::vector<double> ref(rows), l1(rows);
    std::uniform_real_distribution<float> uniform(-0.25f, 0.25f);
    if (encoded) {
        for (float &f : source) f = uniform(rng);
        const quantize_user_data opts{};
        require(ggml_quantize_chunk(type, source.data(), packed.data(), 0, rows, width, nullptr, &opts)
                == packed.size(), "pinned encoder returned wrong size");
    } else {
        for (size_t i = 0; i < rows; ++i) {
            uint8_t *row = packed.data() + i * stride;
            for (size_t j = sizeof(float); j < stride; ++j) row[j] = static_cast<uint8_t>(rng());
            const float d = i == 0 ? 0.0f : std::ldexp(uniform(rng), -14);
            std::memcpy(row, &d, sizeof d);
        }
    }
    for (float &f : x) f = uniform(rng);
    const auto dequant = ggml_internal_get_type_traits(type).to_float;
    require(dequant != nullptr, "pinned CPU decoder unavailable");
    for (size_t i = 0; i < rows; ++i) {
        dequant(packed.data() + i * stride, decoded.data(), width);
        for (size_t j = 0; j < width; ++j) {
            const double term = double(x[j]) * decoded[j];
            ref[i] += term;
            l1[i] += std::abs(term);
        }
    }
    require(ik_kt::matvec_cpu(kind, packed.data(), packed.size(), width, rows, x.data(),
                             x.size(), out.data(), out.size()) == ik_kt::Status::ok,
            "CPU matvec rejected valid inputs");
    for (size_t i = 0; i < rows; ++i) {
        if (!(std::abs(out[i] - ref[i]) <= 1e-7 + 2e-6 * l1[i])) {
            std::fprintf(stderr, "CPU mismatch id=%u width=%zu row=%zu got=%.9g ref=%.9g abs_sum=%.9g\n",
                         unsigned(kind), width, i, out[i], ref[i], l1[i]);
            std::exit(1);
        }
    }
#if defined(KT_TEST_GPU)
    check_device(kind, packed, width, rows, x, ref, l1);
#endif
}

int main() {
    static_assert(int(ik_kt::Kind::iq1) == GGML_TYPE_IQ1_KT && int(ik_kt::Kind::iq2) == GGML_TYPE_IQ2_KT &&
                  int(ik_kt::Kind::iq3) == GGML_TYPE_IQ3_KT && int(ik_kt::Kind::iq4) == GGML_TYPE_IQ4_KT);
    auto *ctx = ggml_init({ggml_tensor_overhead(), nullptr, true});
    require(ctx != nullptr, "cannot initialize pinned GGML");
#if defined(KT_TEST_GPU)
    begin_device();
#endif
    std::mt19937 rng(12377);
    for (auto kind : {ik_kt::Kind::iq1, ik_kt::Kind::iq2, ik_kt::Kind::iq3, ik_kt::Kind::iq4}) {
        for (size_t width = 32; width <= 768; width += 32) check(kind, width, 7, false, rng);
        check(kind, 4128, 129, false, rng);
        for (size_t width : {size_t(32), size_t(256), size_t(544)}) check(kind, width, 3, true, rng);
        std::printf("id=%u passed (24 widths, wide case, 3 upstream-encoded cases)\n", unsigned(kind));
    }
    constexpr auto kind = ik_kt::Kind::iq4;
    std::array<uint8_t, 64> bytes{};
    std::array<float, 32> x{};
    float out = 1;
    const auto n = ik_kt::row_bytes(kind, 32);
    require(ik_kt::row_bytes(kind, 31) == 0 && ik_kt::row_bytes(kind, 0) == 0 &&
            ik_kt::row_bytes(static_cast<ik_kt::Kind>(0), 32) == 0,
            "invalid type or width was accepted");
    require(ik_kt::matvec_cpu(kind, bytes.data(), n - 1, 32, 1, x.data(), 32, &out, 1)
            == ik_kt::Status::invalid_argument && out == 1, "short input changed output");
    require(ik_kt::matvec_cpu(kind, bytes.data(), n, 32, 1, x.data(), 31, &out, 1)
            == ik_kt::Status::invalid_argument, "short vector accepted");
    require(ik_kt::matvec_cpu(kind, bytes.data(), n, 32, 1, x.data(), 32, &out, 0)
            == ik_kt::Status::invalid_argument, "short output accepted");
    ggml_free(ctx);
}
