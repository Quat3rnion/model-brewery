#include "exl3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {
using Block = std::array<float, 128 * 128>;
using Bytes = std::vector<std::uint8_t>;

std::uint32_t next(std::uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

double error(const Block& expected, const Block& actual)
{
    double sum = 0;
    for (std::size_t k = 0; k < expected.size(); ++k) {
        const double delta = static_cast<double>(expected[k]) - actual[k];
        sum += delta * delta;
    }
    return sum;
}

bool quality(const Block& weights, int twice_bits, int book)
{
    Block output, control, zero{};
    Bytes packed(1024 * twice_bits);
    Bytes random_ring(packed.size());
    std::uint32_t rng = 0xf2e34516u;
    for (auto& byte : random_ring) byte = static_cast<std::uint8_t>(next(rng));
    if (exl3_encode(weights.data(), twice_bits, book, packed.data(), packed.size()) ||
        exl3_decode(packed.data(), packed.size(), twice_bits, book, output.data()) ||
        exl3_decode(random_ring.data(), random_ring.size(), twice_bits, book, control.data()))
        return false;
    const double optimized = error(weights, output);
    if (!(optimized < 0.8 * error(weights, zero) && optimized < 0.6 * error(weights, control)))
        return false;
    if (twice_bits == 3) {
        Bytes repeat(packed.size(), 0x5a);
        if (exl3_encode(weights.data(), twice_bits, book, repeat.data(), repeat.size()) || repeat != packed)
            return false;
    }
    return true;
}

bool known_answer()
{
    // The 3inst decode of state 0x0000 is exactly this binary16 value,
    // as specified by the independent tools/exl3_decode.py oracle.
    constexpr float value = 0.76806640625f;
    Block weights, output;
    weights.fill(value);
    Bytes packed(2048);
    return exl3_encode(weights.data(), 2, EXL3_3INST, packed.data(), packed.size()) == 0 &&
           exl3_decode(packed.data(), packed.size(), 2, EXL3_3INST, output.data()) == 0 &&
           output == weights;
}

bool validates()
{
    Block weights{};
    Bytes packed(2048, 0xa5);
    auto* p = packed.data();
    const auto invalid = [&](const float* w, int rate, int book, std::uint8_t* out, std::size_t bytes) {
        return exl3_encode(w, rate, book, out, bytes) != 0;
    };
    if (!(invalid(nullptr, 2, EXL3_3INST, p, packed.size()) &&
          invalid(weights.data(), 2, EXL3_3INST, nullptr, packed.size()) &&
          invalid(weights.data(), 2, -1, p, packed.size()) &&
          invalid(weights.data(), 2, 3, p, packed.size()) &&
          invalid(weights.data(), 0, EXL3_MUL1, p, 0) &&
          invalid(weights.data(), 18, EXL3_MUL1, p, 18432) &&
          invalid(weights.data(), 3, EXL3_MCG, p, 3072) &&
          invalid(weights.data(), 9, EXL3_MUL1, p, 9216) &&
          invalid(weights.data(), 2, EXL3_3INST, p, packed.size() - 1) &&
          invalid(weights.data(), 2, EXL3_3INST, p, packed.size() + 1)))
        return false;
    weights.back() = std::numeric_limits<float>::quiet_NaN();
    if (!invalid(weights.data(), 2, EXL3_3INST, p, packed.size())) return false;
    weights.back() = -std::numeric_limits<float>::infinity();
    return invalid(weights.data(), 2, EXL3_3INST, p, packed.size()) &&
           std::all_of(packed.begin(), packed.end(), [](std::uint8_t b) { return b == 0xa5; });
}

bool finite_range()
{
    // When squared errors are formed without removing the common term, all
    // codebook values appear equal for this finite FP32 input, even in FP64.
    Block weights, output;
    weights.fill(std::numeric_limits<float>::max());
    Bytes packed(2048);
    if (exl3_encode(weights.data(), 2, EXL3_MUL1, packed.data(), packed.size()) ||
        exl3_decode(packed.data(), packed.size(), 2, EXL3_MUL1, output.data()))
        return false;
    double sum = 0;
    for (const auto weight : output) sum += weight;
    return sum > 0;
}
}  // namespace

int main()
{
    if (!validates() || !known_answer() || !finite_range()) {
        std::fputs("EXL3 encode failed validation, known answer, or finite range test\n", stderr);
        return 1;
    }
    Block weights;
    std::uint32_t rng = 0xb3121234u;
    for (auto& weight : weights) {
        const auto sample = static_cast<int>(next(rng) >> 16);
        weight = (sample - 32768) / 16384.0f;
    }
    for (int book : {EXL3_3INST, EXL3_MCG, EXL3_MUL1}) {
        for (int rate = 2; rate <= 16; ++rate) {
            if ((rate & 1) && (book != EXL3_MUL1 || rate > 7)) continue;
            if (!quality(weights, rate, book)) {
                std::fprintf(stderr, "EXL3 encode quality failed (book=%d twice_bits=%d)\n", book, rate);
                return 1;
            }
        }
    }
    std::puts("EXL3 reference encoder tests passed");
}
