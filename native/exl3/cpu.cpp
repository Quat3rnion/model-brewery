#include "exl3.h"

#include <array>
#include <cmath>
#include <cstdint>

namespace {
constexpr int dim = 128;
constexpr int tile_dim = 16;
constexpr int tiles_per_axis = dim / tile_dim;
using Vector = std::array<double, dim>;

bool valid(const std::uint8_t* packed, std::size_t bytes, int twice_bits, int book)
{
    if (!packed || book < EXL3_3INST || book > EXL3_MUL1 || twice_bits < 2 || twice_bits > 16)
        return false;
    if ((twice_bits & 1) && (book != EXL3_MUL1 || twice_bits > 7))
        return false;
    // Each tile has 16 * bits I16 words. There are 64 tiles.
    return bytes == static_cast<std::size_t>(twice_bits) * 1024;
}

std::uint32_t word_le(const std::uint8_t* p)
{
    return std::uint32_t{p[0]} | (std::uint32_t{p[1]} << 8) |
           (std::uint32_t{p[2]} << 16) | (std::uint32_t{p[3]} << 24);
}

// Round exact dyadic integers to binary16, then return the exact result in
// float. The smallest input unit is 2^-18, so no subnormals are needed.
float half_round(int units, int exponent)
{
    const int magnitude = units < 0 ? -units : units;
    int shift = 0;
    for (int n = magnitude; n >= 2048; n >>= 1) ++shift;
    int rounded = magnitude >> shift;
    if (shift) {
        const int remainder = magnitude - (rounded << shift);
        const int middle = 1 << (shift - 1);
        if (remainder > middle || (remainder == middle && (rounded & 1))) ++rounded;
    }
    return std::ldexp(static_cast<float>(units < 0 ? -rounded : rounded), exponent + shift);
}

int half_units(std::uint32_t half)
{
    // The two 3inst/MCG transformed halves have biased exponents 12..15.
    const int n = static_cast<int>(1024 + (half & 1023)) << (((half >> 10) & 31) - 12);
    return (half & 0x8000) ? -n : n;
}
}  // namespace

namespace exl3 {
// Private linkage seam for the native encoder; not part of exl3.h.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("hidden")))
#endif
float codebook_value(std::uint32_t state, int book)
{
    // ExLlamaV3 v1.6.7 exllamav3_ext/quant/codebook.cuh and CPU oracle
    // tests/testlib/trellis.py: exact half add (3inst/MCG) or half FMA (mul1).
    if (book == EXL3_MUL1) {
        const std::uint32_t x = state * 0x83dcd12du;
        const int sum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
        // 0x1eee = 1774 * 2^-18, 0xc931 = -1329 * 2^-7.
        return half_round((1024 + sum) * 1774 - 1329 * 2048, -18);
    }
    const std::uint32_t x = book == EXL3_MCG ? state * 0xcbac1fedu : state * 89226354u + 64248484u;
    const std::uint32_t halves = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    return half_round(half_units(halves) + half_units(halves >> 16), -13);
}
}  // namespace exl3

namespace {
template<typename Consumer>
void visit(const std::uint8_t* packed, int twice_bits, int book, Consumer consume)
{
    const int tile_bytes = twice_bits * 16;
    const int ring_bits = tile_bytes * 8;
    const int words = tile_bytes / 4;
    const int step = twice_bits / 2;
    const bool alternating = twice_bits & 1;
    for (int tr = 0; tr < tiles_per_axis; ++tr) {
        for (int tc = 0; tc < tiles_per_axis; ++tc, packed += tile_bytes) {
            std::array<std::uint32_t, 65> ring;
            for (int w = 0; w < words; ++w) ring[w] = word_le(packed + 4 * w);
            ring[words] = ring[0];

            // Little-endian words carry MSB-first bits. The 16-bit state
            // ends at the current symbol and may cross words or wrap.
            // For half rates the step alternates using upstream mask 0xAAAA.
            int end = 0;
            for (int k = 0; k < tile_dim * tile_dim; ++k) {
                end += step + (alternating ? (k & 1) : 0);
                const int start = end >= 16 ? end - 16 : end - 16 + ring_bits;
                const int w = start / 32;
                const auto pair = (std::uint64_t{ring[w]} << 32) | ring[w + 1];
                const auto state = static_cast<std::uint32_t>(pair >> (48 - (start & 31))) & 0xffffu;
                // Native tensor-core fragment order within each 16x16 tile.
                const int lane = k / 8, element = k % 8;
                const int row = tr * tile_dim + (lane % 4) * 2 + (element & 1) + (element & 2) * 4;
                const int col = tc * tile_dim + lane / 4 + (element & 4) * 2;
                consume(row, col, exl3::codebook_value(state, book));
            }
        }
    }
}

void hadamard(Vector& values)
{
    for (int step = 1; step < dim; step *= 2) {
        for (int base = 0; base < dim; base += 2 * step) {
            for (int j = base; j < base + step; ++j) {
                const double a = values[j], b = values[j + step];
                values[j] = a + b;
                values[j + step] = a - b;
            }
        }
    }
}
}  // namespace

extern "C" int exl3_decode(const std::uint8_t* packed, std::size_t bytes, int twice_bits,
                           int book, float* rotated)
{
    if (!rotated || !valid(packed, bytes, twice_bits, book)) return 1;
    visit(packed, twice_bits, book, [rotated](int row, int col, float weight) {
        rotated[row * dim + col] = weight;
    });
    return 0;
}

extern "C" int exl3_forward(const std::uint8_t* packed, std::size_t bytes, int twice_bits, int book,
                            const float* su, const float* sv, const float* x, float* y)
{
    if (!su || !sv || !x || !y || !valid(packed, bytes, twice_bits, book)) return 1;
    // x diag(su) H W_hat H diag(sv). GEMV consumes packed tiles directly.
    // The two 1/sqrt(dim) normalizations combine to one division by dim.
    Vector a, b{};
    for (int i = 0; i < dim; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(su[i]) || !std::isfinite(sv[i])) return 1;
        a[i] = static_cast<double>(x[i]) * su[i];
    }
    hadamard(a);
    visit(packed, twice_bits, book, [&a, &b](int row, int col, float weight) {
        b[col] += a[row] * weight;
    });
    hadamard(b);
    for (int i = 0; i < dim; ++i) y[i] = static_cast<float>((b[i] / dim) * sv[i]);
    return 0;
}
