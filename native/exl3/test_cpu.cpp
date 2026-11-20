#include "exl3.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {
constexpr int dim = 128;
constexpr std::size_t bytes_per_twice_bit = 1024;
using Block = std::array<float, dim * dim>;
using Vector = std::array<float, dim>;

// Generated independently with v1.6.7 (8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0)
// tests/testlib/trellis.py::dequant_rotated, then an explicit dense Sylvester
// matrix and row-vector matmul in binary64 for forward. Hashes cover all
// output values cast to little-endian binary32, with -0 normalized to +0.
constexpr struct {
    int twice_bits;
    int book;
    std::uint64_t decoded;
    std::uint64_t forward;
} known[] = {
    { 2, 0, 0x72d2f04bed5029e2ULL, 0xc16c7a6a1ea5675cULL},
    { 4, 0, 0x28f78d57187250d3ULL, 0x36366bd453cedc5bULL},
    { 6, 0, 0xe593c6eb5b509517ULL, 0x7c6506400d6213a1ULL},
    { 8, 0, 0x9c795c74cdaeefbeULL, 0xde74a624586ca5feULL},
    {10, 0, 0x265ff50a8e0a1aeeULL, 0x7c3e1a87d2a34b1dULL},
    {12, 0, 0xea2c2c77d1d7e5ddULL, 0xb10db7b381f3320bULL},
    {14, 0, 0x9e75b8627a400d73ULL, 0x1c50420ca725be55ULL},
    {16, 0, 0x39e785cbd39ef2e5ULL, 0x75c809264077e767ULL},
    { 2, 1, 0xf3392340c68f24bcULL, 0xd45d39b58bce7041ULL},
    { 4, 1, 0xe49d9b2a221f7ad9ULL, 0x36d9b755cd7dd79aULL},
    { 6, 1, 0x7850842e1cd3aa64ULL, 0xe98694888e0287dfULL},
    { 8, 1, 0x0348bac59cf8e7c0ULL, 0x024504430a440a68ULL},
    {10, 1, 0xc7f94c18bc1e6d5aULL, 0x3c61fca5191a0301ULL},
    {12, 1, 0x4019e2790c3a80bdULL, 0xebe5688c1a282c69ULL},
    {14, 1, 0xf8455b003bf4f96eULL, 0xdd710280132004d6ULL},
    {16, 1, 0xb92e01b314b02ff8ULL, 0xa075f68c8bd7a827ULL},
    { 2, 2, 0x0b2b5fb65cc09dd1ULL, 0xf28d77df48a90ef7ULL},
    { 3, 2, 0xca2474955bf59d9bULL, 0x6cf83d96bd097aebULL},
    { 4, 2, 0x13854a421fd41a06ULL, 0x1c56709357b9deb3ULL},
    { 5, 2, 0x7cfd93d6cc223b6eULL, 0xbb889ad59c47f564ULL},
    { 6, 2, 0xfcd1c1745831b07cULL, 0xa5b74904f1ec5602ULL},
    { 7, 2, 0x5d7789fb6deec4bbULL, 0xbb1701bad6dd3cb7ULL},
    { 8, 2, 0xafa0bbbc1de3556cULL, 0x1e896c65bad12554ULL},
    {10, 2, 0x93aec2c923ff6c68ULL, 0x1190ac5fddbeafd2ULL},
    {12, 2, 0x8dd8aae3acc8a45bULL, 0x7fca0e59aca6351cULL},
    {14, 2, 0x75d5cd54664d6555ULL, 0xddb2b7ef1213316aULL},
    {16, 2, 0xf259c60aa18218d3ULL, 0x2badcf34b1b5951bULL},
};

std::array<std::uint8_t, dim * dim + 1> fixture()
{
    std::array<std::uint8_t, dim * dim + 1> data{};
    std::uint32_t state = 0x1badb002;
    // Offset by one to exercise unaligned byte input, including word crossings.
    for (std::size_t i = 1; i < data.size(); ++i) {
        state = state * 1664525u + 1013904223u;
        data[i] = static_cast<std::uint8_t>(state >> 24);
    }
    return data;
}

template<std::size_t N>
std::uint64_t digest(const std::array<float, N>& values)
{
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (float v : values) {
        std::uint32_t word = 0;
        if (v != 0) std::memcpy(&word, &v, sizeof(word));
        for (int shift = 0; shift < 32; shift += 8)
            hash = (hash ^ ((word >> shift) & 255u)) * 0x100000001b3ULL;
    }
    return hash;
}
}  // namespace

int main()
{
    const auto packed = fixture();
    const auto* data = packed.data() + 1;
    Block w{};
    Vector su, sv, x, y{};
    for (int i = 0; i < dim; ++i) {
        su[i] = ((i * 7 + 1) % 11 - 5) / 4.f;
        sv[i] = ((i * 5 + 3) % 13 - 6) / 2.f;
        x[i] = ((i * 19 + 2) % 23 - 11) / 8.f;
    }
    for (const auto& kg : known) {
        const auto bytes = bytes_per_twice_bit * static_cast<std::size_t>(kg.twice_bits);
        if (exl3_decode(data, bytes, kg.twice_bits, kg.book, w.data()) != 0 ||
            digest(w) != kg.decoded) {
            std::fprintf(stderr, "decode differs from oracle: twice_bits=%d book=%d\n", kg.twice_bits, kg.book);
            return 1;
        }
        if (exl3_forward(data, bytes, kg.twice_bits, kg.book, su.data(), sv.data(), x.data(), y.data()) != 0 ||
            digest(y) != kg.forward) {
            std::fprintf(stderr, "forward differs from oracle: twice_bits=%d book=%d\n", kg.twice_bits, kg.book);
            return 1;
        }
    }

    constexpr std::size_t valid_bytes = 2 * bytes_per_twice_bit;
    const Block before_w = w;
    const Vector before_y = y;
    const struct {
        const std::uint8_t* data;
        std::size_t bytes;
        int twice_bits;
        int book;
    } invalid[] = {
        {nullptr, valid_bytes, 2, 0}, {data, valid_bytes - 1, 2, 0},
        {data, valid_bytes + 1, 2, 0}, {data, 0, 2, 0},
        {data, std::numeric_limits<std::size_t>::max(), 2, 0},
        {data, valid_bytes, 2, -1}, {data, valid_bytes, 2, 3},
        {data, bytes_per_twice_bit, 1, 2}, {data, 17 * bytes_per_twice_bit, 17, 2},
        {data, 3 * bytes_per_twice_bit, 3, 0}, {data, 5 * bytes_per_twice_bit, 5, 1},
        {data, 9 * bytes_per_twice_bit, 9, 2},
    };
    for (const auto& arg : invalid) {
        if (exl3_decode(arg.data, arg.bytes, arg.twice_bits, arg.book, w.data()) == 0 ||
            exl3_forward(arg.data, arg.bytes, arg.twice_bits, arg.book,
                         su.data(), sv.data(), x.data(), y.data()) == 0 ||
            w != before_w || y != before_y) {
            std::fprintf(stderr, "invalid block not rejected: bytes=%zu twice_bits=%d book=%d\n",
                         arg.bytes, arg.twice_bits, arg.book);
            return 1;
        }
    }
    if (exl3_decode(data, valid_bytes, 2, 0, nullptr) == 0 ||
        exl3_forward(data, valid_bytes, 2, 0, nullptr, sv.data(), x.data(), y.data()) == 0 ||
        exl3_forward(data, valid_bytes, 2, 0, su.data(), nullptr, x.data(), y.data()) == 0 ||
        exl3_forward(data, valid_bytes, 2, 0, su.data(), sv.data(), nullptr, y.data()) == 0 ||
        exl3_forward(data, valid_bytes, 2, 0, su.data(), sv.data(), x.data(), nullptr) == 0 ||
        y != before_y) {
        std::fputs("null pointer not rejected\n", stderr);
        return 1;
    }
    for (auto* input : {&su, &sv, &x}) {
        const float old = (*input)[dim - 1];
        (*input)[dim - 1] = std::numeric_limits<float>::quiet_NaN();
        if (exl3_forward(data, valid_bytes, 2, 0, su.data(), sv.data(), x.data(), y.data()) == 0 ||
            y != before_y) {
            std::fputs("nonfinite forward input not rejected\n", stderr);
            return 1;
        }
        (*input)[dim - 1] = old;
    }
}
