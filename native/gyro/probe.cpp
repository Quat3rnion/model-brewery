#include "gyro.h"
#include "fixtures.h"
#include "tq_ref.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef GYRO_GPU
void probe_device();
void probe_gpu(const gyro::Matrix&, const std::uint8_t*, const float*, gyro::Basis,
               const std::int8_t*, float*);
#endif

static void require(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

static std::vector<std::uint8_t> bytes(const char* hex) {
    std::vector<std::uint8_t> result;
    while (*hex) {
        auto digit = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
        result.push_back(static_cast<std::uint8_t>(16 * digit(hex[0]) + digit(hex[1])));
        hex += 2;
    }
    return result;
}

static void check(const gyro::Matrix& matrix, const std::vector<std::uint8_t>& packed,
                  const float* x, gyro::Basis basis, const std::int8_t* signs,
                  const float* oracle_x) {
    std::vector<float> unpacked(matrix.rows * matrix.columns), result(matrix.rows);
    require(tq_dequantize_row(matrix.type, packed.data(), unpacked.data(), unpacked.size()) == 0,
            "pinned upstream decode");
    require(gyro::forward_cpu(matrix, packed.data(), x, basis, signs, result.data()), "CPU forward");
    auto compare = [&] {
        for (std::size_t row = 0; row < matrix.rows; ++row) {
            double expected = 0;
            for (std::size_t col = 0; col < matrix.columns; ++col)
                expected += double(unpacked[row * matrix.columns + col]) * oracle_x[col];
            const double tolerance = 2e-5 * (1 + std::abs(expected));
            if (!(std::abs(expected - result[row]) <= tolerance)) {
                std::fprintf(stderr, "type=%d width=%zu row=%zu basis=%d expected=%.9g got=%.9g\n",
                             matrix.type, matrix.columns, row, int(basis), expected, double(result[row]));
                require(false, "dense dot parity");
            }
        }
    };
    compare();
#ifdef GYRO_GPU
    probe_gpu(matrix, packed.data(), x, basis, signs, result.data());
    compare();
#endif
}

int main() {
#ifdef GYRO_GPU
    probe_device();
#endif
    float x[256], rotated[256];
    std::int8_t signs[256];
    for (int i = 0; i < 256; ++i) {
        // Sparse plus varied values exercise both pairs, sign, scale and wrap.
        x[i] = (i % 11 == 0 ? 0.f : (i * 31 % 113 - 56) / 64.f);
        signs[i] = i % 5 == 0 || i % 5 == 3 ? 1 : -1;
    }
    for (int type = 144; type <= 151; ++type) {
        const auto packed = bytes(gyro_rows[type - 144]);
        const gyro::Matrix rows{type, 15, 2, 128, packed.size(), 0};
        const gyro::Matrix blocks{type, 15, 1, 256, packed.size(), 0};
        check(rows, packed, x, gyro::Basis::stored, nullptr, x);
        check(blocks, packed, x, gyro::Basis::original, nullptr, x);
        // Change only the scale bytes: negative normal, then FP16 subnormal.
        auto changed_scale = packed;
        changed_scale[1] ^= 0x80;
        changed_scale[packed.size() / 2] = 0xff;
        changed_scale[packed.size() / 2 + 1] = 0x03;
        check(rows, changed_scale, x, gyro::Basis::stored, nullptr, x);
        auto bad = rows;
        --bad.packed_bytes;
        float y[2] = {9.f, 9.f};
        require(!gyro::forward_cpu(bad, packed.data(), x, gyro::Basis::stored, nullptr, y)
                    && y[0] == 9.f && y[1] == 9.f, "reject short packed buffer before writing output");
        bad = rows;
        bad.state_bits = 14;
        require(!gyro::forward_cpu(bad, packed.data(), x, gyro::Basis::stored, nullptr, y), "reject wrong state width");
        std::printf("type=%d bytes=%zu pass\n", type, packed.size());
    }
    const auto packed = bytes(signed_gyro_rows);
    const gyro::Matrix signed_rows{149, 15, 2, 128, packed.size(), 16};
    const gyro::Matrix identity_rows{149, 15, 2, 128, packed.size(), 128};
    // Dense forward H*S, independently implemented, not the kernel butterfly.
    for (const auto& matrix : {signed_rows, identity_rows}) {
        const int block = matrix.rotation_block;
        const auto* s = block == 16 ? signs : nullptr;
        for (int base = 0; base < 128; base += block)
            for (int j = 0; j < block; ++j) {
                float sum = 0.f;
                for (int i = 0; i < block; ++i)
                    sum += (__builtin_parity(i & j) ? -1 : 1) * (s ? s[base + i] : 1) * x[base + i];
                rotated[base + j] = sum / std::sqrt(float(block));
            }
        check(matrix, packed, x, gyro::Basis::original, s, rotated);
        check(matrix, packed, rotated, gyro::Basis::stored, s, rotated);
    }
    signs[2] = 0;
    float y[2];
    require(!gyro::forward_cpu(signed_rows, packed.data(), x, gyro::Basis::original, signs, y), "reject invalid signs");
    std::puts("rotation=explicit,identity input_basis=original,stored pass");
}
