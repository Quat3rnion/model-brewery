#include "gyro.h"
#include "layout.h"

#include <limits>
#include <vector>

namespace gyro {
namespace {
constexpr std::uint16_t codebook[] = {
#include "gyro_codebook.inc"
};
static_assert(sizeof(codebook) / sizeof(codebook[0]) == 4096);
}

bool valid(const Matrix& m, Basis basis) noexcept {
    const auto f = detail::layout(m.type);
    const auto max = std::numeric_limits<std::size_t>::max();
    if (!f.bytes || m.state_bits != 15 || !m.rows || !m.columns || m.columns % 128 ||
        (basis != Basis::original && basis != Basis::stored) ||
        m.columns > max / sizeof(float) || m.rows > max / sizeof(float)) return false;
    const auto row_bytes = (m.columns / 128) * f.bytes;
    return m.rows <= max / row_bytes && m.packed_bytes == m.rows * row_bytes &&
        (!m.rotation_block || (!(m.rotation_block & (m.rotation_block - 1)) &&
                               m.columns % m.rotation_block == 0));
}

bool valid_signs(const Matrix& m, const std::int8_t* signs) noexcept {
    if (!signs) return true;
    if (!m.rotation_block) return false;
    for (std::size_t i = 0; i < m.columns; ++i)
        if (signs[i] != 1 && signs[i] != -1) return false;
    return true;
}

bool forward_cpu(const Matrix& m, const std::uint8_t* packed, const float* input,
                 Basis basis, const std::int8_t* signs, float* output) {
    if (!packed || !input || !output || !valid(m, basis) || !valid_signs(m, signs)) return false;
    std::vector<float> rotated;
    if (m.rotation_block && basis == Basis::original) {
        rotated.resize(m.columns);
        for (std::size_t i = 0; i < m.columns; ++i)
            rotated[i] = signs && signs[i] == -1 ? -input[i] : input[i];
        const auto block = m.rotation_block;
        const float scale = 1.f / std::sqrt(float(block));
        for (std::size_t off = 0; off < m.columns; off += block) {
            for (std::size_t span = 1; span < block; span *= 2)
                for (std::size_t i = 0; i < block; i += 2 * span)
                    for (std::size_t j = off + i; j < off + i + span; ++j) {
                        const float a = rotated[j], b = rotated[j + span];
                        rotated[j] = a + b;
                        rotated[j + span] = a - b;
                    }
            for (std::size_t i = off; i < off + block; ++i) rotated[i] *= scale;
        }
        input = rotated.data();
    }
    const auto f = detail::layout(m.type);
    const std::size_t blocks = m.columns / 128;
    for (std::size_t row = 0; row < m.rows; ++row) {
        double sum = 0.;
        for (std::size_t b = 0; b < blocks; ++b, packed += f.bytes)
            for (unsigned i = 0; i < 128; ++i)
                sum += double(detail::weight(packed, f, i, codebook)) * input[b * 128 + i];
        output[row] = float(sum);
    }
    return true;
}

}  // namespace gyro
