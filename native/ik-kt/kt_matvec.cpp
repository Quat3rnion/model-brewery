#include "kt_packed.hpp"

namespace ik_kt {

size_t row_bytes(Kind kind, size_t width) {
    return detail::valid_stride(kind, width);
}

template<Kind K>
static void run(const uint8_t *packed, size_t stride, size_t width, size_t rows, const float *x, float *y) {
    constexpr auto gs = detail::group_size<K>;
    const auto full = width / 256;
    const unsigned tail = (width % 256) / 32;
    for (size_t row = 0; row < rows; ++row) {
        const auto *p = packed + row * stride;
        const float d = detail::row_scale(p);
        double sum = 0;
        for (size_t g = 0; g < width / gs; ++g) {
            sum += detail::group_dot<K, false>(p + 4, d, full, tail, g, x + gs * g);
        }
        y[row] = static_cast<float>(sum);
    }
}

Status matvec_cpu(Kind kind, const void *packed, size_t bytes, size_t width,
                  size_t rows, const float *x, size_t x_count, float *y, size_t y_count) {
    const auto stride = detail::validate(kind, packed, bytes, width, rows, x, x_count, y, y_count);
    if (!stride) return Status::invalid_argument;
    const auto *p = static_cast<const uint8_t *>(packed);
    switch (kind) {
        case Kind::iq1: run<Kind::iq1>(p, stride, width, rows, x, y); break;
        case Kind::iq2: run<Kind::iq2>(p, stride, width, rows, x, y); break;
        case Kind::iq3: run<Kind::iq3>(p, stride, width, rows, x, y); break;
        case Kind::iq4: run<Kind::iq4>(p, stride, width, rows, x, y); break;
    }
    return Status::ok;
}

} // namespace ik_kt
