#include "glq_detail.h"

#include <vector>

namespace glq {
namespace {
bool size_is(Bytes bytes, std::size_t n) {
    return bytes.size == n && (bytes.data != nullptr) == (n != 0);
}

bool signs(Bytes bytes) {
    for (std::size_t i = 0; i < bytes.size; i += 2) {
        const auto value = trellis_ref::le16(bytes.data + i);
        if (value != 0x3c00 && value != 0xbc00) return false;
    }
    return true;
}
}  // namespace

bool validate_metadata(const Layer& l) {
    // The swizzle pairs 16x16 tiles on both axes. The reference does not use
    // the upstream fused kernel's stricter (cols % 64) dispatch condition.
    if (l.layout != Layout::kernel_3inst_block_diagonal || l.rows <= 0 || l.cols <= 0
            || l.rows > max_axis || l.cols > max_axis || l.rows % 32 || l.cols % 32
            || l.bpw < 2 || l.bpw > 8 || !std::isfinite(l.wscale) || l.wscale <= 0)
        return false;
    const auto eighth = std::size_t(l.rows) * l.cols / 8;
    if (!size_is(l.trellis_packed, eighth * detail::primary(l.bpw))
            || !size_is(l.su, 2 * l.rows) || !size_is(l.sv, 2 * l.cols)) return false;
    if (l.bpw <= 4) return size_is(l.trellis_packed2, 0) && l.inv_resid_scale2 == 0;
    return size_is(l.trellis_packed2, eighth * (l.bpw - 4))
        && std::isfinite(l.inv_resid_scale2) && l.inv_resid_scale2 > 0;
}

bool validate(const Layer& layer) {
    return validate_metadata(layer) && signs(layer.su) && signs(layer.sv);
}

bool linear(const Layer& l, int batch, const float* x, std::size_t nx, float* y, std::size_t ny) {
    if (!validate(l) || !detail::valid_io(l, batch, x, nx, y, ny)) return false;
    std::vector<float> rotated(l.cols);
    for (int b = 0; b < batch; ++b) {
        detail::transform_input(l, x + b * l.cols, rotated.data());
        for (int r = 0; r < l.rows; ++r) y[r] = detail::dot(l, rotated.data(), r);
        detail::transform_output(l, y);
        y += l.rows;
    }
    return true;
}

}  // namespace glq
