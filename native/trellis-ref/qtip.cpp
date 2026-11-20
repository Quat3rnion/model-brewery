#include "qtip_packed.h"

namespace qtip {

Status validate(const Packed& w) {
    const auto s = w.shape;
    auto axis = [](int n) { return n >= 16 && n <= 32768 && (n & (n - 1)) == 0; };
    if (!axis(s.in_features) || !axis(s.out_features) || s.K < 1 || s.K > 4 ||
        !std::isfinite(w.scale) || w.scale <= 0)
        return Status::invalid;
    const bool swizzled = s.layout == Layout::cuda_swizzled;
    if (swizzled) {
        if (s.mode != Mode::quantlut_sym || s.K < 2 || s.Q != 9 ||
            s.in_features < 32 || s.out_features < 32)
            return Status::invalid;
    } else if (s.layout != Layout::reference) {
        return Status::invalid;
    }
    bool mode_ok = false;
    switch (s.mode) {
    case Mode::lut: mode_ok = s.L == 10 && s.V == 2 && s.Q == 10; break;
    case Mode::mad1:
    case Mode::mad2:
    case Mode::inst3: mode_ok = s.L == 16 && s.V == 1 && s.Q == 0; break;
    case Mode::quantlut: mode_ok = s.L == 16 && (s.V == 1 || s.V == 2) && s.Q == 8; break;
    case Mode::quantlut_sym: mode_ok = s.L == 16 && s.V == 2 && s.Q == (swizzled ? 9 : 8); break;
    }
    if (!mode_ok) return Status::invalid;
    const size_t tlut_n = s.Q ? (size_t(1) << s.Q) * s.V : 0;
    return w.trellis && w.su && w.sv && (bool(w.tlut) == bool(tlut_n)) &&
               w.trellis_bytes == size_t(s.in_features) * s.out_features / 8 * s.K &&
               w.su_elements == size_t(s.in_features) && w.sv_elements == size_t(s.out_features) &&
               w.tlut_elements == tlut_n ? Status::ok : Status::invalid;
}

Status linear(const Packed& w, const float* x, float* y, float* work, size_t count) {
    if (!detail::valid_io(w, x, y, work, count)) return Status::invalid;
    const int n = w.shape.in_features, m = w.shape.out_features;
    float* rotated = work + n;
    for (int c = 0; c < n; ++c) work[c] = x[c] * trellis_ref::half_to_float(w.su[c]);
    trellis_ref::fwht(work, n);
    for (int r = 0; r < m; ++r) {
        float acc = 0;
        for (int c = 0; c < n; ++c) acc += work[c] * detail::decode(w, r, c);
        rotated[r] = acc;
    }
    trellis_ref::fwht(rotated, m);
    for (int r = 0; r < m; ++r) y[r] = rotated[r] * (trellis_ref::half_to_float(w.sv[r]) * w.scale);
    return Status::ok;
}

}  // namespace qtip
