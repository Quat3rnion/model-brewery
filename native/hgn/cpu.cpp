#include "packed.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace hgn {
namespace {
using namespace detail;
static_assert(sizeof(size_t) >= 8, "64-bit host required for mapped HGN sizes");
constexpr size_t max_dimension = 1u << 24;
constexpr size_t max_work = 1u << 24;

void check(bool ok, const char* why) {
    if (!ok) throw std::invalid_argument(why);
}

size_t align(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }
size_t mul(size_t a, size_t b) {
    check(b && a <= std::numeric_limits<size_t>::max() / b, "HGN dimension overflow");
    return a * b;
}
void exact(Bytes p, size_t size) {
    check(p.data && p.size == size, "incorrect HGN payload size");
}
Bytes sub(Bytes p, size_t offset, size_t size) {
    check(offset <= p.size && size <= p.size - offset, "HGN payload slice outside span");
    return {p.data + offset, size};
}
bool empty(const Tensor& t) {
    return t.name.empty() && !t.type && !t.variant && t.shape.empty() && !t.bytes.data && !t.bytes.size;
}

const std::array<uint8_t, 2042>& ht_book() {
    static const auto book = [] {
        std::array<uint8_t, 2042> bytes{};
        // Round integer numerator/2^18 to binary16 using only integer
        // arithmetic. Subnormal steps are 2^-24, so no rounding below 2^-14.
        for (int a = 0; a <= 1020; ++a) {
            int number = 1774 * a - 905216;
            unsigned m = unsigned(number < 0 ? -number : number);
            unsigned bits = m * 64;
            if (m >= 16) {
                int e = 0;
                for (auto s = m; s >>= 1;) ++e;
                unsigned sig = m;
                if (e > 10) {
                    unsigned step = 1u << (e - 10), q = m / step, r = m % step;
                    sig = q + (r > step / 2 || (r == step / 2 && (q & 1)));
                } else {
                    sig <<= 10 - e;
                }
                bits = unsigned(e - 4) * 1024 + sig; // carries a rounded 2048
            }
            if (number < 0) bits |= 0x8000;
            bytes[2 * a] = uint8_t(bits);
            bytes[2 * a + 1] = uint8_t(bits >> 8);
        }
        return bytes;
    }();
    return book;
}

Store matrix_store(const Tensor& t) {
    if (t.type == 0 && t.variant == 0) return Store::bf16;
    if (t.type == 1 && t.variant == 0) return Store::f32;
    if (t.type == 2 && t.variant == 0) return Store::f16;
    if (t.type == 5 && t.variant == 2) return Store::q4cp;
    if (t.type == 7 && t.variant == 0) return Store::q8g64;
    if (t.type == 16 && (t.variant == 0x1206 || t.variant == 0x1208)) return Store::ht;
    if (t.type == 23 && t.variant == 128) return Store::i4r;
    if (t.type == 24 && t.variant == 64) return Store::q6g64;
    throw std::invalid_argument("unsupported HGN matvec type/variant " +
                                std::to_string(t.type) + "/" + std::to_string(t.variant));
}

bool ends(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

Bytes plane(const Tensor& t, const std::string& name, size_t n, bool stacked, size_t e) {
    const std::vector<size_t> shape = stacked ? std::vector<size_t>{512, n} : std::vector<size_t>{n};
    check(t.name == name && t.type == 2 && t.variant == 0 && t.shape == shape,
          "missing or incompatible HGN F16 rotation sidecar");
    exact(t.bytes, 2 * n * (stacked ? 512 : 1));
    auto bytes = sub(t.bytes, (stacked ? e : 0) * 2 * n, 2 * n);
    for (size_t i = 0; i < n; ++i)
        check(std::isfinite(half(u16(bytes.data + 2 * i))), "nonfinite HGN rotation value");
    return bytes;
}

Packed matrix(const Tensor& t, const Tensor& su, const Tensor& sv, int expert) {
    Store s = matrix_store(t);
    bool ht = s == Store::ht, rotated = ht || s == Store::i4r;
    bool stacked = t.shape.size() == 3;
    check(t.shape.size() == 2 || stacked, "HGN matvec requires rank two weights or rank three experts");
    size_t n = t.shape[t.shape.size() - 2], k = t.shape.back();
    size_t count = stacked ? t.shape.front() : 1;
    check(n && k && n <= max_dimension && k <= max_dimension && count &&
          count <= 512 && (stacked ? expert >= 0 && size_t(expert) < count : expert == -1),
          "invalid HGN matrix shape or expert selection");
    size_t e = stacked ? size_t(expert) : 0, one = mul(n, k), all = mul(count, one);
    check(one <= max_work, "HGN matvec exceeds selected-matrix work bound");
    Packed p{s, unsigned(n), unsigned(k), 0, 0, false, {}, {}, {}, {}, {}};

    if (rotated) {
        check(n % 128 == 0 && k % 128 == 0 && ends(t.name, ".weight"), "invalid HGN rotated geometry/name");
        bool gate = stacked && count == 512 && n == 1280 && k == 2560 &&
                    ends(t.name, ".experts.gate_up_proj.weight");
        bool down = stacked && count == 512 && n == 2560 && k == 640 &&
                    ends(t.name, ".experts.down_proj.weight");
        check(stacked ? gate || down : ht, "unsupported native HGN rotated expert shape/name");
        std::string base = t.name.substr(0, t.name.size() - 7);
        std::string su_name = gate ?
            base.substr(0, base.size() - std::string(".gate_up_proj").size()) + ".su" : base + ".suh";
        p.su = plane(su, su_name, k, ht && down, e);
        p.sv = plane(sv, base + ".svh", n, ht && stacked, e);
        if (ht) {
            p.bits = (t.variant & 255) / 2;
            size_t bytes = one * p.bits / 8;
            exact(t.bytes, count * bytes);
            p.code = sub(t.bytes, e * bytes, bytes);
            p.paired = gate;
            p.book = {ht_book().data(), ht_book().size()};
        } else {
            exact(t.bytes, all / 2 + all / 64);
            p.code = sub(t.bytes, e * one / 2, one / 2);
            p.scale = sub(t.bytes, all / 2 + e * one / 64, one / 64);
            p.stride = k / 2;
        }
        return p;
    }

    check(empty(su) && empty(sv), "unexpected rotation sidecar for unrotated HGN store");
    if (s == Store::q4cp) {
        check(k % 32 == 0, "invalid Q4C-P column geometry");
        size_t scale_stride = align(k / 16, 16), scales = 64 + align(all / 2, 64);
        exact(t.bytes, scales + count * n * scale_stride);
        p.book = sub(t.bytes, 0, 64);
        p.code = sub(t.bytes, 64 + e * one / 2, one / 2);
        p.scale = sub(t.bytes, scales + e * n * scale_stride, n * scale_stride);
        p.stride = k / 2;
    } else {
        if (s == Store::q8g64 || s == Store::q6g64)
            check(k % 64 == 0, "invalid Q8G64/Q6G64 column geometry");
        size_t stride = s == Store::f32 ? 4 * k :
            s == Store::q8g64 ? k + k / 16 :
            s == Store::q6g64 ? align(13 * k / 16, 16) : 2 * k;
        size_t bytes = mul(n, stride);
        exact(t.bytes, mul(count, bytes));
        p.code = sub(t.bytes, e * bytes, bytes);
        p.stride = stride;
    }
    return p;
}

bool use_device(Backend b) {
    if (b == Backend::cpu) return false;
#if defined(HGN_CUDA)
    if (b == Backend::cuda) return true;
#elif defined(HGN_ROCM)
    if (b == Backend::rocm) return true;
#endif
    throw std::runtime_error("requested HGN backend was not built (no fallback)");
}

void finite(const std::vector<double>& v) {
    for (double x : v) check(std::isfinite(x), "nonfinite HGN operand or contraction");
}
}

std::string device_name(Backend b) {
    if (!use_device(b)) return "CPU";
#ifdef HGN_DEVICE_ENABLED
    return device_description();
#else
    throw std::logic_error("unreachable");
#endif
}

std::vector<double> matvec(const Tensor& t, const Tensor& su, const Tensor& sv, int expert,
                           const std::vector<double>& x, Backend b) {
    const auto p = matrix(t, su, sv, expert);
    check(x.size() == p.cols, "incorrect HGN activation length");
    finite(x);
    std::vector<double> y(p.rows);
    if (use_device(b)) {
#ifdef HGN_DEVICE_ENABLED
        y = device_matvec(p, x);
#endif
    } else {
        std::vector<double> in;
        if (p.su.data) {
            in.resize(p.cols);
            for (unsigned i = 0; i < p.cols; ++i) in[i] = rotated(x.data(), p.su.data, i, true);
        }
        const auto& v = p.su.data ? in : x;
        for (unsigned r = 0; r < p.rows; ++r)
            for (unsigned c = 0; c < p.cols; ++c)
                y[r] += weight(p, r, c) * v[c];
        if (p.sv.data) {
            std::vector<double> out(p.rows);
            for (unsigned i = 0; i < p.rows; ++i) out[i] = rotated(y.data(), p.sv.data, i, false);
            y.swap(out);
        }
    }
    finite(y);
    return y;
}

std::vector<double> gather_fp8(const Tensor& t, size_t shard, size_t row, Backend b) {
    check(t.type == 10 && !t.variant && t.shape.size() == 3 &&
          t.shape[0] == 128 && t.shape[2] == 160 && t.shape[1] &&
          t.shape[1] <= max_dimension && shard < 128 && row < t.shape[1],
          "invalid HGN FP8G table or selection");
    size_t scale = align(128 * t.shape[1] * 160, 64);
    exact(t.bytes, scale + 4);
    auto code = sub(t.bytes, (shard * t.shape[1] + row) * 160, 160);
    auto gain = sub(t.bytes, scale, 4);
    double factor = f32(u32(gain.data));
    check(std::isfinite(factor) && factor > 0, "invalid HGN FP8G scale");
    if (use_device(b)) {
#ifdef HGN_DEVICE_ENABLED
        auto out = device_gather({Store::fp8g, 1, 160, 160, 0, false, code, gain, {}, {}, {}});
        finite(out);
        return out;
#endif
    }
    std::vector<double> out(160);
    for (unsigned i = 0; i < 160; ++i) out[i] = fp8(code.data[i]) * factor;
    return out;
}

std::vector<int64_t> integers(const Tensor& t) {
    check(t.type == 4 && !t.variant && t.shape.size() == 1 &&
          t.shape[0] && t.shape[0] <= 4096, "invalid bounded HGN I64 vector");
    exact(t.bytes, 8 * t.shape[0]);
    std::vector<int64_t> out(t.shape[0]);
    for (size_t i = 0; i < out.size(); ++i) {
        const uint8_t* p = t.bytes.data + 8 * i;
        uint64_t u = u32(p) | (uint64_t(u32(p + 4)) << 32);
        out[i] = u <= INT64_MAX ? int64_t(u) : -1 - int64_t(~u);
    }
    return out;
}
}
