#include "qtip.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
#include <vector>

#ifdef QTIP_GPU
#include "trellis_gpu.h"
#if defined(__HIPCC__)
#define TR_RUNTIME(name) hip##name
using DeviceProperties = hipDeviceProp_t;
#else
#define TR_RUNTIME(name) cuda##name
using DeviceProperties = cudaDeviceProp;
#endif
#endif

namespace {
using namespace qtip;

struct Golden {
    Shape shape;
    float output[64];
};
#include "qtip_vectors.inc"

// All test compounds are exactly representable multiples of powers of two.
uint16_t half_ratio(int x, int scale) {
    const uint16_t sign = x < 0 ? 0x8000 : 0;
    unsigned n = unsigned(std::abs(x));
    if (!n) return sign;
    int exponent = 15 - scale;
    while (n < 1024) {
        n *= 2;
        --exponent;
    }
    return sign | uint16_t((exponent + 10) * 1024 + n - 1024);
}

bool check_constant() {
    uint8_t trellis[32]{};
    uint16_t su[16], sv[16], table[2048]{};
    float x[16]{}, y[16]{}, scratch[32]{};
    for (int i = 0; i < 16; ++i) su[i] = sv[i] = 0x3c00; // 1.0
    table[0] = table[1] = 0x3c00;
    x[0] = 1.0f;
    qtip::Packed w{{16, 16, 10, 1, 2, 10, qtip::Mode::lut, qtip::Layout::reference},
                   trellis, su, sv, table, sizeof(trellis), 16, 16, 2048, 0.5f};
    if (qtip::linear(w, x, y, scratch, 32) != qtip::Status::ok || y[0] != 8.f)
        return false;
    for (int i = 1; i < 16; ++i)
        if (y[i] != 0.f) return false;
    w.trellis_bytes--;
    if (qtip::linear(w, x, y, scratch, 32) != qtip::Status::invalid) return false;
    w.trellis_bytes++;
    if (qtip::linear(w, x, y, scratch, 31) != qtip::Status::invalid) return false;
    w.shape.out_features = 48;
    if (qtip::validate(w) != qtip::Status::invalid) return false;
    w.shape.out_features = 16;
    w.shape.layout = Layout::cuda_swizzled;
    if (qtip::validate(w) != qtip::Status::invalid) return false;
    w.shape.layout = Layout::reference;
    w.scale = std::numeric_limits<float>::infinity();
    return qtip::validate(w) == qtip::Status::invalid;
}

#ifdef QTIP_GPU
bool ok(trellis_ref::Error code, const char* where) {
    if (code == trellis_ref::success) return true;
    std::fprintf(stderr, "%s: %s\n", where, TR_RUNTIME(GetErrorString)(code));
    return false;
}

template<typename T> struct Device {
    T* ptr = nullptr;
    ~Device() { if (ptr) (void)TR_RUNTIME(Free)(ptr); }
    bool init(size_t n, const T* host = nullptr) {
        if (!n) return true;
        if (!ok(TR_RUNTIME(Malloc)(&ptr, n * sizeof(T)), "allocate")) return false;
        return !host || ok(TR_RUNTIME(Memcpy)(ptr, host, n * sizeof(T),
                                             TR_RUNTIME(MemcpyHostToDevice)), "upload");
    }
};

bool device_linear(Packed w, const float* x, float* y, trellis_ref::Stream stream) {
    Device<uint8_t> pack;
    Device<uint16_t> su, sv, lut;
    Device<float> input, output, work;
    const size_t n = w.shape.in_features, m = w.shape.out_features;
    if (!pack.init(w.trellis_bytes, w.trellis) || !su.init(n, w.su) || !sv.init(m, w.sv) ||
        !lut.init(w.tlut_elements, w.tlut) || !input.init(n, x) ||
        !output.init(m) || !work.init(n + m)) return false;
    w.trellis = pack.ptr; w.su = su.ptr; w.sv = sv.ptr; w.tlut = lut.ptr;
    if (linear_gpu(w, input.ptr, output.ptr, work.ptr, n + m - 1, stream) != Status::invalid)
        return false;
    if (linear_gpu(w, input.ptr, output.ptr, work.ptr, n + m, stream) != Status::ok) {
        std::fputs("GPU launch failed\n", stderr);
        return false;
    }
    return ok(TR_RUNTIME(StreamSynchronize)(stream), "execute") &&
           ok(TR_RUNTIME(Memcpy)(y, output.ptr, m * sizeof(float),
                                 TR_RUNTIME(MemcpyDeviceToHost)), "download");
}

struct TestStream {
    trellis_ref::Stream value = nullptr;
    ~TestStream() { if (value) (void)TR_RUNTIME(StreamDestroy)(value); }
};
#endif
}  // namespace

int main() {
    if (!check_constant()) return 1;
    const char* backend = "cpu";
#ifdef QTIP_GPU
    int device;
    DeviceProperties prop{};
    TestStream stream;
    if (!ok(TR_RUNTIME(GetDevice)(&device), "device") ||
        !ok(TR_RUNTIME(GetDeviceProperties)(&prop, device), "properties") ||
        !ok(TR_RUNTIME(StreamCreate)(&stream.value), "stream")) return 1;
    backend = prop.name;
#endif
    float worst = 0.f;
    static_assert(std::size(vectors) == 31);
    for (unsigned trial = 0; trial < std::size(vectors); ++trial) {
        const auto& v = vectors[trial];
        const Shape s = v.shape;
        const int m = s.out_features, n = s.in_features;
        std::vector<uint8_t> bytes(m * n * s.K / 8);
        uint32_t seed = 0x51444950u + trial;
        for (auto& b : bytes) {
            seed = seed * 1664525u + 1013904223u;
            b = uint8_t(seed >> 24);
        }
        std::vector<uint16_t> su(n), sv(m), lut(s.Q ? (1 << s.Q) * s.V : 0);
        std::vector<float> x(n), y(m, std::numeric_limits<float>::quiet_NaN());
        for (int i = 0; i < n; ++i) {
            su[i] = half_ratio(i * 5 % 31 - 15, 4);
            x[i] = float(i * 11 % 53 - 26) / 32;
        }
        for (int i = 0; i < m; ++i) sv[i] = half_ratio(i * 7 % 29 - 14, 4);
        for (size_t i = 0; i < lut.size(); ++i) lut[i] = half_ratio(int(i * 149 % 253) - 126, 7);
        const Packed w{s, bytes.data(), su.data(), sv.data(), lut.empty() ? nullptr : lut.data(),
                       bytes.size(), su.size(), sv.size(), lut.size(), 0.375f};
#ifdef QTIP_GPU
        if (!device_linear(w, x.data(), y.data(), stream.value)) return 1;
#else
        std::vector<float> work(n + m);
        if (linear(w, x.data(), y.data(), work.data(), work.size()) != Status::ok) return 1;
#endif
        for (int i = 0; i < m; ++i) {
            const float err = std::abs(y[i] - v.output[i]);
            if (!(err <= 4e-6f)) {
                std::fprintf(stderr, "%s oracle mismatch case=%u row=%d got=%g expected=%g\n",
                             backend, trial, i, y[i], v.output[i]);
                return 1;
            }
            worst = std::max(worst, err);
        }
    }
    std::printf("QTIP backend=%s pinned_cases=%zu worst_abs=%g\n", backend, std::size(vectors), worst);
}
