#include "glq.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef GLQ_TEST_GPU
#include "glq_gpu.h"
#if defined(__HIPCC__)
#define GPU(name) hip##name
#else
#define GPU(name) cuda##name
#endif
#endif

static void require(bool condition, const char* what) {
    if (!condition) throw std::runtime_error(what);
}

template <class T>
static void read(std::istream& source, T* ptr, std::size_t n) {
    require(bool(source.read(reinterpret_cast<char*>(ptr), n * sizeof(T))), "short fixture");
}

static void compare(const std::vector<float>& actual, const std::vector<float>& expected,
                    const char* surface) {
    float max_error = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::abs(actual[i] - expected[i]);
        if (!(diff <= 1e-4f + 1e-5f * std::abs(expected[i]))) {
            std::cerr << surface << " mismatch at " << i << ": " << actual[i]
                      << " vs " << expected[i] << '\n';
            throw std::runtime_error("packed linear mismatch");
        }
        max_error = std::max(max_error, diff);
    }
    std::cout << ' ' << surface << " max_abs=" << max_error;
}

#ifdef GLQ_TEST_GPU
static void gpu_check(trellis_ref::Error result) {
    if (result != trellis_ref::success) throw std::runtime_error(GPU(GetErrorString)(result));
}

template <class T>
class DeviceBuffer {
public:
    T* ptr = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        if (count) gpu_check(GPU(Malloc)(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr) (void)GPU(Free)(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
};

static void run_gpu(glq::Layer layer, int batch, const std::vector<float>& x,
                    const std::vector<float>& expected, trellis_ref::Stream stream) {
    DeviceBuffer<uint8_t> packed(layer.trellis_packed.size), residual(layer.trellis_packed2.size);
    DeviceBuffer<uint8_t> su(layer.su.size), sv(layer.sv.size);
    auto move = [](glq::Bytes& bytes, DeviceBuffer<uint8_t>& buf) {
        if (bytes.size)
            gpu_check(GPU(Memcpy)(buf.ptr, bytes.data, bytes.size, GPU(MemcpyHostToDevice)));
        bytes.data = buf.ptr;
    };
    move(layer.trellis_packed, packed);
    move(layer.trellis_packed2, residual);
    move(layer.su, su);
    move(layer.sv, sv);
    DeviceBuffer<float> xx(x.size()), work(x.size()), yy(expected.size());
    gpu_check(GPU(Memcpy)(xx.ptr, x.data(), x.size() * sizeof(float), GPU(MemcpyHostToDevice)));
    require(glq::linear_gpu(layer, batch, xx.ptr, x.size(), yy.ptr, expected.size(),
                            work.ptr, x.size() - 1, stream) == trellis_ref::invalid, "workspace accepted");
    gpu_check(glq::linear_gpu(layer, batch, xx.ptr, x.size(), yy.ptr, expected.size(),
                              work.ptr, x.size(), stream));
    gpu_check(GPU(StreamSynchronize)(stream));
    std::vector<float> actual(expected.size());
    gpu_check(GPU(Memcpy)(actual.data(), yy.ptr, actual.size() * sizeof(float), GPU(MemcpyDeviceToHost)));
    compare(actual, expected, "GPU");
}
#endif

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: test_glq glq_vectors.bin");
        const uint16_t endian = 1;
        require(*reinterpret_cast<const uint8_t*>(&endian) == 1 && sizeof(float) == 4
                && std::numeric_limits<float>::is_iec559, "unsupported fixture host");
        require(!glq::validate({}), "empty layer accepted");
        std::ifstream source(argv[1], std::ios::binary);
        char magic[8];
        uint32_t cases;
        read(source, magic, 8);
        read(source, &cases, 1);
        require(std::memcmp(magic, "GLQ3V1\0", 8) == 0 && cases == 8, "bad fixture header");
#ifdef GLQ_TEST_GPU
        trellis_ref::Stream stream = nullptr;
        gpu_check(GPU(StreamCreate)(&stream));
#endif
        for (uint32_t test = 0; test < cases; ++test) {
            uint32_t dims[4];
            float scales[2];
            read(source, dims, 4);
            read(source, scales, 2);
            const int m = dims[0], n = dims[1], bpw = dims[2], batch = dims[3];
            require(batch == 3 && (test < 7 ? m == 32 && n == 64 && dims[2] == test + 2
                                           : m == 96 && n == 96 && bpw == 5), "bad fixture case");
            const std::size_t unit = m * n / 8;
            std::vector<uint8_t> packed(std::min(4, bpw) * unit);
            std::vector<uint8_t> residual(bpw > 4 ? (bpw - 4) * unit : 0), su(m * 2), sv(n * 2);
            for (auto* buf : {&packed, &residual, &su, &sv})
                if (!buf->empty()) read(source, buf->data(), buf->size());
            std::vector<float> x(batch * n), y(batch * m), expected(batch * m);
            read(source, x.data(), x.size());
            read(source, expected.data(), expected.size());
            glq::Layer layer{glq::Layout::kernel_3inst_block_diagonal, m, n, bpw,
                             {packed.data(), packed.size()},
                             {residual.empty() ? nullptr : residual.data(), residual.size()},
                             {su.data(), su.size()}, {sv.data(), sv.size()}, scales[0], scales[1]};
            auto broken = layer;
            broken.layout = glq::Layout::unknown;
            require(!glq::validate(broken), "foreign layout accepted");
            broken = layer;
            --broken.trellis_packed.size;
            require(!glq::validate(broken), "short primary accepted");
            broken = layer;
            broken.wscale = std::numeric_limits<float>::quiet_NaN();
            require(!glq::validate(broken), "NaN scale accepted");
            broken = layer;
            broken.rows -= 16;
            require(!glq::validate(broken), "unpaired tile accepted");
            broken = layer;
            broken.trellis_packed2 = {};
            if (bpw > 4) {
                require(!glq::validate(broken), "missing residual accepted");
                broken = layer;
                broken.inv_resid_scale2 = 0;
                require(!glq::validate(broken), "missing residual scale accepted");
            } else {
                broken.trellis_packed2 = layer.trellis_packed;
                require(!glq::validate(broken), "unexpected residual accepted");
            }
            su[0] = 1;
            require(!glq::validate(layer), "non-sign accepted");
            su[0] = 0;
            require(!glq::linear(layer, batch, x.data(), x.size() - 1, y.data(), y.size()),
                    "short input accepted");
            require(glq::linear(layer, batch, x.data(), x.size(), y.data(), y.size()), "CPU refused fixture");
            std::cout << m << 'x' << n << " bpw=" << bpw;
            compare(y, expected, "CPU");
#ifdef GLQ_TEST_GPU
            run_gpu(layer, batch, x, expected, stream);
#endif
            std::cout << '\n';
        }
        require(source.peek() == EOF, "trailing fixture bytes");
#ifdef GLQ_TEST_GPU
        gpu_check(GPU(StreamDestroy)(stream));
#endif
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
