#include "gemv.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(TEST_HIP)
#include <hip/hip_runtime.h>
#define API(n) hip##n
#define TQ1 brewery_hip_tq1_0_gemv
#define TQ2 brewery_hip_tq2_0_gemv
using Props = hipDeviceProp_t;
#elif defined(TEST_CUDA)
#include <cuda_runtime.h>
#define API(n) cuda##n
#define TQ1 brewery_cuda_tq1_0_gemv
#define TQ2 brewery_cuda_tq2_0_gemv
using Props = cudaDeviceProp;
#endif

static void require(bool ok, const char *msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); }
}

#ifdef API
#define CHECK(call) do { auto e = (call); if (e != API(Success)) { \
    std::fprintf(stderr, "%s: %s\n", #call, API(GetErrorString)(e)); std::exit(1); } } while (0)
static void select_gpu(const char *target) {
    int n;
    CHECK(API(GetDeviceCount)(&n));
    for (int i = 0; i < n; ++i) {
        Props p{};
        CHECK(API(GetDeviceProperties)(&p, i));
#ifdef TEST_HIP
        std::string arch = p.gcnArchName;
        arch = arch.substr(0, arch.find(':'));
#else
        std::string arch = "sm_" + std::to_string(p.major) + std::to_string(p.minor);
#endif
        if (arch != target) continue;
        CHECK(API(SetDevice)(i));
        std::printf("device=%s arch=%s index=%d\n", p.name, target, i);
        return;
    }
    require(false, "required architecture not present");
}
#endif

static unsigned random_word(uint32_t &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

static void test(ggml_type t, int m, int k) {
    const auto *traits = ggml_get_type_traits(t);
    int bytes = t == GGML_TYPE_TQ1_0 ? 54 : 66;
    require(traits && traits->to_float && traits->blck_size == 256 &&
            traits->type_size == size_t(bytes), "stock layout");
    size_t stride = ggml_row_size(t, k);
    size_t size = m * stride;
    std::vector<uint64_t> storage((size+7)/8);
    auto *packed = reinterpret_cast<unsigned char *>(storage.data());
    std::vector<float> w(m*k), x(k), y(m, NAN), row(k);
    uint32_t seed = 17713;
    for (int i = 0; i < m*k; ++i) {
        const int block = i/256;
        float scale = block%17 == 11 ? 0.0f : (1 + block%29) * 0.125f;
        w[i] = scale * (int(random_word(seed)%3) - 1);
    }
    for (auto &v : x) v = (int(random_word(seed)%8191) - 4095) / 4095.0f;
    require(ggml_quantize_chunk(t, w.data(), packed, 0, m, k, nullptr) == size,
            "upstream encoder byte count");

#ifdef API
    unsigned char *dw;
    float *dx, *dy;
    CHECK(API(Malloc)(&dw, size));
    CHECK(API(Malloc)(&dx, k*sizeof(float)));
    CHECK(API(Malloc)(&dy, m*sizeof(float)));
    CHECK(API(Memcpy)(dw, packed, size, API(MemcpyHostToDevice)));
    CHECK(API(Memcpy)(dx, x.data(), k*sizeof(float), API(MemcpyHostToDevice)));
    CHECK(API(Memset)(dy, 0xff, m*sizeof(float)));
    auto launch = t == GGML_TYPE_TQ1_0 ? TQ1 : TQ2;
    require(launch(dw, dx, dy, m, k-1) != 0, "partial block accepted");
    require(launch(dw, dx, dy, m, k) == 0, "packed gemv launch");
    CHECK(API(DeviceSynchronize)());
    CHECK(API(Memcpy)(y.data(), dy, m*sizeof(float), API(MemcpyDeviceToHost)));
    CHECK(API(Free)(dw)); CHECK(API(Free)(dx)); CHECK(API(Free)(dy));
#endif
    double max_error = 0;
    for (int r = 0; r < m; ++r) {
        traits->to_float(packed + r*stride, row.data(), k);
        double expected = 0, l1 = 0;
        for (int j = 0; j < k; ++j) {
            require(row[j] == w[r*k+j], "stock codec roundtrip of exact ternary fixture");
            double term = double(row[j])*x[j];
            expected += term;
            l1 += std::abs(term);
        }
#ifndef API
        // Compare the codec-based FP64 oracle to a sequential FP32 dot on original weights.
        y[r] = 0;
        for (int j = 0; j < k; ++j) y[r] += w[r*k+j]*x[j];
#endif
        double err = std::abs(y[r]-expected);
        require(err <= 1e-5 + 2e-6*l1, "dot error exceeds FP32 tolerance");
        max_error = std::max(max_error, err);
    }
    std::printf("PASS type=%s id=%d rows=%d cols=%d packed_bytes=%zu max_abs=%.9g\n",
                traits->type_name, int(t), m, k, size, max_error);
}

int main(int argc, char **argv) {
    static_assert(GGML_TYPE_TQ1_0 == 34 && GGML_TYPE_TQ2_0 == 35);
    require(!std::strcmp(ggml_commit(), "184f653"), "unexpected CPU oracle revision");
#ifdef API
    require(argc == 2, "supply required device architecture");
    select_gpu(argv[1]);
#else
    (void)argv; require(argc == 1, "unexpected argument");
#endif
    for (auto t : {GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0}) {
        test(t, 1, 256); test(t, 7, 768); test(t, 65, 4096);
    }
}
