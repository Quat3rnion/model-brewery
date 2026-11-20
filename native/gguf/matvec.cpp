#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool ok, const char * reason) {
    if (!ok) {
        throw std::runtime_error(reason);
    }
}

// Keep the numeric ID check independent of enum names in either ggml build.
std::string canonical_name(std::string s) {
    for (auto & c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

struct Metrics {
    double max_abs = 0;
    double max_ratio = 0;
    int64_t count = 0;
};

uint32_t sample(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

void exercise(ggml_backend_t backend, ggml_backend_dev_t device, ggml_type type,
              const ggml_type_traits & traits, bool gpu, bool requires_imatrix,
              int64_t width, int64_t rows, uint32_t seed, Metrics & metrics) {
    uint32_t state = seed;
    std::vector<float> weights(width * rows);
    for (auto & w : weights) {
        w = std::ldexp(static_cast<int>(sample(state) % 253) - 126, -6);
    }
    // The +/-127 anchor gives exact activation values with 127-level GGML
    // quantization (32/64/128/256-element blocks) and FP16/BF16 conversion.
    std::vector<float> input(width), importance(width);
    for (int64_t k = 0; k < width; ++k) {
        const int n = k % 32 == 0 ? ((k / 32 + seed) % 2 == 0 ? 127 : -127)
                                 : static_cast<int>(sample(state) % 253) - 126;
        input[k] = std::ldexp(n, -7);
        importance[k] = 1.0f + static_cast<float>(k % 17) / 16.0f;
    }

    const auto row_size = ggml_row_size(type, width);
    // Aligned storage for ggml's block types; only the exact packed bytes are transferred.
    std::vector<uint64_t> packed((rows * row_size + 7) / 8);
    const size_t packed_size = ggml_quantize_chunk(
        type, weights.data(), packed.data(), 0, rows, width,
        requires_imatrix ? importance.data() : nullptr);
    check(packed_size == rows * row_size, "packed_byte_count");

    const auto ctx_size = 3 * ggml_tensor_overhead() + ggml_graph_overhead();
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({ctx_size, nullptr, true}), ggml_free);
    check(bool(ctx), "context_init");
    auto * w = ggml_new_tensor_2d(ctx.get(), type, width, rows);
    auto * x = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, width);
    auto * y = ggml_mul_mat(ctx.get(), w, x);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, y);
    check(w->type == type && w->nb[1] == row_size && ggml_nbytes(w) == packed_size &&
          y->type == GGML_TYPE_F32 && ggml_nelements(y) == rows, "graph_layout");

    auto * buft = ggml_backend_dev_buffer_type(device);
    check(buft != nullptr, "missing_default_buffer_type");
    if (gpu) {
        check(ggml_backend_buft_get_device(buft) == device, "wrong_buffer_device");
        check(!ggml_backend_buft_is_host(buft), "gpu_host_buffer");
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buf(
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft), ggml_backend_buffer_free);
    check(bool(buf), "buffer_alloc");
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto * node : {w, x, y}) {
        check(node->buffer && ggml_backend_buffer_get_type(node->buffer) == buft,
              "wrong_tensor_buffer");
    }
    check(ggml_backend_dev_supports_op(device, y), "backend_rejects_mul_mat");

    ggml_backend_tensor_set(w, packed.data(), 0, packed_size);
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_memset(y, 0xff, 0, ggml_nbytes(y));
    // Call only this backend; no scheduler is available to route the op to CPU.
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph_compute");
    ggml_backend_synchronize(backend);
    std::vector<float> actual(rows);
    ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * sizeof(float));

    // Decode the very same packed bytes with the host format codec, then take a
    // scalar, double-precision dot. No backend or ggml dot kernels in the oracle.
    const auto * bytes = reinterpret_cast<const unsigned char *>(packed.data());
    std::vector<float> decoded(width);
    bool nonzero = false;
    for (int64_t r = 0; r < rows; ++r) {
        traits.to_float(bytes + r * row_size, decoded.data(), width);
        double ref = 0;
        double l1 = 0;
        for (int64_t k = 0; k < width; ++k) {
            const double term = static_cast<double>(decoded[k]) * input[k];
            ref += term;
            l1 += std::abs(term);
        }
        check(std::isfinite(ref) && std::isfinite(l1) && l1 > 0, "invalid_reference");
        nonzero |= std::abs(ref) > 1.0;
        const double error = std::abs(actual[r] - ref);
        const double limit = 1e-4 + 1e-4 * l1;
        if (!(error <= limit)) {
            std::fprintf(stderr, "width=%lld row=%lld actual=%.9g ref=%.9g error=%.9g limit=%.9g\n",
                         static_cast<long long>(width), static_cast<long long>(r), actual[r], ref,
                         error, limit);
            throw std::runtime_error("dequant_dot_mismatch");
        }
        metrics.max_abs = std::max(metrics.max_abs, error);
        metrics.max_ratio = std::max(metrics.max_ratio, error / limit);
        ++metrics.count;
    }
    check(nonzero, "degenerate_fixture");
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--revision") {
        std::puts(ggml_commit());
        return 0;
    }
    if (argc != 8) {
        std::fprintf(stderr, "usage: matvec REGISTRY DEVICE_INDEX FORMAT ID BLOCK BYTES IMATRIX(0|1)\n");
        return 2;
    }
    const std::string registry = argv[1];
    try {
        check(registry == "CPU" || registry == "CUDA" || registry == "ROCm", "invalid_registry");
        const bool gpu = registry != "CPU";
        const auto index = std::stoul(argv[2]);
        const auto id = std::stol(argv[4]);
        check(id >= 0 && id < GGML_TYPE_COUNT, "missing_type_id");
        const auto type = static_cast<ggml_type>(id);
        const auto & traits = *ggml_get_type_traits(type);
        check(traits.type_name && canonical_name(traits.type_name) == argv[3], "wrong_type_name");
        check(traits.blck_size == std::stoll(argv[5]) && traits.type_size == std::stoul(argv[6]),
              "wrong_block_layout");
        check(traits.to_float != nullptr, "missing_scalar_dequantizer");
        auto * reg = ggml_backend_reg_by_name(registry.c_str());
        check(reg && registry == ggml_backend_reg_name(reg), "missing_backend_registry");
        check(index < ggml_backend_reg_dev_count(reg), "missing_backend_device");
        auto * device = ggml_backend_reg_dev_get(reg, index);
        check(device && ggml_backend_dev_backend_reg(device) == reg, "wrong_device_registry");
        const auto dev_type = ggml_backend_dev_type(device);
        check(gpu ? (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU)
                  : dev_type == GGML_BACKEND_DEVICE_TYPE_CPU, "wrong_device_kind");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_dev_init(device, nullptr), ggml_backend_free);
        check(backend && ggml_backend_get_device(backend.get()) == device, "backend_init_wrong_device");
        if (!gpu) {
            auto * set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
            check(set_threads != nullptr, "missing_cpu_thread_control");
            set_threads(backend.get(), 2);
        }
        const std::string imatrix = argv[7];
        check(imatrix == "0" || imatrix == "1", "invalid_imatrix_flag");
        check(!ggml_quantize_requires_imatrix(type) || imatrix == "1", "missing_catalog_imatrix");
        Metrics metrics;
        exercise(backend.get(), device, type, traits, gpu, imatrix == "1", 512, 33, 187u, metrics);
        exercise(backend.get(), device, type, traits, gpu, imatrix == "1", 1024, 65, 499u, metrics);
        exercise(backend.get(), device, type, traits, gpu, imatrix == "1", 1536, 97, 2731u, metrics);
        ggml_quantize_free();
        std::printf("PASS\t%s\t%lld\t%.9g\t%.9g\t-\n", ggml_backend_dev_name(device),
                    static_cast<long long>(metrics.count), metrics.max_abs, metrics.max_ratio);
        return 0;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "%s: %s/%s\n", ex.what(), registry.c_str(), argv[3]);
        std::printf("FAIL\t-\t-\t-\t-\t%s\n", ex.what());
        return 1;
    }
}
