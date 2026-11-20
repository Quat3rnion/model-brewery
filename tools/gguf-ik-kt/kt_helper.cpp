#include "ggml.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#if !defined(__linux__) || !defined(__x86_64__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "The pinned KT CPU helper requires Linux x86-64, little endian"
#endif

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(GGML_TYPE_IQ1_KT == 158 && GGML_TYPE_IQ2_KT == 153 &&
              GGML_TYPE_IQ3_KT == 154 && GGML_TYPE_IQ4_KT == 155);
static_assert(GGML_TYPE_F32 == 0 && GGML_TYPE_F16 == 1 && GGML_TYPE_BF16 == 30);

constexpr auto commit = "7c00f8788e303bfba2886138e288effb0fd5fbbf";
constexpr int64_t max_cols = 65536;
constexpr int64_t max_elements = 8 << 20;

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static int64_t number(const std::string &arg) {
    int64_t v = 0;
    auto result = std::from_chars(arg.data(), arg.data() + arg.size(), v);
    require(result.ec == std::errc{} && result.ptr == arg.data() + arg.size() && v >= 0,
            "invalid numeric argument");
    return v;
}

static bool kt(int64_t type) {
    return type == GGML_TYPE_IQ1_KT || type == GGML_TYPE_IQ2_KT ||
           type == GGML_TYPE_IQ3_KT || type == GGML_TYPE_IQ4_KT;
}

static void read_bytes(std::ifstream &file, void *dest, size_t size) {
    require(bool(file.read(static_cast<char *>(dest), size)), "short tensor/importance read");
}

struct Source {
    std::ifstream file;
    int64_t kind;
    std::vector<ggml_fp16_t> fp16;
    std::vector<ggml_bf16_t> bf16;
    std::vector<float> floats;

    Source(int64_t type, int64_t width, int64_t offset)
        : file("/source", std::ios::binary), kind(type), floats(width) {
        require(kind == GGML_TYPE_F32 || kind == GGML_TYPE_F16 || kind == GGML_TYPE_BF16,
                "expected F32, F16 or BF16 source");
        require(bool(file.seekg(offset)), "invalid source offset");
        if (kind == GGML_TYPE_F16) fp16.resize(width);
        if (kind == GGML_TYPE_BF16) bf16.resize(width);
    }

    const float *read_row() {
        if (kind == GGML_TYPE_F32) {
            read_bytes(file, floats.data(), floats.size() * sizeof(float));
        } else if (kind == GGML_TYPE_F16) {
            read_bytes(file, fp16.data(), fp16.size() * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(fp16.data(), floats.data(), floats.size());
        } else {
            read_bytes(file, bf16.data(), bf16.size() * sizeof(ggml_bf16_t));
            ggml_bf16_to_fp32_row(bf16.data(), floats.data(), floats.size());
        }
        for (float f : floats) {
            // Prevent arithmetic overflow in the fork's square-weight calculations.
            require(std::isfinite(f) && std::abs(f) <= 1.0e6f, "invalid source weight");
        }
        return floats.data();
    }
};

static void dimensions(int64_t width, int64_t rows) {
    require(width > 0 && width <= max_cols && width % 32 == 0 &&
            rows > 0 && rows <= max_elements / width, "tensor exceeds KT dimensions");
}

static void encode(char **argv) {
    const auto id = number(argv[2]);
    const auto kind = number(argv[3]);
    const auto offset = number(argv[4]);
    const auto width = number(argv[5]);
    const auto rows = number(argv[6]);
    const auto expected_size = number(argv[7]);
    const auto importance = number(argv[8]);
    require(kt(id) && importance <= 1, "expected exact type or invalid importance flag");
    dimensions(width, rows);
    const auto type = static_cast<ggml_type>(id);
    const auto row_size = ggml_row_size(type, width);
    require(ggml_row_blck_size(type) == 32 && row_size * rows == size_t(expected_size),
            "backend type layout mismatch");
    // ggml_fp16_to_fp32_row uses the lookup table initialized by ggml_init.
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({ggml_tensor_overhead(), nullptr, true}), ggml_free);
    require(bool(ctx), "cannot initialize GGML");
    Source source(kind, width, offset);
    std::vector<float> weights;
    if (importance) {
        std::ifstream file("/importance", std::ios::binary);
        weights.resize(width);
        read_bytes(file, weights.data(), weights.size() * sizeof(float));
        require(file.peek() == EOF, "invalid importance length");
        bool any_positive = false;
        for (float f : weights) {
            require(std::isfinite(f) && f >= 0 && f <= 1.0e6f, "invalid importance weight");
            any_positive = any_positive || f > 0;
        }
        require(any_positive, "importance weights are all zero");
    }
    std::vector<unsigned char> row(row_size);
    const quantize_user_data options{};
    // The upstream encoder also writes diagnostics to stdout (notably IQ4_KT).
    // Keep that channel separate from the packed bytes.
    const int output_fd = dup(STDOUT_FILENO);
    require(output_fd >= 0 && dup2(STDERR_FILENO, STDOUT_FILENO) >= 0,
            "cannot separate encoder diagnostics");
    const auto close_stream = [](FILE *f) { std::fclose(f); };
    std::unique_ptr<FILE, decltype(close_stream)> output(fdopen(output_fd, "wb"), close_stream);
    require(bool(output), "cannot open encoder output");
    for (int64_t i = 0; i < rows; ++i) {
        std::fill(row.begin(), row.end(), 0);
        require(ggml_quantize_chunk(type, source.read_row(), row.data(), 0, 1, width,
                                    importance ? weights.data() : nullptr, &options)
                    == row_size, "encoder returned wrong size");
        require(std::fwrite(row.data(), 1, row.size(), output.get()) == row.size(), "short encoder write");
    }
    require(std::fflush(output.get()) == 0, "short encoder write");
}

static void readback() {
    int64_t count, base, alignment;
    require(bool(std::cin >> count >> base >> alignment) && count > 0 && count <= 16384,
            "invalid readback request");
    ggml_context * tensors = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(
        gguf_init_from_file("/output", {true, &tensors}), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(tensors, ggml_free);
    require(gguf && ctx && gguf_get_version(gguf.get()) == 3 &&
            gguf_get_n_tensors(gguf.get()) == count &&
            gguf_get_data_offset(gguf.get()) == size_t(base) &&
            gguf_get_alignment(gguf.get()) == size_t(alignment), "GGUF header readback mismatch");
    std::ifstream payload("/output", std::ios::binary);
    int64_t total = 0;
    for (int i = 0; i < count; ++i) {
        std::string name;
        int64_t id, relative, size, ne[4], source_id, source_offset;
        require(bool(std::cin >> name >> id >> relative >> size >> ne[0] >> ne[1] >> ne[2] >> ne[3]
                                  >> source_id >> source_offset), "invalid readback tensor descriptor");
        const ggml_tensor *t = ggml_get_tensor(ctx.get(), name.c_str());
        require(t && name == gguf_get_tensor_name(gguf.get(), i) && t->type == id &&
                gguf_get_tensor_type(gguf.get(), i) == id &&
                gguf_get_tensor_offset(gguf.get(), i) == size_t(relative) &&
                ggml_nbytes(t) == size_t(size) &&
                std::equal(ne, ne + 4, t->ne), "GGUF tensor readback mismatch");
        if (!kt(id)) continue;
        dimensions(ne[0], ne[1]);
        require(ne[2] == 1 && ne[3] == 1, "unexpected KT tensor rank");
        total += ne[0] * ne[1];
        require(total <= max_elements, "readback exceeds element budget");
        const auto decode = ggml_internal_get_type_traits(t->type).to_float;
        require(decode && bool(payload.seekg(base + relative)), "cannot read KT data");
        Source source(source_id, ne[0], source_offset);
        std::vector<unsigned char> packed(t->nb[1]);
        std::vector<float> decoded(ne[0]);
        double sse = 0, max_error = 0;
        for (int64_t j = 0; j < ne[1]; ++j) {
            read_bytes(payload, packed.data(), packed.size());
            decode(packed.data(), decoded.data(), ne[0]);
            const auto original = source.read_row();
            for (int64_t k = 0; k < ne[0]; ++k) {
                require(std::isfinite(decoded[k]), "non-finite KT readback");
                const double error = std::abs(double(decoded[k]) - original[k]);
                sse += error * error;
                max_error = std::max(max_error, error);
            }
        }
        std::printf("%s %.17g %.17g\n", name.c_str(), std::sqrt(sse / (ne[0] * ne[1])), max_error);
    }
    std::string extra;
    require(!(std::cin >> extra) && std::fflush(stdout) == 0, "invalid readback request or output");
}

int main(int argc, char **argv) {
    try {
        require(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                __builtin_cpu_supports("f16c"), "required CPU extensions: AVX2, FMA, F16C");
        if (argc == 2 && std::string(argv[1]) == "identify") {
            std::printf("brewery-ik-kt-v1 %s 158 153 154 155\n", commit);
        } else if (argc == 9 && std::string(argv[1]) == "encode") {
            encode(argv);
        } else if (argc == 2 && std::string(argv[1]) == "readback") {
            readback();
        } else {
            throw std::runtime_error("unsupported helper command");
        }
        return 0;
    } catch (const std::exception &err) {
        std::fprintf(stderr, "ik-kt-helper: %s\n", err.what());
        return 1;
    }
}
