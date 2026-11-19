// Local experiment, not a native GGUF parity tool. Build against the pinned ggml.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s failed\n", __LINE__, #x); std::exit(2); } } while (0)

static void read(FILE * f, size_t off, void * buf, size_t n) {
    CHECK(fseeko(f, off, SEEK_SET) == 0 && fread(buf, 1, n, f) == n);
}

static std::vector<float> project(ggml_backend_dev_t dev, const std::vector<float> & w,
                                  const std::vector<float> & x) {
    auto * b = ggml_backend_dev_init(dev, nullptr);
    auto * c = ggml_init({ggml_tensor_overhead()*4 + ggml_graph_overhead(), nullptr, true});
    CHECK(b && c);
    auto * tw = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1024, 1024);
    auto * tx = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1024, 1);
    auto * ty = ggml_mul_mat(c, tw, tx);
    auto * g = ggml_new_graph(c);
    ggml_build_forward_expand(g, ty);
    CHECK(ggml_graph_n_nodes(g) == 1 && ggml_backend_supports_op(b, ty));
    auto * buf = ggml_backend_alloc_ctx_tensors(c, b);
    CHECK(buf && ggml_backend_buffer_get_type(ty->buffer) == ggml_backend_dev_buffer_type(dev));
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
        CHECK(!ggml_backend_buffer_is_host(ty->buffer));
    }
    std::vector<float> result(1024);
    ggml_backend_tensor_set(tw, w.data(), 0, ggml_nbytes(tw));
    ggml_backend_tensor_set(tx, x.data(), 0, ggml_nbytes(tx));
    CHECK(ggml_backend_graph_compute(b, g) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(b);
    ggml_backend_tensor_get(ty, result.data(), 0, ggml_nbytes(ty));
    std::fprintf(stderr, "%s: completed F32 MUL_MAT, output buffer %s\n",
                 ggml_backend_dev_description(dev), ggml_backend_buffer_name(ty->buffer));
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    ggml_backend_free(b);
    return result;
}

static int op(const char * input, const char * weight) {
    auto * xf = fopen(input, "rb");
    auto * wf = fopen(weight, "rb");
    CHECK(xf && wf);
    std::vector<float> x(1024), w(1024*1024);
    const auto stride = ggml_row_size(GGML_TYPE_Q4_K, 1024);
    std::vector<uint8_t> packed(stride * 1024);
    read(xf, 0, x.data(), x.size()*4);
    read(wf, 0, packed.data(), packed.size());
    CHECK(fgetc(xf) == EOF && fgetc(wf) == EOF);
    fclose(xf);
    fclose(wf);
    ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(packed.data(), w.data(), w.size());
    ggml_backend_load_all();
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * cuda = ggml_backend_dev_by_name("CUDA0");
    CHECK(cpu && cuda && ggml_backend_dev_type(cuda) == GGML_BACKEND_DEVICE_TYPE_GPU);
    const auto a = project(cpu, w, x);
    const auto b = project(cuda, w, x);
    puts("row\tcpu_f32\tcuda_f32\tfp64");
    for (size_t j=0; j<a.size(); ++j) {
        double ref = 0;
        for (size_t k=0; k<x.size(); ++k) ref += double(x[k])*w[j*x.size()+k];
        CHECK(std::isfinite(a[j]) && std::isfinite(b[j]) && std::isfinite(ref));
        printf("%zu\t%.9g\t%.9g\t%.17g\n",j,a[j],b[j],ref);
    }
    return 0;
}

static int convert(const char * model, const char * output, const char * parent_sha) {
    CHECK(strlen(parent_sha) == 64 && strspn(parent_sha, "0123456789abcdef") == 64);
    ggml_context * tensors = nullptr;
    auto * src = gguf_init_from_file(model, {true, &tensors});
    CHECK(src && tensors && gguf_find_key(src, "split.count") < 0);
    auto * dst = gguf_init_empty();
    CHECK(dst);
    gguf_set_kv(dst, src);
    gguf_remove_key(dst, "general.quantization_version");
    gguf_set_val_u32(dst, "general.file_type", 0); // ALL_F32
    const auto name_id = gguf_find_key(src, "general.name");
    CHECK(name_id >= 0 && gguf_get_kv_type(src, name_id) == GGUF_TYPE_STRING);
    const auto name = std::string(gguf_get_val_str(src, name_id)) + " (derived F32 reference)";
    gguf_set_val_str(dst, "general.name", name.c_str());
    gguf_set_val_str(dst, "brewery.reference.source_sha256", parent_sha);
    gguf_set_val_str(dst, "brewery.reference.mode", "dequantized-f32-v1");
    const auto n = gguf_get_n_tensors(src);
    CHECK(n > 0);
    size_t weight_bytes = 0;
    for (int64_t i=0; i<n; ++i) {
        auto * t = ggml_get_tensor(tensors, gguf_get_tensor_name(src, i));
        CHECK(t);
        const auto * traits = ggml_get_type_traits(t->type);
        CHECK(t->type == GGML_TYPE_F32 || (traits && traits->to_float));
        gguf_add_tensor(dst, t);
        gguf_set_tensor_type(dst, t->name, GGML_TYPE_F32);
        weight_bytes += gguf_get_tensor_size(dst, i);
    }
    printf("tensors=%ld f32_tensor_bytes=%zu gguf_bytes=%zu\n", n, weight_bytes,
           gguf_get_meta_size(dst) + gguf_get_tensor_offset(dst, n-1)
           + GGML_PAD(gguf_get_tensor_size(dst, n-1), gguf_get_alignment(dst)));
    if (strcmp(output, "--size-only") == 0) return 0;
    auto * input = fopen(model, "rb");
    // Never overwrite an original or an earlier diagnostic.
    int fd = open(output, O_WRONLY | O_CREAT | O_EXCL, 0600);
    CHECK(input && fd >= 0);
    auto * out = fdopen(fd, "wb");
    CHECK(out && gguf_write_to_file_ptr(dst, out, true));
    const auto start = ftello(out);
    CHECK(start >= 0 && size_t(start) == gguf_get_meta_size(dst));
    const auto data_off = gguf_get_data_offset(src);
    const auto alignment = gguf_get_alignment(dst);
    std::vector<char> padding(alignment);
    for (int64_t i=0; i<n; ++i) {
        const auto * t = ggml_get_tensor(tensors, gguf_get_tensor_name(src, i));
        CHECK(ftello(out) == start + off_t(gguf_get_tensor_offset(dst, i)));
        const auto * traits = ggml_get_type_traits(t->type);
        const auto row_bytes = ggml_row_size(t->type, t->ne[0]);
        // Work in blocks of contiguous rows (including MoE expert matrices).
        constexpr int64_t chunk_rows = 1024;
        std::vector<uint8_t> packed(row_bytes * std::min(chunk_rows, ggml_nrows(t)));
        std::vector<float> unpacked(t->ne[0] * std::min(chunk_rows, ggml_nrows(t)));
        for (int64_t j=0; j<ggml_nrows(t); j+=chunk_rows) {
            const auto rows = std::min(chunk_rows, ggml_nrows(t)-j);
            read(input, data_off + gguf_get_tensor_offset(src, i) + j * row_bytes,
                 packed.data(), rows * row_bytes);
            if (t->type == GGML_TYPE_F32) {
                CHECK(fwrite(packed.data(), row_bytes, rows, out) == size_t(rows));
            } else {
                traits->to_float(packed.data(), unpacked.data(), rows*t->ne[0]);
                CHECK(fwrite(unpacked.data(), sizeof(float)*t->ne[0], rows, out) == size_t(rows));
            }
        }
        const auto bytes = gguf_get_tensor_size(dst, i);
        const auto pad = GGML_PAD(bytes,alignment)-bytes;
        CHECK(fwrite(padding.data(), 1, pad, out) == pad);
    }
    CHECK(fclose(out) == 0);
    fclose(input);
    gguf_free(dst);
    gguf_free(src);
    ggml_free(tensors);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 4 && strcmp(argv[1], "op") == 0) return op(argv[2], argv[3]);
    if (argc == 5 && strcmp(argv[1], "convert") == 0) return convert(argv[2], argv[3], argv[4]);
    fprintf(stderr, "usage: reference_f32 op X.f32 Q4_K.bin | convert MODEL OUTPUT_OR_--size-only SOURCE_SHA256\n");
    return 2;
}
