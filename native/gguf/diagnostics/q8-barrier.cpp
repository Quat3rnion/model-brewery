// Diagnostic only: change ordering at one boundary without modifying either pin.
#include "ggml.h"
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void quantize_row_q8_1_cuda(const float * x, const int32_t * ids, void * packed,
                          ggml_type type, int64_t width, int64_t stride1,
                          int64_t stride2, int64_t stride3, int64_t padded_width,
                          int64_t n1, int64_t n2, int64_t n3, hipStream_t stream) {
    using Fn = decltype(&quantize_row_q8_1_cuda);
    static const auto original = reinterpret_cast<Fn>(
        dlsym(RTLD_NEXT, "_Z22quantize_row_q8_1_cudaPKfPKiPv9ggml_typellllllllP12ihipStream_t"));
    const char * mode = std::getenv("GGUF_Q8_BARRIER");
    const bool before = mode && std::strcmp(mode, "before") == 0;
    const bool after = mode && std::strcmp(mode, "after") == 0;
    const bool none = mode && std::strcmp(mode, "none") == 0;
    if (!original || !(before || after || none)) {
        std::fprintf(stderr, "gguf-q8-barrier: unable to resolve quantizer or select a mode\n");
        std::abort();
    }
    const auto sync = [&] {
        if (const auto err = hipStreamSynchronize(stream); err != hipSuccess) {
            std::fprintf(stderr, "gguf-q8-barrier: %s\n", hipGetErrorString(err));
            std::abort();
        }
    };
    std::fprintf(stderr, "gguf-q8-barrier: mode=%s width=%lld\n", mode, static_cast<long long>(width));
    if (before) {
        sync();
    }
    original(x, ids, packed, type, width, stride1, stride2, stride3, padded_width, n1, n2, n3,
             stream);
    if (after) {
        sync();
    }
}
