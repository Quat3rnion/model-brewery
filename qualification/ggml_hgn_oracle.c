/* Native half of a small, offline dequantization comparison. */
#define _POSIX_C_SOURCE 200809L
#include "ggml.h"
#include "gguf.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

enum { MAX_SOURCE = 8 << 20, MAX_VALUES = 4 << 20, MAX_TENSORS = 256 };

static void refuse(const char *reason) {
    fprintf(stderr, "native oracle refused: %s\n", reason);
    exit(2);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        refuse("usage: ggml_hgn_oracle SOURCE.gguf");
    }
    const struct rlimit ram = {256 << 20, 256 << 20};
    if (setrlimit(RLIMIT_AS, &ram) != 0) {
        refuse("cannot set address-space limit");
    }
    FILE *file = fopen(argv[1], "rb");
    struct stat info;
    if (!file || fstat(fileno(file), &info) != 0 || !S_ISREG(info.st_mode)
        || info.st_size < 4 || info.st_size > MAX_SOURCE) {
        refuse("source must be a regular GGUF file at most 8 MiB");
    }
    struct gguf_context *ctx = gguf_init_from_file_ptr(file, (struct gguf_init_params) {
        .no_alloc = true, .ctx = NULL,
    });
    if (!ctx) {
        refuse("pinned GGUF parser rejected the source");
    }
    const int64_t tensors = gguf_get_n_tensors(ctx);
    if (tensors < 1 || tensors > MAX_TENSORS) {
        refuse("expected 1–256 tensors");
    }
    size_t remaining = MAX_VALUES;
    for (int64_t id = 0; id < tensors; ++id) {
        const char *name = gguf_get_tensor_name(ctx, id);
        const enum ggml_type type = gguf_get_tensor_type(ctx, id);
        if ((unsigned)type >= GGML_TYPE_COUNT || !name[0] || strlen(name) > 512) {
            refuse("invalid tensor name or type");
        }
        const struct ggml_type_traits *trait = ggml_get_type_traits(type);
        if (!trait->type_size || trait->blck_size <= 0 || (!trait->to_float && type != GGML_TYPE_F32)) {
            fprintf(stderr, "%s: no supported F32 decoder for %s\n", name, trait->type_name);
            return 2;
        }
        const int64_t *ne = gguf_get_tensor_ne(ctx, id);
        size_t count = 1;
        for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
            if (ne[axis] < 1 || (uint64_t)ne[axis] > remaining / count) {
                refuse("fixture exceeds 4 Mi decoded values");
            }
            count *= ne[axis];
        }
        remaining -= count;
        const size_t data = gguf_get_data_offset(ctx);
        const size_t offset = gguf_get_tensor_offset(ctx, id);
        const size_t bytes = gguf_get_tensor_size(ctx, id);
        if (ne[0] % trait->blck_size != 0 || bytes != count / trait->blck_size * trait->type_size
            || data > (size_t)info.st_size || offset > (size_t)info.st_size - data
            || bytes > (size_t)info.st_size - data - offset) {
            refuse("invalid tensor extent or block dimensions");
        }
        void *raw = malloc(bytes);
        float *decoded = malloc(count * sizeof(float));
        if (!raw || !decoded || fseeko(file, (off_t)(data + offset), SEEK_SET)
            || fread(raw, 1, bytes, file) != bytes) {
            refuse("cannot read bounded tensor bytes");
        }
        if (type == GGML_TYPE_F32) {
            memcpy(decoded, raw, bytes);
        } else {
            size_t stride = (size_t)(ne[0] / trait->blck_size) * trait->type_size;
            for (size_t i = 0; i < count / (size_t)ne[0]; ++i) {
                trait->to_float((const unsigned char *)raw + i * stride, decoded + i * ne[0], ne[0]);
            }
        }
        printf("%u %s %zu ", (unsigned)type, trait->type_name, count);
        for (const unsigned char *c = (const unsigned char *)name; *c; ++c) {
            printf("%02x", (unsigned)*c);
        }
        if (putchar('\n') == EOF || fwrite(decoded, sizeof(float), count, stdout) != count) {
            refuse("cannot emit decoded tensor");
        }
        free(decoded);
        free(raw);
    }
    gguf_free(ctx);
    fclose(file);
    return fflush(stdout) == EOF ? 2 : 0;
}
