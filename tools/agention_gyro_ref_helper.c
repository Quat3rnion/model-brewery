/* Private binary protocol for agention_gyro_ref.py; no model/runtime access. */
#include "tq_ref.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_WEIGHTS (32 * 128)

static int number(const char *s, int max) {
    char *end;
    long value = strtol(s, &end, 10);
    return *s && !*end && value >= 0 && value <= max ? (int) value : -1;
}

int main(int argc, char **argv) {
    const uint32_t one = 1;
    if (*(const uint8_t *) &one != 1 || sizeof(float) != 4 || FLT_MANT_DIG != 24) return 2;
    if (argc == 2 && !strcmp(argv[1], "identity")) {
        puts("brewery-agention-gyro-ref-v1 " GYRO_REVISION);
        return 0;
    }
    if (argc != 7 || strcmp(argv[1], "encode")) return 2;
    const int type = number(argv[2], 151);
    const int width = number(argv[3], MAX_WEIGHTS);
    const int n = number(argv[4], MAX_WEIGHTS);
    const int block = number(argv[5], MAX_WEIGHTS);
    const int mode = number(argv[6], 4);  /* bit 0: rotate first; bit 1: explicit signs */
    if (type < 144 || width <= 0 || n <= 0 || width % 128 || n % width
            || block < 0 || (block && (width % block || (block & (block - 1))))
            || mode < 0 || mode > 3 || (!block && mode)) return 2;

    float values[MAX_WEIGHTS];
    uint8_t packed[(MAX_WEIGHTS / 128) * 58];
    int8_t signs[MAX_WEIGHTS];
    const int8_t *s = mode & 2 ? signs : NULL;
    if (fread(values, sizeof(float), n, stdin) != (size_t) n
            || (s && fread(signs, 1, width, stdin) != (size_t) width)
            || fgetc(stdin) != EOF) return 2;
    if (s) for (int i = 0; i < width; ++i) if (s[i] != 1 && s[i] != -1) return 2;
    if (mode & 1) for (int i = 0; i < n; i += width)
        if (tq_hadamard_rotate(values + i, width, block, s)) return 2;
    for (int i = 0; i < n; ++i) if (!isfinite(values[i])) return 2;

    const size_t size = tq_row_bytes(type, n);
    if (!size || size > sizeof(packed)
            || tq_quantize_row_ref(type, values, packed, n, NULL) != size
            || tq_dequantize_row(type, packed, values, n)) return 2;
    if (fwrite(packed, 1, size, stdout) != size
            || fwrite(values, sizeof(float), n, stdout) != (size_t) n) return 2;
    if (block) {
        for (int i = 0; i < n; i += width)
            if (tq_hadamard_unrotate(values + i, width, block, s)) return 2;
        if (fwrite(values, sizeof(float), n, stdout) != (size_t) n) return 2;
    }
    return fflush(stdout) == EOF ? 2 : 0;
}
