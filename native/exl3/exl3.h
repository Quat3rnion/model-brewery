#ifndef BREWERY_EXL3_H
#define BREWERY_EXL3_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One aligned 128-input x 128-output block, input-major. The packed buffer
 * is 64 consecutive native 16x16 EXL3 tiles, input tile major. Gather its
 * eight stripes before calling if the checkpoint has more than 128 outputs.
 * No checkpoint parser, Python interpreter or tensor framework in this ABI.
 * twice_bits = twice the per-weight rate; 2,4,...,16 for every codebook,
 * and 3,5,7 for MUL1 only. Size in bytes = 1024 * twice_bits.
 * 0 means success. Buffers are host-accessible, non-overlapping and sized
 * by the caller. Factors and activations must be finite. No backend fallback.
 */
enum exl3_codebook { EXL3_3INST = 0, EXL3_MCG = 1, EXL3_MUL1 = 2 };
int exl3_encode(const float *rotated, int twice_bits, int book, uint8_t *packed, size_t bytes);
int exl3_decode(const uint8_t *packed, size_t bytes, int twice_bits, int book, float *rotated);
/* y = x diag(su) H W_rot H diag(sv); H is normalized, order 128.
 * su, sv, x and y each contain 128 floats. Bias is not applied here. */
int exl3_forward(const uint8_t *packed, size_t bytes, int twice_bits, int book,
                 const float *su, const float *sv, const float *x, float *y);
int exl3_cuda_forward(const uint8_t *packed, size_t bytes, int twice_bits, int book,
                      const float *su, const float *sv, const float *x, float *y);
int exl3_rocm_forward(const uint8_t *packed, size_t bytes, int twice_bits, int book,
                      const float *su, const float *sv, const float *x, float *y);
#ifdef __cplusplus
}
#endif
#endif
