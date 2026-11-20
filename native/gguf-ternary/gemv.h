#pragma once

// Enqueue one FP32 GEMV on the default stream. Device buffers must not alias;
// weights are contiguous packed GGML rows, x[K], y[M]. Caller synchronizes.
// Returns zero on successful enqueue, nonzero for invalid args/launch failure.
#define BREWERY_TQ_PARAMS const unsigned char *w, const float *x, float *y, int m, int k
extern "C" int brewery_cuda_tq1_0_gemv(BREWERY_TQ_PARAMS);
extern "C" int brewery_cuda_tq2_0_gemv(BREWERY_TQ_PARAMS);
extern "C" int brewery_hip_tq1_0_gemv(BREWERY_TQ_PARAMS);
extern "C" int brewery_hip_tq2_0_gemv(BREWERY_TQ_PARAMS);
