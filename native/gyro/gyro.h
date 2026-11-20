#pragma once

#include <cstddef>
#include <cstdint>

namespace gyro {

enum class Basis { original, stored };

// One dense logical matrix in the Agention namespace. The caller resolves
// GGUF namespace, version and tensor rotation metadata before making this.
// Packed rows have no inter-row padding; rotation_block==0 means unrotated.
struct Matrix {
    int type;
    unsigned state_bits;
    std::size_t rows, columns, packed_bytes;
    std::size_t rotation_block;  // Normalized Walsh-Hadamard block width.
};

bool valid(const Matrix& matrix, Basis basis) noexcept;
// For a rotated matrix, signs may be null (identity) or columns values of +/-1.
// The GPU caller can use this on the host copy before upload.
bool valid_signs(const Matrix& matrix, const std::int8_t* signs) noexcept;

// All CPU pointers are host pointers. Input has columns floats, output rows
// floats. Buffers do not overlap. For original-basis input apply H*(signs*x);
// stored-basis input is already transformed, so it is not transformed again.
// Returns false without writing on rejection.
bool forward_cpu(const Matrix& matrix, const std::uint8_t* packed, const float* input,
                 Basis basis, const std::int8_t* signs, float* output);

// Link exactly one CUDA or HIP implementation. All buffer pointers below are
// on the current device and do not overlap. Upload validated signs, or pass
// null for identity. Only original-basis rotated input needs workspace
// (columns floats); stored-basis input is used as-is. Input copies to the
// buffers must be complete or ordered before this call's work on the stream.
// stream is a cudaStream_t or hipStream_t, passed as void* (null = default).
// Returns 0 if enqueued, otherwise the backend's error value. Does not sync:
// keep buffers alive and synchronize/check the stream for execution errors.
int forward_gpu(const Matrix& matrix, const std::uint8_t* packed, const float* input,
                Basis basis, const std::int8_t* signs, float* output,
                float* workspace, void* stream = nullptr) noexcept;

}  // namespace gyro
