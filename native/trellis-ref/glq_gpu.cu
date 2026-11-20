#include "glq_gpu.h"
#include "glq_detail.h"

namespace glq {
namespace {
// Reference schedule: one thread per sample for each FWHT, one thread per
// sample/output row for the packed dot product. No dense staging or ISA calls.
__global__ void prepare(Layer l, const float* x, float* scratch) {
    const int b = blockIdx.x;
    detail::transform_input(l, x + b * l.cols, scratch + b * l.cols);
}

__global__ void multiply(Layer l, const float* scratch, float* y) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (row < l.rows) y[b * l.rows + row] = detail::dot(l, scratch + b * l.cols, row);
}

__global__ void restore(Layer l, float* y) {
    detail::transform_output(l, y + blockIdx.x * l.rows);
}
}  // namespace

trellis_ref::Error linear_gpu(const Layer& l, int batch,
    const float* x, std::size_t nx, float* y, std::size_t ny,
    float* scratch, std::size_t ns, trellis_ref::Stream stream) {
    using namespace trellis_ref;
    if (!validate_metadata(l) || !detail::valid_io(l, batch, x, nx, y, ny)
            || !scratch || scratch == x || scratch == y || ns != nx) return invalid;
    prepare<<<batch, 1, 0, stream>>>(l, x, scratch);
    if (const auto error = launch_status(); error != success) return error;
    multiply<<<dim3((l.rows + 127) / 128, batch), 128, 0, stream>>>(l, scratch, y);
    if (const auto error = launch_status(); error != success) return error;
    restore<<<batch, 1, 0, stream>>>(l, y);
    return launch_status();
}

}  // namespace glq
