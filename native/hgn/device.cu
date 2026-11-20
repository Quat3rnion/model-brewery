#if defined(HGN_ROCM)
#include <hip/hip_runtime.h>
#define API(name) hip##name
#else
#include <cuda_runtime.h>
#define API(name) cuda##name
#endif

#include "packed.h"

#include <stdexcept>

namespace hgn::detail {
namespace {
constexpr unsigned threads = 128;

void checked(API(Error_t) result, const char* step) {
    if (result != API(Success))
        throw std::runtime_error(std::string(step) + ": " + API(GetErrorString)(result));
}

class Buffer {
    void* ptr = nullptr;
public:
    explicit Buffer(size_t size) {
        if (size) checked(API(Malloc)(&ptr, size), "allocate HGN device buffer");
    }
    ~Buffer() { if (ptr) (void)API(Free)(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T = double> T* as() const { return static_cast<T*>(ptr); }
    Bytes upload(Bytes source) {
        if (source.size)
            checked(API(Memcpy)(ptr, source.data, source.size, API(MemcpyHostToDevice)), "upload HGN bytes");
        return {as<uint8_t>(), source.size};
    }
    std::vector<double> get(size_t n) {
        std::vector<double> out(n);
        checked(API(Memcpy)(out.data(), ptr, n * sizeof(double), API(MemcpyDeviceToHost)), "download HGN result");
        return out;
    }
};

__global__ void rotate_kernel(const double* src, const uint8_t* scale, double* out,
                              unsigned n, bool input) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = rotated(src, scale, i, input);
}

__global__ void matvec_kernel(Packed p, const double* x, double* y) {
    // One block per row; its lanes directly contract packed weights. Store
    // only one reduction buffer, never a decoded matrix in device memory.
    __shared__ double partial[threads];
    const unsigned row = blockIdx.x, lane = threadIdx.x;
    double sum = 0;
    for (unsigned c = lane; c < p.cols; c += threads) sum += weight(p, row, c) * x[c];
    partial[lane] = sum;
    for (unsigned offset = threads / 2; offset; offset /= 2) {
        __syncthreads();
        if (lane < offset) partial[lane] += partial[lane + offset];
    }
    if (lane == 0) y[row] = partial[0];
}

__global__ void gather_kernel(Packed p, double* out) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.cols) out[i] = fp8(p.code.data[i]) * f32(u32(p.scale.data));
}
}

std::string device_description() {
    int ordinal = -1;
    checked(API(GetDevice)(&ordinal), "select HGN test device");
#if defined(HGN_ROCM)
    hipDeviceProp_t prop{};
#else
    cudaDeviceProp prop{};
#endif
    checked(API(GetDeviceProperties)(&prop, ordinal), "identify HGN test device");
#if defined(HGN_ROCM)
    return std::string("ROCm ") + prop.name + " " + prop.gcnArchName;
#else
    return std::string("CUDA ") + prop.name + " sm_" + std::to_string(prop.major) +
           std::to_string(prop.minor);
#endif
}

std::vector<double> device_matvec(const Packed& source, const std::vector<double>& x) {
    Packed p = source;
    Buffer code(p.code.size), scale(p.scale.size), book(p.book.size), su(p.su.size), sv(p.sv.size);
    p.code = code.upload(p.code);
    p.scale = scale.upload(p.scale);
    p.book = book.upload(p.book);
    p.su = su.upload(p.su);
    p.sv = sv.upload(p.sv);
    Buffer in(x.size() * sizeof(double)), out(p.rows * sizeof(double));
    in.upload({reinterpret_cast<const uint8_t*>(x.data()), x.size() * sizeof(double)});
    Buffer rotated_in(p.su.size ? x.size() * sizeof(double) : 0);
    Buffer rotated_out(p.sv.size ? p.rows * sizeof(double) : 0);
    if (p.su.size) {
        rotate_kernel<<<(p.cols + threads - 1) / threads, threads>>>(
            in.as(), p.su.data, rotated_in.as(), p.cols, true);
        checked(API(GetLastError)(), "launch HGN input rotation");
    }
    matvec_kernel<<<p.rows, threads>>>(p, p.su.size ? rotated_in.as() : in.as(), out.as());
    checked(API(GetLastError)(), "launch HGN packed contraction");
    if (p.sv.size) {
        rotate_kernel<<<(p.rows + threads - 1) / threads, threads>>>(
            out.as(), p.sv.data, rotated_out.as(), p.rows, false);
        checked(API(GetLastError)(), "launch HGN output rotation");
    }
    checked(API(DeviceSynchronize)(), "finish HGN packed contraction");
    return p.sv.size ? rotated_out.get(p.rows) : out.get(p.rows);
}

std::vector<double> device_gather(const Packed& source) {
    Packed p = source;
    Buffer code(p.code.size), scale(p.scale.size), out(p.cols * sizeof(double));
    p.code = code.upload(p.code);
    p.scale = scale.upload(p.scale);
    gather_kernel<<<(p.cols + threads - 1) / threads, threads>>>(p, out.as());
    checked(API(GetLastError)(), "launch HGN FP8G gather");
    checked(API(DeviceSynchronize)(), "finish HGN FP8G gather");
    return out.get(p.cols);
}
}
