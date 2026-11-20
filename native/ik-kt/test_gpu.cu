#if defined(__HIPCC__) || defined(__HIP__)
#include <hip/hip_runtime.h>
#define RT(name) hip##name
#define HOST_ALLOC hipHostMalloc
#define HOST_FREE hipHostFree
#else
#include <cuda_runtime.h>
#define RT(name) cuda##name
#define HOST_ALLOC cudaHostAlloc
#define HOST_FREE cudaFreeHost
#endif

#include "kt_matvec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void status(RT(Error_t) rc) {
    if (rc != RT(Success)) {
        std::fprintf(stderr, "%s\n", RT(GetErrorString)(rc));
        std::exit(1);
    }
}

void begin_device() {
    int dev = -1;
    status(RT(GetDevice)(&dev));
#if defined(__HIPCC__) || defined(__HIP__)
    hipDeviceProp_t properties{};
    status(hipGetDeviceProperties(&properties, dev));
    std::printf("HIP device %d: %s (%s)\n", dev, properties.name, properties.gcnArchName);
#else
    cudaDeviceProp properties{};
    status(cudaGetDeviceProperties(&properties, dev));
    std::printf("CUDA device %d: %s (sm_%d%d)\n", dev, properties.name, properties.major, properties.minor);
#endif
}

template<class T>
struct Buffer {
    T *ptr = nullptr;
    explicit Buffer(size_t count) { status(RT(Malloc)(reinterpret_cast<void **>(&ptr), count * sizeof(T))); }
    ~Buffer() { status(RT(Free)(ptr)); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
};

template<class T>
struct HostBuffer {
    T *ptr = nullptr;
    explicit HostBuffer(size_t count) { status(HOST_ALLOC(reinterpret_cast<void **>(&ptr), count * sizeof(T), 0)); }
    ~HostBuffer() { status(HOST_FREE(ptr)); }
    HostBuffer(const HostBuffer &) = delete;
    HostBuffer &operator=(const HostBuffer &) = delete;
};

void check_device(ik_kt::Kind kind, const std::vector<uint8_t> &packed, size_t width, size_t rows,
                  const std::vector<float> &x, const std::vector<double> &ref,
                  const std::vector<double> &l1) {
    Buffer<uint8_t> q(packed.size());
    Buffer<float> dx(width), dy(rows);
    HostBuffer<uint8_t> hq(packed.size());
    HostBuffer<float> hx(width), out(rows);
    std::copy(packed.begin(), packed.end(), hq.ptr);
    std::copy(x.begin(), x.end(), hx.ptr);
    RT(Stream_t) stream;
    status(RT(StreamCreateWithFlags)(&stream, RT(StreamNonBlocking)));
    // Stage test inputs completely before checking a kernel launch. In particular,
    // don't mix a pageable upload on the default stream with a nonblocking kernel.
    status(RT(MemcpyAsync)(q.ptr, hq.ptr, packed.size(), RT(MemcpyHostToDevice), stream));
    status(RT(MemcpyAsync)(dx.ptr, hx.ptr, width * sizeof(float), RT(MemcpyHostToDevice), stream));
    status(RT(MemsetAsync)(dy.ptr, 0xFF, rows * sizeof(float), stream));
    status(RT(StreamSynchronize)(stream));
    if (ik_kt::matvec_gpu(kind, q.ptr, packed.size() - 1, width, rows, dx.ptr, width, dy.ptr, rows, stream)
            != ik_kt::Status::invalid_argument ||
        ik_kt::matvec_gpu(kind, q.ptr, packed.size(), width, rows, dx.ptr, width, dy.ptr, rows, stream)
            != ik_kt::Status::ok) {
        std::fprintf(stderr, "GPU argument rejection or launch failure\n");
        std::exit(1);
    }
    status(RT(MemcpyAsync)(out.ptr, dy.ptr, rows * sizeof(float), RT(MemcpyDeviceToHost), stream));
    status(RT(StreamSynchronize)(stream));
    status(RT(StreamDestroy)(stream));
    for (size_t i = 0; i < rows; ++i) {
        if (!(std::abs(out.ptr[i] - ref[i]) <= 1e-7 + 2e-6 * l1[i])) {
            std::fprintf(stderr, "GPU mismatch id=%u width=%zu row=%zu got=%.9g ref=%.9g abs_sum=%.9g\n",
                         unsigned(kind), width, i, out.ptr[i], ref[i], l1[i]);
            std::exit(1);
        }
    }
}
