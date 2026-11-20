#include "gpu_backend.h"
#include "gyro.h"

#include <cstdio>
#include <cstdlib>

static void checked(int code) {
    if (code != GYRO_API(Success)) {
        std::fprintf(stderr, "%s error: %s\n", GYRO_BACKEND,
                     GYRO_API(GetErrorString)(static_cast<GYRO_API(Error_t)>(code)));
        std::exit(1);
    }
}

void probe_device() {
    int device, runtime;
    GyroDeviceProp props{};
    checked(GYRO_API(GetDevice)(&device));
    checked(GYRO_API(GetDeviceProperties)(&props, device));
    checked(GYRO_API(RuntimeGetVersion)(&runtime));
    std::printf("backend=%s device=%s runtime=%d ", GYRO_BACKEND, props.name, runtime);
#if defined(__HIPCC__)
    std::printf("arch=%s\n", props.gcnArchName);
#else
    std::printf("arch=sm_%d%d\n", props.major, props.minor);
#endif
}

void probe_gpu(const gyro::Matrix& m, const std::uint8_t* packed, const float* x,
               gyro::Basis basis, const std::int8_t* signs, float* y) {
    GYRO_API(Stream_t) stream{};
    checked(GYRO_API(StreamCreateWithFlags)(&stream, GYRO_API(StreamNonBlocking)));
    std::uint8_t* dp = nullptr;
    float *dx = nullptr, *dy = nullptr, *work = nullptr;
    std::int8_t* ds = nullptr;
    const std::size_t nx = m.columns * sizeof(float), ny = m.rows * sizeof(float);
    checked(GYRO_API(Malloc)(reinterpret_cast<void**>(&dp), m.packed_bytes));
    checked(GYRO_API(Malloc)(reinterpret_cast<void**>(&dx), nx));
    checked(GYRO_API(Malloc)(reinterpret_cast<void**>(&dy), ny));
    // Keep transfers synchronous so this probe isolates the launcher and stream.
    checked(GYRO_API(Memcpy)(dp, packed, m.packed_bytes, GYRO_API(MemcpyHostToDevice)));
    checked(GYRO_API(Memcpy)(dx, x, nx, GYRO_API(MemcpyHostToDevice)));
    if (signs) {
        if (!gyro::valid_signs(m, signs)) std::exit(1);
        checked(GYRO_API(Malloc)(reinterpret_cast<void**>(&ds), m.columns));
        checked(GYRO_API(Memcpy)(ds, signs, m.columns, GYRO_API(MemcpyHostToDevice)));
    }
    if (m.rotation_block && basis == gyro::Basis::original) {
        if (gyro::forward_gpu(m, dp, dx, basis, ds, dy, nullptr, stream) != GYRO_API(ErrorInvalidValue))
            std::exit(1);
        checked(GYRO_API(Malloc)(reinterpret_cast<void**>(&work), nx));
    }
    checked(gyro::forward_gpu(m, dp, dx, basis, ds, dy, work, stream));
    checked(GYRO_API(StreamSynchronize)(stream));
    checked(GYRO_API(Memcpy)(y, dy, ny, GYRO_API(MemcpyDeviceToHost)));
    if (work) checked(GYRO_API(Free)(work));
    if (ds) checked(GYRO_API(Free)(ds));
    checked(GYRO_API(Free)(dx));
    checked(GYRO_API(Free)(dy));
    checked(GYRO_API(Free)(dp));
    checked(GYRO_API(StreamDestroy)(stream));
}
