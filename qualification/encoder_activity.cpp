// Build as a CUDA 13.4 CUPTI injection library; see the workbench README.
#include <cupti.h>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>

namespace {
CUpti_SubscriberHandle subscriber;
const char *path;
const char *expected;
pid_t pid;
struct stat executable;
std::mutex lock;
std::atomic<bool> bad{false};
bool armed;
uint32_t device = UINT32_MAX, kernel_device = UINT32_MAX, major, minor;
uint64_t counts[4];

void physical_device(const CUpti_ActivityDevice6 &d) {
    if (!d.isCudaVisible || d.isMigEnabled) return;
    char uuid[41] = "GPU-";
    static const char hex[] = "0123456789abcdef";
    size_t pos = 4;
    for (size_t i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) uuid[pos++] = '-';
        auto byte = static_cast<unsigned char>(d.uuid.bytes[i]);
        uuid[pos++] = hex[byte >> 4];
        uuid[pos++] = hex[byte & 15];
    }
    if (strcmp(uuid, expected) != 0) return;
    if (device != UINT32_MAX &&
        (device != d.id || major != d.computeCapabilityMajor || minor != d.computeCapabilityMinor)) {
        bad = true;
        return;
    }
    device = d.id;
    major = d.computeCapabilityMajor;
    minor = d.computeCapabilityMinor;
}

void kernel(const CUpti_ActivityKernel13 &k) {
    if (!k.name || k.start == 0 || k.end <= k.start ||
        k.channelType != CUPTI_CHANNEL_TYPE_COMPUTE ||
        !strstr(k.name, "_ht_trellis_cuda_cu_")) return;
    // Match the converter's CUDA symbols, not launches, decoder kernels or copies.
    unsigned codec;
    if (strstr(k.name, "12encode_tilesILi3E")) codec = 0;
    else if (strstr(k.name, "12encode_tilesILi4E")) codec = 1;
    else if (strstr(k.name, "12i4r_quantizeE")) codec = 2;
    else if (strstr(k.name, "11q4_quantizeE")) codec = 3;
    else return;
    if ((kernel_device != UINT32_MAX && kernel_device != k.deviceId) ||
        counts[codec] == UINT64_MAX) {
        bad = true;
        return;
    }
    kernel_device = k.deviceId;
    ++counts[codec];
}

void CUPTIAPI request(uint8_t **buffer, size_t *size, size_t *max_records,
                      CUpti_BufferCallbackRequestInfo *) {
    constexpr size_t bytes = 1 << 20;
    *buffer = static_cast<uint8_t *>(malloc(bytes));
    if (!*buffer) bad = true;
    *size = *buffer ? bytes : 0;
    *max_records = 0;
}

void CUPTIAPI complete(uint8_t *buffer, size_t, size_t valid,
                       CUpti_BufferCallbackCompleteInfo *) {
    std::lock_guard<std::mutex> guard(lock);
    CUpti_Activity *record = nullptr;
    while (valid) {
        const auto status = cuptiActivityGetNextRecord_v2(subscriber, buffer, valid, &record);
        if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        if (status != CUPTI_SUCCESS || !record) {
            bad = true;
            break;
        }
        if (record->kind == CUPTI_ACTIVITY_KIND_DEVICE)
            physical_device(*reinterpret_cast<const CUpti_ActivityDevice6 *>(record));
        else if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
            kernel(*reinterpret_cast<const CUpti_ActivityKernel13 *>(record));
    }
    free(buffer);
}

void finish() {
    if (!armed || getpid() != pid ||
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED) != CUPTI_SUCCESS) return;
    std::lock_guard<std::mutex> guard(lock);
    if (bad || device == UINT32_MAX || (kernel_device != UINT32_MAX && kernel_device != device)) return;
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) return;
    // One bounded record. A missing or partial write does not validate.
    dprintf(fd,
        "{\"schema\":\"brewery-cupti-encoder-v1\",\"pid\":%ld,\"exe_dev\":%" PRIuMAX
        ",\"exe_ino\":%" PRIuMAX ",\"gpu_uuid\":\"%s\",\"compute_major\":%u,\"compute_minor\":%u"
        ",\"counts\":{\"ht3\":%" PRIu64 ",\"ht4\":%" PRIu64
        ",\"i4r\":%" PRIu64 ",\"q4cp\":%" PRIu64 "}}\n",
        static_cast<long>(pid), static_cast<uintmax_t>(executable.st_dev),
        static_cast<uintmax_t>(executable.st_ino), expected, major, minor,
        counts[0], counts[1], counts[2], counts[3]);
    close(fd);
}

void CUPTIAPI callback(void *, CUpti_CallbackDomain, CUpti_CallbackId, const void *) {}
} // namespace

extern "C" int InitializeInjection() {
    path = getenv("BREWERY_CUPTI_CAPTURE");
    expected = getenv("CUDA_VISIBLE_DEVICES");
    uint32_t version = 0;
    if (!path || path[0] != '/' || !expected || strncmp(expected, "GPU-", 4) != 0 ||
        strlen(expected) != 40 || stat("/proc/self/exe", &executable) != 0 ||
        cuptiGetVersion(&version) != CUPTI_SUCCESS || version != CUPTI_API_VERSION) return 0;
    pid = getpid();
    CUpti_SubscriberParams params{};
    params.structSize = sizeof(params);
    if (cuptiSubscribe_v2(&subscriber, callback, nullptr, &params) != CUPTI_SUCCESS ||
        cuptiActivityRegisterCallbacks_v2(subscriber, request, complete) != CUPTI_SUCCESS ||
        cuptiActivityEnable_v2(subscriber, CUPTI_ACTIVITY_KIND_DEVICE, nullptr) != CUPTI_SUCCESS ||
        cuptiActivityEnable_v2(subscriber, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, nullptr) != CUPTI_SUCCESS ||
        atexit(finish) != 0) return 0;
    armed = true;
    return 1;
}
