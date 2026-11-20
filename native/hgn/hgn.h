#ifndef BREWERY_PACKED_HGN_H
#define BREWERY_PACKED_HGN_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hgn {
struct Bytes {
    const uint8_t* data = nullptr;
    size_t size = 0;
};

// Checked borrowed payload from a container reader, not a whole HGN file.
struct Tensor {
    std::string name;
    uint32_t type = 0, variant = 0;
    std::vector<size_t> shape;
    Bytes bytes;
};

enum class Backend { cpu, cuda, rocm };
std::string device_name(Backend backend);

// Expert number required for rank 3; -1 for unstacked matrices.
// Supply both F16 sidecars for rotated stores and neither otherwise.
std::vector<double> matvec(const Tensor& tensor, const Tensor& su, const Tensor& sv,
                           int expert, const std::vector<double>& x,
                           Backend backend = Backend::cpu);
std::vector<double> gather_fp8(const Tensor& tensor, size_t shard, size_t row,
                               Backend backend = Backend::cpu);
std::vector<int64_t> integers(const Tensor& tensor);
}

#endif
