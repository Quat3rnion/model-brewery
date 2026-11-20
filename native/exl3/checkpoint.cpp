#include "exl3.h"

#include <json-c/json.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using Byte = std::uint8_t;
using Bytes = std::vector<Byte>;
using Size = std::uint64_t;
using Json = std::unique_ptr<json_object, decltype(&json_object_put)>;
constexpr int dim = 128, tile = 16, stripes = dim / tile;
constexpr Size header_limit = 32u << 20, file_limit = Size{2} << 40;
constexpr Size axis_limit = 32768, block_limit = 4096;
constexpr std::uint32_t mcg_marker = 0xcbac1fedu, mul1_marker = 0x83dcd12du;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE FP32 required");
static_assert(sizeof(off_t) >= 8, "large-file offsets required");

void need(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

[[noreturn]] void system_error(const std::string& operation)
{
    const int error = errno;
    throw std::runtime_error(operation + ": " + std::strerror(error));
}

Size get_le(const Byte* p, unsigned width)
{
    Size result = 0;
    for (unsigned i = 0; i < width; ++i) result |= Size{p[i]} << (8 * i);
    return result;
}

void put_le(Bytes& bytes, Size value, unsigned width)
{
    for (unsigned i = 0; i < width; ++i) bytes.push_back(static_cast<Byte>(value >> (8 * i)));
}

class Fd {
    int fd_;
public:
    explicit Fd(int fd) : fd_(fd) {}
    ~Fd() { if (fd_ >= 0) ::close(fd_); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    int get() const { return fd_; }
    void close()
    {
        if (::close(std::exchange(fd_, -1)) != 0) system_error("close output");
    }
};

class Input {
    Fd fd_;
    struct stat initial_ {};
public:
    explicit Input(const char* path) : fd_(::open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC))
    {
        if (fd_.get() < 0) system_error(std::string("open ") + path);
        if (::fstat(fd_.get(), &initial_) != 0) system_error("stat input");
        need(S_ISREG(initial_.st_mode) && initial_.st_size >= 0 &&
             static_cast<Size>(initial_.st_size) <= file_limit, "expected bounded regular input file");
    }
    Size size() const { return static_cast<Size>(initial_.st_size); }
    void read(Size offset, Byte* data, std::size_t count) const
    {
        need(offset <= size() && count <= size() - offset, "read exceeds input extent");
        while (count != 0) {
            const auto n = ::pread(fd_.get(), data, count, static_cast<off_t>(offset));
            if (n < 0 && errno == EINTR) continue;
            if (n < 0) system_error("read input");
            need(n != 0, "truncated input");
            count -= n;
            offset += n;
            data += n;
        }
    }
    void check_unchanged() const
    {
        struct stat current {};
        if (::fstat(fd_.get(), &current) != 0) system_error("stat input");
        need(current.st_size == initial_.st_size &&
             current.st_mtim.tv_sec == initial_.st_mtim.tv_sec &&
             current.st_mtim.tv_nsec == initial_.st_mtim.tv_nsec &&
             current.st_ctim.tv_sec == initial_.st_ctim.tv_sec &&
             current.st_ctim.tv_nsec == initial_.st_ctim.tv_nsec, "input changed during read");
    }
};

void publish(const char* path, const Bytes& bytes)
{
    // The temporary file is on the destination filesystem. link() publishes
    // the complete file but fails if anything already occupies the destination.
    auto temp = std::string(path) + ".tmp.XXXXXX";
    Fd fd(::mkstemp(temp.data()));
    if (fd.get() < 0) system_error("create temporary output");
    struct Cleanup {
        const char* path;
        ~Cleanup() { ::unlink(path); }
    } cleanup{temp.c_str()};
    for (std::size_t done = 0; done < bytes.size();) {
        const auto n = ::write(fd.get(), bytes.data() + done, bytes.size() - done);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) system_error("write output");
        need(n != 0, "short output write");
        done += n;
    }
    if (::fsync(fd.get()) != 0) system_error("sync output");
    fd.close();
    if (::link(temp.c_str(), path) != 0) system_error("publish new output");
}

void check_new_output(const char* path)
{
    struct stat st {};
    need(::lstat(path, &st) != 0, "output already exists");
    if (errno != ENOENT) system_error("check output path");
}

bool named(std::string_view name)
{
    return !name.empty() && name.size() <= 4096 &&
           std::none_of(name.begin(), name.end(), [](unsigned char c) { return c < 32 || c == 127; });
}

bool is(const json_object* o, json_type type) { return json_object_is_type(o, type); }

json_object* member(const json_object* o, const char* name)
{
    json_object* result = nullptr;
    json_object_object_get_ex(o, name, &result);
    return result;
}

Size integer(const json_object* o, Size limit)
{
    need(is(o, json_type_int) && json_object_get_int64(o) >= 0 &&
         json_object_get_uint64(o) <= limit, "invalid SafeTensors shape or offset integer");
    return json_object_get_uint64(o);
}

std::string_view text(json_object* o)
{
    need(is(o, json_type_string), "expected SafeTensors JSON string");
    return {json_object_get_string(o), static_cast<std::size_t>(json_object_get_string_len(o))};
}

unsigned dtype_width(std::string_view dtype)
{
    for (const auto name : {"BOOL", "I8", "U8", "F8_E4M3", "F8_E5M2", "F8_E8M0"})
        if (dtype == name) return 1;
    for (const auto name : {"I16", "U16", "F16", "BF16"})
        if (dtype == name) return 2;
    for (const auto name : {"I32", "U32", "F32"})
        if (dtype == name) return 4;
    for (const auto name : {"I64", "U64", "F64", "C64"})
        if (dtype == name) return 8;
    return 0;
}

// json-c overwrites duplicate object keys. Count the actual unquoted colons
// in the already parsed header and compare against its surviving member count.
// The selected schema only allows objects at the root and one level below it.
Size count_keys(std::string_view header)
{
    bool in_string = false, nul = false;
    Size count = 0;
    for (std::size_t i = 0; i < header.size(); ++i) {
        const char c = header[i];
        if (in_string) {
            if (c == '\\') {
                if (header.substr(i + 1, 5) == "u0000") nul = true;
                ++i;
            } else if (c == '"') {
                in_string = false;
            }
        } else if (c == '"') {
            in_string = true;
            nul = false;
        } else if (c == ':') {
            need(!nul, "NUL in SafeTensors key");
            ++count;
        }
    }
    return count;
}

struct Tensor {
    std::string dtype;
    std::vector<Size> shape;
    Size begin, end;
};

class SafeTensors {
    const Input& file_;
    Size payload_;
    std::map<std::string, Tensor> tensors_;
public:
    explicit SafeTensors(const Input& file) : file_(file)
    {
        std::array<Byte, 8> bytes;
        file.read(0, bytes.data(), bytes.size());
        const Size length = get_le(bytes.data(), 8);
        need(2 <= length && length <= header_limit && length <= file.size() - 8,
             "SafeTensors header exceeds bounds");
        std::string header(length, '\0');
        file.read(8, reinterpret_cast<Byte*>(header.data()), header.size());
        need(header.front() == '{', "SafeTensors header must begin with an object");
        std::unique_ptr<json_tokener, decltype(&json_tokener_free)> tok(json_tokener_new_ex(8),
                                                                      json_tokener_free);
        need(bool(tok), "allocate SafeTensors parser");
        json_tokener_set_flags(tok.get(), JSON_TOKENER_STRICT | JSON_TOKENER_VALIDATE_UTF8);
        Json root(json_tokener_parse_ex(tok.get(), header.data(), static_cast<int>(header.size())),
                  json_object_put);
        need(json_tokener_get_error(tok.get()) == json_tokener_success &&
             json_tokener_get_parse_end(tok.get()) == header.size() && is(root.get(), json_type_object),
             "invalid SafeTensors JSON header");
        const Size root_keys = json_object_object_length(root.get());
        need(root_keys > 0 && root_keys <= 100001, "invalid SafeTensors tensor count");
        Size keys = root_keys;
        payload_ = 8 + length;
        const Size capacity = file.size() - payload_;
        std::vector<std::pair<Size, Size>> ranges;
        json_object_object_foreach(root.get(), name, entry) {
            need(is(entry, json_type_object), "invalid SafeTensors record");
            keys += json_object_object_length(entry);
            if (std::string_view(name) == "__metadata__") {
                json_object_object_foreach(entry, key, value) {
                    (void)key;
                    need(is(value, json_type_string), "invalid SafeTensors metadata value");
                }
                continue;
            }
            need(named(name) && json_object_object_length(entry) == 3, "invalid SafeTensors tensor fields");
            const auto dtype = text(member(entry, "dtype"));
            const auto width = dtype_width(dtype);
            auto* shape = member(entry, "shape");
            auto* offsets = member(entry, "data_offsets");
            need(width != 0 && is(shape, json_type_array) && json_object_array_length(shape) <= 8 &&
                 is(offsets, json_type_array) && json_object_array_length(offsets) == 2,
                 "invalid SafeTensors dtype, shape or offsets");
            Tensor t{std::string(dtype), {},
                     integer(json_object_array_get_idx(offsets, 0), capacity),
                     integer(json_object_array_get_idx(offsets, 1), capacity)};
            for (std::size_t i = 0; i < json_object_array_length(shape); ++i)
                t.shape.push_back(integer(json_object_array_get_idx(shape, i), file_limit));
            const bool empty = std::find(t.shape.begin(), t.shape.end(), 0) != t.shape.end();
            Size expected = empty ? 0 : width;
            if (!empty) for (Size axis : t.shape) {
                need(expected <= capacity / axis, "SafeTensors tensor size exceeds payload");
                expected *= axis;
            }
            need(t.begin <= t.end && t.end - t.begin == expected, "SafeTensors payload size mismatch");
            ranges.emplace_back(t.begin, t.end);
            tensors_.emplace(name, std::move(t));
        }
        need(keys == count_keys(header), "duplicate SafeTensors JSON key");
        need(!tensors_.empty(), "no SafeTensors tensors");
        std::sort(ranges.begin(), ranges.end());
        Size cursor = 0;
        for (const auto& range : ranges) {
            need(range.first == cursor, "SafeTensors payload has a gap or overlap");
            cursor = range.second;
        }
        need(cursor == capacity, "SafeTensors payload has unindexed bytes");
    }

    const Tensor* find(const std::string& name) const
    {
        const auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }
    void read(const Tensor& t, Size offset, Byte* data, std::size_t bytes) const
    {
        need(offset <= t.end - t.begin && bytes <= t.end - t.begin - offset,
             "read exceeds SafeTensors tensor extent");
        file_.read(payload_ + t.begin + offset, data, bytes);
    }
};

float from_half(unsigned word)
{
    const int exponent = (word >> 10) & 31;
    const int mantissa = word & 1023;
    need(exponent != 31, "non-finite F16 companion value");
    const float value = std::ldexp(static_cast<float>(exponent ? 1024 + mantissa : mantissa),
                                   exponent ? exponent - 25 : -24);
    return word & 0x8000 ? -value : value;
}

unsigned to_half(double value)
{
    if (value == 0) return 0;
    const double magnitude = std::abs(value);
    need(magnitude < 65520.0, "weight scale exceeds F16 range");
    int exponent;
    std::frexp(magnitude, &exponent);
    const int grid = std::max(exponent - 11, -24);
    const double units = std::ldexp(magnitude, -grid);
    unsigned n = static_cast<unsigned>(units);
    const double fraction = units - n;
    if (fraction > 0.5 || (fraction == 0.5 && (n & 1))) ++n;
    need(n != 0, "nonzero weight scale underflows F16");
    // For normal grid units, the implicit bit (or the carry if n == 2048)
    // adds one to the biased exponent. Subnormals already have the right bits.
    const unsigned bits = grid == -24 ? n : ((grid + 24) << 10) + n;
    return bits | (value < 0 ? 0x8000u : 0u);
}

float from_f32(const Byte* bytes)
{
    const auto bits = static_cast<std::uint32_t>(get_le(bytes, 4));
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    need(std::isfinite(result), "non-finite F32 input or companion value");
    return result;
}

void put_f32(Bytes& bytes, double value)
{
    need(std::isfinite(value) && std::abs(value) <= std::numeric_limits<float>::max(),
         "non-finite or out-of-range F32 output");
    const float result = static_cast<float>(value);
    std::uint32_t bits;
    std::memcpy(&bits, &result, sizeof(bits));
    put_le(bytes, bits, 4);
}

std::vector<float> read_f32(const char* path, Size count)
{
    Input file(path);
    need(file.size() == 4 * count, "raw F32 input has wrong byte count");
    Bytes bytes(file.size());
    file.read(0, bytes.data(), bytes.size());
    file.check_unchanged();
    std::vector<float> out(count);
    for (Size i = 0; i < count; ++i) out[i] = from_f32(bytes.data() + 4 * i);
    return out;
}

bool supported(int twice_bits, int book)
{
    return twice_bits >= 2 && twice_bits <= 16 &&
           (!(twice_bits & 1) || (book == EXL3_MUL1 && twice_bits <= 7));
}

std::vector<float> rotation(const SafeTensors& sf, const std::string& prefix, char axis, Size size)
{
    const auto packed_name = prefix + ".s" + axis;
    const auto* half = sf.find(packed_name + "h");
    const auto* packed = sf.find(packed_name);
    need(bool(half) != bool(packed), packed_name + ": expected exactly one rotation tensor");
    const Size count = half ? size : size / tile;
    const auto& tensor = half ? *half : *packed;
    need(tensor.dtype == (half ? "F16" : "I16") && tensor.shape == std::vector<Size>{count},
         packed_name + ": invalid rotation dtype or shape");
    Bytes bytes(count * 2);
    sf.read(tensor, 0, bytes.data(), bytes.size());
    std::vector<float> result(size);
    for (Size i = 0; i < count; ++i) {
        const unsigned word = get_le(bytes.data() + 2 * i, 2);
        if (half) result[i] = from_half(word);
        else for (int bit = 0; bit < tile; ++bit) result[tile * i + bit] = word & (1u << bit) ? -1.0f : 1.0f;
    }
    return result;
}

decltype(&exl3_forward) backend(std::string_view name)
{
    if (name == "cpu") return exl3_forward;
    if (name == "cuda") {
#ifdef EXL3_HAVE_CUDA
        return exl3_cuda_forward;
#else
        throw std::runtime_error("CUDA backend not built");
#endif
    }
    if (name == "rocm") {
#ifdef EXL3_HAVE_ROCM
        return exl3_rocm_forward;
#else
        throw std::runtime_error("ROCm backend not built");
#endif
    }
    throw std::runtime_error("unknown backend; expected cpu, cuda or rocm");
}

void forward(const char* device, const char* checkpoint, const std::string& prefix,
             const char* activation, const char* output)
{
    const auto kernel = backend(device);
    check_new_output(output);
    Input file(checkpoint);
    const SafeTensors sf(file);
    const auto* trellis = sf.find(prefix + ".trellis");
    need(trellis && trellis->dtype == "I16" && trellis->shape.size() == 3,
         "expected co-located rank-3 I16 linear trellis");
    for (const auto suffix : {".scale", ".weight", ".signs"})
        need(!sf.find(prefix + suffix), "conflicting tensors in EXL3 linear group");
    const Size rows = trellis->shape[0], cols = trellis->shape[1], words = trellis->shape[2];
    need(rows > 0 && cols > 0 && rows % stripes == 0 && cols % stripes == 0 &&
         rows <= axis_limit / tile && cols <= axis_limit / tile &&
         (rows / stripes) * (cols / stripes) <= block_limit, "linear axes exceed aligned EXL3 bounds");
    const auto* mcg = sf.find(prefix + ".mcg");
    const auto* mul1 = sf.find(prefix + ".mul1");
    need(!mcg || !mul1, "ambiguous EXL3 codebook markers");
    const int book = mcg ? EXL3_MCG : mul1 ? EXL3_MUL1 : EXL3_3INST;
    if (const auto* marker = mcg ? mcg : mul1) {
        need(marker->dtype == "I32" && marker->shape.empty(), "expected I32 scalar EXL3 marker");
        std::array<Byte, 4> bits;
        sf.read(*marker, 0, bits.data(), bits.size());
        need(get_le(bits.data(), 4) == (mcg ? mcg_marker : mul1_marker), "incorrect EXL3 codebook marker");
    }
    need(words <= 128 && words % 8 == 0 && supported(static_cast<int>(words / 8), book),
         "unsupported EXL3 words per tile for codebook");
    const int twice_bits = words / 8;
    const Size k = rows * tile, n = cols * tile;
    const auto su = rotation(sf, prefix, 'u', k), sv = rotation(sf, prefix, 'v', n);
    const auto x = read_f32(activation, k);
    std::vector<double> y(n);
    if (const auto* bias = sf.find(prefix + ".bias")) {
        need((bias->dtype == "F16" || bias->dtype == "F32") && bias->shape == std::vector<Size>{n},
             "invalid EXL3 bias dtype or shape");
        const auto width = dtype_width(bias->dtype);
        Bytes bytes(n * width);
        sf.read(*bias, 0, bytes.data(), bytes.size());
        for (Size i = 0; i < n; ++i)
            y[i] = width == 2 ? from_half(get_le(bytes.data() + 2 * i, 2)) : from_f32(bytes.data() + 4 * i);
    }
    const Size tile_bytes = words * 2, stripe_bytes = stripes * tile_bytes;
    Bytes packed(stripes * stripe_bytes);
    for (Size row = 0; row < rows; row += stripes) {
        for (Size col = 0; col < cols; col += stripes) {
            for (Size r = 0; r < stripes; ++r) {
                const Size offset = ((row + r) * cols + col) * tile_bytes;
                sf.read(*trellis, offset, packed.data() + r * stripe_bytes, stripe_bytes);
            }
            std::array<float, dim> partial;
            need(kernel(packed.data(), packed.size(), twice_bits, book, su.data() + row * tile,
                        sv.data() + col * tile, x.data() + row * tile, partial.data()) == 0,
                 std::string(device) + " EXL3 forward failed");
            for (int j = 0; j < dim; ++j) y[col * tile + j] += partial[j];
        }
    }
    file.check_unchanged();
    Bytes bytes;
    bytes.reserve(n * 4);
    for (double value : y) put_f32(bytes, value);
    publish(output, bytes);
}

void hadamard(double* matrix, int stride)
{
    for (int step = 1; step < dim; step *= 2)
        for (int i = 0; i < dim; i += 2 * step)
            for (int j = i; j < i + step; ++j) {
                auto &a = matrix[j * stride], &b = matrix[(j + step) * stride];
                const double sum = a + b, diff = a - b;
                a = sum;
                b = diff;
            }
}

void add_tensor(json_object* root, const std::string& key, const char* dtype,
                std::initializer_list<int> shape, Bytes& payload, const Bytes& bytes)
{
    const auto add = [](json_object* parent, const char* name, json_object* value) {
        Json owned(value, json_object_put);
        need(bool(owned), "allocate SafeTensors output descriptor");
        const auto status = name ? json_object_object_add(parent, name, value)
                                 : json_object_array_add(parent, value);
        need(status == 0, "build SafeTensors output descriptor");
        return owned.release();
    };
    auto* record = add(root, key.c_str(), json_object_new_object());
    add(record, "dtype", json_object_new_string(dtype));
    auto* axes = add(record, "shape", json_object_new_array());
    for (int axis : shape) add(axes, nullptr, json_object_new_int(axis));
    auto* offsets = add(record, "data_offsets", json_object_new_array());
    add(offsets, nullptr, json_object_new_uint64(payload.size()));
    payload.insert(payload.end(), bytes.begin(), bytes.end());
    add(offsets, nullptr, json_object_new_uint64(payload.size()));
}

void encode(int book, int twice_bits, const std::string& prefix, const char* weights, const char* output)
{
    need(supported(twice_bits, book), "unsupported twice-bits for EXL3 codebook");
    check_new_output(output);
    const auto w = read_f32(weights, dim * dim);
    double sum = 0;
    for (double v : w) sum += v * v;
    constexpr double codebook_rms = 1.24371088;
    const double base = std::sqrt(std::sqrt(sum / (dim * dim)) / codebook_rms);
    // Deterministic signs spread structured matrices in the rotated basis.
    std::uint32_t seed = 0x19fa215du;
    const auto sign = [&seed] {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 31 ? -1.0 : 1.0;
    };
    std::array<double, dim * dim> matrix;
    Bytes suh, svh;
    // Balance the normalization over both axes to make better use of the F16
    // range. Use the exact stored half values for the inverse transform.
    for (int c = 0; c < dim; ++c) {
        double sq = 0;
        for (int r = 0; r < dim; ++r) {
            const double value = w[r * dim + c];
            sq += value * value;
        }
        const unsigned word = to_half(sq == 0 ? 0 : sign() * std::sqrt(sq / dim) / (base * codebook_rms));
        put_le(svh, word, 2);
        const double sv = from_half(word);
        for (int r = 0; r < dim; ++r) matrix[r * dim + c] = sv == 0 ? 0 : w[r * dim + c] / sv;
    }
    for (int r = 0; r < dim; ++r) {
        double sq = 0;
        for (int c = 0; c < dim; ++c) {
            const double value = matrix[r * dim + c];
            sq += value * value;
        }
        const unsigned word = to_half(sign() * std::sqrt(sq / dim) / codebook_rms);
        put_le(suh, word, 2);
        const double su = from_half(word);
        if (su != 0) for (int c = 0; c < dim; ++c) matrix[r * dim + c] /= su;
        hadamard(matrix.data() + r * dim, 1);
    }
    for (int c = 0; c < dim; ++c) hadamard(matrix.data() + c, dim);
    std::array<float, dim * dim> rotated;
    for (std::size_t i = 0; i < matrix.size(); ++i) rotated[i] = static_cast<float>(matrix[i] / dim);
    Bytes packed(1024 * twice_bits);
    need(exl3_encode(rotated.data(), twice_bits, book, packed.data(), packed.size()) == 0,
         "EXL3 encoding failed");
    Json header(json_object_new_object(), json_object_put);
    need(bool(header), "allocate SafeTensors header");
    Bytes payload;
    add_tensor(header.get(), prefix + ".suh", "F16", {dim}, payload, suh);
    add_tensor(header.get(), prefix + ".svh", "F16", {dim}, payload, svh);
    if (book != EXL3_3INST) {
        Bytes marker;
        put_le(marker, book == EXL3_MCG ? mcg_marker : mul1_marker, 4);
        add_tensor(header.get(), prefix + (book == EXL3_MCG ? ".mcg" : ".mul1"), "I32", {}, payload, marker);
    }
    add_tensor(header.get(), prefix + ".trellis", "I16", {stripes, stripes, 8 * twice_bits}, payload, packed);
    const auto* json = json_object_to_json_string_ext(header.get(), JSON_C_TO_STRING_PLAIN);
    need(json != nullptr, "serialize SafeTensors header");
    std::string text = json;
    text.append((8 - text.size() % 8) % 8, ' ');
    Bytes checkpoint;
    put_le(checkpoint, text.size(), 8);
    checkpoint.insert(checkpoint.end(), text.begin(), text.end());
    checkpoint.insert(checkpoint.end(), payload.begin(), payload.end());
    publish(output, checkpoint);
}

const char usage[] =
    "usage:\n"
    "  exl3-checkpoint encode <3inst|mcg|mul1> <twice-bits> <prefix> <weights.f32> <new.safetensors>\n"
    "  exl3-checkpoint forward <cpu|cuda|rocm> <checkpoint.safetensors> <prefix> <x.f32> <new-y.f32>\n"
    "Raw F32 files are little-endian; encode takes one input-major 128x128 block.\n"
    "Forward supports co-located 128-aligned layers, at most 32768 features per axis and 4096 blocks.\n";
} // namespace

int main(int argc, char** argv)
{
    try {
        const std::string_view command = argc > 1 ? argv[1] : "";
        if (command == "--help" && argc == 2) {
            std::fputs(usage, stdout);
            return 0;
        }
        if (command == "encode" && argc == 7) {
            const std::string_view book = argv[2], bits = argv[3], prefix = argv[4];
            need(named(prefix) && prefix.size() <= 87 &&
                 std::all_of(prefix.begin(), prefix.end(), [](unsigned char c) { return c < 127; }),
                 "encode prefix must be printable ASCII (1 to 87 bytes)");
            int twice_bits = 0;
            const auto parsed = std::from_chars(bits.data(), bits.data() + bits.size(), twice_bits);
            need(parsed.ec == std::errc{} && parsed.ptr == bits.data() + bits.size(),
                 "twice-bits must be an integer");
            need(book == "3inst" || book == "mcg" || book == "mul1", "unknown EXL3 codebook");
            encode(book == "3inst" ? EXL3_3INST : book == "mcg" ? EXL3_MCG : EXL3_MUL1,
                   twice_bits, argv[4], argv[5], argv[6]);
        } else if (command == "forward" && argc == 7) {
            const std::string_view prefix = argv[4];
            need(named(prefix) && prefix.size() <= 4096 - 8, "invalid EXL3 module prefix");
            forward(argv[2], argv[3], argv[4], argv[5], argv[6]);
        } else {
            std::fputs(usage, stderr);
            return 2;
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "exl3-checkpoint: %s\n", error.what());
        return 1;
    }
    return 0;
}
