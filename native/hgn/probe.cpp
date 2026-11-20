#include "hgn.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>

using namespace hgn;
static Backend backend;

static void near(const std::vector<double>& got, const std::vector<double>& want) {
    if (got.size() != want.size()) throw std::runtime_error("output shape");
    for (size_t i = 0; i < got.size(); ++i)
        if (!(std::abs(got[i] - want[i]) <= 1e-9 * (1 + std::abs(want[i]))))
            throw std::runtime_error("numerical mismatch at " + std::to_string(i) +
                                     ": " + std::to_string(got[i]) + " vs " + std::to_string(want[i]));
}

static void q6() {
    // 64 columns, 48 packed code bytes, 4 affine bytes, 12 padding bytes.
    // All four high-bit positions differ; use two unequal rows.
    const unsigned k = 64;
    std::vector<uint8_t> payload(128);
    std::vector<double> x(k);
    for (unsigned c = 0; c < k; ++c) x[c] = double(int(c % 7) - 3) / 8;
    std::vector<double> want(2);
    for (unsigned r = 0; r < 2; ++r) {
        auto* p = payload.data() + 64 * r;
        for (unsigned c = 0; c < k; ++c) {
            unsigned code = (c * 13 + r * 7) % 64;
            p[c / 2] |= (code % 16) << ((c % 2) * 4);
            p[32 + c / 4] |= (code / 16) << ((c % 4) * 2);
            want[r] += (code / 8.0 - 0.5) * x[c];
        }
        p[49] = 0x30;  // 1/8
        p[51] = 0xb8;  // -1/2
    }
    Tensor tensor{"q6.weight", 24, 64, {2, k}, {payload.data(), payload.size()}};
    near(matvec(tensor, {}, {}, -1, x, backend), want);
    std::cout << "PASS Q6G64 24/64 packed matvec\n";
}

static void put(uint8_t* dst, uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i) dst[i] = uint8_t(value >> (8 * i));
}
static void float_bits(uint8_t* dst, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    put(dst, bits, 4);
}

static void dense() {
    for (unsigned type : {0, 1, 2}) {
        unsigned width = type == 1 ? 4 : 2;
        std::vector<uint8_t> p(6 * width);
        const unsigned bf[] = {0x3fc0, 0xbf00, 0x4000, 0xc040, 0x3f80, 0xbe80};
        const unsigned hf[] = {0x3e00, 0xb800, 0x4000, 0xc200, 0x3c00, 0xb400};
        for (unsigned i = 0; i < 6; ++i) {
            put(p.data() + i * width, type == 2 ? hf[i] : type == 1 ? bf[i] << 16 : bf[i], width);
        }
        near(matvec({"a.weight", type, 0, {2, 3}, {p.data(), p.size()}}, {}, {}, -1,
                    {0.5, -1, 2}, backend), {5.25, -3.0});
    }
    std::cout << "PASS BF16 0/0, F32 1/0, F16 2/0 packed matvec\n";
}

static void affine() {
    const unsigned k = 128;
    std::vector<double> x(k), q4want(2), q8want(2);
    std::vector<uint8_t> q4(64 + 128 + 32), q8(136 * 2);
    for (unsigned i = 0; i < 16; ++i) float_bits(q4.data() + 4 * i, (int(i) - 4) / 4.0f);
    for (unsigned c = 0; c < k; ++c) x[c] = (int(c % 13) - 6) / 16.0;
    for (unsigned r = 0; r < 2; ++r) {
        for (unsigned c = 0; c < k; ++c) {
            unsigned a = (3 * c + 5 * r) % 16, b = (71 * c + 31 * r) % 256;
            q4[64 + r * 64 + c / 2] |= a << (4 * (c % 2));
            q8[136 * r + c] = b;
            q4want[r] += x[c] * (int(a) - 4) / 4 * (c / 32 % 2 ? -0.25 : 0.5);
            q8want[r] += x[c] * (b * (c < 64 ? 0.25 : 0.5) - 1.5);
        }
        for (unsigned g = 0; g < 4; ++g) q4[64 + 128 + r * 16 + g * 2 + 1] = g % 2 ? 0xb4 : 0x38;
        for (unsigned g = 0; g < 2; ++g) {
            q8[136 * r + 128 + 4 * g + 1] = g ? 0x38 : 0x34;
            q8[136 * r + 128 + 4 * g + 3] = 0xbe;
        }
    }
    near(matvec({"q4.weight", 5, 2, {2, k}, {q4.data(), q4.size()}}, {}, {}, -1, x, backend), q4want);
    near(matvec({"q8.weight", 7, 0, {2, k}, {q8.data(), q8.size()}}, {}, {}, -1, x, backend), q8want);
    // The MTP expert uses the same planar store. Select expert 1 of 3 and
    // ensure the codebook and scale-plane offset are global, not per expert.
    std::vector<uint8_t> stacked(288);
    float_bits(stacked.data() + 4, 3.0f);
    for (unsigned i = 96; i < 128; ++i) stacked[i] = 0x11;
    stacked[225] = 0x38;
    stacked[241] = 0xbc;
    near(matvec({"mtp.layers.0.mlp.experts.down_proj.weight", 5, 2, {3, 2, 32},
                 {stacked.data(), stacked.size()}}, {}, {}, 1, std::vector<double>(32, 1), backend),
         {48, -96});
    std::cout << "PASS Q4C-P 5/2 and Q8G64 7/0 packed matvec\n";
}

static void tables() {
    std::vector<uint8_t> table(128 * 3 * 160 + 4);
    const size_t offset = (117 * 3 + 1) * 160;
    const uint8_t codes[] = {0, 1, 0x81, 0x38, 0x3c, 0x7e, 0x7f, 0xff};
    const double values[] = {0, 1.0/512, -1.0/512, 1, 1.5, 448, 480, -480};
    std::vector<double> want(160);
    for (unsigned i = 0; i < 160; ++i) {
        table[offset + i] = codes[i % 8];
        want[i] = values[i % 8] / 8;
    }
    float_bits(table.data() + table.size() - 4, 0.125);
    near(gather_fp8({"layers.1.ple.ngram_embedding.weight", 10, 0, {128, 3, 160},
                    {table.data(), table.size()}}, 117, 1, backend), want);
    uint8_t bytes[24]{};
    put(bytes, 0x8000000000000000ULL, 8);
    put(bytes + 8, 0x7fedcba987654321ULL, 8);
    put(bytes + 16, 0xffffffffffffffffULL, 8);
    if (integers({"ple_embedding.layer_multipliers", 4, 0, {3}, {bytes, sizeof(bytes)}}) !=
        std::vector<int64_t>{INT64_MIN, 0x7fedcba987654321LL, -1}) throw std::runtime_error("inexact I64");
    std::cout << "PASS FP8G 10/0 packed gather and I64 4/0 exact host metadata\n";
}

struct Mapping {
    size_t size;
    uint8_t* data;
    explicit Mapping(size_t n) : size(n), data(static_cast<uint8_t*>(
        mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0))) {
        if (data == MAP_FAILED) throw std::runtime_error("cannot map synthetic HGN expert tensor");
    }
    ~Mapping() { munmap(data, size); }
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
};
static void fwht(std::vector<double>& v) {
    for (size_t base = 0; base < v.size(); base += 128)
        for (size_t stride = 1; stride < 128; stride *= 2)
            for (size_t i = base; i < base + 128; i += 2 * stride)
                for (size_t j = i; j < i + stride; ++j) {
                    double a = v[j], b = v[j + stride];
                    v[j] = a + b; v[j + stride] = a - b;
                }
    for (auto& value : v) value /= std::sqrt(128.0);
}
static unsigned random_step(uint32_t& seed) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return seed;
}

static void rounded_half_levels(double* levels) {
    // Independent oracle: enumerate the exact binary16 candidates and round
    // with a nearest-value search rather than the kernel's integer algorithm.
    std::vector<double> candidates;
    for (unsigned bits = 0; bits < 0x4800; ++bits) {
        unsigned exponent = bits / 1024, fraction = bits % 1024;
        candidates.push_back(exponent ? std::ldexp(1.0 + fraction / 1024.0, int(exponent) - 15) :
                             std::ldexp(double(fraction), -24));
    }
    for (unsigned a = 0; a <= 1020; ++a) {
        double v = (1774.0 * a - 905216) / 262144.0;
        double magnitude = std::abs(v);
        auto hi = std::lower_bound(candidates.begin(), candidates.end(), magnitude);
        auto lo = hi == candidates.begin() ? hi : hi - 1;
        auto idx = hi - candidates.begin();
        double lower = magnitude - *lo, upper = *hi - magnitude;
        levels[a] = std::copysign(lower < upper || (lower == upper && (idx & 1)) ? *lo : *hi, v);
    }
}

static void published_ht_tile() {
    // HGN TEST-VECTORS.md "ht tile". Select columns in the rotated basis
    // via a Hadamard input, then invert the observed output rotation.
    const unsigned row[] = {0, 0, 0, 0, 1, 15, 0, 15};
    const unsigned col[] = {0, 1, 2, 7, 0, 7, 8, 15};
    const double values[][8] = {
        {-0.1236572265625, 1.7578125, -0.61083984375, -0.1981201171875,
         -0.1236572265625, -0.495849609375, 1.2431640625, 0.74267578125},
        {1.48046875, 1.5888671875, -0.32666015625, 0.09967041015625,
         0.1064453125, 1.642578125, -1.341796875, 1.568359375},
    };
    std::vector<uint8_t> ones(256);
    for (size_t i = 1; i < ones.size(); i += 2) ones[i] = 0x3c;
    Tensor su{"gold.suh", 2, 0, {128}, {ones.data(), ones.size()}};
    Tensor sv{"gold.svh", 2, 0, {128}, {ones.data(), ones.size()}};
    for (unsigned bits : {3, 4}) {
        std::vector<uint8_t> packed(128 * 128 * bits / 8);
        for (unsigned j = 0; j < 8 * bits; ++j)
            put(packed.data() + j * 4, 0x9e3779b9U * (j + 1), 4);
        Tensor t{"gold.weight", 16, 0x1200u + 2 * bits, {128, 128}, {packed.data(), packed.size()}};
        for (unsigned i = 0; i < 8; ++i) {
            std::vector<double> x(128);
            x[col[i]] = 1;
            fwht(x);
            auto y = matvec(t, su, sv, -1, x, backend);
            fwht(y);
            near({y[row[i]]}, {values[bits - 3][i]});
        }
    }
    std::cout << "PASS independent published HT3/HT4 tile values through matvec\n";
}

static void rotated_case(unsigned bits, bool experts, bool gate) {
    const bool i4r = bits == 0;
    size_t n = experts ? (gate ? 1280 : 2560) : 256;
    size_t k = experts ? (gate ? 2560 : 640) : 256;
    size_t one = n * k, count = experts ? 512 : 1, e = experts ? 173 : 0;
    size_t codes_per = one * (i4r ? 4 : bits) / 8;
    Mapping codes(count * (codes_per + (i4r ? one / 64 : 0)));
    uint8_t* code = codes.data + e * codes_per;
    bool st_u = experts && !i4r && !gate, st_v = experts && !i4r;
    std::vector<uint8_t> su(2 * k * (st_u ? count : 1)), sv(2 * n * (st_v ? count : 1));
    std::vector<double> x(k), xu(k), ys(n), gain(n);
    std::vector<double> dense_weights(one);
    for (size_t i = 0; i < k; ++i) {
        double sign = (i * 17 % 29 < 13) ? -1 : 1;
        su[2 * ((st_u ? e * k : 0) + i) + 1] = sign < 0 ? 0xbc : 0x3c;
        x[i] = (int(i % 27) - 13) / 4.0;
        xu[i] = sign * x[i];
    }
    for (size_t i = 0; i < n; ++i) {
        gain[i] = i % 3 == 0 ? -0.5 : i % 7 ? 2 : 0;
        sv[2 * ((st_v ? e * n : 0) + i) + 1] = i % 3 == 0 ? 0xb8 : i % 7 ? 0x40 : 0;
    }
    uint32_t seed = 0x734395ae;
    if (i4r) {
        auto* scales = codes.data + count * codes_per + e * one / 64;
        for (size_t i = 0; i < one; ++i) {
            unsigned nib = random_step(seed) % 16;
            code[i / 2] |= nib << ((i % 2) * 4);
            bool large = ((i / 128) % 3) == 0;
            if (i % 128 == 0) scales[(i / 128) * 2 + 1] = large ? 0x38 : 0x34;
            dense_weights[i] = (int(nib) - 8) * (large ? 0.5 : 0.25);
        }
    } else {
        double levels[1021];
        rounded_half_levels(levels);
        // Encode random symbols into native LE words MSB first, tracking the
        // circular state independently with a per-symbol shift register.
        auto tile = [&](size_t row, size_t col) {
            unsigned symbols[256], state = 0;
            for (auto& symbol : symbols) symbol = random_step(seed) % (1u << bits);
            for (unsigned i = 250; i < 256; ++i) state = (state << bits | symbols[i]) & 65535;
            for (unsigned i = 0; i < 256; ++i) {
                state = (state << bits | symbols[i]) & 65535;
                uint32_t hash = state * 0x83dcd12dU, byte_sum = 0;
                for (int j = 0; j < 4; ++j, hash /= 256) byte_sum += hash % 256;
                size_t c = col + (i >= 128 ? 8 : 0) + i % 8, r = row + (i % 128) / 8;
                dense_weights[r * k + c] = levels[byte_sum];
                for (unsigned j = 0; j < bits; ++j) {
                    unsigned at = i * bits + j;
                    unsigned byte = (at / 32) * 4 + 3 - (at % 32) / 8;
                    code[byte] |= ((symbols[i] >> (bits - j - 1)) & 1) << (7 - at % 8);
                }
            }
            code += 32 * bits;
        };
        size_t width = gate ? n / 2 : n;
        for (size_t band = 0; band < width; band += 128)
            for (size_t c = 0; c < k; c += 16)
                for (size_t half = 0; half < (gate ? 2u : 1u); ++half)
                    for (size_t r = band; r < band + 128; r += 16) tile(r + half * width, c);
    }
    fwht(xu);
    for (size_t r = 0; r < n; ++r)
        for (size_t c = 0; c < k; ++c) ys[r] += dense_weights[r * k + c] * xu[c];
    fwht(ys);
    for (size_t r = 0; r < n; ++r) ys[r] *= gain[r];
    std::string base = !experts ? "model.layers.0.self_attn.q_proj" : gate ?
        "layers.12.mlp.experts.gate_up_proj" : "layers.12.mlp.experts.down_proj";
    Tensor t{base + ".weight", i4r ? 23u : 16u, i4r ? 128u : 0x1200u + 2 * bits,
             experts ? std::vector<size_t>{count, n, k} : std::vector<size_t>{n, k},
             {codes.data, codes.size}};
    Tensor u{gate ? "layers.12.mlp.experts.su" : base + ".suh", 2, 0,
             st_u ? std::vector<size_t>{count, k} : std::vector<size_t>{k}, {su.data(), su.size()}};
    Tensor v{base + ".svh", 2, 0, st_v ? std::vector<size_t>{count, n} : std::vector<size_t>{n},
             {sv.data(), sv.size()}};
    near(matvec(t, u, v, experts ? int(e) : -1, x, backend), ys);
    std::cout << "PASS " << (i4r ? "I4R 23/128" : bits == 3 ? "HT3 16/0x1206" : "HT4 16/0x1208")
              << ' ' << (!experts ? "dense" : gate ? "paired gate/up expert" : "down expert") << '\n';
}

template<class Run> static void reject(Run run, const char* expected) {
    try {
        run();
    } catch (const std::exception& err) {
        if (std::string(err.what()).find(expected) != std::string::npos) return;
        throw;
    }
    throw std::runtime_error(std::string("unexpected HGN acceptance: ") + expected);
}

static void invalid() {
    std::vector<uint8_t> raw(128 * 128 / 2), signs(256);
    std::vector<double> x(128, 1);
    auto run = [&](Tensor t, Tensor su = {}, Tensor sv = {}, int expert = -1) {
        return matvec(t, su, sv, expert, x, backend);
    };
    Tensor ht{"layer.q.weight", 16, 0x1208, {128, 128}, {raw.data(), raw.size()}};
    Tensor u{"layer.q.suh", 2, 0, {128}, {signs.data(), signs.size()}};
    Tensor v{"layer.q.svh", 2, 0, {128}, {signs.data(), signs.size()}};
    reject([&] { auto t = ht; t.variant = 0x1308; run(t); }, "type/variant");
    reject([&] { auto t = ht; t.type = 17; run(t); }, "type/variant");
    reject([&] { auto t = ht; t.shape[1] = 127; run(t); }, "geometry");
    reject([&] { auto t = ht; t.bytes.size--; run(t, u, v); }, "payload size");
    reject([&] { run(ht); }, "sidecar");
    reject([&] { auto bad = u; bad.name += "2"; run(ht, bad, v); }, "sidecar");
    reject([&] { auto bad = u; bad.shape = {1, 128}; run(ht, bad, v); }, "sidecar");
    reject([&] { auto bad = u; bad.type = 0; run(ht, bad, v); }, "sidecar");
    reject([&] { run(ht, u, v, 512); }, "expert selection");
    reject([&] { auto t = ht; t.shape = {512, 128, 128}; run(t, u, v, 0); }, "expert shape/name");
    reject([&] { auto t = ht; t.type = 23; t.variant = 128; run(t, u, v); }, "expert shape/name");
    reject([&] { auto t = ht; t.type = 4; t.variant = 0; run(t); }, "type/variant");
    reject([&] { auto t = ht; t.type = 10; t.variant = 0; run(t); }, "type/variant");
    reject([&] { auto t = ht; t.shape = {8192, 4096}; run(t, u, v); }, "work bound");
    signs[1] = 0x7c;
    reject([&] { run(ht, u, v); }, "nonfinite");
    std::cout << "PASS fail-closed type/variant, layout, expert, sidecar and bound cases\n";
}

int main(int argc, char** argv) {
    try {
        std::string which = argc == 2 ? argv[1] : "";
        if (which == "cpu") backend = Backend::cpu;
        else if (which == "cuda") backend = Backend::cuda;
        else if (which == "rocm") backend = Backend::rocm;
        else throw std::invalid_argument("usage: probe cpu|cuda|rocm");
        std::cout << device_name(backend) << '\n';
        dense();
        q6();
        affine();
        tables();
        published_ht_tile();
        for (unsigned bits : {3, 4}) {
            rotated_case(bits, false, false);
            rotated_case(bits, true, false);
            rotated_case(bits, true, true);
        }
        rotated_case(0, true, false);
        rotated_case(0, true, true);
        invalid();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
