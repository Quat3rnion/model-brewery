#include "exl3.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <new>

namespace exl3 {
// Private CPU codebook primitive defined in cpu.cpp.
float codebook_value(std::uint32_t state, int book);
}

namespace {
constexpr int dim = 128;
constexpr int tile_dim = 16;
constexpr int steps = tile_dim * tile_dim;
constexpr int state_bits = 16;
constexpr int state_count = 1 << state_bits;
constexpr int beam_width = 16;

using Targets = std::array<float, steps>;
using Path = std::array<std::uint8_t, steps>;

struct Node {
    double score;
    std::uint16_t state;
    std::uint8_t parent;
};

int step_bits(int twice_bits, int k)
{
    return twice_bits / 2 + (twice_bits & k & 1);
}

// State immediately before symbol k: the 16 preceding bits of the ring.
std::uint16_t anchor(const Path& path, int twice_bits, int k)
{
    std::uint32_t state = 0;
    int bits = 0;
    while (bits < state_bits) {
        k = (k + steps - 1) % steps;
        state |= std::uint32_t{path[k]} << bits;
        bits += step_bits(twice_bits, k);
    }
    return static_cast<std::uint16_t>(state);
}

double cost(double value, double twice_target)
{
    // Squared error minus the constant target^2. Removing the common term
    // keeps the search meaningful even for finite inputs near FLT_MAX.
    return value * value - twice_target * value;
}

double score_path(const Path& path, const Targets& target, int twice_bits, const float* table)
{
    double score = 0;
    std::uint16_t state = anchor(path, twice_bits, 0);
    for (int k = 0; k < steps; ++k) {
        state = static_cast<std::uint16_t>((std::uint32_t{state} << step_bits(twice_bits, k)) | path[k]);
        score += cost(table[state], 2.0 * target[k]);
    }
    return score;
}

// Bounded beam search, not an exact Viterbi solver. For a fixed anchor,
// every candidate's last 16 appended bits must repeat the initial state.
// That makes every traceback a valid tail-biting path, with no unscored seam.
Path optimize(const Targets& target, int twice_bits, const float* table, int start,
              std::uint16_t seed)
{
    using Beam = std::array<Node, beam_width>;
    std::array<Beam, steps + 1> trace;
    trace[0][0] = {0, seed, 0};
    int count = 1;
    int offset = 0;
    const int free_bits = steps * twice_bits / 2 - state_bits;
    for (int i = 0; i < steps; ++i) {
        const int k = (start + i) % steps;
        const int shift = step_bits(twice_bits, k);
        offset += shift;
        int fixed = offset - free_bits;
        if (fixed < 0) fixed = 0;
        if (fixed > shift) fixed = shift;
        const unsigned mask = (1u << fixed) - 1;
        const unsigned suffix = fixed ? (seed >> (state_bits + free_bits - offset)) & mask : 0;

        const double twice_target = 2.0 * target[k];
        const auto& previous = trace[i];
        auto& next = trace[i + 1];
        int next_count = 0;
        int worst = 0;
        for (int j = 0; j < count; ++j) {
            const auto prefix = std::uint32_t{previous[j].state} << shift;
            for (unsigned appended = suffix; appended < (1u << shift); appended += mask + 1) {
                const auto state = static_cast<std::uint16_t>(prefix | appended);
                const double score = previous[j].score + cost(table[state], twice_target);
                const bool full = next_count == beam_width;
                if (full && score >= next[worst].score) continue;

                // Keep only the cheapest prefix for each state; its possible
                // continuations (including the forced suffix) are identical.
                int slot = 0;
                while (slot < next_count && next[slot].state != state) ++slot;
                if (slot < next_count) {
                    if (score >= next[slot].score) continue;
                } else if (!full) {
                    ++next_count;
                } else {
                    slot = worst;
                }
                next[slot] = {score, state, static_cast<std::uint8_t>(j)};
                if (next_count == beam_width && (!full || slot == worst)) {
                    worst = 0;
                    for (int n = 1; n < next_count; ++n) {
                        if (next[n].score > next[worst].score) worst = n;
                    }
                }
            }
        }
        count = next_count;
    }

    // Closure guarantees all survivors end in seed, hence exactly one remains.
    Path path;
    int j = 0;
    for (int i = steps - 1; i >= 0; --i) {
        const int k = (start + i) % steps;
        const auto& node = trace[i + 1][j];
        path[k] = static_cast<std::uint8_t>(node.state & ((1u << step_bits(twice_bits, k)) - 1));
        j = node.parent;
    }
    return path;
}

void pack_tile(const Path& path, int twice_bits, std::uint8_t* out)
{
    std::uint64_t word = 0;
    int bits = 0;
    for (int k = 0; k < steps; ++k) {
        const int shift = step_bits(twice_bits, k);
        word = (word << shift) | path[k];
        bits += shift;
        if (bits >= 32) {
            bits -= 32;
            const auto w = static_cast<std::uint32_t>(word >> bits);
            for (int n = 0; n < 4; ++n) *out++ = static_cast<std::uint8_t>(w >> (8 * n));
        }
    }
}

void encode_tile(const float* base, int twice_bits, const float* table, std::uint8_t* out)
{
    Targets target;
    for (int i = 0; i < steps; ++i) {
        const int lane = i / 8, element = i % 8;
        const int row = 2 * (lane % 4) + (element & 1) + 4 * (element & 2);
        const int col = lane / 4 + 2 * (element & 4);
        target[i] = base[row * dim + col];
    }
    Path best{};
    double score = score_path(best, target, twice_bits, table);
    // Move the second anchor halfway round the ring, freeing the first cut
    // to improve. Retain the best complete ring, including the zero control.
    for (int start : {0, steps / 2}) {
        const auto candidate = optimize(target, twice_bits, table, start, anchor(best, twice_bits, start));
        const double candidate_score = score_path(candidate, target, twice_bits, table);
        if (candidate_score < score) {
            best = candidate;
            score = candidate_score;
        }
    }
    pack_tile(best, twice_bits, out);
}
}  // namespace

extern "C" int exl3_encode(const float* rotated, int twice_bits, int book,
                            std::uint8_t* packed, std::size_t bytes)
{
    if (!rotated || !packed || book < EXL3_3INST || book > EXL3_MUL1 ||
        twice_bits < 2 || twice_bits > 16 ||
        ((twice_bits & 1) && (book != EXL3_MUL1 || twice_bits > 7)) ||
        bytes != 1024u * static_cast<unsigned>(twice_bits))
        return 1;
    for (int i = 0; i < dim * dim; ++i) {
        if (!std::isfinite(rotated[i])) return 1;
    }

    // Fixed budget: 256 KiB codebook + ~65 KiB traceback. At most
    // 2 * 64 * 256 * 16 * 256 candidate transitions per block.
    const std::unique_ptr<float[]> table(new (std::nothrow) float[state_count]);
    if (!table) return 1;
    for (std::uint32_t state = 0; state < state_count; ++state)
        table[state] = exl3::codebook_value(state, book);

    for (int row = 0; row < dim; row += tile_dim) {
        for (int col = 0; col < dim; col += tile_dim) {
            encode_tile(rotated + row * dim + col, twice_bits, table.get(), packed);
            packed += twice_bits * 16;
        }
    }
    return 0;
}
