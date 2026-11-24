// CPU-only teacher-forced scoring with the same llama.cpp runtime as tracing.
#include <llama.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

constexpr int context_size = 512;

static void hex_piece(std::ostream & out, const std::string & text) {
    out << std::hex << std::setfill('0');
    for (unsigned char byte : text) {
        out << std::setw(2) << static_cast<unsigned>(byte);
    }
    out << std::dec;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text,
                                         bool chat = false) {
    std::vector<llama_token> tokens(context_size);
    const int size = llama_tokenize(vocab, text.data(), text.size(), tokens.data(),
                                    tokens.size(), true, chat);
    if (size <= 0) {
        throw std::runtime_error("empty or overlong token sequence");
    }
    tokens.resize(size);
    return tokens;
}

static const float * logits(llama_context * ctx, int index, int vocab_size) {
    const float * row = llama_get_logits_ith(ctx, index);
    if (!row || !std::all_of(row, row + vocab_size, [](float v) { return std::isfinite(v); })) {
        throw std::runtime_error("invalid logits");
    }
    return row;
}

static void generate(llama_context * ctx, const llama_vocab * vocab, int vocab_size) {
    auto prefix = tokenize(vocab, "The color of a clear sky is");
    llama_memory_clear(llama_get_memory(ctx), false);
    if (llama_decode(ctx, llama_batch_get_one(prefix.data(), prefix.size()))) {
        throw std::runtime_error("smoke prompt decode failed");
    }
    std::string text;
    for (int i = 0; i < 8; ++i) {
        const float * row = logits(ctx, -1, vocab_size);
        auto token = static_cast<llama_token>(std::max_element(row, row + vocab_size) - row);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        char buf[512];
        const int size = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
        if (size < 0 || llama_decode(ctx, llama_batch_get_one(&token, 1))) {
            throw std::runtime_error("generation failed");
        }
        text.append(buf, size);
    }
    if (text.find_first_not_of(" \t\n\r") == std::string::npos) {
        throw std::runtime_error("generation produced no text");
    }
    std::cout << "generation\t";
    hex_piece(std::cout, text);
    std::cout << '\n';
}

static void score(llama_context * ctx, const llama_vocab * vocab, int vocab_size,
                  const std::string & prefix, const std::string & suffix) {
    const auto prompt = tokenize(vocab, prefix);
    auto whole = tokenize(vocab, prefix + suffix);
    if (whole.size() <= prompt.size() || !std::equal(prompt.begin(), prompt.end(), whole.begin())) {
        throw std::runtime_error("prompt/answer is not a token boundary: " + prefix);
    }
    llama_memory_clear(llama_get_memory(ctx), false);
    auto batch = llama_batch_get_one(whole.data(), whole.size() - 1);
    // Ask for logits at positions that predict each suffix token, no others.
    std::vector<int8_t> outputs(batch.n_tokens, 0);
    std::fill(outputs.begin() + prompt.size() - 1, outputs.end(), 1);
    batch.logits = outputs.data();
    if (llama_decode(ctx, batch)) {
        throw std::runtime_error("scoring decode failed");
    }
    double logp = 0, prob = 0;
    uint64_t ranks = 0;
    bool top1 = true;
    std::ostringstream details;
    details << std::setprecision(17);
    for (size_t i = 0; i < whole.size(); ++i) {
        if (i > 0) {
            std::cout << ',';
        }
        std::cout << whole[i];
    }
    for (size_t i = prompt.size(); i < whole.size(); ++i) {
        const float * row = logits(ctx, i - 1, vocab_size);
        const float maximum = *std::max_element(row, row + vocab_size);
        const float answer = row[whole[i]];
        double sum = 0;
        unsigned rank = 1;
        for (int j = 0; j < vocab_size; ++j) {
            sum += std::exp(static_cast<double>(row[j]) - maximum);
            rank += row[j] > answer;
        }
        const double lp = static_cast<double>(answer) - maximum - std::log(sum);
        logp += lp;
        prob += std::exp(lp);
        ranks += rank;
        top1 &= rank == 1;
        char buf[512];
        const int size = llama_token_to_piece(vocab, whole[i], buf, sizeof(buf), 0, false);
        if (size < 0) {
            throw std::runtime_error("token piece exceeds size limit");
        }
        if (i > prompt.size()) {
            details << ',';
        }
        details << lp << ':';
        hex_piece(details, std::string(buf, size));
    }
    std::cout << '\t' << prompt.size() << '\t' << logp << '\t' << prob << '\t' << ranks
              << '\t' << top1 << '\t' << details.str() << '\n';
}

struct Model {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> weights{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx{nullptr, llama_free};
    const llama_vocab * vocab;
    int size;

    Model(const char * path, int threads) {
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        weights.reset(llama_model_load_from_file(path, mp));
        if (!weights || llama_model_has_encoder(weights.get())) {
            throw std::runtime_error("expected a loadable decoder-only GGUF model");
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = context_size;
        cp.n_batch = context_size;
        cp.n_ubatch = context_size;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.offload_kqv = false;
        cp.op_offload = false;
        ctx.reset(llama_init_from_model(weights.get(), cp));
        if (!ctx) {
            throw std::runtime_error("cannot create llama.cpp context");
        }
        vocab = llama_model_get_vocab(weights.get());
        size = llama_vocab_n_tokens(vocab);
    }

    std::vector<llama_token> chat(const std::string & system, const std::string & user) const {
        const char * tmpl = llama_model_chat_template(weights.get(), nullptr);
        if (!tmpl || !*tmpl) {
            throw std::runtime_error("GGUF is missing a chat template");
        }
        llama_chat_message messages[] = {{"system", system.c_str()}, {"user", user.c_str()}};
        const auto first = system.empty() ? 1 : 0;
        std::string text(2 * (system.size() + user.size()) + 256, '\0');
        const int n = llama_chat_apply_template(tmpl, messages + first, 2 - first, true,
                                               text.data(), text.size());
        if (n <= 0 || static_cast<size_t>(n) >= text.size()) {
            throw std::runtime_error("unsupported or oversized GGUF chat template");
        }
        text.resize(n);
        return tokenize(vocab, text, true);
    }

    void feed(llama_token * tokens, size_t n) const {
        if (llama_decode(ctx.get(), llama_batch_get_one(tokens, n))) {
            throw std::runtime_error("behavior decode failed");
        }
    }

    const float * row() const { return logits(ctx.get(), -1, size); }
    llama_token greedy(const float * p) const { return std::max_element(p, p + size) - p; }
};

constexpr int gen_tokens = 64, kl_tokens = 4;

static std::string unhex(std::string_view hex) {
    auto digit = [](char ch) {
        if (ch >= '0' && ch <= '9') return ch - '0';
        if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
        throw std::runtime_error("invalid chat input hex");
    };
    if (hex.size() % 2 || hex.size() > 8192) {
        throw std::runtime_error("invalid chat input length");
    }
    std::string text;
    for (size_t i = 0; i < hex.size(); i += 2) {
        const auto byte = digit(hex[i]) * 16 + digit(hex[i + 1]);
        if (!byte) throw std::runtime_error("NUL in chat input");
        text.push_back(static_cast<char>(byte));
    }
    return text;
}

static void reply(const Model & model, int limit) {
    int count = 0;
    std::cout << '\t' << std::hex << std::setfill('0');
    for (; count < limit; ++count) {
        auto token = model.greedy(model.row());
        if (llama_vocab_is_eog(model.vocab, token)) break;
        char buf[512];
        const int len = llama_token_to_piece(model.vocab, token, buf, sizeof(buf), 0, false);
        if (len < 0) throw std::runtime_error("oversized token piece");
        for (int i = 0; i < len; ++i) {
            std::cout << std::setw(2) << static_cast<unsigned char>(buf[i]) + 0u;
        }
        if (count + 1 < limit) model.feed(&token, 1);
    }
    std::cout << std::dec << '\t' << count;
}

static double kl(const float * p, const float * q, int size) {
    const double pmax = *std::max_element(p, p + size), qmax = *std::max_element(q, q + size);
    double p_sum = 0, q_sum = 0, weighted_delta = 0;
    for (int i = 0; i < size; ++i) {
        const double lp = p[i] - pmax, lq = q[i] - qmax;
        const double weight = std::exp(lp);
        p_sum += weight;
        q_sum += std::exp(lq);
        weighted_delta += weight * (lp - lq);
    }
    return std::max(0., weighted_delta / p_sum + std::log(q_sum / p_sum));
}

static void behavior(const Model & parent, const Model & child) {
    if (parent.size != child.size) throw std::runtime_error("GGUF vocabulary sizes differ");
    for (int i = 0; i < parent.size; ++i) {
        if (std::string_view(llama_vocab_get_text(parent.vocab, i)) !=
            llama_vocab_get_text(child.vocab, i) ||
            llama_vocab_get_attr(parent.vocab, i) != llama_vocab_get_attr(child.vocab, i) ||
            llama_vocab_is_eog(parent.vocab, i) != llama_vocab_is_eog(child.vocab, i)) {
            throw std::runtime_error("GGUF vocabulary IDs differ");
        }
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        const auto a = line.find('\t'), b = line.find('\t', a + 1);
        if (a == std::string::npos || b == std::string::npos) {
            throw std::runtime_error("expected behavior mode and two hex chat fields");
        }
        const auto mode = line.substr(0, a);
        int limit = gen_tokens;
        if (mode.compare(0, 7, "marker:") == 0) {
            const auto cap = mode.substr(7);
            if (cap.empty() || cap.size() > 3 || cap.find_first_not_of("0123456789") != cap.npos) {
                throw std::runtime_error("invalid marker token cap");
            }
            limit = std::stoi(cap);
            if (limit < 1 || limit > 128) throw std::runtime_error("invalid marker token cap");
        } else if (mode != "generate" && mode != "kl") {
            throw std::runtime_error("invalid behavior mode");
        }
        const std::string_view input(line);
        const auto system = unhex(input.substr(a + 1, b - a - 1));
        const auto user = unhex(input.substr(b + 1));
        auto tokens = parent.chat(system, user);
        if (tokens != child.chat(system, user) || tokens.size() + limit > context_size) {
            throw std::runtime_error("chat token IDs differ or leave no room for generation");
        }
        for (const auto * model : {&parent, &child}) {
            llama_memory_clear(llama_get_memory(model->ctx.get()), false);
            model->feed(tokens.data(), tokens.size());
        }
        if (mode != "kl") {
            std::cout << mode;
            reply(parent, limit);
            reply(child, limit);
            std::cout << '\n';
            continue;
        }
        const double first = kl(parent.row(), child.row(), parent.size);
        double total = 0;
        int positions = 0;
        for (; positions < kl_tokens; ++positions) {
            auto token = parent.greedy(parent.row());
            if (llama_vocab_is_eog(parent.vocab, token)) break;
            parent.feed(&token, 1);
            child.feed(&token, 1);
            total += kl(parent.row(), child.row(), parent.size);
        }
        std::cout << "kl\t" << first << '\t' << total << '\t' << positions << '\n';
    }
}

int main(int argc, char ** argv) try {
    if (argc != 3 && !(argc == 5 && std::string_view(argv[3]) == "--behavior-v1")) {
        throw std::runtime_error("usage: scratch-check-logits MODEL.gguf THREADS [--behavior-v1 CHILD.gguf]");
    }
    const int threads = std::stoi(argv[2]);
    if (threads < 1 || threads > 16) {
        throw std::runtime_error("expected 1 to 16 CPU threads");
    }
    llama_backend_init();
    const Model model(argv[1], threads);
    std::cout << std::setprecision(17);
    if (argc == 5) {
        behavior(model, Model(argv[4], threads));
        return 0;
    }
    generate(model.ctx.get(), model.vocab, model.size);
    std::string line;
    while (std::getline(std::cin, line)) {
        const auto tab = line.find('\t');
        if (tab == std::string::npos || line.find('\t', tab + 1) != std::string::npos) {
            throw std::runtime_error("expected one tab-delimited prompt/answer per row");
        }
        score(model.ctx.get(), model.vocab, model.size, line.substr(0, tab), line.substr(tab + 1));
    }
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
