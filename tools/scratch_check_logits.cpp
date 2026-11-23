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
#include <vector>

constexpr int context_size = 512;

static void hex_piece(std::ostream & out, const std::string & text) {
    out << std::hex << std::setfill('0');
    for (unsigned char byte : text) {
        out << std::setw(2) << static_cast<unsigned>(byte);
    }
    out << std::dec;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> tokens(context_size);
    const int size = llama_tokenize(vocab, text.data(), text.size(), tokens.data(),
                                    tokens.size(), true, false);
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

int main(int argc, char ** argv) try {
    if (argc != 3) {
        throw std::runtime_error("usage: scratch-check-logits MODEL.gguf THREADS");
    }
    const int threads = std::stoi(argv[2]);
    if (threads < 1 || threads > 16) {
        throw std::runtime_error("expected 1 to 16 CPU threads");
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model || llama_model_has_encoder(model.get())) {
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
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) {
        throw std::runtime_error("cannot create llama.cpp context");
    }
    const auto * vocab = llama_model_get_vocab(model.get());
    const int vocab_size = llama_vocab_n_tokens(vocab);
    std::cout << std::setprecision(17);
    generate(ctx.get(), vocab, vocab_size);
    std::string line;
    while (std::getline(std::cin, line)) {
        const auto tab = line.find('\t');
        if (tab == std::string::npos || line.find('\t', tab + 1) != std::string::npos) {
            throw std::runtime_error("expected one tab-delimited prompt/answer per row");
        }
        score(ctx.get(), vocab, vocab_size, line.substr(0, tab), line.substr(tab + 1));
    }
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
