#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "llama.h"

struct Config {
    std::string user_prompt = "Analyze the environment and report.";
    std::string model_path = "model.gguf";
    std::string backend = "native-linear";
    bool chat_mode = false;
};

inline Config parse_cli(int argc, const char* argv[]) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) config.user_prompt = argv[++i];
        else if (arg == "--model" && i + 1 < argc) config.model_path = argv[++i];
        else if (arg == "--backend" && i + 1 < argc) config.backend = argv[++i];
        else if (arg == "--chat") config.chat_mode = true;
    }
    return config;
}

class LlamaCppEngine {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_sampler* smpl = nullptr;
    bool is_chat_;

public:
    LlamaCppEngine(const std::string& model_path, bool is_chat = false) : is_chat_(is_chat) {
        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 99; // try GPU if available
        model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!model) {
            throw std::runtime_error("llama_model_load_from_file failed to load model from " + model_path);
        }
        vocab = llama_model_get_vocab(model);

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 4096; // generous context size for AutoResearch
        ctx_params.n_batch = 512;
        ctx_params.no_perf = true;

        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            std::cerr << "Failed to init context\n";
            return;
        }

        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = true;
        smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    ~LlamaCppEngine() {
        if (smpl) llama_sampler_free(smpl);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
    }

    std::string generate(const std::string& prompt) {
        if (!model || !ctx) return "Error: Model not loaded.";

        std::string full_prompt = prompt;
        if (!is_chat_ && (prompt.find("Draft an execution") != std::string::npos || prompt.find("Generating thought/action") != std::string::npos)) {
            full_prompt += "\nThought: ";
        }

        const int n_prompt_estimate = -llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(), NULL, 0, true, true);
        std::vector<llama_token> prompt_tokens(n_prompt_estimate);
        if (llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            return "Error: tokenization failed";
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

        if (llama_model_has_encoder(model)) {
            if (llama_encode(ctx, batch)) {
                return "Error encoding";
            }
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = llama_vocab_bos(vocab);
            }
            batch = llama_batch_get_one(&decoder_start_token_id, 1);
        }

        int n_predict = 256;
        llama_token new_token_id;
        std::string result = "";
        
        if (!is_chat_ && (full_prompt.find("\nThought: ") == full_prompt.size() - 10)) {
            result = "Thought: ";
        }

        for (int n_pos = 0; n_pos + batch.n_tokens < static_cast<int>(prompt_tokens.size()) + n_predict; ) {
            if (llama_decode(ctx, batch)) {
                break;
            }
            n_pos += batch.n_tokens;

            new_token_id = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n >= 0) {
                result += std::string(buf, n);
                
                // Early stopping checks for ReAct Agent Output
                if (!is_chat_) {
                    if (result.find("Observation:") != std::string::npos) {
                        result = result.substr(0, result.find("Observation:"));
                        break;
                    }
                    if (result.find("<|im_end|>") != std::string::npos) {
                        result = result.substr(0, result.find("<|im_end|>"));
                        break;
                    }
                    if (result.find("User:") != std::string::npos) {
                        result = result.substr(0, result.find("User:"));
                        break;
                    }
                }
            }
            batch = llama_batch_get_one(&new_token_id, 1);
        }
        return result;
    }
};
