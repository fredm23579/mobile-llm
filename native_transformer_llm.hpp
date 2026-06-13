#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include "agent.hpp"
#include "gguf_parser.hpp"
#include "tokenizer.hpp"

class NativeTransformerLLM {
public:
    NativeTransformerLLM(int d_model, int vocab_size, const std::string& model_path) 
        : d_model_(d_model), vocab_size_(vocab_size), 
          parser_(model_path), tokenizer_("vocab.json") {
        
        W_out = parser_.load_tensor("output.weight", d_model_ * vocab_size_);
        token_embd = parser_.load_tensor("token_embd.weight", d_model_ * vocab_size_);
        
        // Load projection matrices for Self-Attention (Query, Key, Value)
        W_q = parser_.load_tensor("attn_q.weight", d_model_ * d_model_);
        W_k = parser_.load_tensor("attn_k.weight", d_model_ * d_model_);
        W_v = parser_.load_tensor("attn_v.weight", d_model_ * d_model_);
    }

    std::string generate(const std::string& prompt) {
        // Encode user prompt into token sequence
        std::vector<int> tokens = tokenizer_.encode(prompt);
        if (tokens.empty()) return ""; 
        
        int max_tokens = 64; 
        std::vector<int> generated_tokens;
        
        for (int step = 0; step < max_tokens; ++step) {
            std::vector<int> context_tokens = tokens;
            context_tokens.insert(context_tokens.end(), generated_tokens.begin(), generated_tokens.end());
            int seq_len = context_tokens.size();
            
            // O(N^2) Self-Attention Forward Pass Simulation
            // We pre-allocate Q, K, V matrices for the entire sequence
            std::vector<std::vector<float>> Q(seq_len, std::vector<float>(d_model_, 0.0f));
            std::vector<std::vector<float>> K(seq_len, std::vector<float>(d_model_, 0.0f));
            std::vector<std::vector<float>> V(seq_len, std::vector<float>(d_model_, 0.0f));
            
            for (int t = 0; t < seq_len; ++t) {
                int token = context_tokens[t];
                // Defensive check to avoid segmentation faults on unknown tokens
                if (token < 0 || token >= vocab_size_) token = 0;
                
                // Q, K, V projections
                for (int i = 0; i < d_model_; ++i) {
                    float emb_val = token_embd[token * d_model_ + i];
                    Q[t][i] = emb_val * 0.9f; // Simplified projection
                    K[t][i] = emb_val * 0.9f;
                    V[t][i] = emb_val * 0.9f;
                }
            }
            
            // Scaled Dot-Product Attention: softmax(Q * K^T / sqrt(d)) * V
            std::vector<float> last_hidden(d_model_, 0.0f);
            
            for (int i = 0; i < d_model_; ++i) {
                float val = 0.0f;
                // We focus our attention calculation on the most recent token (seq_len - 1)
                // computing its attention scores against all past tokens
                float sum_exp = 0.0f;
                for (int past = 0; past < seq_len; ++past) {
                    float score = 0.0f;
                    for (int d = 0; d < d_model_; ++d) {
                        score += Q[seq_len - 1][d] * K[past][d];
                    }
                    score /= std::sqrt(static_cast<float>(d_model_));
                    float weight = std::exp(score);
                    sum_exp += weight;
                    val += weight * V[past][i];
                }
                last_hidden[i] = val / (sum_exp + 1e-9f);
            }

            int predicted_token = 0;
            float max_val = -1e9;
            for (int i = 0; i < vocab_size_; ++i) {
                float val = 0.0f;
                for (int j = 0; j < d_model_; ++j) {
                    val += last_hidden[j] * W_out[j * vocab_size_ + i];
                }
                if (val > max_val) {
                    max_val = val;
                    predicted_token = i;
                }
            }
            
            generated_tokens.push_back(predicted_token);
            
            if (predicted_token == 0) break;
            
            std::string current_text = tokenizer_.decode(generated_tokens);
            if (current_text.find("Observation:") != std::string::npos || 
                current_text.find("<|im_end|>") != std::string::npos ||
                current_text.find("User:") != std::string::npos) {
                break;
            }
        }
        
        return tokenizer_.decode(generated_tokens);
    }

private:
    int d_model_;
    int vocab_size_;
    std::vector<float> W_out;
    std::vector<float> token_embd;
    std::vector<float> W_q, W_k, W_v;
    GGUFParser parser_;
    BPETokenizer tokenizer_;
};
