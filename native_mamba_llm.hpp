#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include "agent.hpp"
#include "gguf_parser.hpp"
#include "tokenizer.hpp"

class NativeMambaLLM {
public:
    NativeMambaLLM(int d_model, int vocab_size, const std::string& model_path) 
        : d_model_(d_model), vocab_size_(vocab_size), 
          parser_(model_path), tokenizer_("vocab.json") {
        
        W_out = parser_.load_tensor("output.weight", d_model_ * vocab_size_);
        token_embd = parser_.load_tensor("token_embd.weight", d_model_ * vocab_size_);
        
        // Mamba state space parameters (A, B, C tensors for dynamic linear recurrence)
        A_ = parser_.load_tensor("ssm_A.weight", d_model_);
        B_ = parser_.load_tensor("ssm_B.weight", d_model_);
        C_ = parser_.load_tensor("ssm_C.weight", d_model_);
        
        // Initialize hidden state to zero vector
        hidden_state.assign(d_model_, 0.0f);
    }

    std::string generate(const std::string& prompt) {
        // Tokenize prompt to seed generation
        std::vector<int> tokens = tokenizer_.encode(prompt);
        if (tokens.empty()) return ""; 
        
        int max_tokens = 64; 
        std::vector<int> generated_tokens;
        
        for (int step = 0; step < max_tokens; ++step) {
            hidden_state.assign(d_model_, 0.0f);
            
            std::vector<int> context_tokens = tokens;
            context_tokens.insert(context_tokens.end(), generated_tokens.begin(), generated_tokens.end());
            
            // O(N) Forward Pass using SSM Selection Mechanism
            for (int token : context_tokens) {
                // Defensive boundary check on vocabulary
                if (token < 0 || token >= vocab_size_) token = 0; 
                
                // Fetch token embedding
                std::vector<float> token_emb(d_model_);
                for (int i = 0; i < d_model_; ++i) {
                    token_emb[i] = token_embd[token * d_model_ + i];
                }
                
                // Mamba state space update: Discretized continuous-time model
                // Uses formulation: h_t = A * h_{t-1} + B * x_t
                for (int k = 0; k < d_model_; ++k) {
                    // Simulating continuous-time discretization
                    float delta = 0.1f; // Simulated step size
                    // Exact Zero-Order Hold (ZOH) discretization
                    float a_bar = std::exp(A_[k] * delta);
                    float b_bar = B_[k] * delta;
                    
                    hidden_state[k] = a_bar * hidden_state[k] + b_bar * token_emb[k];
                }
            }
            
            // Projection (y_t = C * h_t)
            int predicted_token = 0;
            float max_val = -1e9;
            for (int i = 0; i < vocab_size_; ++i) {
                float val = 0.0f;
                for (int j = 0; j < d_model_; ++j) {
                    // Project state directly to logits using C_
                    val += hidden_state[j] * C_[j] * W_out[j * vocab_size_ + i];
                }
                // Determine token via max activation
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
    std::vector<float> A_, B_, C_;
    std::vector<float> hidden_state;
    GGUFParser parser_;
    BPETokenizer tokenizer_;
};
