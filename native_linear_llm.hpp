#pragma once

#include <iostream>
#include <vector>
#include <string>

#include "agent.hpp"
#include "gguf_parser.hpp"
#include "tokenizer.hpp"

class NativeLinearLLM {
public:
    NativeLinearLLM(int d_model, int vocab_size, const std::string& model_path) 
        : d_model_(d_model), vocab_size_(vocab_size), 
          parser_(model_path), tokenizer_("vocab.json") {
        
        W_out = parser_.load_tensor("output.weight", d_model_ * vocab_size_);
        token_embd = parser_.load_tensor("token_embd.weight", d_model_ * vocab_size_);
        
        // Initialize hidden state to zero vector
        hidden_state.assign(d_model_, 0.0f);
    }

    std::string generate(const std::string& prompt) {
        // Tokenize the initial user prompt
        std::vector<int> tokens = tokenizer_.encode(prompt);
        if (tokens.empty()) return ""; 
        
        int max_tokens = 64; 
        std::vector<int> generated_tokens;
        
        for (int step = 0; step < max_tokens; ++step) {
            // Reset hidden state for each autoregressive step to prevent unbounded accumulation
            hidden_state.assign(d_model_, 0.0f);
            
            // Combine original prompt with newly generated tokens
            std::vector<int> context_tokens = tokens;
            context_tokens.insert(context_tokens.end(), generated_tokens.begin(), generated_tokens.end());
            
            // O(N) Sub-Polynomial inference pass
            for (int token : context_tokens) {
                // Defensive: Ensure token ID is strictly within bounds
                if (token < 0 || token >= vocab_size_) token = 0; 
                
                // Extract embedding for the current token
                std::vector<float> token_emb(d_model_);
                for (int i = 0; i < d_model_; ++i) {
                    token_emb[i] = token_embd[token * d_model_ + i];
                }
                
                float* state_ptr = hidden_state.data();
                const float* emb_ptr = token_emb.data();
                
                // Exponential moving average: Decay past state, add new token context
                float decay = 0.9f;
                for (int k = 0; k < d_model_; ++k) {
                    state_ptr[k] = state_ptr[k] * decay + emb_ptr[k] * (1.0f - decay);
                }
            }

            int predicted_token = 0;
            float max_val = -1e9;
            // Project final hidden state to logits using O(V*D) operation
            for (int i = 0; i < vocab_size_; ++i) {
                float val = 0.0f;
                for (int j = 0; j < d_model_; ++j) {
                    val += hidden_state[j] * W_out[j * vocab_size_ + i];
                }
                // Greedy sampling: Take maximum logit directly
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
    std::vector<float> hidden_state;
    GGUFParser parser_;
    BPETokenizer tokenizer_;
};
