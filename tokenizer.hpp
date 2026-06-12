#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>

/**
 * Native C++ Byte-Pair Encoding (BPE) Tokenizer
 * Designed for fast, zero-dependency text splitting and token decoding.
 */
class BPETokenizer {
public:
    BPETokenizer(const std::string& vocab_path) {
        // In a real implementation, we would parse a tokenizer.json or read directly from the GGUF metadata
        std::cout << "[Tokenizer] Initializing BPE vocabulary from: " << vocab_path << std::endl;
        
        // Setup mock vocab for basic operation
        vocab_["<|endoftext|>"] = 0;
        vocab_["The"] = 101;
        vocab_["Agent"] = 2045;
        vocab_["is"] = 1032;
        vocab_["running"] = 4501;
        
        for (const auto& pair : vocab_) {
            inverse_vocab_[pair.second] = pair.first;
        }
    }

    // Encode a raw string into a stream of integer tokens
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        // Simple word-boundary split for scaffolding (Real BPE merges byte pairs)
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            if (vocab_.find(word) != vocab_.end()) {
                tokens.push_back(vocab_[word]);
            } else {
                // Fallback to unknown or arbitrary mock token
                tokens.push_back(999); 
            }
        }
        return tokens;
    }

    // Decode an integer stream back into human-readable text
    std::string decode(const std::vector<int>& tokens) {
        std::string text = "";
        for (int t : tokens) {
            if (inverse_vocab_.find(t) != inverse_vocab_.end()) {
                text += inverse_vocab_[t] + " ";
            } else {
                text += "<UNK> ";
            }
        }
        return text;
    }

private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> inverse_vocab_;
};

#endif // TOKENIZER_HPP
