#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <array>

class LlamaServerAdapter {
    bool is_chat_;
    std::string backend_;
    std::string model_;
public:
    LlamaServerAdapter(bool is_chat = false, std::string backend = "llama.cpp", std::string model = "llama3") 
        : is_chat_(is_chat), backend_(backend), model_(model) {}
    
    std::string generate(const std::string& prompt) {
        std::ofstream out("/tmp/llm_prompt.txt");
        out << prompt;
        out.close();
        
        std::string cmd = "python3 /root/mobile-llm/request_llama.py";
        if (is_chat_) cmd += " --chat";
        cmd += " --backend " + backend_ + " --model " + model_;
        cmd += " < /tmp/llm_prompt.txt";
        
        std::array<char, 128> buffer;
        std::string result;
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) return "Error";
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }
        pclose(pipe);
        return result;
    }

    bool get_is_chat() const { return is_chat_; }
    std::string get_backend() const { return backend_; }
    std::string get_model() const { return model_; }
};

struct Config {
    std::string user_prompt = "Analyze the environment and report.";
    std::string model_path = "model.gguf";
    std::string backend = "llama.cpp";
    bool chat_mode = false;
    bool llama_mode = false;
};

inline Config parse_cli(int argc, const char* argv[]) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) config.user_prompt = argv[++i];
        else if (arg == "--model" && i + 1 < argc) config.model_path = argv[++i];
        else if (arg == "--backend" && i + 1 < argc) { config.backend = argv[++i]; config.llama_mode = true; }
        else if (arg == "--chat") config.chat_mode = true;
        else if (arg == "--llama") config.llama_mode = true;
    }
    return config;
}
