#include <iostream>
#include <vector>
#include <string>

#include "agent.hpp"
#include "gguf_parser.hpp"
#include "tokenizer.hpp"
#include "llama_adapter.hpp"

// Fortran math dependency removed in favor of native C++ inline implementation

#include "native_linear_llm.hpp"

int main(int argc, char* argv[]) {
    std::cout << "===========================================" << std::endl;
    std::cout << " Native C++ Mobile-Optimized LLM Engine" << std::endl;
    std::cout << " Complexity: O(N) Linear Time (Polynomial) " << std::endl;
    std::cout << " Backends: Native C++, NumPy C++ (Eigen)  " << std::endl;
    std::cout << "===========================================" << std::endl;

    Config config = parse_cli(argc, const_cast<const char**>(argv));
    std::string user_prompt = config.user_prompt;
    std::string model_path = config.model_path;
    std::string backend = config.backend;
    bool chat_mode = config.chat_mode;
    bool llama_mode = config.llama_mode;

    try {
        int d_model = 256;
        int vocab_size = 32000;
        
        if (llama_mode) {
            std::cout << "[Translation Layer] Routing inference to local Llama backend" << std::endl;
            LlamaCppEngine model(model_path, chat_mode);
            if (chat_mode) {
                std::cout << "\n[Interactive Chat Mode Started. Type 'exit' to quit.]\n";
                std::string input;
                while (true) {
                    std::cout << "\nUser> ";
                    if (!std::getline(std::cin, input) || input == "exit") break;
                    if (input.empty()) continue;
                    std::cout << "MobileLLM> " << model.generate("User: " + input) << "\n";
                }
            } else {
                Agent<LlamaCppEngine> agent(model);
                std::string final_answer = agent.run_autoresearch_loop(user_prompt);
                std::cout << "\n[Execution Complete]\n" << final_answer << std::endl;
            }
        } else {
            std::cout << "Initializing Native model weights from: " << model_path << std::endl;
            NativeLinearLLM model(d_model, vocab_size, model_path);

            if (chat_mode) {
                std::cout << "\n[Interactive Chat Mode Started. Type 'exit' to quit.]\n";
                std::string input;
                while (true) {
                    std::cout << "\nUser> ";
                    if (!std::getline(std::cin, input) || input == "exit") break;
                    if (input.empty()) continue;
                    std::cout << "MobileLLM> " << model.generate("User: " + input) << "\n";
                }
            } else {
                Agent<NativeLinearLLM> agent(model);
                std::string final_answer = agent.run_autoresearch_loop(user_prompt);
                std::cout << "\n[Execution Complete]\n" << final_answer << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
