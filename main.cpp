#include <iostream>
#include <vector>
#include <string>

#include "agent.hpp"
#include "gguf_parser.hpp"
#include "tokenizer.hpp"
#include "llama_adapter.hpp"

// Fortran math dependency removed in favor of native C++ inline implementation

#include "native_linear_llm.hpp"
#include "native_transformer_llm.hpp"
#include "native_mamba_llm.hpp"

template <typename ModelType>
void run_mode(ModelType& model, bool chat_mode, bool oneshot_mode, const std::string& user_prompt) {
    if (chat_mode) {
        std::cout << "\n[Interactive Chat Mode Started. Type 'exit' to quit.]\n";
        std::string input;
        while (true) {
            std::cout << "\nUser> ";
            if (!std::getline(std::cin, input) || input == "exit") break;
            if (input.empty()) continue;
            // Generate response and stream directly to standard out
            std::cout << "MobileLLM> " << model.generate("User: " + input) << "\n";
        }
    } else if (oneshot_mode) {
        // Direct raw inference without agent wrappers
        std::cout << model.generate(user_prompt) << std::endl;
    } else {
        // Wrap model in the Mythos Agent Protocol for autonomous tool-use
        Agent<ModelType> agent(model);
        std::string final_answer = agent.run_autoresearch_loop(user_prompt);
        std::cout << "\n[Execution Complete]\n" << final_answer << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cerr << "===========================================" << std::endl;
    std::cerr << " Native C++ Mobile-Optimized LLM Engine" << std::endl;
    std::cerr << " Models: Linear, Transformer, Mamba, Llama" << std::endl;
    std::cerr << " Backends: Native C++, llama.cpp  " << std::endl;
    std::cerr << "===========================================" << std::endl;

    Config config = parse_cli(argc, const_cast<const char**>(argv));
    std::string user_prompt = config.user_prompt;
    std::string model_path = config.model_path;
    std::string backend = config.backend;
    bool chat_mode = config.chat_mode;

    try {
        // Defensive: hardcoded base values for demonstration network architecture
        // In a full implementation, these are read dynamically from GGUF metadata.
        int d_model = 256;
        int vocab_size = 32000;
        
        std::cerr << "Initializing Backend Engine: [" << backend << "] from path: " << model_path << std::endl;

        // Dynamic multi-model routing based on CLI flags
        if (backend == "llama.cpp") {
            LlamaCppEngine model(model_path, chat_mode);
            run_mode(model, chat_mode, config.oneshot_mode, user_prompt);
        } else if (backend == "native-transformer") {
            NativeTransformerLLM model(d_model, vocab_size, model_path);
            run_mode(model, chat_mode, config.oneshot_mode, user_prompt);
        } else if (backend == "native-mamba") {
            NativeMambaLLM model(d_model, vocab_size, model_path);
            run_mode(model, chat_mode, config.oneshot_mode, user_prompt);
        } else {
            // Defensive graceful degradation: default to fastest O(N) Sub-Polynomial architecture
            if (backend != "native-linear") {
                std::cout << "Warning: Unknown backend '" << backend << "'. Falling back to 'native-linear'." << std::endl;
            }
            NativeLinearLLM model(d_model, vocab_size, model_path);
            run_mode(model, chat_mode, config.oneshot_mode, user_prompt);
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
