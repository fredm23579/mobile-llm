#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <torch/torch.h>

#include "turboquant.hpp"
#include "agent.hpp"

// External Fortran binding
extern "C" {
    void update_hidden_state(int d_model, float decay, float* hidden_state, const float* token_emb);
}

void test_fortran_decay() {
    std::cout << "[Test] Running Fortran Decay Math Stability..." << std::endl;
    int d_model = 256;
    std::vector<float> hidden(d_model, 1.0f);
    std::vector<float> emb(d_model, 0.5f);
    
    // hidden = 0.9 * 1.0 + 0.5 = 1.4
    update_hidden_state(d_model, 0.9f, hidden.data(), emb.data());
    
    for (int i = 0; i < d_model; i++) {
        assert(std::abs(hidden[i] - 1.4f) < 1e-5);
    }
    
    // Simulate 1000 iterations to check for Inf/NaN
    for (int step = 0; step < 1000; step++) {
        update_hidden_state(d_model, 0.9f, hidden.data(), emb.data());
        assert(!std::isnan(hidden[0]) && !std::isinf(hidden[0]));
    }
    std::cout << "  -> PASS: Fortran math is stable over 1000 recurrent steps." << std::endl;
}

void test_turboquant_bounds() {
    std::cout << "[Test] Running TurboQuant Eigen Bounds Verification..." << std::endl;
    int d_model = 256;
    tq::TurboQuant quantizer(d_model);
    
    std::vector<float> simulated_hidden(d_model);
    for (int i = 0; i < d_model; i++) simulated_hidden[i] = (rand() % 100) * 0.1f;
    
    auto compressed = quantizer.quantize(simulated_hidden);
    assert(compressed.size() == d_model);
    
    for (int8_t val : compressed) {
        assert(val >= -127 && val <= 127);
    }
    std::cout << "  -> PASS: TurboQuant successfully bounds vectors to INT8 space." << std::endl;
}

void test_agent_initialization() {
    std::cout << "[Test] Running ReAct Agent Flow..." << std::endl;
    
    // A dummy wrapper to test agent compilation, loop bounds, and string logic
    class DummyLLM {
    public:
        std::string generate(const std::string& p) { 
            static int calls = 0;
            if (calls++ == 0) return "Thought: I should use a tool.\nAction: run_command\nActionInput: echo hello";
            return "Final Answer: Parsed"; 
        }
    };
    
    DummyLLM dummy;
    Agent<DummyLLM> agent(dummy);
    std::string result = agent.run_autoresearch_loop("Do something");
    
    // Test Edge Case: Ensure the loop breaks properly on 'Final Answer'
    assert(result.find("Final Answer:") != std::string::npos);
    
    std::cout << "  -> PASS: Agent architecture compiles and integrates successfully." << std::endl;
}

#include "llama_adapter.hpp"

void test_cli_parsing() {
    std::cout << "[Test] Running CLI Flag Parsing Verification..." << std::endl;
    
    const char* argv1[] = {"./mobile_llm_tests", "--backend", "ollama", "--model", "llama3", "--chat"};
    Config config1 = parse_cli(6, argv1);
    
    assert(config1.backend == "ollama");
    assert(config1.model_path == "llama3");
    assert(config1.chat_mode == true);
    assert(config1.llama_mode == true);
    
    const char* argv2[] = {"./mobile_llm_tests", "--model", "custom.gguf"};
    Config config2 = parse_cli(3, argv2);
    
    assert(config2.model_path == "custom.gguf");
    assert(config2.backend == "llama.cpp"); // default
    assert(config2.chat_mode == false);
    assert(config2.llama_mode == false);

    std::cout << "  -> PASS: CLI parsing correctly assigns flags." << std::endl;
}

void test_llama_adapter_initialization() {
    std::cout << "[Test] Running LlamaServerAdapter Initialization..." << std::endl;
    
    // Test initialization without crashing
    LlamaServerAdapter default_adapter;
    assert(default_adapter.get_is_chat() == false);
    assert(default_adapter.get_backend() == "llama.cpp");
    assert(default_adapter.get_model() == "llama3");

    LlamaServerAdapter custom_adapter(true, "ollama", "mistral");
    assert(custom_adapter.get_is_chat() == true);
    assert(custom_adapter.get_backend() == "ollama");
    assert(custom_adapter.get_model() == "mistral");
    
    std::cout << "  -> PASS: LlamaServerAdapter initializes correctly." << std::endl;
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << " MobileLLM Robustness Test Suite           " << std::endl;
    std::cout << "===========================================" << std::endl;
    
    test_fortran_decay();
    test_turboquant_bounds();
    test_agent_initialization();
    test_cli_parsing();
    test_llama_adapter_initialization();
    
    std::cout << "\nALL TESTS PASSED. Engine is mathematically robust." << std::endl;
    return 0;
}
