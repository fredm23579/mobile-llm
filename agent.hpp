#ifndef AGENT_HPP
#define AGENT_HPP

#include <string>
#include <iostream>
#include <vector>

// Forward declaration of our LLM engine
class LinearLLM;

class Agent {
public:
    Agent(LinearLLM& llm) : llm_(llm) {}

    std::string run_autonomous_loop(const std::string& user_prompt) {
        std::cout << "[Agent] Received task: " << user_prompt << std::endl;
        
        std::string context = "System: You are an autonomous agent. Thought/Action/Observation loop.\nUser: " + user_prompt + "\n";
        
        // Mock ReAct Loop
        for (int i = 0; i < 3; ++i) {
            std::cout << "[Agent] Generating thought/action..." << std::endl;
            // In reality, llm_.generate(context) would run inference
            std::string response = "Thought: I need to use a tool.\nAction: search_web\nActionInput: termux cpp\n";
            context += response;
            
            std::cout << "[Agent] Executing tool: search_web" << std::endl;
            std::string observation = "Observation: Termux is an Android terminal emulator.\n";
            context += observation;
        }
        
        return "Final Answer: The task is complete.";
    }

private:
    LinearLLM& llm_;
};

#endif // AGENT_HPP
