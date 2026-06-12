#ifndef AGENT_HPP
#define AGENT_HPP

#include <string>
#include <iostream>
#include <vector>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>

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
            std::string response = "Thought: I need to use a tool.\nAction: run_command\nActionInput: echo \"hello from termux\"\n";
            context += response;
            
            // Parse response
            std::string thought, action, action_input;
            size_t thought_pos = response.find("Thought: ");
            size_t action_pos = response.find("Action: ");
            size_t input_pos = response.find("ActionInput: ");
            
            if (thought_pos != std::string::npos) {
                size_t end_pos = response.find('\n', thought_pos);
                thought = response.substr(thought_pos + 9, end_pos - (thought_pos + 9));
            }
            if (action_pos != std::string::npos) {
                size_t end_pos = response.find('\n', action_pos);
                action = response.substr(action_pos + 8, end_pos - (action_pos + 8));
            }
            if (input_pos != std::string::npos) {
                size_t end_pos = response.find('\n', input_pos);
                action_input = response.substr(input_pos + 13, end_pos - (input_pos + 13));
            }
            
            std::cout << "[Agent] Parsed Action: " << action << " | Input: " << action_input << std::endl;
            
            std::string observation = "Observation: ";
            if (action == "run_command") {
                std::array<char, 128> buffer;
                std::string result;
                FILE* pipe = popen(action_input.c_str(), "r");
                if (!pipe) {
                    observation += "Error executing command.\n";
                } else {
                    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                        result += buffer.data();
                    }
                    pclose(pipe);
                    observation += result;
                    if (result.empty() || result.back() != '\n') {
                        observation += "\n";
                    }
                }
            } else {
                observation += "Unknown action.\n";
            }
            context += observation;
            std::cout << "[Agent] " << observation;
        }
        
        return "Final Answer: The task is complete.";
    }

private:
    LinearLLM& llm_;
};

#endif // AGENT_HPP
