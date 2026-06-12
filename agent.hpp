#ifndef AGENT_HPP
#define AGENT_HPP

#include <string>
#include <iostream>
#include <vector>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <fstream>

// Forward declaration of our LLM engine
class LinearLLM;

class Agent {
public:
    Agent(LinearLLM& llm) : llm_(llm) {}

    std::string run_autoresearch_loop(const std::string& user_prompt) {
        std::cout << "[AutoResearch] Initializing Deep Research Protocol..." << std::endl;
        std::cout << "[AutoResearch] Drafting execution specifications..." << std::endl;
        
        // Phase 1: AutoResearch Specification
        std::string spec_context = "System: Create a detailed, multi-step execution specification. Output 'Specification: [steps]'.\nUser: " + user_prompt;
        std::string specification = "Specification:\n1. Analyze requirements\n2. Execute tool sequences\n3. Validate observations\n4. Terminate with Final Answer"; // Mock LLM gen
        std::cout << "[AutoResearch] " << specification << std::endl;
        
        std::string context = "System: You are an AutoResearch agent. Follow this specification strictly:\n" + specification + "\nUser: " + user_prompt + "\n";
        
        int max_iterations = 50; // Deep continuous looping
        for (int i = 0; i < max_iterations; ++i) {
            std::cout << "\n[AutoResearch] Iteration " << (i+1) << "/" << max_iterations << " | Generating thought/action..." << std::endl;
            
            // In reality, llm_.generate(context) would run inference
            std::string response;
            if (i == 0) response = "Thought: I need to use a tool.\nAction: run_command\nActionInput: echo \"hello from termux\"\n";
            else response = "Thought: Done.\nFinal Answer: Task completed.";
            
            context += response;
            
            // Phase 2: Termination Check
            size_t final_pos = response.find("Final Answer: ");
            if (final_pos != std::string::npos) {
                std::cout << "[AutoResearch] Protocol Complete." << std::endl;
                return response.substr(final_pos);
            }
            
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
            } else if (action == "read_file") {
                std::ifstream file(action_input);
                if (file.is_open()) {
                    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                    observation += content.empty() ? "[Empty File]\n" : content;
                    if (observation.back() != '\n') observation += "\n";
                } else {
                    observation += "Error: Could not open file " + action_input + "\n";
                }
            } else if (action == "search_web") {
                std::array<char, 128> buffer;
                std::string result;
                // Basic curl wrapper for web search (requires curl and grep on host system)
                std::string cmd = "curl -s \"https://html.duckduckgo.com/html/?q=" + action_input + "\" | grep -oP '(?<=class=\"result__snippet\">).*?(?=</a>)' | head -n 3";
                FILE* pipe = popen(cmd.c_str(), "r");
                if (!pipe) {
                    observation += "Error executing web search.\n";
                } else {
                    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                        result += buffer.data();
                    }
                    pclose(pipe);
                    observation += result.empty() ? "No results found.\n" : result;
                    if (observation.back() != '\n') observation += "\n";
                }
            } else if (action == "write_file") {
                size_t space_pos = action_input.find(' ');
                if (space_pos != std::string::npos) {
                    std::string filename = action_input.substr(0, space_pos);
                    std::string content = action_input.substr(space_pos + 1);
                    std::ofstream file(filename);
                    if (file.is_open()) {
                        file << content;
                        observation += "File written successfully.\n";
                    } else {
                        observation += "Error: Could not open file for writing.\n";
                    }
                } else {
                    observation += "Error: write_file requires 'filename content' format.\n";
                }
            } else if (action == "move_file") {
                size_t space_pos = action_input.find(' ');
                if (space_pos != std::string::npos) {
                    std::string src = action_input.substr(0, space_pos);
                    std::string dest = action_input.substr(space_pos + 1);
                    if (std::rename(src.c_str(), dest.c_str()) == 0) {
                        observation += "File moved successfully.\n";
                    } else {
                        observation += "Error: Could not move file.\n";
                    }
                } else {
                    observation += "Error: move_file requires 'src dest' format.\n";
                }
            } else if (action == "copy_file") {
                size_t space_pos = action_input.find(' ');
                if (space_pos != std::string::npos) {
                    std::string src = action_input.substr(0, space_pos);
                    std::string dest = action_input.substr(space_pos + 1);
                    std::ifstream src_file(src, std::ios::binary);
                    std::ofstream dest_file(dest, std::ios::binary);
                    if (src_file && dest_file) {
                        dest_file << src_file.rdbuf();
                        observation += "File copied successfully.\n";
                    } else {
                        observation += "Error: Could not copy file.\n";
                    }
                } else {
                    observation += "Error: copy_file requires 'src dest' format.\n";
                }
            } else if (action == "pattern_match") {
                size_t space_pos = action_input.find(' ');
                if (space_pos != std::string::npos) {
                    std::string pattern = action_input.substr(0, space_pos);
                    std::string filename = action_input.substr(space_pos + 1);
                    std::string cmd = "grep -n \"" + pattern + "\" \"" + filename + "\" | head -n 50";
                    FILE* pipe = popen(cmd.c_str(), "r");
                    if (!pipe) {
                        observation += "Error executing pattern match.\n";
                    } else {
                        std::array<char, 128> buffer;
                        std::string result;
                        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                            result += buffer.data();
                        }
                        pclose(pipe);
                        observation += result.empty() ? "No matches found.\n" : result;
                        if (observation.back() != '\n') observation += "\n";
                    }
                } else {
                    observation += "Error: pattern_match requires 'pattern file' format.\n";
                }
            } else if (action == "mathematics") {
                std::string cmd = "echo \"" + action_input + "\" | bc -l";
                FILE* pipe = popen(cmd.c_str(), "r");
                if (!pipe) {
                    observation += "Error executing mathematics.\n";
                } else {
                    std::array<char, 128> buffer;
                    std::string result;
                    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                        result += buffer.data();
                    }
                    pclose(pipe);
                    observation += result.empty() ? "Calculation failed.\n" : result;
                    if (observation.back() != '\n') observation += "\n";
                }
            } else if (action == "text_parsing") {
                size_t space_pos = action_input.find(' ');
                if (space_pos != std::string::npos) {
                    std::string awk_cmd = action_input.substr(0, space_pos);
                    std::string filename = action_input.substr(space_pos + 1);
                    std::string cmd = "awk '" + awk_cmd + "' \"" + filename + "\" | head -n 50";
                    FILE* pipe = popen(cmd.c_str(), "r");
                    if (!pipe) {
                        observation += "Error executing text parsing.\n";
                    } else {
                        std::array<char, 128> buffer;
                        std::string result;
                        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                            result += buffer.data();
                        }
                        pclose(pipe);
                        observation += result.empty() ? "No parsed text found.\n" : result;
                        if (observation.back() != '\n') observation += "\n";
                    }
                } else {
                    observation += "Error: text_parsing requires 'awk_cmd filename' format.\n";
                }
            } else if (action == "universal_parse") {
                std::string filename = action_input;
                std::string cmd;
                if (filename.find(".json") != std::string::npos) {
                    cmd = "python3 -c \"import json, sys; print(json.dumps(json.load(open(sys.argv[1])), indent=2)[:2000])\" \"" + filename + "\"";
                } else if (filename.find(".csv") != std::string::npos) {
                    cmd = "python3 -c \"import csv, sys; r=csv.reader(open(sys.argv[1])); [print(row) for _,row in zip(range(20), r)]\" \"" + filename + "\"";
                } else if (filename.find(".xml") != std::string::npos) {
                    cmd = "python3 -c \"import xml.etree.ElementTree as ET, sys; tree=ET.parse(sys.argv[1]); print(ET.tostring(tree.getroot(), encoding='unicode')[:2000])\" \"" + filename + "\"";
                } else if (filename.find(".bin") != std::string::npos || filename.find(".gguf") != std::string::npos) {
                    cmd = "xxd -l 256 \"" + filename + "\"";
                } else {
                    cmd = "file \"" + filename + "\" && head -n 20 \"" + filename + "\"";
                }
                
                FILE* pipe = popen(cmd.c_str(), "r");
                if (!pipe) {
                    observation += "Error executing universal parser.\n";
                } else {
                    std::array<char, 128> buffer;
                    std::string result;
                    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                        result += buffer.data();
                    }
                    pclose(pipe);
                    observation += result.empty() ? "Unparseable or empty file.\n" : result;
                    if (observation.back() != '\n') observation += "\n";
                }
            } else if (action == "delete_file") {
                if (std::remove(action_input.c_str()) == 0) {
                    observation += "File deleted successfully.\n";
                } else {
                    observation += "Error: Could not delete file.\n";
                }
            } else if (action == "list_dir") {
                std::string cmd = "ls -la \"" + action_input + "\" | head -n 50";
                FILE* pipe = popen(cmd.c_str(), "r");
                if (!pipe) {
                    observation += "Error executing list_dir.\n";
                } else {
                    std::array<char, 128> buffer;
                    std::string result;
                    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                        result += buffer.data();
                    }
                    pclose(pipe);
                    observation += result.empty() ? "Empty directory.\n" : result;
                    if (observation.back() != '\n') observation += "\n";
                }
            } else if (action == "fetch_url") {
                std::string cmd = "curl -sL -A \"Mozilla/5.0\" \"" + action_input + "\" | head -n 100";
                FILE* pipe = popen(cmd.c_str(), "r");
                if (!pipe) {
                    observation += "Error executing fetch_url.\n";
                } else {
                    std::array<char, 128> buffer;
                    std::string result;
                    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                        result += buffer.data();
                    }
                    pclose(pipe);
                    observation += result.empty() ? "No content returned.\n" : result;
                    if (observation.back() != '\n') observation += "\n";
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
