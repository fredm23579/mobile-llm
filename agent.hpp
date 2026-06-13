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

template <typename LLM>
class Agent {
public:
    Agent(LLM& llm) : llm_(llm) {}


    std::string escape_shell_arg(const std::string& arg) {
        std::string escaped = "'";
        for (char c : arg) {
            if (c == '\'') {
                escaped += "'\\''";
            } else {
                escaped += c;
            }
        }
        escaped += "'";
        return escaped;
    }

    std::string sanitize_python_eval(const std::string& input) {
        std::string result;
        for (char c : input) {
            if (c == '\'' || c == '"' || c == '\\') {
                result += '\\';
            }
            result += c;
        }
        return result;
    }

    std::string run_autoresearch_loop(const std::string& user_prompt) {
        std::cout << "[AutoResearch] Initializing Deep Research Protocol..." << std::endl;
        std::cout << "[AutoResearch] Drafting execution specifications..." << std::endl;
        
        std::string tools_desc = 
            "AVAILABLE TOOLS:\n"
            "- run_command: Executes a raw bash command (e.g. ActionInput: echo hello)\n"
            "- read_file: Reads file contents (e.g. ActionInput: /path/to/file)\n"
            "- write_file: Writes file (e.g. ActionInput: filename content)\n"
            "- move_file: Moves file (e.g. ActionInput: src dest)\n"
            "- copy_file: Copies file (e.g. ActionInput: src dest)\n"
            "- delete_file: Deletes file (e.g. ActionInput: filename)\n"
            "- list_dir: Lists files in a directory (e.g. ActionInput: /path/to/dir)\n"
            "- mathematics: Evaluates a Python math expression. Use Python syntax like ** for exponents (e.g. ActionInput: ((1+math.sqrt(5))**10 - (1-math.sqrt(5))**10) / (2**10 * math.sqrt(5)))\n"
            "- search_web: DuckDuckGo search (e.g. ActionInput: search term)\n"
            "- fetch_url: cURL a URL (e.g. ActionInput: https://example.com)\n"
            "- pattern_match: grep file (e.g. ActionInput: pattern file)\n"
            "- text_parsing: awk file (e.g. ActionInput: '{print $1}' file)\n"
            "- universal_parse: parses json/csv/xml/bin (e.g. ActionInput: file.json)\n\n"
            "RULES:\n"
            "1. You MUST only generate ONE Action per turn. DO NOT generate Final Answer until you see an Observation.\n"
            "2. Wait for the C++ engine to return an Observation before generating your next Thought.\n";
            
        // Phase 1: AutoResearch Specification
        std::string spec_context = "System: Create a detailed execution specification. Output 'Specification: [steps]'.\n" + tools_desc + "User: " + user_prompt;
        
        std::string specification = llm_.generate(spec_context);
        std::cout << "[AutoResearch] " << specification << std::endl;
        
        std::string context = "System: You are an AutoResearch agent. Follow this specification:\n" + specification + "\n" + tools_desc + "\nUser: " + user_prompt + "\n";
        
        int max_iterations = 50; // Deep continuous looping
        bool has_observation = false;
        for (int i = 0; i < max_iterations; ++i) {
            std::cout << "\n[AutoResearch] Iteration " << (i+1) << "/" << max_iterations << " | Generating thought/action..." << std::endl;

            // Run raw $O(N)$ linear-time inference through the LibTorch/Fortran core
            std::string response = llm_.generate(context);

            if (response.find("Action:") == std::string::npos && response.find("Final Answer:") == std::string::npos) {
                if (has_observation && !response.empty()) {
                    // Model answered directly after seeing an Observation — treat as Final Answer.
                    response = "Final Answer: " + response;
                } else {
                    std::cout << "[Agent Warning] LLM generated invalid ReAct formatting (expected Action or Final Answer)." << std::endl;
                    response = "Final Answer: Error - Invalid syntax from LLM generator.";
                }
            }
            
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

            try {
                if (action == "run_command") {
                    std::array<char, 128> buffer;
                    std::string result;
                    std::string cmd = action_input + " 2>&1";
                    FILE* pipe = popen(cmd.c_str(), "r");
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
                    std::string query = "https://html.duckduckgo.com/html/?q=" + action_input;
                    std::string cmd = "curl -s " + escape_shell_arg(query) + " | grep -oP '(?<=class=\"result__snippet\">).*?(?=</a>)' | head -n 3 2>&1";
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
                        std::string write_content = action_input.substr(space_pos + 1);
                        std::ofstream file(filename);
                        if (file.is_open()) {
                            file << write_content;
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
                        std::string cmd = "grep -n " + escape_shell_arg(pattern) + " " + escape_shell_arg(filename) + " | head -n 50 2>&1";
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
                    std::string py_cmd = "import math; print(eval('''" + sanitize_python_eval(action_input) + "'''))";
                    std::string cmd = "python3 -c " + escape_shell_arg(py_cmd) + " 2>&1";
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
                        std::string cmd = "awk " + escape_shell_arg(awk_cmd) + " " + escape_shell_arg(filename) + " | head -n 50 2>&1";
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
                        cmd = "python3 -c 'import json, sys; print(json.dumps(json.load(open(sys.argv[1])), indent=2)[:2000])' " + escape_shell_arg(filename) + " 2>&1";
                    } else if (filename.find(".csv") != std::string::npos) {
                        cmd = "python3 -c 'import csv, sys; r=csv.reader(open(sys.argv[1])); [print(row) for _,row in zip(range(20), r)]' " + escape_shell_arg(filename) + " 2>&1";
                    } else if (filename.find(".xml") != std::string::npos) {
                        cmd = "python3 -c \"import xml.etree.ElementTree as ET, sys; tree=ET.parse(sys.argv[1]); print(ET.tostring(tree.getroot(), encoding='unicode')[:2000])\" " + escape_shell_arg(filename) + " 2>&1";
                    } else if (filename.find(".bin") != std::string::npos || filename.find(".gguf") != std::string::npos) {
                        cmd = "xxd -l 256 " + escape_shell_arg(filename) + " 2>&1";
                    } else {
                        cmd = "file " + escape_shell_arg(filename) + " && head -n 20 " + escape_shell_arg(filename) + " 2>&1";
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
                    std::string cmd = "ls -la " + escape_shell_arg(action_input) + " | head -n 50 2>&1";
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
                    std::string cmd = "curl -sL -A 'Mozilla/5.0' " + escape_shell_arg(action_input) + " | head -n 100 2>&1";
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
            } catch (const std::exception& e) {
                observation += std::string("C++ Exception caught: ") + e.what() + "\n";
            } catch (...) {
                observation += "Unknown C++ Exception caught.\n";
            }
            context += observation;
            has_observation = true;
            std::cout << "[Agent] " << observation;
        }
        
        return "Final Answer: The task is complete.";
    }

private:
    LLM& llm_;
};

#endif // AGENT_HPP
