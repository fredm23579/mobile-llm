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
        std::cout << "[Mythos Protocol] Initializing Deep Agentic Reasoning..." << std::endl;
        
        std::string tools_desc = 
            "<available_tools>\n"
            "  <tool>\n"
            "    <name>run_command</name>\n"
            "    <description>Executes a bash command and returns stdout/stderr.</description>\n"
            "  </tool>\n"
            "  <tool>\n"
            "    <name>python_interpreter</name>\n"
            "    <description>Executes a Python script in a sandboxed environment. Provide raw python code as input.</description>\n"
            "  </tool>\n"
            "  <tool>\n"
            "    <name>read_file</name>\n"
            "    <description>Reads a file's contents into context.</description>\n"
            "  </tool>\n"
            "  <tool>\n"
            "    <name>write_file</name>\n"
            "    <description>Writes content to a file. Input format MUST be: filename|content</description>\n"
            "  </tool>\n"
            "  <tool>\n"
            "    <name>search_web</name>\n"
            "    <description>Searches the web via DuckDuckGo/Wikipedia API.</description>\n"
            "  </tool>\n"
            "</available_tools>\n"
            "\n"
            "RULES:\n"
            "1. You MUST enclose your internal reasoning inside <thought>...</thought> tags before taking any action.\n"
            "2. To use a tool, you MUST use the following XML structure:\n"
            "<tool_call>\n"
            "  <name>tool_name</name>\n"
            "  <input>tool_arguments</input>\n"
            "</tool_call>\n"
            "3. You can execute multiple tools in a single turn by emitting multiple <tool_call> blocks.\n"
            "4. When the task is fully complete, emit <final_answer>your solution</final_answer>.\n";
            
        std::string context = "System: You are an advanced autonomous agent.\n" + tools_desc + "\nUser: " + user_prompt + "\n";
        
        int max_iterations = 50; 
        for (int i = 0; i < max_iterations; ++i) {
            std::cout << "\n[Mythos Protocol] Iteration " << (i+1) << " | Reasoning Phase..." << std::endl;

            std::string response = llm_.generate(context);
            context += response;
            
            size_t final_pos = response.find("<final_answer>");
            if (final_pos != std::string::npos) {
                size_t end_final = response.find("</final_answer>", final_pos);
                std::cout << "[Mythos Protocol] Task Accomplished." << std::endl;
                if (end_final != std::string::npos) {
                    return response.substr(final_pos + 14, end_final - (final_pos + 14));
                }
                return response.substr(final_pos + 14);
            }
            
            // Extract all XML tool calls natively
            std::vector<std::pair<std::string, std::string>> tool_calls;
            size_t search_offset = 0;
            while (true) {
                size_t call_start = response.find("<tool_call>", search_offset);
                if (call_start == std::string::npos) break;
                
                size_t name_start = response.find("<name>", call_start);
                size_t name_end = response.find("</name>", name_start);
                size_t input_start = response.find("<input>", call_start);
                size_t input_end = response.find("</input>", input_start);
                size_t call_end = response.find("</tool_call>", call_start);
                
                if (name_start != std::string::npos && name_end != std::string::npos &&
                    input_start != std::string::npos && input_end != std::string::npos) {
                    
                    std::string name = response.substr(name_start + 6, name_end - (name_start + 6));
                    std::string input = response.substr(input_start + 7, input_end - (input_start + 7));
                    tool_calls.push_back({name, input});
                }
                search_offset = call_end != std::string::npos ? call_end : call_start + 11;
            }
            
            if (tool_calls.empty()) {
                std::cout << "[Agent Warning] No valid <tool_call> or <final_answer> emitted." << std::endl;
                context += "\nSystem: You did not format a tool call using XML or provide a final answer. Correct your format.";
                continue;
            }
            
            std::string observations = "";
            for (const auto& call : tool_calls) {
                std::string action = call.first;
                std::string action_input = call.second;
                
                std::cout << "[Agent] Executing Tool: " << action << std::endl;
                std::string observation = "<observation tool=\"" + action + "\">\n";

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
                    std::string py_script = "import sys, json, urllib.request, urllib.parse\\n"
                                            "try:\\n"
                                            "  q=sys.argv[1]\\n"
                                            "  req=urllib.request.Request(f'https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(q)}&limit=3&format=json', headers={'User-Agent':'Mozilla/5.0'})\\n"
                                            "  res=json.loads(urllib.request.urlopen(req).read())\\n"
                                            "  if len(res) > 2 and res[2]:\\n"
                                            "    print('\\n'.join([x for x in res[2] if x]))\\n"
                                            "  else:\\n"
                                            "    print('No results found.')\\n"
                                            "except Exception as e:\\n"
                                            "  print('Error:', e)";
                    std::string cmd = "python3 -c \"" + py_script + "\" " + escape_shell_arg(action_input) + " 2>&1";
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
                } else if (action == "python_interpreter") {
                    std::ofstream py_file("agent_sandbox.py");
                    if (py_file.is_open()) {
                        py_file << action_input;
                        py_file.close();
                        std::string cmd = "python3 agent_sandbox.py 2>&1";
                        FILE* pipe = popen(cmd.c_str(), "r");
                        if (!pipe) {
                            observation += "Error executing sandboxed Python code.\n";
                        } else {
                            std::array<char, 128> buffer;
                            std::string result;
                            while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                                result += buffer.data();
                            }
                            pclose(pipe);
                            observation += result.empty() ? "[Code executed successfully with no stdout]\n" : result;
                            if (observation.back() != '\n') observation += "\n";
                        }
                    } else {
                        observation += "Error: Could not allocate python sandbox.\n";
                    }
                } else if (action == "write_file") {
                    size_t pipe_pos = action_input.find('|');
                    if (pipe_pos != std::string::npos) {
                        std::string filename = action_input.substr(0, pipe_pos);
                        std::string write_content = action_input.substr(pipe_pos + 1);
                        std::ofstream file(filename);
                        if (file.is_open()) {
                            file << write_content;
                            observation += "File written successfully.\n";
                        } else {
                            observation += "Error: Could not open file for writing.\n";
                        }
                    } else {
                        observation += "Error: write_file requires 'filename|content' format.\n";
                    }
                } else {
                    observation += "Unknown tool action.\n";
                }
            } catch (const std::exception& e) {
                observation += std::string("C++ Exception caught: ") + e.what() + "\n";
            } catch (...) {
                observation += "Unknown C++ Exception caught.\n";
            }
            
            observation += "</observation>\n";
            observations += observation;
            } // end multi-tool loop
            
            context += "\n" + observations;
        }
        
        return "Task terminated or completed max iterations.";
    }

private:
    LLM& llm_;
};

#endif // AGENT_HPP
