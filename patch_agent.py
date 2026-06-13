import re

with open('/root/mobile-llm/agent.hpp', 'r') as f:
    content = f.read()

# Add sanitization helpers
helpers = """
    std::string escape_shell_arg(const std::string& arg) {
        std::string escaped = "'";
        for (char c : arg) {
            if (c == '\\'') {
                escaped += "'\\\\''";
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
            if (c == '\\'' || c == '"' || c == '\\\\') {
                result += '\\\\';
            }
            result += c;
        }
        return result;
    }

    std::string run_autoresearch_loop(const std::string& user_prompt) {"""

content = content.replace("    std::string run_autoresearch_loop(const std::string& user_prompt) {", helpers)

# Wrap tool execution in try/catch and apply sanitization
tool_block_start = '            std::string observation = "Observation: ";'
tool_block_end = '            context += observation;\n            std::cout << "[Agent] " << observation;'

# Extract the tool block
start_idx = content.find(tool_block_start) + len(tool_block_start)
end_idx = content.find(tool_block_end)

original_tool_block = content[start_idx:end_idx]

# Modify the tool block:
new_tool_block = """
            try {
                if (action == "run_command") {
                    std::array<char, 128> buffer;
                    std::string result;
                    std::string cmd = action_input + " 2>&1";
                    FILE* pipe = popen(cmd.c_str(), "r");
                    if (!pipe) {
                        observation += "Error executing command.\\n";
                    } else {
                        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                            result += buffer.data();
                        }
                        pclose(pipe);
                        observation += result;
                        if (result.empty() || result.back() != '\\n') {
                            observation += "\\n";
                        }
                    }
                } else if (action == "read_file") {
                    std::ifstream file(action_input);
                    if (file.is_open()) {
                        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                        observation += content.empty() ? "[Empty File]\\n" : content;
                        if (observation.back() != '\\n') observation += "\\n";
                    } else {
                        observation += "Error: Could not open file " + action_input + "\\n";
                    }
                } else if (action == "search_web") {
                    std::array<char, 128> buffer;
                    std::string result;
                    std::string query = "https://html.duckduckgo.com/html/?q=" + action_input;
                    std::string cmd = "curl -s " + escape_shell_arg(query) + " | grep -oP '(?<=class=\"result__snippet\">).*?(?=</a>)' | head -n 3 2>&1";
                    FILE* pipe = popen(cmd.c_str(), "r");
                    if (!pipe) {
                        observation += "Error executing web search.\\n";
                    } else {
                        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                            result += buffer.data();
                        }
                        pclose(pipe);
                        observation += result.empty() ? "No results found.\\n" : result;
                        if (observation.back() != '\\n') observation += "\\n";
                    }
                } else if (action == "write_file") {
                    size_t space_pos = action_input.find(' ');
                    if (space_pos != std::string::npos) {
                        std::string filename = action_input.substr(0, space_pos);
                        std::string write_content = action_input.substr(space_pos + 1);
                        std::ofstream file(filename);
                        if (file.is_open()) {
                            file << write_content;
                            observation += "File written successfully.\\n";
                        } else {
                            observation += "Error: Could not open file for writing.\\n";
                        }
                    } else {
                        observation += "Error: write_file requires 'filename content' format.\\n";
                    }
                } else if (action == "move_file") {
                    size_t space_pos = action_input.find(' ');
                    if (space_pos != std::string::npos) {
                        std::string src = action_input.substr(0, space_pos);
                        std::string dest = action_input.substr(space_pos + 1);
                        if (std::rename(src.c_str(), dest.c_str()) == 0) {
                            observation += "File moved successfully.\\n";
                        } else {
                            observation += "Error: Could not move file.\\n";
                        }
                    } else {
                        observation += "Error: move_file requires 'src dest' format.\\n";
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
                            observation += "File copied successfully.\\n";
                        } else {
                            observation += "Error: Could not copy file.\\n";
                        }
                    } else {
                        observation += "Error: copy_file requires 'src dest' format.\\n";
                    }
                } else if (action == "pattern_match") {
                    size_t space_pos = action_input.find(' ');
                    if (space_pos != std::string::npos) {
                        std::string pattern = action_input.substr(0, space_pos);
                        std::string filename = action_input.substr(space_pos + 1);
                        std::string cmd = "grep -n " + escape_shell_arg(pattern) + " " + escape_shell_arg(filename) + " | head -n 50 2>&1";
                        FILE* pipe = popen(cmd.c_str(), "r");
                        if (!pipe) {
                            observation += "Error executing pattern match.\\n";
                        } else {
                            std::array<char, 128> buffer;
                            std::string result;
                            while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                                result += buffer.data();
                            }
                            pclose(pipe);
                            observation += result.empty() ? "No matches found.\\n" : result;
                            if (observation.back() != '\\n') observation += "\\n";
                        }
                    } else {
                        observation += "Error: pattern_match requires 'pattern file' format.\\n";
                    }
                } else if (action == "mathematics") {
                    std::string py_cmd = "import math; print(eval('''" + sanitize_python_eval(action_input) + "'''))";
                    std::string cmd = "python3 -c " + escape_shell_arg(py_cmd) + " 2>&1";
                    FILE* pipe = popen(cmd.c_str(), "r");
                    if (!pipe) {
                        observation += "Error executing mathematics.\\n";
                    } else {
                        std::array<char, 128> buffer;
                        std::string result;
                        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                            result += buffer.data();
                        }
                        pclose(pipe);
                        observation += result.empty() ? "Calculation failed.\\n" : result;
                        if (observation.back() != '\\n') observation += "\\n";
                    }
                } else if (action == "text_parsing") {
                    size_t space_pos = action_input.find(' ');
                    if (space_pos != std::string::npos) {
                        std::string awk_cmd = action_input.substr(0, space_pos);
                        std::string filename = action_input.substr(space_pos + 1);
                        std::string cmd = "awk " + escape_shell_arg(awk_cmd) + " " + escape_shell_arg(filename) + " | head -n 50 2>&1";
                        FILE* pipe = popen(cmd.c_str(), "r");
                        if (!pipe) {
                            observation += "Error executing text parsing.\\n";
                        } else {
                            std::array<char, 128> buffer;
                            std::string result;
                            while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                                result += buffer.data();
                            }
                            pclose(pipe);
                            observation += result.empty() ? "No parsed text found.\\n" : result;
                            if (observation.back() != '\\n') observation += "\\n";
                        }
                    } else {
                        observation += "Error: text_parsing requires 'awk_cmd filename' format.\\n";
                    }
                } else if (action == "universal_parse") {
                    std::string filename = action_input;
                    std::string cmd;
                    if (filename.find(".json") != std::string::npos) {
                        cmd = "python3 -c \"import json, sys; print(json.dumps(json.load(open(sys.argv[1])), indent=2)[:2000])\" " + escape_shell_arg(filename) + " 2>&1";
                    } else if (filename.find(".csv") != std::string::npos) {
                        cmd = "python3 -c \"import csv, sys; r=csv.reader(open(sys.argv[1])); [print(row) for _,row in zip(range(20), r)]\" " + escape_shell_arg(filename) + " 2>&1";
                    } else if (filename.find(".xml") != std::string::npos) {
                        cmd = "python3 -c \"import xml.etree.ElementTree as ET, sys; tree=ET.parse(sys.argv[1]); print(ET.tostring(tree.getroot(), encoding='unicode')[:2000])\" " + escape_shell_arg(filename) + " 2>&1";
                    } else if (filename.find(".bin") != std::string::npos || filename.find(".gguf") != std::string::npos) {
                        cmd = "xxd -l 256 " + escape_shell_arg(filename) + " 2>&1";
                    } else {
                        cmd = "file " + escape_shell_arg(filename) + " && head -n 20 " + escape_shell_arg(filename) + " 2>&1";
                    }
                    
                    FILE* pipe = popen(cmd.c_str(), "r");
                    if (!pipe) {
                        observation += "Error executing universal parser.\\n";
                    } else {
                        std::array<char, 128> buffer;
                        std::string result;
                        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                            result += buffer.data();
                        }
                        pclose(pipe);
                        observation += result.empty() ? "Unparseable or empty file.\\n" : result;
                        if (observation.back() != '\\n') observation += "\\n";
                    }
                } else if (action == "delete_file") {
                    if (std::remove(action_input.c_str()) == 0) {
                        observation += "File deleted successfully.\\n";
                    } else {
                        observation += "Error: Could not delete file.\\n";
                    }
                } else if (action == "list_dir") {
                    std::string cmd = "ls -la " + escape_shell_arg(action_input) + " | head -n 50 2>&1";
                    FILE* pipe = popen(cmd.c_str(), "r");
                    if (!pipe) {
                        observation += "Error executing list_dir.\\n";
                    } else {
                        std::array<char, 128> buffer;
                        std::string result;
                        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                            result += buffer.data();
                        }
                        pclose(pipe);
                        observation += result.empty() ? "Empty directory.\\n" : result;
                        if (observation.back() != '\\n') observation += "\\n";
                    }
                } else if (action == "fetch_url") {
                    std::string cmd = "curl -sL -A \"Mozilla/5.0\" " + escape_shell_arg(action_input) + " | head -n 100 2>&1";
                    FILE* pipe = popen(cmd.c_str(), "r");
                    if (!pipe) {
                        observation += "Error executing fetch_url.\\n";
                    } else {
                        std::array<char, 128> buffer;
                        std::string result;
                        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                            result += buffer.data();
                        }
                        pclose(pipe);
                        observation += result.empty() ? "No content returned.\\n" : result;
                        if (observation.back() != '\\n') observation += "\\n";
                    }
                } else {
                    observation += "Unknown action.\\n";
                }
            } catch (const std::exception& e) {
                observation += std::string("C++ Exception caught: ") + e.what() + "\\n";
            } catch (...) {
                observation += "Unknown C++ Exception caught.\\n";
            }
"""

content = content[:start_idx] + "\n" + new_tool_block + content[end_idx:]

with open('/root/mobile-llm/agent.hpp', 'w') as f:
    f.write(content)
