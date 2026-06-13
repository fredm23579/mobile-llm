#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <regex>

int main() {
    std::unordered_map<std::string, int> vocab;
    std::string json = "{\"hello\": 1, \" world\": 2, \"\\\"\": 3}";
    
    // Simple json parser for string to int
    size_t pos = 0;
    while (pos < json.size()) {
        size_t start_quote = json.find('"', pos);
        if (start_quote == std::string::npos) break;
        
        size_t end_quote = start_quote + 1;
        while (end_quote < json.size() && (json[end_quote] != '"' || json[end_quote-1] == '\\')) {
            end_quote++;
        }
        
        std::string key = json.substr(start_quote + 1, end_quote - start_quote - 1);
        
        size_t colon = json.find(':', end_quote);
        if (colon == std::string::npos) break;
        
        size_t comma = json.find(',', colon);
        size_t end_brace = json.find('}', colon);
        size_t end_val = (comma == std::string::npos) ? end_brace : std::min(comma, end_brace);
        
        if (end_val == std::string::npos) break;
        
        std::string val_str = json.substr(colon + 1, end_val - colon - 1);
        try {
            vocab[key] = std::stoi(val_str);
        } catch (...) {}
        
        pos = end_val + 1;
    }
    
    for (auto& p : vocab) std::cout << p.first << ":" << p.second << "\n";
    return 0;
}
