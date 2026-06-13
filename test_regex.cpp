#include <iostream>
#include <string>
#include <regex>
#include <vector>

int main() {
    std::string text = "Hello, world! 123";
    std::regex word_regex(R"( ?[a-zA-Z]+| ?[0-9]+| ?[^ \t\n\ra-zA-Z0-9]+|[ \t\n\r]+)");
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), word_regex);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::cout << "[" << i->str() << "]\n";
    }
    return 0;
}
