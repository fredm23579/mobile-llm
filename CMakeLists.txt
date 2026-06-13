cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(MobileLLM_LlamaCpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Llama.cpp includes and libraries
include_directories(/root/llama.cpp/include /root/llama.cpp/ggml/include /root/llama.cpp/common)
link_directories(/root/llama.cpp/build/bin)

add_executable(mobile_llm main.cpp)
add_executable(mobile_llm_tests tests.cpp)

# Link Llama.cpp libraries
target_link_libraries(mobile_llm 
    llama
    ggml
    llama-common
)

target_link_libraries(mobile_llm_tests 
    llama
    ggml
    llama-common
)

# Mobile Architectures Optimization & Strict Invariants
target_compile_options(mobile_llm PRIVATE -O3 -march=native -ffast-math -Wall -Wextra -Werror)
target_compile_options(mobile_llm_tests PRIVATE -O3 -march=native -ffast-math -Wall -Wextra -Werror)
