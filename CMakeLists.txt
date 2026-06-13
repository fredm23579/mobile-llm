cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(MobileLLM_LlamaCpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(LLAMA_CPP_DIR "../llama.cpp" CACHE STRING "Path to llama.cpp source directory")
if(DEFINED ENV{LLAMA_CPP_DIR})
    set(LLAMA_CPP_DIR $ENV{LLAMA_CPP_DIR})
endif()

# Fallback for the current host machine environment
if(NOT EXISTS "${LLAMA_CPP_DIR}/include" AND EXISTS "/root/llama.cpp/include")
    set(LLAMA_CPP_DIR "/root/llama.cpp")
endif()

# Llama.cpp includes and libraries
include_directories(${LLAMA_CPP_DIR}/include ${LLAMA_CPP_DIR}/ggml/include ${LLAMA_CPP_DIR}/common)
link_directories(${LLAMA_CPP_DIR}/build/bin)

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
