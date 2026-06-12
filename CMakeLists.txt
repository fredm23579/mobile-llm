cmake_minimum_required(VERSION 3.10)
project(MobileLLM CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Boost
find_package(Boost REQUIRED COMPONENTS timer system)

# Find BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Include ITensor
# Note: ITensor provides an options.mk but we will map it to CMake for modern builds
# We assume ITensor is installed globally or we will link it directly
include_directories(/root/mobile-llm/ITensor)
link_directories(/root/mobile-llm/ITensor/lib)

add_executable(mobile_llm main.cpp)

# Link everything together
target_link_libraries(mobile_llm 
    Boost::timer 
    Boost::system
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    # -litensor # We will compile ITensor separately and link it
)

# Termux optimization flags: -O3, NEON SIMD for ARM64
target_compile_options(mobile_llm PRIVATE -O3 -march=native -ffast-math)
