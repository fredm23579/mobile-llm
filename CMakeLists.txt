cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(MobileLLM_LibTorch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- Dependency Equivalents ---
# PyTorch C++ Equivalent (LibTorch)
find_package(Torch REQUIRED)

# NumPy C++ Equivalent (Eigen3)
find_package(Eigen3 REQUIRED)

# Scikit-Learn C++ Equivalent (mlpack)
# find_package(mlpack REQUIRED)

# Theano/PyTensor C++ Equivalent (Xtensor or direct LibTorch autograd)
# find_package(xtensor REQUIRED)

add_executable(mobile_llm main.cpp)

# Link LibTorch and Eigen
target_link_libraries(mobile_llm 
    ${TORCH_LIBRARIES} 
    Eigen3::Eigen
    # mlpack
)

# Termux / Mobile Architectures Optimization
# -O3 for max optimization, -ffast-math for floating point polynomial speedups
target_compile_options(mobile_llm PRIVATE -O3 -march=native -ffast-math)
set_property(TARGET mobile_llm PROPERTY CXX_STANDARD 17)
