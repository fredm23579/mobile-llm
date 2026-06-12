cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(MobileLLM_LibTorch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable Fortran compilation alongside C++
enable_language(Fortran)

# --- Dependency Equivalents ---
# PyTorch C++ Equivalent (LibTorch)
find_package(Torch REQUIRED)

# NumPy C++ Equivalent (Eigen3)
find_package(Eigen3 REQUIRED)

# Scikit-Learn C++ Equivalent (mlpack)
# find_package(mlpack REQUIRED)

# Theano/PyTensor C++ Equivalent (Xtensor or direct LibTorch autograd)
# find_package(xtensor REQUIRED)

add_executable(mobile_llm main.cpp fast_math.f90)
add_executable(mobile_llm_tests tests.cpp fast_math.f90)

# Link LibTorch and Eigen
target_link_libraries(mobile_llm 
    ${TORCH_LIBRARIES} 
    Eigen3::Eigen
)

target_link_libraries(mobile_llm_tests 
    ${TORCH_LIBRARIES} 
    Eigen3::Eigen
)

# Termux / Mobile Architectures Optimization
target_compile_options(mobile_llm PRIVATE -O3 -march=native -ffast-math)
set_property(TARGET mobile_llm PROPERTY CXX_STANDARD 17)

target_compile_options(mobile_llm_tests PRIVATE -O3 -march=native -ffast-math)
set_property(TARGET mobile_llm_tests PROPERTY CXX_STANDARD 17)
