cmake_minimum_required(VERSION 3.18)

# Enable both C++ and CUDA languages
project(cuGrayScott LANGUAGES CXX CUDA)

# Enforce C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Target Ampere architecture (Compute Capability 8.6 for an RTX 3080)
# This allows the compiler to optimize specifically for this hardware
set(CMAKE_CUDA_ARCHITECTURES 86)

# Define our executable
add_executable(cuGrayScott 
    src/main.cpp 
    # src/solver.cu  <-- uncomment in Phase 2
)