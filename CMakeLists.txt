cmake_minimum_required(VERSION 3.18)

# Enable both C++ and CUDA languages
project(cuGrayScott LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

# Force CMake to put -I include paths directly in compile_commands.json
# instead of hiding them inside .rsp response files
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# Generate compile_commands.json for Clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enforce C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Target Ampere architecture (Compute Capability 8.6 for an RTX 3080)
set(CMAKE_CUDA_ARCHITECTURES 86)

# TLook for shared header files
include_directories(src/common)

# Define Phase 1 executable
add_executable(cuGrayScott_phase1 
    src/phase1_naive/main.cpp 
    src/phase1_naive/solver.cu 
)
target_link_libraries(cuGrayScott_phase1 PRIVATE CUDA::cudart)