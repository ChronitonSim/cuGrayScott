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

# Phase 1 naive kernel executable
add_executable(cuGrayScott_phase1 
    src/phase1_naive/main.cpp 
    src/phase1_naive/solver.cu 
)
target_include_directories(cuGrayScott_phase1 PRIVATE src/common)
target_link_libraries(cuGrayScott_phase1 PRIVATE CUDA::cudart)

# Phase 2 shared memory executable
add_executable(cuGrayScott_phase2 
    src/phase2_shared_mem/main.cpp 
    src/phase2_shared_mem/solver.cu 
)
target_include_directories(cuGrayScott_phase2 PRIVATE src/common)
target_link_libraries(cuGrayScott_phase2 PRIVATE CUDA::cudart)

# Phase 3 asynchronous pipeline executable
add_executable(cuGrayScott_phase3 
    src/phase3_async/main.cpp 
    src/phase3_async/solver.cu 
)
target_include_directories(cuGrayScott_phase3 PRIVATE src/common)
target_link_libraries(cuGrayScott_phase3 PRIVATE CUDA::cudart)