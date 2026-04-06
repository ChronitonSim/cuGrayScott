#pragma once

#include <string>
#include <format>
#include <source_location>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cmath>

// Inline allows us to define this function in a header without violating the ODR.
inline void cudaCheck(cudaError_t err,
                      std::source_location location = std::source_location::current()) {
    
    if (err != cudaSuccess) {
        std::string msg = std::format("CUDA error at {}:{} code={} \"{}\"\n",
                                      location.file_name(),
                                      location.line(),
                                      static_cast<int>(err),
                                      cudaGetErrorString(err));
        throw std::runtime_error(msg);
    }
}

inline dim3 computeHardwareGridDimensions(
    int sqBlockSize,
    int numSMs,
    int width,
    int height
) {
    // Cap the size of the grid to hardware bounds.
    // A standard heuristic is launching 32 blocks per SM to hide latency.
    // We take the square root of this maximum number of blocks to find the
    // maximum size of the 2D grid of blocks.
    int sqMaxNumBlocks { static_cast<int>(std::sqrt(numSMs * 32)) };
    
    // Cap hardware grid size to simulation grid size so we don't launch
    // more threads than simulation nodes (in case the hardware can spwan
    // enough blocks to cover the entire simulation grid).
    int gridDimX {std::min(
        (width + sqBlockSize - 1) / sqBlockSize,
        sqMaxNumBlocks
    )};
    int gridDimY {std::min(
        (height + sqBlockSize - 1) / sqBlockSize,
        sqMaxNumBlocks
    )};

    dim3 blocksPerGrid(gridDimX, gridDimY);
    return blocksPerGrid;
}