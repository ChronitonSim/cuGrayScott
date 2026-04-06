#pragma once
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>


// Row-major linearization of grid nodes.
// The x dim is indexed by i, so i is the column index;
// The y dim is indexed by j, so j is the row index.
// Thus, a node of grid coordinates (j, i) has 
// linearized coordinate k = j * N_x + i
__host__ __device__
inline int getIndex(const int col, const int row, const int width){
    return row * width + col;
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

__device__
inline float computeLaplacian(
    const float center,
    const float south,
    const float north,
    const float east,
    const float west,
    const float grid_spacing
) {
    return (south + north + east + west - 4 * center) / (grid_spacing * grid_spacing);
}

__device__
inline float computeExplicitEulerStepU(
    const float centerU,
    const float centerV,
    const float laplacian,
    const float dt,
    const float D,
    const float F
) {
    return centerU + dt * (D * laplacian - centerU * centerV * centerV + F * (1 - centerU));
}

__device__
inline float computeExplicitEulerStepV(
    const float centerU,
    const float centerV,
    const float laplacian,
    const float dt,
    const float D,
    const float F,
    const float k
) {
    return centerV + dt * (D * laplacian + centerU * centerV * centerV - (F + k) * centerV);
}

