#pragma once
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

