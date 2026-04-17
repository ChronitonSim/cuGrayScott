#pragma once
#include <cuda_runtime.h>
#include <driver_types.h>

// Public API for the host to launch the Gray-Scott simulation step.
void runGrayScottStep(
    const float* d_U,
    const float* d_V,
    float* d_next_U,
    float* d_next_V,
    dim3 threadsPerBlock, 
    dim3 blocksPerGrid,
    cudaStream_t stream
);