#include "solver.cuh"
#include "parameters.hpp"
#include "utils.hpp"
#include "io_utils.hpp"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <driver_types.h>

__global__
void grayScottKernelShared(
    const float* __restrict__ d_U, 
    const float* __restrict__ d_V, 
    float* __restrict__ d_next_U,  
    float* __restrict__ d_next_V 
) {
    // One thread block computes exactly one 16 x 16
    // tile of the simulation grid.

    // Thread indices at the block level.
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global thread indices.
    int globalTx = blockIdx.x * blockDim.x + tx;
    int globalTy = blockIdx.y * blockDim.y + ty;

    // --- Shared memory allocation ---
    // Add one border cell on all sides of the 16 x 16 tile.
    constexpr int TILE_SIZE {16};
    constexpr int SHARED_DIM {TILE_SIZE + 2};

    __shared__ float s_U[SHARED_DIM][SHARED_DIM];
    __shared__ float s_V[SHARED_DIM][SHARED_DIM];

    // --- Collaborative loading ---
    // Every thread loads exactly one cell from global memory
    // into shared memory. 

    // Define shared memory indices for the core 16 x 16
    // compute tile. Add 1 to tx and ty to avoid the border rim.
    int sx {tx + 1};
    int sy {ty + 1};

    // Define the global memory index for the core tile.
    // Use getWrappedIndex so that if the block is on the edge 
    // of the simulation grid, the index wraps.
    int globalIdx {getWrappedIndex(globalTx, globalTy, Params::N_x, Params::N_y)};
    s_U[sy][sx] = d_U[globalIdx];
    s_V[sy][sx] = d_V[globalIdx];

    // --– Load the border (halo) cells ---
    // Threads on the edges must supply the border cell too.
    // Coming soon.

    // --- Synchronization barrier ---
    // Force all 256 threads in the block to wait here until 
    // the entire shared 18 x 18 array has been loaded.
    __syncthreads();

    // --- Compute and write back ---
    // Perform the 5-point stencil reading only from s_U and
    // s_V, then write the result to d_next_U[global_idx].
    // Coming soon.
    
}

// Kernel launcher wrapper for the host.
void runGrayScottStep(
    const float* d_U, 
    const float* d_V, 
    float* d_next_U,  
    float* d_next_V,  
    dim3 threadsPerBlock, 
    dim3 blocksPerGrid
) {
    // Launch the kernel.
    grayScottKernelShared<<<blocksPerGrid, threadsPerBlock>>> (
        d_U,
        d_V,
        d_next_U,
        d_next_V
    );

    // Catch any launch errors immediately.
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}