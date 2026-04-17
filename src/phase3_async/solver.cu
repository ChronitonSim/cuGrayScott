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
    

    // Thread indices at the block level. 
    // Range (0 - 15).
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // --- Shared Memory Allocation ---
    // Add one border cell on all sides of the 16 x 16 tile.
    constexpr int TILE_SIZE {16};
    constexpr int SHARED_DIM {TILE_SIZE + 2};

    __shared__ float s_U[SHARED_DIM][SHARED_DIM]; // Range (0 - 17).
    __shared__ float s_V[SHARED_DIM][SHARED_DIM];

    // Define shared memory indices for the core 16 x 16
    // compute tile. Add 1 to tx and ty to avoid the border rim.
    // Range (1 - 16).
    int sx {tx + 1};
    int sy {ty + 1};

    // --- Block Stride Loop ---
    // The entire block jumps by the total grid size (gridDim.x * blockDim.x)
    // (as measured in number of threads).
    int block_stride_x = blockDim.x * gridDim.x;
    int block_stride_y = blockDim.y * gridDim.y;

    int block_start_x = blockDim.x * blockIdx.x;
    int block_start_y = blockDim.y * blockIdx.y;

    for (int y {block_start_y}; y < Params::N_y; y += block_stride_y) {
        for (int x {block_start_x}; x < Params::N_x; x += block_stride_x) {
            
            // Global thread indices.
            int globalTx = block_start_x + tx;
            int globalTy = block_start_y + ty;
            
            // --- Collaborative loading ---
            // For every stride step, every thread loads exactly 
            // one cell from global memory into shared memory. 

            // Define the global memory index for the core tile.
            // Use getWrappedIndex so that if the block is on the edge 
            // of the simulation grid, the index wraps.
            int globalIdx {getWrappedIndex(
                globalTx, 
                globalTy, 
                Params::N_x, 
                Params::N_y)};
            // Load from global memory into shared memory.
            s_U[sy][sx] = d_U[globalIdx];
            s_V[sy][sx] = d_V[globalIdx];

            // --– Load the border (halo) cells ---
            // Threads on the edges must supply the border cell too.
            // Use thread-block indices tx, ty to identify threads on 
            // the edges of the tile.
            // Note: the four corners of the tile (0,0), (0, 17), (17,0), 
            // and (17, 17) are left uninitialized, because they are not 
            // needed to compute the Laplacian stencil 
            // (which does not read diagonally).

            // West edge
            if (tx == 0) {
                int westIdx {getWrappedIndex(globalTx - 1, globalTy, Params::N_x, Params::N_y)};
                s_U[sy][0] = d_U[westIdx];
                s_V[sy][0] = d_V[westIdx];
            }

            // East edge
            if (tx == TILE_SIZE - 1) {
                int eastIdx {getWrappedIndex(globalTx + 1, globalTy, Params::N_x, Params::N_y)};
                s_U[sy][SHARED_DIM - 1] = d_U[eastIdx];
                s_V[sy][SHARED_DIM - 1] = d_V[eastIdx];
            }

            // North edge
            if (ty == 0) {
                int northIdx {getWrappedIndex(globalTx, globalTy - 1, Params::N_x, Params::N_y)};
                s_U[0][sx] = d_U[northIdx];
                s_V[0][sx] = d_V[northIdx];
            }

            // South Edge
            if (ty == TILE_SIZE - 1) {
                int southIdx {getWrappedIndex(globalTx, globalTy + 1, Params::N_x, Params::N_y)};
                s_U[SHARED_DIM - 1][sx] = d_U[southIdx];
                s_V[SHARED_DIM - 1][sx] = d_V[southIdx];
            }

            // --- Synchronization barrier 1 ---
            // Force all 256 threads in the block to wait here until 
            // the entire shared 18 x 18 array has been loaded
            // (to prevent race conditions).
            __syncthreads();

            // --- Compute and write back ---
            // Perform the 5-point stencil reading only from s_U and
            // s_V, then write the result to d_next_U/V[global_idx].

            float u_c {s_U[sy][sx]};
            float u_south {s_U[sy + 1][sx]};
            float u_north {s_U[sy - 1][sx]};
            float u_east {s_U[sy][sx + 1]};
            float u_west {s_U[sy][sx - 1]};

            float v_c {s_V[sy][sx]};
            float v_south {s_V[sy + 1][sx]};
            float v_north {s_V[sy - 1][sx]};
            float v_east {s_V[sy][sx + 1]};
            float v_west {s_V[sy][sx - 1]};
            
            // Compute spatial Laplacians for U and V.
            float laplacian_u {computeLaplacian(
                u_c, 
                u_south, 
                u_north, 
                u_east, 
                u_west, 
                Params::h)
            };

            float laplacian_v {computeLaplacian(
                v_c, 
                v_south, 
                v_north, 
                v_east, 
                v_west, 
                Params::h)
            };

            // Compute explicit Euler time steps and write them to d_next_U and d_next_V
            d_next_U[globalIdx] = computeExplicitEulerStepU(
                u_c, 
                v_c, 
                laplacian_u, 
                Params::dt, 
                Params::D_u, 
                Params::F
            );

            d_next_V[globalIdx] = computeExplicitEulerStepV(
                u_c, 
                v_c, 
                laplacian_v, 
                Params::dt, 
                Params::D_v, 
                Params::F, 
                Params::k
            );

            // --- Synchronization barrier 2 ---
            // Prevent threads from looping back and overwriting 
            // shared memory (s_U/s_V) with the next tile's data
            // until all threads have finished computing this tile.
            __syncthreads();
        
        }
    }
}

// Kernel launcher wrapper for the host.
void runGrayScottStep(
    const float* d_U, 
    const float* d_V, 
    float* d_next_U,  
    float* d_next_V,  
    dim3 threadsPerBlock, 
    dim3 blocksPerGrid,
    cudaStream_t stream
) {
    // Launch the kernel.
    grayScottKernelShared<<<blocksPerGrid, threadsPerBlock, 0, stream>>> (
        d_U,
        d_V,
        d_next_U,
        d_next_V
    );

    // Catch any launch errors immediately.
    cudaCheck(cudaGetLastError());
    
    // We no longer call cudaDeviceSynchronize()
    // here, to let the CPU move on immediately 
    // after launching the kernel.

}