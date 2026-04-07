#include "solver.cuh"
#include "parameters.hpp"
#include "utils.hpp"
#include "io_utils.hpp"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <driver_types.h>

__global__
void grayScottKernel(
    const float* __restrict__ d_U, 
    const float* __restrict__ d_V, 
    float* __restrict__ d_next_U,  
    float* __restrict__ d_next_V 
) {

    // Calculate thread's initial starting coordinates.
    int start_x = blockIdx.x * blockDim.x + threadIdx.x;
    int start_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the total hardware grid span, i.e. the stride.
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    // Note: when the hardware grid covers the entire simulation
    // grid, stride_x and stride_y evaluate to exactly width and
    // height, so the for loop below executes only once per thread.

    // 2D grid-stride loop to cover the entire simulation grid.
    for (int y {start_y}; y < Params::N_y; y += stride_y) {
        for (int x {start_x}; x < Params::N_x; x += stride_x) {
            
            // Fetch values of U and V for the center of the stencil.
            int centerStencilIndex = getIndex(x, y, Params::N_x);
            float u_c {d_U[centerStencilIndex]};
            float v_c {d_V[centerStencilIndex]};

            // Fetch neighbor values. 
            float u_south {d_U[getWrappedIndex(x, y - 1, Params::N_x, Params::N_y)]};
            float u_north {d_U[getWrappedIndex(x, y + 1, Params::N_x, Params::N_y)]};
            float u_east {d_U[getWrappedIndex(x + 1, y, Params::N_x, Params::N_y)]};
            float u_west {d_U[getWrappedIndex(x - 1, y, Params::N_x, Params::N_y)]};
            
            float v_south {d_V[getWrappedIndex(x, y - 1, Params::N_x, Params::N_y)]};
            float v_north {d_V[getWrappedIndex(x, y + 1, Params::N_x, Params::N_y)]};
            float v_east {d_V[getWrappedIndex(x + 1, y, Params::N_x, Params::N_y)]};
            float v_west {d_V[getWrappedIndex(x - 1, y, Params::N_x, Params::N_y)]};

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
            d_next_U[centerStencilIndex] = computeExplicitEulerStepU(
                u_c, 
                v_c, 
                laplacian_u, 
                Params::dt, 
                Params::D_u, 
                Params::F
            );
            d_next_V[centerStencilIndex] = computeExplicitEulerStepV(
                u_c, 
                v_c, 
                laplacian_v, 
                Params::dt, 
                Params::D_v, 
                Params::F, 
                Params::k
            );
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
    dim3 blocksPerGrid
) {
    // Launch the kernel.
    grayScottKernel<<<blocksPerGrid, threadsPerBlock>>> (
        d_U,
        d_V,
        d_next_U,
        d_next_V
    );

    // Catch any launch errors immediately.
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}