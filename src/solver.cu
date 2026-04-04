#include "solver.cuh"
#include "utils.hpp"

__global__
void grayScottKernel(
    const float* __restrict__ d_U, 
    const float* __restrict__ d_V, 
    float* __restrict__ d_next_U,  
    float* __restrict__ d_next_V,  
    int width, 
    int height, 
    float Du, 
    float Dv, 
    float F, 
    float k, 
    float dt, 
    float h) {

    // Calculate thread's initial starting coordinates.
    int start_x = blockIdx.x * blockDim.x + threadIdx.x;
    int start_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the total hardware grid span, i.e. the stride.
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    // 2D grid-stride loop to cover the entire simulation grid.
    for (int y {start_y}; y < height; y += stride_y) {
        for (int x {start_x}; x < width; x += stride_x) {

            // Calculate the 1D simulation grid index for (x, y).
            int simGridIndex = getIndex(x, y, width);

            // Handle boundaries in an elementary way (for now).
            // If a thread is on the edge of the simulation grid,
            // simply copy d_U to d_next_U and d_V to d_next_V.
            if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
                d_next_U[simGridIndex] = d_U[simGridIndex];
                d_next_V[simGridIndex] = d_V[simGridIndex];
                continue;
            }
            
            // Fetch values of U and V for the center of the stencil.
            float u_c {d_U[simGridIndex]};
            float v_c {d_V[simGridIndex]};

            // Fetch neighbor values. 
            float u_south {d_U[getIndex(x, y - 1, width)]};
            float u_north {d_U[getIndex(x, y + 1, width)]};
            float u_east {d_U[getIndex(x + 1, y, width)]};
            float u_west {d_U[getIndex(x - 1, y, width)]};
            
            float v_south {d_V[getIndex(x, y - 1, width)]};
            float v_north {d_V[getIndex(x, y + 1, width)]};
            float v_east {d_V[getIndex(x + 1, y, width)]};
            float v_west {d_V[getIndex(x - 1, y, width)]};

            // Compute spatial Laplacians for U and V.
            float laplacian_u {computeLaplacian(u_c, u_south, u_north, u_east, u_west, h)};

            float laplacian_v {computeLaplacian(v_c, v_south, v_north, v_east, v_west, h)};

            // Compute explicit Euler time steps and write them to d_next_U and d_next_V
            d_next_U[simGridIndex] = computeExplicitEulerStepU(u_c, v_c, laplacian_u, dt, Du, F);
            d_next_V[simGridIndex] = computeExplicitEulerStepV(u_c, v_c, laplacian_v, dt, Dv, F, k);
        }
    }

}