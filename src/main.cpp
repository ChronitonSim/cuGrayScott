#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "io_utils.hpp"
#include "utils.hpp"

#include <cuda_runtime.h>


// Grid dimensions.
constexpr int N_x {256};
constexpr int N_y {256};

// Kinetic parameters.
// These specific values produce a "Coral" Turing pattern.
constexpr float D_u {0.16f};
constexpr float D_v {0.08f};
constexpr float F {0.035f};
constexpr float k {0.065f};

// Discretization parameters.
constexpr float h {1.0f};  // Spatial step dx = dy
constexpr float dt {1.0f}; // Time step

int main() {
    std::cout << "Starting cuGrayScott Initialization...\n";

    // --- STABILITY ANALYSIS CHECK ---
    float max_D {std::max(D_u, D_v)};
    float dt_limit { (h * h)/(4.0f * max_D) };

    if (dt > dt_limit)
        throw std::runtime_error("Numerical Instability Warning: dt exceeds the von Neumann stability limit.");

    std::cout << "Stability check passed. dt = " << dt << " is <= dt_limit = " << dt_limit << "\n";

    // --- HOST MEMORY ALLOCATION ---   
    // Total number of grid points.
    const int numNodes {N_x * N_y};

    // Host vectors for U and V concentrations.
    // Flattened 1D arrays for contiguous memory mapping.
    std::vector<float> h_U(numNodes, 1.0f);
    std::vector<float> h_V(numNodes, 0.0f);

    // Initialize a central square to seed the reaction.
    int centerX {N_x / 2};  // Floor division.
    int centerY {N_y / 2};
    int squareSize {20};

    for (int y {centerY - squareSize / 2}; y < centerY + squareSize / 2; ++y) {
        for (int x {centerX - squareSize / 2}; x < centerX + squareSize / 2; ++x) {
            int index = getIndex(x, y, N_x);
            h_U[index] = 0.5f;  // Deplete U.
            h_V[index] = 0.25f; // Inject V.
        }
    }

    std::cout << "Host memory allocated and initialized successfully.\n";
    std::cout << "Grid size: " << N_x << " x " << N_y << " points.\n";

    // --- DEVICE MEMORY ALLOCATION ---
    std::cout << "Allocating device memory...\n";
    std::size_t bytes = numNodes * sizeof(float);

    // In explicit Euler, state n+1 depends entirely on the current 
    // state n. If we overwrite the current state while other threads
    // are still reading from it, we expose ourselves to race conditions.
    // Thus, we need two separate state arrays.

    // Device pointers for the current time step n
    float* d_U;
    float* d_V;
    // Device pointers for the next time step n + 1
    float* d_next_U;
    float* d_next_V;

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_U), bytes));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_V), bytes));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_next_U), bytes));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_next_V), bytes));

    // Host-to-device data transfer.
    std::cout << "Copying initial state from host to device...\n";
    cudaCheck(cudaMemcpy(d_U, h_U.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));
    
    // Just for safety, we initialize d_next_U/V too, even if it is not strictly necessary.
    cudaCheck(cudaMemcpy(d_next_U, h_U.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_next_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    // --- SIMULATION LOOP ---
    // coming next

    // --- CLEAN UP ---
    std::cout << "Freeing device memory...\n";
    cudaCheck(cudaFree(d_U));
    cudaCheck(cudaFree(d_V));
    cudaCheck(cudaFree(d_next_U));
    cudaCheck(cudaFree(d_next_V));

    return 0;

}



