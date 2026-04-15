#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <format>
#include <string>
#include <cuda_runtime.h>
#include <filesystem>
#include "io_utils.hpp"
#include "utils.hpp"
#include "solver.cuh"
#include "parameters.hpp"

int main() {
    std::cout << "Starting cuGrayScott Initialization...\n";

    // --- STABILITY ANALYSIS CHECK ---
    float max_D {std::max(Params::D_u, Params::D_v)};
    float dt_limit { (Params::h * Params::h)/(4.0f * max_D) };

    if (Params::dt > dt_limit)
        throw std::runtime_error("Numerical Instability Warning: dt exceeds the von Neumann stability limit.");

    std::cout << "Stability check passed. dt = " << Params::dt << " is <= dt_limit = " << dt_limit << "\n";

    // --- HOST MEMORY ALLOCATION ---   
    // Total number of grid points.
    const int numNodes {Params::N_x * Params::N_y};

    // Host vectors for U and V concentrations.
    // Flattened 1D arrays for contiguous memory mapping.
    std::vector<float> h_U(numNodes, 1.0f);
    std::vector<float> h_V(numNodes, 0.0f);

    // Initialize a central square to seed the reaction.
    int centerX {Params::N_x / 2};  // Floor division.
    int centerY {Params::N_y / 2};
    int squareSize {20};

    for (int y {centerY - squareSize / 2}; y < centerY + squareSize / 2; ++y) {
        for (int x {centerX - squareSize / 2}; x < centerX + squareSize / 2; ++x) {
            int index = getIndex(x, y, Params::N_x);
            h_U[index] = 0.5f;  // Deplete U.
            h_V[index] = 0.25f; // Inject V.
        }
    }

    std::cout << "Host memory allocated and initialized successfully.\n";
    std::cout << "Grid size: " << Params::N_x << " x " << Params::N_y << " points.\n";

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

    // --- SIMULATION PARAMETERS ---
    constexpr int numSteps {50'000};
    constexpr int outputFrequency {100}; 

    // --- HARDWARE GRID CONFIGURATION ---
    // Query the GPU hardware for the number of SMs.
    int deviceId;
    cudaCheck(cudaGetDevice(&deviceId));
    int numSMs;
    cudaCheck(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    
    // We use 16 by 16 2D thread blocks, for a total of 256 threads.
    int sqBlockSize {16};
    dim3 threadsPerBlock(sqBlockSize, sqBlockSize);
    dim3 blocksPerGrid = computeHardwareGridDimensions(sqBlockSize, numSMs, Params::N_x, Params::N_y);

    // --- CREATE OUTPUT DIRECTORY ---
    std::string outDir = "../out_phase1";
    if (!std::filesystem::exists(outDir)) {
        std::filesystem::create_directories(outDir);
        std::cout << "Created output directory: " << outDir << "\n";
    }
    
    // --- SIMULATION LOOP ---
    std::string config_msg = std::format(
        "Starting simulation loop with:\n{} time steps\n{} output frequency\n({},{}) 2D block\n({},{}) 2D grid",
        numSteps,
        outputFrequency,
        threadsPerBlock.x,
        threadsPerBlock.y,
        blocksPerGrid.x,
        blocksPerGrid.y
    );
    std::cout << config_msg << "\n";

    for (int step{0}; step < numSteps; ++step) {

        // Output data periodically.
        if (step % outputFrequency == 0) {
            std::cout << "Step " << step << " / " << numSteps << " - Extracting frame...\n";

            // Copying V is enough to visualize the pattern.
            cudaCheck(cudaMemcpy(h_V.data(), d_V, bytes, cudaMemcpyDeviceToHost));

            writeBinaryFrame(h_V, step, outDir);
        }

        runGrayScottStep(
            d_U, 
            d_V, 
            d_next_U, 
            d_next_V, 
            threadsPerBlock, 
            blocksPerGrid
        );

        // Ping-Pong the pointers.
        // The newly computed state becomes the current state for the next loop iteration.
        std::swap(d_U, d_next_U);
        std::swap(d_V, d_next_V);
    }

    // --- CLEAN UP ---
    std::cout << "Freeing device memory...\n";
    cudaCheck(cudaFree(d_U));
    cudaCheck(cudaFree(d_V));
    cudaCheck(cudaFree(d_next_U));
    cudaCheck(cudaFree(d_next_V));

    std::cout << "Simulation complete.\n";

    return 0;

}



