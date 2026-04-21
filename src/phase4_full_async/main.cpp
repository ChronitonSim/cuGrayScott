#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <format>
#include <string>
#include <future>
#include <cuda_runtime.h>
#include <filesystem>
#include "io_utils.hpp"
#include "utils.hpp"
#include "solver.cuh"
#include "parameters.hpp"
#include "timer.hpp"

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
    // Use pinned (page-locked) memory for asynchronous 
    // host/device data transfer.
    float *h_U, *h_V;
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&h_U), numNodes * sizeof(float)));
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&h_V), numNodes * sizeof(float)));

    std::fill_n(h_U, numNodes, 1.0f);
    std::fill_n(h_V, numNodes, 0.0f);

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

    // Device pointers for the current time step n.
    float* d_U;
    float* d_V;
    // Device pointers for the next time step n + 1.
    float* d_next_U;
    float* d_next_V;
    // Buffer device pointer for asynchronous
    // CPU/GPU data transfer.
    float* d_output_V;

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_U), bytes));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_V), bytes));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_next_U), bytes));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_next_V), bytes));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_output_V), bytes));

    // Host-to-device data transfer.
    std::cout << "Copying initial state from host to device...\n";
    cudaCheck(cudaMemcpy(d_U, h_U, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));
    
    // Just for safety, we initialize d_next_U/V too, even if it is not strictly necessary.
    cudaCheck(cudaMemcpy(d_next_U, h_U, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_next_V, h_V, bytes, cudaMemcpyHostToDevice));

    // --- SIMULATION PARAMETERS ---
    constexpr int numSteps {1000};
    constexpr int outputFrequency {100}; 
    constexpr bool ENABLE_IO {true};

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
    std::string outDir = "../out_phase4";
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

    // Create separate streams for computation
    // and data transfer.
    cudaStream_t computeStream;
    cudaStream_t transferStream;
    cudaCheck(cudaStreamCreate(&computeStream));
    cudaCheck(cudaStreamCreate(&transferStream));

    // Create an event to guide the asynchronous
    // GPU/CPU timeline.
    cudaEvent_t computeDoneEvent;
    cudaCheck(cudaEventCreate(&computeDoneEvent));

    // Variable to hold the background
    // I/O thread.
    std::future<void> io_thread;

    // Initialize and start the CUDA timer 
    // on the compute stream, to measure 
    // only computations and not data transfers.
    CudaTimer timer;
    timer.start(computeStream);

    for (int step{0}; step < numSteps; ++step) {

        // Launch the compute kernel
        // in the compute stream.
        runGrayScottStep(
            d_U, 
            d_V, 
            d_next_U, 
            d_next_V, 
            threadsPerBlock, 
            blocksPerGrid,
            computeStream
        );

        // Output data periodically.
        if (ENABLE_IO && step % outputFrequency == 0) {
            
            // Wait for any previous disk writes to 
            // finish, to prevent overwriting h_V
            // while the CPU is still saving it.
            if (io_thread.valid()) // Check the std::future 
                                   // is associated with a task.
                // Pause the main thread until the 
                // result becomes available.
                io_thread.wait();

            // Make a device snapshot of the
            // buffer (d_V) we intend to dump. 
            // This is crucial since without it:
            // 1. The CPU would swap d_V with d_next_V
            // at the end of this loop.
            // 2. The CPU would launch the next kernel,
            // which would write to d_next_V, formerly
            // d_V, while the buffer at that VRAM address
            // was still being copied to the host.
            cudaCheck(cudaMemcpyAsync(
                d_output_V, 
                d_V, 
                bytes, 
                cudaMemcpyDeviceToDevice,
                computeStream
            ));

            // --- GPU-TO-GPU SYNC ---

            // Drop a marker in the compute 
            // stream right after the kernel.
            cudaCheck(cudaEventRecord(
                computeDoneEvent, 
                computeStream
            ));

            // Instruct the transfer stream 
            // to wait for that marker,
            // signaling completion of kernel 
            // computation.
            // The CPU queues this rule on the 
            // stream and moves on immediately.
            cudaCheck(cudaStreamWaitEvent(
                transferStream, 
                computeDoneEvent
            ));
            
            // Queue the async device-to-host
            // copy in the transfer stream,
            // using the snapshot buffer.
            // It won't begin until the
            // computeDoneEvent is reached.
            // Note: this leaverages on h_V 
            // being pinned memory.
            cudaCheck(cudaMemcpyAsync(
                h_V, 
                d_output_V, 
                bytes, 
                cudaMemcpyDeviceToHost,
                transferStream
            ));

            // --- DELEGATE WAITING FOR THE
            //     TRANSFER TO A BACKGROUND THREAD ---
            
            // Launch a background CPU thread to write 
            // to SSD, after waiting for the async 
            // device-to-host transfer to complete.
            // The main thread immediately loops
            // back and launches the next compute kernel
            // while this file is saving.
            io_thread = std::async(
                std::launch::async,  // ensure a dedicated thread is spawn
                [h_V, bytes, step, outDir, transferStream]() {

                    // Pause the background thread 
                    // until the PCIe copy in the
                    // transfer stream has completed.
                    cudaCheck(cudaStreamSynchronize(transferStream));

                    // Only then, write the
                    // frame to disk.
                    writeBinaryFrameAsync(h_V, bytes, step, outDir);
                    std::cout << "Step " << step << " saved.\n";
                }
            );
        }

        // Ping-Pong the pointers.
        // The newly computed state becomes 
        // the current state for the next 
        // loop iteration.
        std::swap(d_U, d_next_U);
        std::swap(d_V, d_next_V);
    }

    // Make sure the very last frame finishes
    // saving before the program exits.
    if (io_thread.valid())
        io_thread.wait();

    // Stop the CUDA timer and
    // print the results.
    float elapsed_ms {timer.stop(computeStream)};

    std::cout << "Simulation complete.\n";
    std::cout << "========================================\n";
    std::cout << "Total Compute Time: " << elapsed_ms << " ms\n";
    std::cout << "Average Time/Step : " << elapsed_ms / numSteps << " ms\n";
    std::cout << "========================================\n";

    // --- CLEAN UP ---
    std::cout << "Freeing host memory...\n";
    cudaCheck(cudaFreeHost(h_U));
    cudaCheck(cudaFreeHost(h_V));

    std::cout << "Freeing device memory...\n";
    cudaCheck(cudaFree(d_U));
    cudaCheck(cudaFree(d_V));
    cudaCheck(cudaFree(d_next_U));
    cudaCheck(cudaFree(d_next_V));
    cudaCheck(cudaFree(d_output_V));
    cudaCheck(cudaStreamDestroy(computeStream));
    cudaCheck(cudaStreamDestroy(transferStream));
    cudaCheck(cudaEventDestroy(computeDoneEvent));

    std::cout << "Simulation complete.\n";

    return 0;

}