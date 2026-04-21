# Asynchronous Host/Device Pipelining

## Architecture Overview
Phase 3 introduces asynchronous execution to decouple GPU computation from the host's file system I/O. While Phase 2 optimized the spatial finite difference method on the GPU silicon, its data extraction pipeline remained strictly synchronous. Phase 3 leverages Pinned (Page-Locked) memory, multiple CUDA Streams, and C++ multithreading (`std::async`) to overlap SSD writes with GPU kernel execution.

## Hardware Execution Flow and Memory Costs

### 1. Pinned Memory (`cudaMallocHost`)
Standard host memory (e.g., `std::vector`) is pageable, meaning the OS is permitted to move its physical location in RAM. The GPU's Direct Memory Access (DMA) engine cannot safely pull from pageable memory asynchronously. By allocating "Pinned" memory, we lock the host array's physical address. This allows `cudaMemcpyAsync` to transfer data over the PCIe bus in the background without requiring active CPU cycles.

### 2. Multi-Stream Choreography
The architecture utilizes two distinct CUDA streams:
* **Compute Stream:** Handles the execution of the `grayScottKernelShared`.
* **Transfer Stream:** Handles the `cudaMemcpyAsync` payload deliveries over the PCIe bus.

### 3. The Execution Flow
1. **Kernel Launch:** The main CPU thread dispatches the compute kernel into the Compute Stream and immediately continues execution.
2. **Compute Synchronization:** When an I/O frame is required, the CPU halts (`cudaStreamSynchronize`) until the Compute Stream finishes calculating the current time step.
3. **PCIe Transfer:** The CPU queues a `cudaMemcpyAsync` in the Transfer Stream.
4. **Transfer Synchronization:** The CPU halts again to ensure the PCIe transfer to Pinned Memory is completely finished.
5. **Disk Delegation:** The CPU spawns a detached background thread (`std::async`) to write the pinned array to the SSD. 
6. **Overlap:** The main CPU thread instantly loops back, commanding the GPU to calculate the next frame *while* the background thread writes the previous frame to disk.

## Architectural Limitations and Improvements

### Why this Pipeline is Still Limited
While disk I/O is successfully hidden, the CPU's main thread remains tethered to the PCIe bus. Because the main loop calls `cudaStreamSynchronize(transferStream)`, the CPU is physically blocked until the host receives the data. Consequently, the GPU's compute cores sit completely idle during the PCIe transfer because the CPU hasn't yet looped back around to launch the next compute kernel.

### Path to Improvement (Phase 4: Full Async)
To resolve this lingering idle time, we must eliminate all host-side synchronization from the main simulation loop. 
* **Implementation:** We will introduce Hardware Events (`cudaEvent_t`). The Transfer Stream will be instructed to wait for the Compute Stream directly on the silicon (`cudaStreamWaitEvent`), entirely bypassing the CPU. The CPU-side synchronization will be moved *inside* the background `std::async` thread.
* **Benefit:** The main CPU thread will never pause. It will rapidly queue math, events, and transfers, maximizing parallel utilization of the GPU compute cores, the PCIe DMA engine, and the SSD.