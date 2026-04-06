# Naive 5-Point Stencil Kernel

## Architecture Overview
The Phase 1 kernel maps a continuous 2D finite difference method (FDM) onto a 1D linearized memory layout. It employs a 2D grid-stride loop, decoupling the hardware thread topology from the simulation grid dimensions. State updates use ping-pong buffers, ensuring thread synchronization and preventing race conditions during explicit Euler integration.

## The `__restrict__` Qualifier
The `__restrict__` keyword asserts to the compiler that memory accessed through a pointer is not accessed by any other pointer during kernel execution, eliminating pointer aliasing.

### Read-Only Buffers (`const float* __restrict__`)
Without `__restrict__` the compiler must assume the pointers overlap. To prevent corruption, it generates conservative machine code, forcing the hardware to route memory requests (e.g. for `u_south` or `u_east`) through the standard L1/L2 cache hierarchy. This ensures coherence and prevents anyone from reading stale values.

Combining `const` and `__restrict__` guarantees the data is neither aliased nor modified. The compiler can then replace standard global memory load instructions (`LD.E`) with the `__ldg()` intrinsic. Data fetches bypass the L1 cache and route through the Read-Only Cache (shared with the texture unit). 

**Benefit:** Reduces L1 cache congestion and avoids hardware checks for cache coherence.

### Output Buffers (`float* __restrict__`)

Without `__restrict__`, the compiler must assume that writing to `d_next_U[idx]` might accidentally overwrite a value in `d_U` that another line of code is about to read. Therefore, it is forced to execute reads and writes in the exact order they appear in source code.

* **Optimal Instruction Reordering**: Applying `__restrict__` without `const` signals write-only or read-write access without aliasing to input buffers. The no-aliasing promise allows the compiler to optimally reorder read and write instructions, overlapping memory latency with arithmetic operations.

* **Preventing L1 Thrashing**: If the compiler suspects an output pointer aliases with an input pointer, every single write to the L1 cache forces the hardware to check if it needs to invalidate the data currently being held for the read operations. 

    By explicitly separating them via the `__restrict__` contract, the GPU memory controller can stream data in (via the Read-Only cache) and stream data out (via the standard L1/L2 cache hierarchy) asynchronously, without stalling to cross-check for overlapping addresses.

## Hardware Execution Flow and Memory Costs

### 1. Warp Scheduling and Control Flow
* The CUDA runtime groups blocks into chunks of 32 threads (Warps) and assigns them to Streaming Multiprocessors (SMs).
* Warps execute instructions in lockstep.
* The grid-stride loop limits execution to grid bounds. Threads evaluating boundary conditions execute a separate branch. This causes warp divergence at the grid edges, forcing the SM to serialize execution paths and mask inactive threads.

### 2. Memory Fetches
For each cell update, the kernel executes 5 reads (`center`, `north`, `south`, `east`, `west`) and 2 writes (`next_U`, `next_V`).
* **Center, East, West Fetches:** Threads in a warp request contiguous memory addresses. The memory controller coalesces these requests into single 128-byte transactions.
* **North, South Fetches:** Threads request addresses separated by the grid `width`. These requests are uncoalesced, requiring multiple memory transactions per warp.
* **Cost Assessment:** Global memory accesses cost hundreds of clock cycles. The Read-Only cache mitigates latency via spatial locality, but the sheer volume of global transactions dominates execution time. The kernel is memory-bound.

## Architectural Limitations and Improvements

### Why this Kernel is Naive
The current architecture suffers from redundant global memory transactions. A single grid node is read 5 distinct times from global VRAM: once as the center cell by its assigned thread, and 4 times as a neighbor by adjacent threads. The compute-to-memory-access ratio is low.

### Path to Improvement (Phase 2)
To resolve the memory bottleneck, the architecture must transition from global memory to Shared Memory (SMEM). 
* **Implementation:** Thread blocks cooperatively load a 2D tile of the simulation grid (including a halo of ghost cells) from global memory into the SM's on-chip shared memory.
* **Benefit:** Shared memory provides order-of-magnitude lower latency and higher bandwidth. Redundant neighbor fetches occur in SMEM instead of global VRAM, reducing global memory reads by approximately 80%.