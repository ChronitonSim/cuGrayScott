# CUDA Gray-Scott Reaction-Diffusion Engine

## Introduction
This repository contains a high-performance, GPU-accelerated implementation of the Gray-Scott reaction-diffusion model. The Gray-Scott system describes the continuous complex spatial interactions between two chemical species ($U$ and $V$), capable of generating intricate, biologically-inspired Turing patterns (such as coral, mitosis, and labyrinthine structures). 

The mathematical engine relies on a finite difference method (FDM) for spatial Laplacian approximations and an explicit Euler integrator for temporal evolution. For a rigorous derivation of the governing partial differential equations and numerical stability conditions, please refer to the project's accompanying white paper.

This project demonstrates iterative optimization techniques in CUDA C++, transforming a naive global memory kernel into a hardware-optimized, asynchronous data pipeline.

## Hardware Specifications
For reproducibility, all benchmarks and architectural decisions documented in this repository were profiled on the following target system:
* **GPU:** NVIDIA GeForce RTX 3080 Laptop GPU (Ampere Architecture)
* **VRAM:** 8 GB GDDR6
* **CPU:** 11th Gen Intel Core i7-11800H (8 Cores, 16 Threads)
* **Host RAM:** 32 GB
* **Storage:** NVMe SSD

## Optimization Phases

### Phase 1: Naive Global Memory Kernel
The baseline implementation maps the 2D simulation grid to a 1D linearized memory array. It utilizes a grid-stride loop, safely decoupling execution from hardware limits. Pointers are heavily constrained using `const __restrict__` qualifiers, forcing the compiler to bypass the L1 cache in favor of the GPU's Read-Only data cache. While numerically stable, this architecture is strictly memory-bound. The 5-point FDM stencil requires five distinct global memory fetches per cell, creating a severe VRAM bandwidth bottleneck for large domains.
*(For deep architectural mechanics, see `src/phase1_naive/README.md`)*

### Phase 2: Tiled Shared Memory (SMEM)
Phase 2 fundamentally restructures memory access by leveraging the Streaming Multiprocessor's (SM) on-chip Shared Memory. It replaces the grid-stride loop with a block-stride loop. Thread blocks cooperatively load 16x16 compute tiles—padded by a 1-cell border to accommodate the stencil's "halo" cells—directly into L1-speed SMEM. After hardware synchronization (`__syncthreads()`), the FDM math executes entirely on-chip, reducing redundant global VRAM transactions by roughly 80%.
*(For deep architectural mechanics, including bank conflict resolution and cache behaviors, see `src/phase2_shared_mem/README.md`)*

### Phase 3: Asynchronous Pipelining
While Phase 2 optimizes the compute kernel, the host-to-device data pipeline remains strictly synchronous, bottlenecked by PCIe transfers and SSD I/O limits. Phase 3 introduces Asynchronous Execution. By allocating Pinned (Page-Locked) host memory and deploying concurrent CUDA Streams, the pipeline hides file I/O latency entirely, allowing the GPU to compute $t_{n+1}$ simultaneously while the CPU writes $t_{n}$ to physical storage.
*(For flow execution and stream analysis, see `src/phase3_async/README.md`)*

### Phase 4: Full Hardware Asynchronous Pipelining
Phase 3 still requires the CPU's main loop to synchronize and wait for the PCIe transfer to complete before delegating the disk write. Phase 4 introduces CUDA Hardware Events (`cudaEvent_t`), completely decoupling the CPU from the GPU streams. By moving all synchronization inside the background threads and utilizing GPU-to-GPU event waiting, the main thread never pauses, keeping the compute cores fed at absolute maximum capacity.
*(For full hardware decoupling details, see `src/phase4_full_async/README.md`)*

## Performance Benchmarks & Hardware Analysis

The following tables detail the compute time per temporal step (measured via hardware `cudaEvent_t` timestamps). Data is aggregated over 10 execution trials.

| Grid Dimensions | Phase 1 (Global Memory) | Phase 2 (Shared Memory) | Speedup Factor |
| :--- | :--- | :--- | :--- |
| **256 x 256** | 0.028 ± 0.003 ms      | 0.028 ± 0.002 ms      | **1.00x** |
| **4096 x 4096** | 1.20 ± 0.01 ms      | 0.40 ± 0.02 ms      | **3.02x** |

### Discussion: The L2 Cache Trap vs. True Bandwidth
The benchmark data reveals a critical threshold in GPU memory architecture. 

At a grid size of **256 x 256**, the performance of Phase 1 and Phase 2 is statistically identical. This occurs due to the Ampere architecture's massive 5 MB L2 cache. The total footprint of a 256x256 simulation is approximately 512 KB. After the initial "cold" read, the entire domain is trapped securely within the L2 silicon. Phase 1 never interacts with the global GDDR6 VRAM again, effectively executing at near-SMEM speeds automatically. The hardware caching mechanism perfectly masks the naive architecture.

At a grid size of **4096 x 4096**, the simulation footprint explodes to ~134 MB. This thoroughly overwhelms the 5 MB L2 cache, forcing the hardware to continuously evict data and fetch cache lines across the board from the physical VRAM. 
* Under these conditions, **Phase 1** collapses under the weight of its 5 redundant global memory fetches per thread. 
* **Phase 2**, however, demonstrates its architectural superiority. By loading contiguous tiles into Shared Memory once, it completely bypasses the VRAM bandwidth bottleneck. The result is a robust **>3x hardware acceleration**, exemplifying the need for explicit cache management in production-scale simulations.

### Discussion: Asynchronous I/O and Hardware Limits (Phase 3 vs. Phase 4)
When scaling up to the 4096 x 4096 grid and activating intensive disk I/O (dumping 16 million floats per frame), a synchronous pipeline would halt the GPU completely during PCIe and SSD transfers. By utilizing asynchronous execution streams, we can successfully overlap computation with memory transfers.

| Grid Dimensions (I/O Enabled) | Phase 3 (Host Delegation) | Phase 4 (Hardware Events) |
| :--- | :--- | :--- |
| **4096 x 4096** | 0.48 ± 0.01 ms      | 0.47 ± 0.02 ms      |

*(Note: The Phase 2 baseline with zero I/O overhead is 0.40 ± 0.02 ms)*

Phase 3's asynchronous CPU delegation achieves an average compute time of 0.48 ms per step, representing a mere ~0.08 ms penalty per frame to completely hide the heavy SSD writes.

However, moving from Phase 3's host-level decoupling to Phase 4's perfect hardware-level decoupling yields only a marginal improvement. The software architecture is flawless, but the pipeline has hit the physical limits of the hardware:

1. **VRAM Controller Contention:** Because the stream overlap is now perfectly concurrent, the Streaming Multiprocessors (SMs) are furiously reading and writing to GDDR6 VRAM to calculate step $t_{n+1}$ at the *same time* the GPU's DMA engine is reading from that same VRAM to send step $t_n$ over the PCIe bus. Both components are fighting for the same physical memory controllers, creating a hardware-level traffic jam that slightly throttles the compute kernels.
2. **PCIe Bus Saturation:** A 4096 x 4096 frame represents roughly 67 MB of data. Pushing this volume of data across the motherboard continuously saturates the bandwidth limits of the PCIe bus. Perfect software optimization cannot overcome this physical hardware ceiling.