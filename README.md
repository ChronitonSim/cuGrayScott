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

### Phase 3: Asynchronous Pipelining (Planned)
While Phase 2 optimizes the compute kernel, the host-to-device data pipeline remains strictly synchronous, bottlenecked by PCIe transfers and SSD I/O limits. Phase 3 introduces Asynchronous Execution. By allocating Pinned (Page-Locked) host memory and deploying concurrent CUDA Streams, the pipeline will hide file I/O latency entirely, allowing the GPU to compute $t_{n+1}$ simultaneously while the CPU writes $t_{n}$ to physical storage.
*(For flow execution and stream analysis, see `src/phase3_async/README.md`)*

### Phase 4: Full Hardware Asynchronous Pipelining (Planned)
Phase 3 still requires the CPU's main loop to synchronize and wait for the PCIe transfer to complete before delegating the disk write. Phase 4 will introduce CUDA Hardware Events (`cudaEvent_t`), completely decoupling the CPU from the GPU streams. By moving all synchronization inside the background threads and utilizing GPU-to-GPU event waiting, the main thread will never pause, keeping the compute cores fed at absolute maximum capacity.

## Performance Benchmarks & Hardware Analysis

The following table details the compute time per temporal step (measured via hardware `cudaEvent_t` timestamps, completely isolating the GPU from CPU and disk I/O overhead). Data is aggregated over 10 execution trials.

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

### Discussion: Asynchronous I/O Overhead (Phase 3)
When scaling up to the 4096 x 4096 grid and activating intensive disk I/O (dumping 16 million floats per frame), a synchronous pipeline would halt the GPU completely during PCIe and SSD transfers. 

Phase 3's asynchronous CPU delegation achieves an average compute time of **0.48 ± 0.01 ms** per step (over 10 trials). Compared to Phase 2's baseline of 0.40 ± 0.02 ms (which had zero I/O overhead), we observe a mere ~0.08 ms penalty per frame. The heavy lifting of disk writing is successfully offloaded to the background, demonstrating that PCIe transfers of Pinned Memory can be effectively overlapped with active compute kernels.