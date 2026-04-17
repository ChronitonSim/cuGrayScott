# Tiled Shared Memory Kernel

## Architecture Overview
The Phase 2 kernel maps the 2D finite difference method onto a blocked shared memory architecture. It replaces the thread-level grid-stride loop with a block-stride loop. Thread blocks cooperatively load discrete 2D simulation tiles, including a 1-cell boundary halo, into on-chip shared memory (SMEM). This architecture minimizes redundant global memory transactions by performing the 5-point stencil entirely within SMEM.

## Shared Memory and Halo Cells

### Cooperative Loading (`__shared__`)
Shared memory provides L1-level latency and high bandwidth but requires explicit software management. Each thread block allocates a 2D array of size `(TILE_SIZE + 2) x (TILE_SIZE + 2)`. The interior `TILE_SIZE x TILE_SIZE` grid maps 1:1 with the thread block. Threads fetch their corresponding global memory index and write it to the shifted SMEM address.

### Halo Zone Population
To compute the Laplacian at the tile edges, threads require data from adjacent tiles. Threads positioned on the boundaries of the block compute secondary global indices to fetch these "halo" or "ghost" cells. The physical corners of the SMEM allocation remain uninitialized, as the 5-point stencil does not perform diagonal reads.

## Hardware Execution Flow and Memory Costs

### 1. Synchronization Barriers (`__syncthreads()`)
Data hazards (specifically Read-After-Write) are mitigated using hardware synchronization barriers.
* **Pre-Compute Barrier:** Halts thread execution until all core tile and halo cells are committed to SMEM.
* **Post-Compute Barrier:** Halts fast threads from looping back and overwriting SMEM with the next tile's data before slow threads complete the current spatial Laplacian.

### 2. SMEM Bank Conflicts
Shared memory is divided into 32 distinct memory banks, each 4 bytes wide. Simultaneous access to the same bank by different threads within a 32-thread warp causes serialization (bank conflicts).

* **Memory Mapping and Bank Calculation:** A 2D SMEM array translates to a flat 1D contiguous block of bytes. The hardware calculates the bank using the exact byte address:
    1. `Byte Address = Base Address + (1D Index * sizeof(Type))`
    2. `Bank = (Byte Address / 4) % 32`
* **The Float Simplification:** For a 32-bit `float`, `sizeof(Type) = 4`. Assuming the compiler aligns the `__shared__` allocation perfectly to the start of Bank 0 (Base Address is a multiple of 128 bytes, effectively 0 for the modulo), the equation simplifies:
    1. `Bank = ((0 + 1D Index * 4) / 4) % 32`
    2. `Bank = 1D Index % 32`
    This 1:1 mapping means the C++ array index directly translates to the hardware bank index.
* **Halo Shift:** The linear 1D index (`Idx`) relies on a `+1` shift: `Idx = (y + 1) * 18 + (x + 1)`, where `x` and `y` are shared-memory tile coordinates. This centers the 16x16 compute tile inside the 18x18 allocation, preserving row 0 and column 0 for halo cells.
* **Warp Execution & 2-Way Conflict:**
    1. A 32-thread warp executes as two 16-thread rows (`y=0` and `y=1`).
    2. The first half-warp (`y=0`) calculates indices 19 through 34, mapping to Banks 19-31 and 0-2 (according to the fomulas above).
    3. The second half-warp (`y=1`) calculates indices 37 through 52, mapping to Banks 5-20.
    4. An overlap occurs at Banks 19 (threads 0 and 30) and 20 (threads 1 and 31), causing a 2-way conflict for 4 threads per warp.
* **Resolution via Padding:** Padding the SMEM array width to an odd number (e.g., 19) shifts the modulo calculation and eliminates the conflict entirely. Padding is omitted here, as a 2-way conflict on 4 threads per warp incurs negligible cycle latency compared to global memory constraints.

### 3. The L2 Cache Factor
Global memory fetches are executed in 128-byte cache lines. The hardware relies on spatial locality to cache data in the L2 cache. For sufficiently small domains, this mechanism masks the latency of naive architectures.

* **Quantitative Footprint:** A 256x256 grid requires 512 KB of total memory (65,536 cells * 4 bytes * 2 grids for U and V).
* **Cache Capacity:** An Ampere architecture (e.g., RTX 3080) features a 5,120 KB (5 MB) L2 Cache. 
* **L2 Fill Mechanism:**
    1. **Cold Cache (Step 0):** The initial kernel execution fetches 128-byte lines (32 contiguous floats) from VRAM, incurring high latency.
    2. **Cumulative Fill:** The spatial locality of the grid guarantees the 512 KB footprint is fully absorbed into the L2 cache.
    3. **Warm Cache (Steps 1-50,000):** Because the footprint consumes only 10% of the L2 capacity, the Least Recently Used (LRU) eviction policy never triggers.
    4. **Execution:** Subsequent global memory reads resolve directly from the L2 cache. 
* **Architectural Threshold:** To benchmark true VRAM bandwidth vs. SMEM efficiency, the domain must saturate the L2 cache. A 4096x4096 grid expands the memory footprint to ~134 MB, overwhelming the cache, forcing VRAM fetches, and yielding a >3x speedup for the SMEM architecture.

## Architectural Limitations and Improvements

### Why this Pipeline is Limited
While the compute kernel is memory-optimized, the host-device data pipeline remains synchronous. When frame extraction occurs, the CPU commands a `cudaMemcpy` across the PCIe bus and writes the payload to disk. The GPU execution queue halts during this transfer, leaving the SMs idle. Memory operations and compute operations are strictly serialized.

### Path to Improvement (Phase 3)
To resolve the I/O bottleneck, the pipeline must implement asynchronous execution.
* **Implementation:** Utilize CUDA Streams to concurrently schedule compute kernels and memory transfers. Replace pageable host memory with Pinned (Page-Locked) memory.
* **Benefit:** Allows the GPU to execute step `N+1` while the DMA engine asynchronously transfers step `N` across the PCIe bus, entirely hiding the latency of disk I/O and memory transfers.