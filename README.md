# Accelerated Numerical Computing

A collection of CUDA C++ and CPU SIMD implementations focusing on core hardware bottlenecks: memory coalescing, cache limits, and warp divergence.

## 🔑 Key Insights & Results

### 1. Sparse Storage: Memory Coalescing vs. Footprint
**The Problem:** Storing the 3D discrete Laplacian operator as a dense matrix requires massive memory overhead ($\mathcal{O}(N^6)$) and redundant computation. 
**The Progression:** 
- **Dense $\rightarrow$ CRS:** Implemented Compressed Row Storage (CRS) to reduce the memory footprint. However, the irregular row lengths inherent to CRS caused severe warp divergence and un-coalesced memory reads on the GPU (78% data transfer waste).
- **CRS $\rightarrow$ SELL-C-σ:** Re-structured the data into chunks, sorted by non-zero elements, and padded to uniform lengths.
**The Result:** SELL-C-σ reduced un-coalesced waste to just 3%, increasing effective SpMV memory bandwidth to ~148 GB/s and allowing the Lanczos algorithm to run entirely on-device.

### 2. Dense Linear Algebra: Atomic Contention & Tiling
**The Problem:** Naive GPU implementations of Matrix-Vector (GEMV) operations suffer from severe memory bottlenecks and atomic write contention when multiple threads write to the same output index.
**The Progression:** 
- Shifted from naive global memory reads to **shared memory tiling**.
- Replaced direct atomic accumulations with **tree-based reductions** within the warp to minimize contention.
**The Result:** The optimized kernels saturated 78.9% of the hardware's theoretical peak bandwidth (252.25 GB/s), successfully matching and—in the case of narrow matrix geometries—outperforming standard `cuBLAS` implementations.

### 3. Hardware Limits: The STREAM Triad
**The Problem:** Before optimizing complex algorithms, the absolute memory bandwidth ceilings of the CPU and GPU memory hierarchies must be established.
**The Progression:** 
- Profiled the STREAM Triad benchmark starting from scalar CPU code.
- Scaled to auto-vectorized SIMD (`-O3`), explicit AVX intrinsics, and finally bare-metal CUDA.
**The Result:** Mapped the exact performance drop-offs across L1, L2, and L3 caches as array sizes grew to $10^9$ elements, establishing the baseline limits used to profile the sparse and dense algorithms above.