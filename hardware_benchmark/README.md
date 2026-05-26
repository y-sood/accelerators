# Hardware Architecture Profiling

### 📖 The Engineering Story
Before optimizing complex software (like the SELL-C-σ sparse format or tiled GEMV kernels), it is critical to map the physical constraints of the target hardware. Using standard STREAM Triad benchmarking methodologies, I profiled the underlying limits of the UPPMAX Snowy cluster (Intel Xeon CPU & NVIDIA Tesla T4 GPU).

*(Note: This section focuses purely on performance analysis and architectural profiling using standard benchmark suites, rather than custom software implementation).*

### 🛠️ The Profiling Methodology
- **CPU Vectorization:** Profiled the differences between scalar code, compiler auto-vectorization (`-O3`, `-march=sandybridge`), and explicit AVX SIMD intrinsics.
- **GPU Bandwidth:** Benchmarked theoretical vs. achievable VRAM limits for both single (`float`) and double (`double`) precision workloads.

### 🚀 The Results
The profiling successfully mapped the exact boundaries of the L1 (32KB), L2 (256KB), and L3 (20MB) CPU caches. As the array sizes grew to $10^9$ elements, the data perfectly visualized the cache fall-off cliffs, establishing the absolute DRAM bandwidth ceilings that were used as targets for the custom CUDA kernels developed in this repository.

![Cache Drop-off Boundaries](./assets/stream_triad_cache_dropoff.png)
*Figure: Visualization of L1/L2/L3 cache fall-offs as problem size scales.*