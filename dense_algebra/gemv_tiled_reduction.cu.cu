#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>
#include <iostream>
#include <fstream>

const int chunk_size = 1;

__global__ void ye_olde_matrix_vector_transpose(const float* A, const float* x, float* y, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    float sum = 0.0f;
    int col_base = col * M;
    for (int row = 0; row < M; ++row) {
        sum += A[col_base + row] * x[row];
    }
    y[col] = sum;
}

__global__ void matrix_vector_transpose(const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y,  int M, int N) {
    const int yey = blockSize.x;
    __shared__ float sdata[128 * 4];
    int tid = threadIdx.x;
    int col_block_start = blockIdx.x * 4;

    for (int c = 0; c < 4; ++c) {
        int col = col_block_start + c;
        if (col >= N) return;

        float partial = 0.0f;
        for (int row = tid; row < M; row += blockDim.x) 
        {
            partial += A[col * (size_t)M + row] * x[row];
        }
        //Writing to shared array for reduction
        sdata[tid + c * blockDim.x] = partial;
    }
    __syncthreads();

    //Reduction across thread
    for (int col_i = 0; col_i < 4; ++col_i) 
    { 
        if (col_block_start + col_i >= N) break; //Overflow

        //Tree based reduction - ex1
        int index = col_i * blockDim.x + tid;
        for (int s = blockDim.x / 2; s > 0; s >>= 1) 
        {
            if (tid < s) 
            {
                sdata[index] += sdata[index + s];
            }
            __syncthreads();
        }

        //Write into output vector
        if (tid == 0) 
        {
            atomicAdd(&y[col_block_start + col_i], sdata[col_i * blockSize]);
        }
    }
}

__global__ void set_vector(const int N, const float val, float *x, const int bsy){
  const int idx_base = threadIdx.x + blockIdx.x * (blockDim.x * chunk_size);
  for (unsigned int i = 0; i < chunk_size; ++i)
    {
      const int idx = idx_base + i * bsy;
      if (idx < N)
        x[idx] = val;
    }
}

//Sets an M*N => M==N hilbert matrix for testing
__global__ void set_hilbert_matrix(float *mat, int N, int M){
        int idx_base = threadIdx.x + blockIdx.x * blockDim.x;

        // Each thread initializes multiple elements strided by the total number of threads
        int stride = blockDim.x * gridDim.x;
        for (int idx = idx_base; idx < M*N; idx += stride) 
        {
            int row = idx % N;
            int col = idx / N;
            mat[idx] = 1.0f / (row + col + 1);     
        }
}

void wtf(const std::vector<float>& data, const std::string& filename){
    std::ofstream ofs(filename);
    if(!ofs)
    {
        throw std::runtime_error("Cannot open file for writing\n");
    }
    for(size_t i=0; i<data.size(); ++i)
    {
        ofs << data[i];
        if(i != data.size() - 1) ofs << "\n";
    }
    ofs.close();
}

// Run the actual benchmark
void benchmark_multiplication(const std::size_t M, const std::size_t N, const long long repeat, const int bs, const int num){
    float *cu_vec, *cu_mat, *cu_out; 

    // Device allocations: x has length M, A is MxN, y has length N
    cudaMalloc(&cu_vec, M * sizeof(float));
    cudaMalloc(&cu_mat, M * N * sizeof(float));
    cudaMalloc(&cu_out, N * sizeof(float));

    const unsigned int n_blocks = (M + bs - 1) / bs;

    // Initialize y = 0, x = 1, and build A (Hilbert) in column-major as A[row + col*M]
    set_vector<<<n_blocks, bs>>>(N, 0.0f, cu_out, bs);
    set_vector<<<n_blocks, bs>>>(M, 1.0f, cu_vec, bs);
    set_hilbert_matrix<<<n_blocks, bs>>>(cu_mat, N, M); // assumes kernel writes column-major

    // Host buffer for correctness (length N for y)
    std::vector<float> result_host(N);

    // Benchmark parameters
    const unsigned int n_tests = 50;
    const unsigned long long int n_repeat = repeat;

    double best = 1e10, worst = 0.0, avg = 0.0;

    const int threadsPerBlock = bs;
    const int blocksPerGrid = (N+num-1)/num;

    for (unsigned int t = 0; t < n_tests; ++t)
    {
        //Begin measuring time   
        const auto t1 = std::chrono::steady_clock::now();
        //Run many repetitions
        for (unsigned int rep = 0; rep < n_repeat; ++rep)
        {
            //cudaMemset(cu_out, 0, M * sizeof(float));
            matrix_vector_transpose<<<blocksPerGrid, threadsPerBlock>>>(cu_mat, cu_vec, cu_out, (int)M, (int)N);
            cudaDeviceSynchronize();
        }

    //End time measurement 
     const double time = std::chrono::duration_cast<std::chrono::duration<double>>( std::chrono::steady_clock::now() - t1).count();
     best  = std::min(best, time / n_repeat);
     worst = std::max(worst, time / n_repeat);
     avg += time / n_repeat;
    }

    //Bandwidth results
    std::cout << M << "*" << N << " matrix " << best << "s  " << avg/n_tests << "s  " << worst <<"s  " << ((1e-9 * (M * N + M + N) * sizeof(float)) / best) << "GB/s  " <<std::endl;

    // Copy back N outputs
    cudaMemcpy(result_host.data(), cu_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(cu_vec);
    cudaFree(cu_mat);
    cudaFree(cu_out);

    // Save result for correctness checks
    wtf(result_host, "result.txt");
}

int main(int argc, char **argv){
    //Check inputs
    if (argc != 6)
    {
        std::cout << "Expected 3 arguments, got" << argc << std::endl;
        std::cout << "Format :" << std::atoi(argv[0]) << "M N reps num_rows_per_block loop_length_per_thread" << std::endl;
        return 1;
    };
    
    //Number of repetitions and size of problem
    int reps = std::atoi(argv[3]);
    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    //Thread and block arrangements
    int block_size = std::atoi(argv[4]);
    int num = std::atoi(argv[5]);

    //Run benchmark
    benchmark_multiplication(M, N, reps, block_size, num);
    return 0;
}
