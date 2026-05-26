#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>
#include <iostream>
#include <fstream>
#include "cublas_v2.h"

const int block_size = 64;
const int chunk_size = 1;

__global__ void set_vector(const int N, const float val, float *x)
{
  const int idx_base = threadIdx.x + blockIdx.x * (blockDim.x * chunk_size);
  for (unsigned int i = 0; i < chunk_size; ++i)
    {
      const int idx = idx_base + i * block_size;
      if (idx < N)
        x[idx] = val;
    }
}

//Sets an M*N => M==N hilbert matrix for testing
__global__ void set_hilbert_matrix(float *mat, int N, int M)
{
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

void wtf(const std::vector<float>& data, const std::string& filename)
{
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
void benchmark_multiplication(const std::size_t M, const std::size_t N, const long long repeat)
{
  float *cu_vec, *cu_mat, *cu_out; 
  
  //Allocate memory on the device
  cudaMalloc(&cu_vec, N * sizeof(float));
  cudaMalloc(&cu_mat, M * N * sizeof(float));
  cudaMalloc(&cu_out, M * sizeof(float));  

  //Number of blocks based on block_size
  const unsigned int n_blocks = (N + block_size - 1) / block_size;

  //Set vector and matrix
  set_vector<<<n_blocks, block_size>>>(N, 1.0, cu_vec);
  set_hilbert_matrix<<<n_blocks, block_size>>>(cu_mat, N, M);
  set_vector<<<n_blocks, block_size>>>(N, 0.0, cu_out);

  //For results post-processing
  std::vector<float> result_host(M);

  //Run a lot of tests
  const unsigned int n_tests = 20; //20 outer loops
  const unsigned long long int n_repeat = repeat > 0 ? repeat : std::max(1UL, 100000000U / N); //Repeat value fix
  double best = 1e10, worst = 0, avg = 0; //For storing results
  
  //Setup cuBLAS
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
      std::cout << "CUBLAS initialization failed\n";
      std::abort();
  }
  float alpha = 1.f;
  float beta = 0.;
  
  //Run 20 tests  
  for (unsigned int t = 0; t < n_tests; ++t)
    {
     //Begin measuring time   
     const auto t1 = std::chrono::steady_clock::now();
        
     //Run many repetitions
     for (unsigned int rep = 0; rep < n_repeat; ++rep) 
     {
       //Run kernel
       cublasSgemv(handle, CUBLAS_OP_T, M, N, &alpha, cu_mat, M, cu_vec, 1, &beta, cu_out, 1); 
       //Best to do before measuring time
       cudaDeviceSynchronize();
     
     };  

      
     //End time measurement 
     const double time = std::chrono::duration_cast<std::chrono::duration<double>>( std::chrono::steady_clock::now() - t1).count();
    
     //Store time measurement  
     best  = std::min(best, time / n_repeat);
     worst = std::max(worst, time / n_repeat);
     avg += time / n_repeat;
   }
   //Bandwidth results
   std::cout << N << " elements  " << best << "s  " << avg/n_tests << "s  " << worst <<"s  " << ((1e-9 * (M * N + M + N) * sizeof(float)) / best) << "GB/s  " <<std::endl;

   //Copy the result back to the host
   cudaMemcpy(result_host.data(), cu_out, M * sizeof(float), cudaMemcpyDeviceToHost);

   // Free the memory on the device
   cudaFree(cu_vec);
   cudaFree(cu_mat);
   cudaFree(cu_out); 

   //Output array to file for correctness test
   wtf(result_host, "result.txt");
}

int main(int argc, char **argv)
{
    //Check inputs
    if (argc != 4)
    {
        std::cout << "Expected 3 arguments, got" << argc << std::endl;
        std::cout << "Format :" << std::atoi(argv[0]) << "M N reps" << std::endl;
        return 1;
    };
    
    //Number of repetitions and size of problem
    int reps = std::atoi(argv[3]);
    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);

    //Run benchmark
    benchmark_multiplication(M, N, reps);
    return 0;
}
