#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <iomanip>

//Error cheq - Very helpful!!
inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void checkCublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << ": CuBLAS error code " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

//Chunk size fixed to 32 for GPU
const int SELL_C = 32;

//struct to hold all sellcsigma related things
struct SELLCSigmaMatrix {
    int *chunk_ptr;
    int *col_idx;
    float *values;
    int num_rows;
    int num_chunks;
    int chunk_size;
    int *chunk_lengths;
    int *row_mapping;
};

//CRS assembler kernel (GPU) - kept for initial matrix assembly only
__global__ void assemble_crs(const int *row_ptr, int *col_idx, float *values, int n, float h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * n * n;
    if (idx >= total_elements) return;


    int k = idx / (n * n);
    int j = (idx % (n * n)) / n;
    int i = idx % n;


    int row_start = row_ptr[idx];
    float coeff = 1.0f / (h * h);
    int nnz = 0;


    if (k > 0) { col_idx[row_start + nnz] = idx - n * n; values[row_start + nnz] = coeff; nnz++; }
    if (j > 0) { col_idx[row_start + nnz] = idx - n; values[row_start + nnz] = coeff; nnz++; }
    if (i > 0) { col_idx[row_start + nnz] = idx - 1; values[row_start + nnz] = coeff; nnz++; }
    col_idx[row_start + nnz] = idx; values[row_start + nnz] = -6.0f * coeff; nnz++;
    if (i < n - 1) { col_idx[row_start + nnz] = idx + 1; values[row_start + nnz] = coeff; nnz++; }
    if (j < n - 1) { col_idx[row_start + nnz] = idx + n; values[row_start + nnz] = coeff; nnz++; }
    if (k < n - 1) { col_idx[row_start + nnz] = idx + n * n; values[row_start + nnz] = coeff; nnz++; }
}

//SELL-C-Sigma SpMV kernel
__global__ void spmv_sellcsigma(const int *chunk_ptr, const int *chunk_lengths, const int *col_idx, const float *values, const float *x, float *y, int num_rows, int chunk_size, const int *row_mapping, int chunks_per_block, int num_chunks){
    int thread_id = threadIdx.x;
    int warp_id   = thread_id / chunk_size;
    int local_row = thread_id % chunk_size;


    int global_chunk_id = blockIdx.x * chunks_per_block + warp_id;


    if (global_chunk_id >= num_chunks) return;


    int mapped_global_row = row_mapping[global_chunk_id * chunk_size + local_row];
    bool valid_row = (mapped_global_row >= 0 && mapped_global_row < num_rows);


    int chunk_start = chunk_ptr[global_chunk_id];
    int chunk_len   = chunk_lengths[global_chunk_id];


    float sum = 0.0f;
    for (int j = 0; j < chunk_len; ++j) {
        int idx = chunk_start + j * chunk_size + local_row;
        float v = values[idx];
        int  c = col_idx[idx];
        sum += v * x[c];
    }


    if (valid_row) {
        y[mapped_global_row] = sum;
    }
}

//Function to count number of non-zeros per row in 3D grid
void countnnz(std::vector<int>& nnz_per_row, int n) {
    int total_points = n * n * n;
    for (int idx = 0; idx < total_points; ++idx) {
        int k = idx / (n * n);
        int j = (idx % (n * n)) / n;
        int i = idx % n;
        int count = 1;
        if (i > 0) count++;
        if (i < n - 1) count++;
        if (j > 0) count++;
        if (j < n - 1) count++;
        if (k > 0) count++;
        if (k < n - 1) count++;
        nnz_per_row[idx] = count;
    }
}

// Function to convert CRS to SELL-C-Sigma (CPU)
SELLCSigmaMatrix convert_crs_to_sellcsigma( const std::vector<int>& h_row_ptr, const std::vector<int>& h_col_idx, const std::vector<float>& h_values, int num_rows, int chunk_size, std::vector<int>& nnz_per_row) {


    SELLCSigmaMatrix sell;
    sell.num_rows = num_rows;
    sell.chunk_size = chunk_size;
    sell.num_chunks = (num_rows + chunk_size - 1) / chunk_size;


    std::vector<int> h_chunk_ptr(sell.num_chunks + 1);
    std::vector<int> h_chunk_lengths(sell.num_chunks);
    std::vector<int> h_row_mapping(sell.num_chunks * sell.chunk_size);


    h_chunk_ptr[0] = 0;
    int total_sell_entries = 0;


    for (int chunk = 0; chunk < sell.num_chunks; chunk++) {
        int chunk_start_row = chunk * chunk_size;
        int chunk_end_row = std::min(chunk_start_row + chunk_size, num_rows);


        int max_row_len = 0;
        for (int row = chunk_start_row; row < chunk_end_row; row++) {
            max_row_len = std::max(max_row_len, nnz_per_row[row]);
        }


        h_chunk_lengths[chunk] = max_row_len;
        int chunk_entries = max_row_len * chunk_size;
        total_sell_entries += chunk_entries;
        h_chunk_ptr[chunk + 1] = h_chunk_ptr[chunk] + chunk_entries;
    }


    std::vector<int> h_sell_col_idx(total_sell_entries, 0);
    std::vector<float> h_sell_values(total_sell_entries, 0.0f);


    for (int chunk = 0; chunk < sell.num_chunks; chunk++) {
        int chunk_start_row = chunk * chunk_size;
        int chunk_end_row = std::min(chunk_start_row + chunk_size, num_rows);


        std::vector<int> chunk_row_indices(chunk_end_row - chunk_start_row);
        for (int i = chunk_start_row; i < chunk_end_row; i++) {
            chunk_row_indices[i - chunk_start_row] = i;
        }
        std::sort(chunk_row_indices.begin(), chunk_row_indices.end(), [&](int a, int b) {
            return nnz_per_row[a] > nnz_per_row[b];
        });


        int max_row_len = h_chunk_lengths[chunk];
        int chunk_offset = h_chunk_ptr[chunk];


        for (int local_row = 0; local_row < chunk_size; local_row++) {
            int row = (local_row < (int)chunk_row_indices.size()) ? chunk_row_indices[local_row] : -1;


            h_row_mapping[chunk * chunk_size + local_row] = row;


            int row_start = (row != -1) ? h_row_ptr[row] : 0;
            int row_len = (row != -1) ? (h_row_ptr[row + 1] - h_row_ptr[row]) : 0;


            for (int j = 0; j < max_row_len; j++) {
                int sell_idx = chunk_offset + j * chunk_size + local_row;


                if (row != -1 && j < row_len) {
                    h_sell_col_idx[sell_idx] = h_col_idx[row_start + j];
                    h_sell_values[sell_idx] = h_values[row_start + j];
                } else {
                    h_sell_col_idx[sell_idx] = 0;
                    h_sell_values[sell_idx] = 0.0f;
                }
            }
        }
    }


    cudaMalloc(&sell.chunk_ptr, (sell.num_chunks + 1) * sizeof(int));
    cudaMalloc(&sell.chunk_lengths, sell.num_chunks * sizeof(int));
    cudaMalloc(&sell.col_idx, total_sell_entries * sizeof(int));
    cudaMalloc(&sell.values, total_sell_entries * sizeof(float));
    cudaMalloc(&sell.row_mapping, sell.num_chunks * sell.chunk_size * sizeof(int));


    cudaMemcpy(sell.chunk_ptr, h_chunk_ptr.data(), (sell.num_chunks + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sell.chunk_lengths, h_chunk_lengths.data(), sell.num_chunks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sell.col_idx, h_sell_col_idx.data(), total_sell_entries * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sell.values, h_sell_values.data(), total_sell_entries * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sell.row_mapping, h_row_mapping.data(), sell.num_chunks * sell.chunk_size * sizeof(int), cudaMemcpyHostToDevice);


    return sell;
}

// Lanczos routine: expects pre-allocated buffers and a cuBLAS handle
void lanczos_sellcsigma(const SELLCSigmaMatrix& sell, int total_points, int m, float *d_q_current, float *d_w, double *alpha, double *beta, int num_threads, cublasHandle_t handle, float *d_q_prev, float *d_q_next, double *d_alpha, double *d_beta, std::vector<std::vector<float>>* Q_vectors_ptr, float* local_time, bool print_iter_times=false){

    //Starting vector initiliased on host - [1,0,0,0,0...0]
    std::vector<float> h_q_init(total_points, 0.0f);
    h_q_init[0] = 1; //Norm = 1

    //Transfer initial vector to GPU
    checkCuda(cudaMemcpy(d_q_current, h_q_init.data(), total_points * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_q_current failed");

    //Q-vectors storage
    if(Q_vectors_ptr != nullptr){
    (*Q_vectors_ptr).assign(m, std::vector<float>(total_points));
    }

    //Initialise loop
    double beta_zero = 0.0;
    checkCuda(cudaMemcpy(d_beta, &beta_zero, sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy d_beta[0] failed");
    checkCuda(cudaMemset(d_q_prev, 0, total_points * sizeof(float)), "cudaMemset d_q_prev failed");

    //SpMV kernel config.
    int threads_per_block = num_threads;
    int chunks_per_block = (threads_per_block / SELL_C);
    if (chunks_per_block <= 0) chunks_per_block = 1;
    int num_blocks = (sell.num_chunks + chunks_per_block - 1)/chunks_per_block;

    // Per-iteration timing events
    cudaEvent_t it_start, it_stop;
    checkCuda(cudaEventCreate(&it_start), "cudaEventCreate it_start failed");
    checkCuda(cudaEventCreate(&it_stop), "cudaEventCreate it_stop failed");

    //Run timer
    double run_time = 0.0;

    //  LANCZOS LOOP  //
    for(int k = 0; k < m; ++k)
    {
        //Storing current Q_k
        if (Q_vectors_ptr != nullptr){
        checkCuda(cudaMemcpy((*Q_vectors_ptr)[k].data(), d_q_current, total_points * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy store Q_vectors failed");        
        }

        // record iteration start
        checkCuda(cudaEventRecord(it_start), "cudaEventRecord it_start failed");

        //Step 1 : w = A * q_current
        spmv_sellcsigma<<<num_blocks, threads_per_block>>>(sell.chunk_ptr, sell.chunk_lengths, sell.col_idx, sell.values, d_q_current,
        d_w, total_points, SELL_C, sell.row_mapping, chunks_per_block, sell.num_chunks);
        checkCuda(cudaGetLastError(), "spmv_sellcsigma kernel launch failed");

        //Step 2 : w = w - beta[k] * q_prev
        if(k > 0){
            double beta_k;
            checkCuda(cudaMemcpy(&beta_k, d_beta + k, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy beta_k failed");
            float beta_k_f = static_cast<float>(-beta_k);
            checkCublas(cublasSaxpy(handle, total_points, &beta_k_f, d_q_prev, 1, d_w, 1), "cublasSaxpy (beta*q_prev) failed");
        }

        //Step 3 : alpha[k] = q_current^T * w
        float alpha_k_f;
        checkCublas(cublasSdot(handle, total_points, d_q_current, 1, d_w, 1, &alpha_k_f), "cublasSdot (alpha) failed");
        double alpha_k = static_cast<double>(alpha_k_f);
        checkCuda(cudaMemcpy(d_alpha + k, &alpha_k, sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy d_alpha failed");

        //Step 4 : w = w - alpha[k] * q_current
        float neg_alpha_k = static_cast<float>(-alpha_k);
        checkCublas(cublasSaxpy(handle, total_points, &neg_alpha_k, d_q_current, 1, d_w, 1), "cublasSaxpy (alpha*q_curr) failed");

        //Step 5.1 : Reorthogonalisation against current Q
        float dot_curr;
        checkCublas(cublasSdot(handle, total_points, d_q_current, 1, d_w, 1, &dot_curr), "cublasSdot (dot_curr) failed");
        float neg_dot_curr = -dot_curr;
        checkCublas(cublasSaxpy(handle, total_points, &neg_dot_curr, d_q_current, 1, d_w, 1), "cublasSaxpy (reorth curr) failed");
        //Step 5.2 : Reorthogonalisation against previous Q
        if (k > 0) {
            float dot_prev;
            checkCublas(cublasSdot(handle, total_points, d_q_prev, 1, d_w, 1, &dot_prev), "cublasSdot (dot_prev) failed");
            float neg_dot_prev = -dot_prev;
            checkCublas(cublasSaxpy(handle, total_points, &neg_dot_prev, d_q_prev, 1, d_w, 1), "cublasSaxpy (reorth prev) failed");}

        //Step 6 : betaa[k+1] = norm2(w)
        float beta_norm_f;
        checkCublas(cublasSnrm2(handle, total_points, d_w, 1, &beta_norm_f), "cublasSnrm2 failed");
        double beta_norm = static_cast<double>(beta_norm_f);

        //Convergence check
        if(beta_norm < 1e-14)
        {
            std::cout << "Lanczos breakdown at iteration " << k << "; Beta norm is " << beta_norm << std::endl;
            checkCuda(cudaMemcpy(d_beta + k + 1, &beta_norm, sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy d_beta failed");
            // record iteration end before breaking
            checkCuda(cudaEventRecord(it_stop), "cudaEventRecord it_stop failed");
            checkCuda(cudaEventSynchronize(it_stop), "cudaEventSynchronize it_stop failed");
            float it_ms; cudaEventElapsedTime(&it_ms, it_start, it_stop);
            if (print_iter_times) std::cout << "Iteration " << k << " time: " << it_ms << " ms\n";
            break; //End loop
        }
        checkCuda(cudaMemcpy(d_beta + k + 1, &beta_norm, sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy d_beta failed");

        //Step 7 : Prepare for next iteration
        float inv_beta = static_cast<float>(1.0 / beta_norm);
        checkCuda(cudaMemcpy(d_q_next, d_w, total_points * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy d_q_next failed");
        checkCublas(cublasSscal(handle, total_points, &inv_beta, d_q_next, 1), "cublasSscal failed");
        float *temp = d_q_prev;
        d_q_prev = d_q_current;
        d_q_current = d_q_next;
        d_q_next = temp;

        // record iteration end
        checkCuda(cudaEventRecord(it_stop), "cudaEventRecord it_stop failed");
        checkCuda(cudaEventSynchronize(it_stop), "cudaEventSynchronize it_stop failed");
        float it_ms; cudaEventElapsedTime(&it_ms, it_start, it_stop);
        run_time += it_ms;
        if (print_iter_times) std::cout << "Iteration " << k << " time: " << std::fixed << std::setprecision(6) << it_ms << " ms\n";
    }

    //Time for this instance of lanczos
    *local_time = run_time;

    //Transfer back results
    checkCuda(cudaMemcpy(alpha, d_alpha, m * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy alpha host failed");
    checkCuda(cudaMemcpy(beta, d_beta, (m + 1) * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy beta host failed");

    checkCuda(cudaEventDestroy(it_start), "cudaEventDestroy it_start failed");
    checkCuda(cudaEventDestroy(it_stop), "cudaEventDestroy it_stop failed");
}

// Verification function: Compute A*Q and Q*T and compare
void verify_lanczos(const SELLCSigmaMatrix& sell, const std::vector<std::vector<float>>& Q_vectors, const double *alpha, const double *beta, int num_rows, int m, int num_threads) {
    //std::cout << "\n=== Lanczos Verification: A*Q vs Q*T ===\n";
    float *d_q, *d_aq;
    cudaMalloc(&d_q, num_rows * sizeof(float));
    cudaMalloc(&d_aq, num_rows * sizeof(float));

    std::vector<float> h_aq(num_rows);
    std::vector<float> h_qt(num_rows);

    int threads_per_block = num_threads;
    int chunks_per_block = threads_per_block / SELL_C;
    int num_blocks = (sell.num_chunks + chunks_per_block - 1) / chunks_per_block;

    double max_error = 0.0;
    double avg_error = 0.0;

    for (int j = 0; j < m-1; j++) {
        // Compute A * q_j
        cudaMemcpy(d_q, Q_vectors[j].data(), num_rows * sizeof(float), cudaMemcpyHostToDevice);
        
        spmv_sellcsigma<<<num_blocks, threads_per_block>>>(
            sell.chunk_ptr, sell.chunk_lengths, sell.col_idx, sell.values,
            d_q, d_aq, num_rows, SELL_C, sell.row_mapping, chunks_per_block, sell.num_chunks);
        cudaDeviceSynchronize();

        cudaMemcpy(h_aq.data(), d_aq, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

        // Compute Q*T column j: T[i,j] = beta[i] if i=j-1, alpha[i] if i=j, beta[i+1] if i=j+1
        // Q*T[:,j] = beta[j]*Q[:,j-1] + alpha[j]*Q[:,j] + beta[j+1]*Q[:,j+1]
        std::fill(h_qt.begin(), h_qt.end(), 0.0f);

        // beta[j] * q_{j-1}
        if (j > 0) {
            for (int i = 0; i < num_rows; i++) {
                h_qt[i] += beta[j] * Q_vectors[j-1][i];
            }
        }

        // alpha[j] * q_j
        for (int i = 0; i < num_rows; i++) {
            h_qt[i] += alpha[j] * Q_vectors[j][i];
        }

        // beta[j+1] * q_{j+1}
        if (j + 1 < m) {
            for (int i = 0; i < num_rows; i++) {
                h_qt[i] += beta[j+1] * Q_vectors[j+1][i];
            }
        }
        // Compare A*q_j with Q*T[:,j]
        double col_error = 0.0;
        for (int i = 0; i < num_rows; i++) {
            double diff = fabs(h_aq[i] - h_qt[i]);
            col_error += diff * diff;
        }
        col_error = sqrt(col_error);
        max_error = std::max(max_error, col_error);
        avg_error += col_error;
        //std::cout << "Column " << j << " error (||A*q_j - Q*T[:,j]||): " << col_error << "\n";
        
    }
    avg_error /= m;
    std::cout << "\nMax column error: " << max_error << "\n";
    std::cout << "Average column error: " << avg_error << "\n";
    cudaFree(d_q);
    cudaFree(d_aq);
}

//Main Lanczos test function
int run_lanczos_test(int n, int num_threads, int lanczos_iterations, int num_runs) {
    int num_rows = n * n * n;
    float h = 1.0f / (n + 1);

    //For time measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Build CRS matrix structure for initial assembly
    cudaEventRecord(start);
    //Count number of non-zeros per row
    std::vector<int> nnz_per_row(num_rows);
    countnnz(nnz_per_row, n);

    //Reduce over the former to get pointers
    std::vector<int> h_row_ptr(num_rows + 1);
    h_row_ptr[0] = 0;
    std::partial_sum(nnz_per_row.begin(), nnz_per_row.end(), h_row_ptr.begin() + 1);
    int total_nnz = h_row_ptr[num_rows];

    //Allocate device arrays to assemble crs into
    int *d_row_ptr, *d_col_idx;
    float *d_values;
    cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, total_nnz * sizeof(int));
    cudaMalloc(&d_values, total_nnz * sizeof(float));
    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    //Launch kernel and record time elapsed
    int blocks = (num_rows + num_threads - 1) / num_threads;
    assemble_crs<<<blocks, num_threads>>>(d_row_ptr, d_col_idx, d_values, n, h);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float assembly_time;
    cudaEventElapsedTime(&assembly_time, start, stop);
    std::cout << "CRS assembly kernel time: " << assembly_time << " ms\n";

    //Copy result into host memory for conversion
    std::vector<int> h_col_idx(total_nnz);
    std::vector<float> h_values(total_nnz);
    cudaMemcpy(h_col_idx.data(), d_col_idx, total_nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values.data(), d_values, total_nnz * sizeof(float), cudaMemcpyDeviceToHost);

    //Clean up device arrays - No longer needed
    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);
    cudaFree(d_values);
    
    //Convert to SELL-C-Sigma format
    cudaEventRecord(start);
    SELLCSigmaMatrix sell = convert_crs_to_sellcsigma(h_row_ptr, h_col_idx, h_values, num_rows, SELL_C, nnz_per_row);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float conversion_time;
    cudaEventElapsedTime(&conversion_time, start, stop);
    std::cout << "SCS conversion time: " << conversion_time << " ms\n";

    // ================= L A N C Z O S  M E T H O D ====================                                                   
    //Allocate device memory for Lanczos vectors
    float *d_q_current, *d_w;
    checkCuda(cudaMalloc(&d_q_current, num_rows * sizeof(float)), "cudaMalloc d_q_current failed");
    checkCuda(cudaMalloc(&d_w, num_rows * sizeof(float)), "cudaMalloc d_w failed");

    //Previous/next vectors and tri-diagonal arrays
    float *d_q_prev, *d_q_next;
    double *d_alpha, *d_beta;
    checkCuda(cudaMalloc(&d_q_prev, num_rows * sizeof(float)), "cudaMalloc d_q_prev failed");
    checkCuda(cudaMalloc(&d_q_next, num_rows * sizeof(float)), "cudaMalloc d_q_next failed");
    checkCuda(cudaMalloc(&d_alpha, lanczos_iterations * sizeof(double)), "cudaMalloc d_alpha failed");
    checkCuda(cudaMalloc(&d_beta, (lanczos_iterations + 1) * sizeof(double)), "cudaMalloc d_beta failed");

    //To re-assemble tridiagonal matrix (host)
    std::vector<double> alpha(lanczos_iterations, 0.0);
    std::vector<double> beta(lanczos_iterations + 1, 0.0);

    // Prepare host storage for Q vectors (for verification)
    std::vector<std::vector<float>>* Q_vectors_ptr = nullptr;
    if (Q_vectors_ptr != nullptr) {
        Q_vectors_ptr = new std::vector<std::vector<float>>(
            lanczos_iterations, std::vector<float>(num_rows));
    }

    //Create cuBLAS handle
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate failed");

    //Testing loop
    double sell_total_time = 0.0;
    for(int run = 0; run < num_runs; run++) {
        //Reset for this instance (host arrays)
        std::fill(alpha.begin(), alpha.end(), 0.0);
        std::fill(beta.begin(), beta.end(), 0.0);
        float local_time;

        //Record: run Lanczos (per-iteration timings printed inside)
        lanczos_sellcsigma(sell, num_rows, lanczos_iterations, d_q_current, d_w, alpha.data(), beta.data(), num_threads,
            handle, d_q_prev, d_q_next, d_alpha, d_beta, Q_vectors_ptr, &local_time, /*print_iter_times=*/ false);

        sell_total_time += local_time;
    }

    //Timing for this run
    float sell_avg_time = sell_total_time / num_runs;
    std::cout << "Average SELL-C-Sigma Lanczos time: " << sell_avg_time << " ms (" << num_runs << " runs)\n";
    std::cout << "Average time per iterations: " << sell_avg_time / lanczos_iterations << "ms (" << lanczos_iterations << " iterations)\n";

    //Print tridiagonal elements
    //for(int i = 0; i < std::min(10, lanczos_iterations); i++) {
    //    printf("%5d | %12.6f | %12.6f\n", i, alpha[i], beta[i]);
    //}

    // VERIFICATION: Check if A*Q = Q*T
    if (Q_vectors_ptr != nullptr){
    verify_lanczos(sell, *Q_vectors_ptr, alpha.data(), beta.data(), 
    num_rows, lanczos_iterations, num_threads);}

    //Cleanup
    cudaFree(d_q_current);
    cudaFree(d_w);
    cudaFree(d_q_prev);
    cudaFree(d_q_next);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cublasDestroy(handle);
    cudaFree(sell.chunk_ptr);
    cudaFree(sell.chunk_lengths);
    cudaFree(sell.col_idx);
    cudaFree(sell.values);
    cudaFree(sell.row_mapping);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <grid_size> <num_threads> <num_iterations> <num_runs>\n";
        return 1;
    }

    int n = std::atoi(argv[1]);
    int num_threads = std::atoi(argv[2]);
    int lanczos_iterations = std::atoi(argv[3]);
    int num_runs = std::atoi(argv[4]);

    return run_lanczos_test(n, num_threads, lanczos_iterations, num_runs);
}