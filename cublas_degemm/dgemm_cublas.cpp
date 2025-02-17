#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "performance.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

// Error checking macro
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error at %s %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct PerformanceResult {
    int size;
    double transfer_to_device_time;
    double computation_time;
    double transfer_from_device_time;
    double gflops;
    double bandwidth_to_device;
    double bandwidth_from_device;
};

void runDGEMM(cublasHandle_t handle, int size, std::vector<PerformanceResult>& results) {
    const int m = size;  // rows of A and C
    const int k = size;  // cols of A and rows of B
    const int n = size;  // cols of B and C
    
    // Calculate sizes and number of operations
    size_t size_A = m * k * sizeof(double);
    size_t size_B = k * n * sizeof(double);
    size_t size_C = m * n * sizeof(double);
    size_t total_data_size = size_A + size_B + size_C;
    size_t num_operations = 2ULL * m * n * k;
    
    std::cout << "\nMatrix size: " << size << "x" << size << "\n";
    std::cout << "Memory required: " << std::fixed << std::setprecision(2) 
              << (total_data_size / (1024.0 * 1024.0)) << " MB\n";
    
    // Host matrices
    double *h_A = (double*)malloc(size_A);
    double *h_B = (double*)malloc(size_B);
    double *h_C = (double*)malloc(size_C);
    
    // Initialize matrices
    for(int i = 0; i < m * k; i++) h_A[i] = 1.0;
    for(int i = 0; i < k * n; i++) h_B[i] = 2.0;
    for(int i = 0; i < m * n; i++) h_C[i] = 0.0;
    
    // Device matrices
    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    
    PerformanceResult result;
    result.size = size;
    
    // Memory transfer timing
    auto transfer_start = getCurrentTime();
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    auto transfer_end = getCurrentTime();
    
    result.transfer_to_device_time = calculateDurationInSeconds(transfer_start, transfer_end);
    result.bandwidth_to_device = ((size_A + size_B) / (1024.0 * 1024.0 * 1024.0)) / result.transfer_to_device_time;
    
    std::cout << "Memory Transfer (Host to Device):\n";
    computeAndPrintMetrics(transfer_start, transfer_end, size_A + size_B, 0);
    
    // DGEMM parameters
    double alpha = 1.0;
    double beta = 0.0;
    
    // Synchronize and time computation
    cudaDeviceSynchronize();
    auto compute_start = getCurrentTime();
    
    CHECK_CUBLAS(cublasDgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha,
        d_A, m,
        d_B, k,
        &beta,
        d_C, m
    ));
    
    cudaDeviceSynchronize();
    auto compute_end = getCurrentTime();
    
    result.computation_time = calculateDurationInSeconds(compute_start, compute_end);
    result.gflops = static_cast<double>(num_operations) / (result.computation_time * 1e9);
    
    std::cout << "DGEMM Computation:\n";
    computeAndPrintMetrics(compute_start, compute_end, 0, num_operations);
    
    // Transfer result back timing
    auto transfer_back_start = getCurrentTime();
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    auto transfer_back_end = getCurrentTime();
    
    result.transfer_from_device_time = calculateDurationInSeconds(transfer_back_start, transfer_back_end);
    result.bandwidth_from_device = (size_C / (1024.0 * 1024.0 * 1024.0)) / result.transfer_from_device_time;
    
    std::cout << "Memory Transfer (Device to Host):\n";
    computeAndPrintMetrics(transfer_back_start, transfer_back_end, size_C, 0);
    
    // Verify result (check first and last elements)
    double expected_value = k * 2.0; // Each element should be k * 1.0 * 2.0
    if (std::abs(h_C[0] - expected_value) > 1e-5 || 
        std::abs(h_C[m*n-1] - expected_value) > 1e-5) {
        std::cout << "Result verification failed!\n";
        std::cout << "Expected: " << expected_value << "\n";
        std::cout << "Got: First=" << h_C[0] << ", Last=" << h_C[m*n-1] << "\n";
    }
    
    results.push_back(result);
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
}

void printSummary(const std::vector<PerformanceResult>& results) {
    std::cout << "\n=== Performance Summary ===\n";
    std::cout << std::setw(8) << "Size" 
              << std::setw(12) << "GFLOPS"
              << std::setw(15) << "Compute(ms)"
              << std::setw(15) << "H2D BW(GB/s)"
              << std::setw(15) << "D2H BW(GB/s)"
              << "\n";
    std::cout << std::string(65, '-') << "\n";
    
    for (const auto& result : results) {
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(8) << result.size
                  << std::setw(12) << result.gflops
                  << std::setw(15) << result.computation_time * 1000
                  << std::setw(15) << result.bandwidth_to_device
                  << std::setw(15) << result.bandwidth_from_device
                  << "\n";
    }
}

int main() {
    std::vector<int> sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    std::vector<PerformanceResult> results;
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // Print CUDA device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "=== CUDA Device Information ===\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB\n\n";
    
    // Run tests for each matrix size
    for(int size : sizes) {
        try {
            runDGEMM(handle, size, results);
        } catch (const std::exception& e) {
            std::cerr << "Error testing size " << size << ": " << e.what() << std::endl;
            continue;
        }
    }
    
    // Print performance summary
    printSummary(results);
    
    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    
    return 0;
}
