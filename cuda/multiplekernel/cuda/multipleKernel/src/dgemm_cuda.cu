#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

#include "../includes/error_checking.h"
#include "../includes/performance_result.h"
#include "../includes/performance_utils.h"
#include "../includes/matrix_mul_kernel.h"

void runDGEMM(int size, std::vector<PerformanceResult>& results) {
    const int m = size;
    const int k = size;
    const int n = size;

    size_t size_A = m * k * sizeof(double);
    size_t size_B = k * n * sizeof(double);
    size_t size_C = m * n * sizeof(double);
    size_t total_memory = (size_A + size_B + size_C) / (1024.0 * 1024.0);

    printf("\nMatrix size: %d x %d (%zu MB)\n", size, size, total_memory);

    // Allocate host memory
    double *h_A = (double*)malloc(size_A);
    double *h_B = (double*)malloc(size_B);
    double *h_C = (double*)malloc(size_C);

    // Initialize matrices
    for(int i = 0; i < m * k; i++) h_A[i] = 1.0;
    for(int i = 0; i < k * n; i++) h_B[i] = 2.0;
    for(int i = 0; i < m * n; i++) h_C[i] = 0.0;

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    PerformanceResult result;
    result.size = size;

    // Transfer data to device
    cudaDeviceSynchronize();
    auto transfer_start = getCurrentTime();
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    auto transfer_end = getCurrentTime();

    result.transfer_to_device_time = calculateDurationInSeconds(transfer_start, transfer_end);
    result.bandwidth_to_device = ((size_A + size_B) / (1024.0 * 1024.0 * 1024.0)) / result.transfer_to_device_time;

    printf("H2D Transfer: %.2f ms (%.2f GB/s)\n",
           result.transfer_to_device_time * 1000,
           result.bandwidth_to_device);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16); // 16x16 threads per block
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y); // Number of blocks

    // Launch the first matrix multiplication kernel (Kernel 1)
    auto compute_start1 = getCurrentTime();
    matrixMulKernel1<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
    auto compute_end1 = getCurrentTime();

    result.computation_time_kernel1 = calculateDurationInSeconds(compute_start1, compute_end1);
    double flops_kernel1 = 2.0 * m * n * k;
    result.gflops_kernel1 = (result.computation_time_kernel1 > 0) ? (flops_kernel1 / (result.computation_time_kernel1 * 1e9)) : 0.0;

    printf("Kernel 1 Computation: %.2f ms (%.2f GFLOPS)\n",
           result.computation_time_kernel1 * 1000,
           result.gflops_kernel1);

    // Transfer result back to host after Kernel 1 computation
    cudaDeviceSynchronize();
    auto transfer_back_start1 = getCurrentTime();
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    auto transfer_back_end1 = getCurrentTime();

    result.transfer_from_device_time_kernel1 = calculateDurationInSeconds(transfer_back_start1, transfer_back_end1);
    result.bandwidth_from_device_kernel1 = (size_C / (1024.0 * 1024.0 * 1024.0)) / result.transfer_from_device_time_kernel1;

    printf("D2H Transfer Kernel 1: %.2f ms (%.2f GB/s)\n", result.transfer_from_device_time_kernel1 * 1000, result.bandwidth_from_device_kernel1);

    // Launch the second matrix multiplication kernel (Kernel 2)
    auto compute_start2 = getCurrentTime();
    matrixMulKernel2<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
    auto compute_end2 = getCurrentTime();

    result.computation_time_kernel2 = calculateDurationInSeconds(compute_start2, compute_end2);
    double flops_kernel2 = 2.0 * m * n * k;
    result.gflops_kernel2 = (result.computation_time_kernel2 > 0) ? (flops_kernel2 / (result.computation_time_kernel2 * 1e9)) : 0.0;

    printf("Kernel 2 Computation: %.2f ms (%.2f GFLOPS)\n",
           result.computation_time_kernel2 * 1000,
           result.gflops_kernel2);

    // Transfer result back to host after Kernel 2 computation
    cudaDeviceSynchronize();
    auto transfer_back_start2 = getCurrentTime();
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    auto transfer_back_end2 = getCurrentTime();

    result.transfer_from_device_time_kernel2 = calculateDurationInSeconds(transfer_back_start2, transfer_back_end2);
    result.bandwidth_from_device_kernel2 = (size_C / (1024.0 * 1024.0 * 1024.0)) / result.transfer_from_device_time_kernel2;

    printf("D2H Transfer Kernel 2: %.2f ms (%.2f GB/s)\n", result.transfer_from_device_time_kernel2 * 1000, result.bandwidth_from_device_kernel2);

    // Result verification
    double expected_value = k * 2.0;
    if (std::abs(h_C[0] - expected_value) > 1e-5 ||
        std::abs(h_C[m * n - 1] - expected_value) > 1e-5) {
        printf("Verification FAILED: Expected %.2f, Got %.2f (first) %.2f (last)\n",
               expected_value, h_C[0], h_C[m * n - 1]);
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
    printf("\n=== Performance Summary ===\n");
    printf("%8s %12s %15s %15s %15s %15s %15s\n",
           "Size", "GFLOPS1", "Compute1(ms)", "H2D BW1(GB/s)", "D2H BW1(GB/s)", "GFLOPS2", "Compute2(ms)");
    printf("%s\n", std::string(90, '-').c_str());

    for (const auto& result : results) {
        printf("%8d %12.2f %15.2f %15.2f %15.2f %12.2f %15.2f\n",
               result.size,
               result.gflops_kernel1,
               result.computation_time_kernel1 * 1000,
               result.bandwidth_to_device,
               result.bandwidth_from_device_kernel1,
               result.gflops_kernel2,
               result.computation_time_kernel2 * 1000);
    }
}

int main() {
    std::vector<int> sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    std::vector<PerformanceResult> results;

    // Print device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=== CUDA Device Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Global Memory: %.2f GB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Run tests
    for (int size : sizes) {
        try {
            runDGEMM(size, results);
        } catch (const std::exception& e) {
            std::cerr << "Error testing size " << size << ": " << e.what() << std::endl;
            continue;
        }
    }

    // Print final summary
    printSummary(results);

    return 0;
}

