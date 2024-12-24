#include "include/gemm.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include "include/utility.cuh"
#include "include/utility.h"

// Declaration of the kernel launch function from gemm_cuda_kernel.cu
extern "C" void launchGemmKernel(const float* A, const float* B, float* C, int M, int N, int K);

void benchmark_gemm_cuda(const float* h_A, const float* h_B, float* h_C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate GPU memory
    checkCudaErrors(cudaMalloc(&d_A, size_A));
    checkCudaErrors(cudaMalloc(&d_B, size_B));
    checkCudaErrors(cudaMalloc(&d_C, size_C));

    // Copy data from host to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Launch the GEMM kernel
    launchGemmKernel(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();  // Wait for kernel to complete

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));


    // Free GPU memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
}

