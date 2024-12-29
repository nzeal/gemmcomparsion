#include <cuda_runtime.h>
#include <iostream>
#include "../include/aid-cuda.cuh"  // Include the error-checking header

__global__ void cuda_gemm_kernel_optimized(const float* A, const float* B, float* C,
                                         int M, int N, int K) {
    const int BLOCK_SIZE = 32;
    const int THREAD_ITEMS = 4;  // Each thread computes 4 elements
    
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Compute base indices
    int row = by * BLOCK_SIZE + ty;
    int col = bx * (BLOCK_SIZE * THREAD_ITEMS) + tx;
    
    float sum[THREAD_ITEMS] = {0.0f};
    
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // Collaborative loading of A and B into shared memory
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i += 16) {
            if (row < M && (k + tx + i) < K)
                As[ty][tx + i] = A[row * K + (k + tx + i)];
        }
        
        #pragma unroll
        for (int i = 0; i < THREAD_ITEMS; i++) {
            if ((k + ty) < K && (col + i * BLOCK_SIZE) < N)
                Bs[ty][tx + i * 16] = B[(k + ty) * N + (col + i * BLOCK_SIZE)];
        }
        
        __syncthreads();
        
        // Compute partial results
        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE; ++kk) {
            #pragma unroll
            for (int t = 0; t < THREAD_ITEMS; ++t) {
                sum[t] += As[ty][kk] * Bs[kk][tx + t * 16];
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int t = 0; t < THREAD_ITEMS; ++t) {
        if (row < M && (col + t * BLOCK_SIZE) < N) {
            C[row * N + (col + t * BLOCK_SIZE)] = sum[t];
        }
    }
}

// C wrapper functions for the kernel launch
extern "C" {
    void launchGemmKernel(const float* A, const float* B, float* C, int M, int N, int K) {
        const int BLOCK_SIZE = 32;
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        cuda_gemm_kernel_optimized<<<dimGrid, dimBlock>>>(A, B, C, M, N, K);
        
	// Check for errors after kernel launch
        chkErr(cudaGetLastError());  // Check if kernel launch succeeded
        chkErr(cudaDeviceSynchronize());  // Synchronize and check for any runtime errors
    }
}

