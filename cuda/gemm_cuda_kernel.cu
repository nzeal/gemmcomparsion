#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel for matrix multiplication
__global__ void gemmKernel(const float* A, const float* B, float* C, 
                          const int M, const int N, const int K) {
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread is within bounds
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Each thread computes one element of C
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

// Helper function to check for CUDA errors
inline void checkCuda(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d: %s\n",
                file, line, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA(x) checkCuda(x, __FILE__, __LINE__)

// This is the wrapper function that's called from gemm_cuda.cpp
extern "C" void launchGemmKernel(const float* A, const float* B, float* C, 
                                int M, int N, int K) {
    // Define block and grid dimensions
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y
    );
    
    // Launch kernel
    gemmKernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}