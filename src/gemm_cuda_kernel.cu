#include <cuda_runtime.h>
#include <iostream>

__global__ void cuda_gemm_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    const int BLOCK_SIZE = 32;
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles of A and B
    for (int k = 0; k < (K + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE; k += BLOCK_SIZE) {
        // Load data into shared memory
        if (row < M && k + tx < K)
            As[ty][tx] = A[row * K + k + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && k + ty < K)
            Bs[ty][tx] = B[(k + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        // Synchronize threads within the block
        __syncthreads();

        // Perform the multiplication
        for (int kk = 0; kk < BLOCK_SIZE; ++kk) {
            sum += As[ty][kk] * Bs[kk][tx];
        }

        // Synchronize threads within the block
        __syncthreads();
    }

    // Store the result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// C wrapper functions for the kernel launch
extern "C" {
    void launchGemmKernel(const float* A, const float* B, float* C, int M, int N, int K) {
        const int BLOCK_SIZE = 32;
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        cuda_gemm_kernel<<<dimGrid, dimBlock>>>(A, B, C, M, N, K);
        cudaDeviceSynchronize();
    }
}

