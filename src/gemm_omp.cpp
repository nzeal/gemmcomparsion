// gemm_serial.cpp
#include <iostream>
#include <chrono>
#include <algorithm>

// Include the header file to match the declarations
#include "../include/gemm.hpp"

#define BLOCK_SIZE 32

void benchmark_gemm_openmp(const float* A, const float* B, float* C, 
                 int M, int N, int K) {
      #pragma omp target data map(to:A[0:M*K], B[0:K*N]) map(tofrom:C[0:M*N])
    {
        #pragma omp target teams distribute collapse(2) thread_limit(BLOCK_SIZE*BLOCK_SIZE)
        for (int by = 0; by < M; by += BLOCK_SIZE) {
            for (int bx = 0; bx < N; bx += BLOCK_SIZE) {
                #pragma omp parallel for collapse(2)
                for (int ty = 0; ty < BLOCK_SIZE; ty++) {
                    for (int tx = 0; tx < BLOCK_SIZE; tx++) {
                        int row = by + ty;
                        int col = bx + tx;
                        
                        if (row < M && col < N) {
                            float sum = 0.0f;
                            #pragma omp simd reduction(+:sum)
                            for (int k = 0; k < K; k++) {
                                sum += A[row * K + k] * B[k * N + col];
                            }
                            C[row * N + col] = sum;
                        }
                    }
                }
            }
        }
    }
}	


