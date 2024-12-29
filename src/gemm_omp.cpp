// gemm_serial.cpp
#include <iostream>
#include <chrono>
#include <algorithm>

// Include the header file to match the declarations
#include "../include/gemm.hpp"

void benchmark_gemm_openmp(const float* restrict A, const float* restrict B, float* restrict C,
              int M, int N, int K) {
    #pragma omp target teams distribute parallel for collapse(2) \
            map(to:A[0:M*K], B[0:K*N]) map(from:C[0:M*N])
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/*
void benchmark_gemm_openmp(const float* A, const float* B, float* C, int M, int N, int K) {
    #pragma omp target data map(to: A[0:M*K], B[0:K*N]) map(from: C[0:M*N])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
}
*/
