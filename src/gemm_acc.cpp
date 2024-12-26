// gemm_serial.cpp
#include <iostream>
#include <chrono>
#include <algorithm>

// Include the header file to match the declarations
#include "../include/gemm.hpp"

void benchmark_gemm_openacc(const float* A, const float* B, float* C, int M, int N, int K) {
    #pragma acc data copyin(A[0:M*K], B[0:K*N]) copyout(C[0:M*N])
    #pragma acc parallel loop collapse(2) // Parallelize both i and j loops on the GPU
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

