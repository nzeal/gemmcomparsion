// gemm_serial.cpp
#include <iostream>
#include <chrono>
#include <algorithm>

// Include the header file to match the declarations
#include "../include/gemm.hpp"


void benchmark_gemm_serial(const float* A, const float* B, float* C, 
                 int M, int N, int K) {
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


