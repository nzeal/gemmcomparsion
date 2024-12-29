// gemm.hpp

#ifndef GEMM_HPP
#define GEMM_HPP

#include <cstddef>  // for size_t

// Declare function prototypes
template <typename T>
void allocateMemory(T** A, T** B, T** C, size_t size);

void benchmark_gemm_serial(const float* A, const float* B, float* C,
                           int M, int N, int K);

void benchmark_gemm_sycl(const float* A, const float* B, float* C,
                           int M, int N, int K);

template <typename T>
void freeMemory(T* A, T* B, T* C);

#endif // GEMM_HPP

