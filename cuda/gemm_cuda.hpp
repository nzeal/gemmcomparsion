// gemm.hpp

#ifndef GEMM_HPP
#define GEMM_HPP

#include <cstddef>  // for size_t

#ifdef __CUDACC__  // Check if CUDA is available
#include <cuda_runtime.h>
#endif


// Declare function prototypes
template <typename T>
void allocateMemory(T** A, T** B, T** C, size_t size);

void benchmark_gemm_serial(const float* A, const float* B, float* C,
                           int M, int N, int K);

void benchmark_gemm_openmp(const float* A, const float* B, float* C,
                           int M, int N, int K);

void benchmark_gemm_openacc(const float* A, const float* B, float* C,
                           int M, int N, int K);

template <typename T>
void freeMemory(T* A, T* B, T* C);


// Add CUDA-specific functions
void allocateGPUmemory(float*& d_A, float*& d_B, float*& d_C,
                      size_t size_A, size_t size_B, size_t size_C,
                      const float* h_A, const float* h_B);

void freeGPUmemory(float* d_A, float* d_B, float* d_C);

void benchmark_gemm_cuda(const float* A, const float* B, float* C,
                        int M, int N, int K);

#endif // GEMM_HPP

