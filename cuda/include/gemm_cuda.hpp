// gemm.hpp

#ifndef GEMM_HPP
#define GEMM_HPP

#include <cstddef>  // for size_t

#ifdef __CUDACC__  // Check if CUDA is available
#include <cuda_runtime.h>
#endif


// Declare function prototypes
void allocateGPUmemory(float*& d_A, float*& d_B, float*& d_C,
                      size_t size_A, size_t size_B, size_t size_C,
                      const float* h_A, const float* h_B);

void freeGPUmemory(float* d_A, float* d_B, float* d_C);

void benchmark_gemm_cuda(const float* A, const float* B, float* C,
                        int M, int N, int K);

void benchmark_gemm_cublas(const float* h_A, const float* h_B, float* h_C, int M, int N, int K);                        

#endif // GEMM_HPP

