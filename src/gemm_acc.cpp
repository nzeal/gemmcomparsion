// gemm_serial.cpp
#include <iostream>
#include <chrono>
#include <algorithm>
#include "../include/gemm.hpp"

#define BLOCK_SIZE 64

/*
 * This function performs the matrix multiplication C = A * B using OpenACC for parallelization.
 *
 * @param A Pointer to the first input matrix (M x K).
 * @param B Pointer to the second input matrix (K x N).
 * @param C Pointer to the output matrix (M x N).
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 */

/*
void benchmark_gemm_openacc(const float* A, const float* B, float* C, int M, int N, int K) {
    #pragma acc kernels copyin(A[0:M*K], B[0:K*N]) copyout(C[0:M*N])
    {
    #pragma acc loop independent collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
	    #pragma acc loop reduction(+:sum)
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
     }
    }
}
*/

void blocked_dgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    #pragma acc data copyin(A[0:M*K], B[0:K*N]) copy(C[0:M*N])
    {
        // Initialize matrix C to zero
        #pragma acc parallel loop collapse(2)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] = 0.0f;
            }
        }

        // Main computation
        for (int bi = 0; bi < M; bi += BLOCK_SIZE) {
            for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
                for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
                    #pragma acc parallel loop collapse(2)
                    for (int i = bi; i < ((bi + BLOCK_SIZE < M) ? bi + BLOCK_SIZE : M); i++) {
                        for (int j = bj; j < ((bj + BLOCK_SIZE < N) ? bj + BLOCK_SIZE : N); j++) {
                            float sum = 0.0f;
                            #pragma acc loop reduction(+:sum)
                            for (int k = bk; k < ((bk + BLOCK_SIZE < K) ? bk + BLOCK_SIZE : K); k++) {
                                sum += A[i * K + k] * B[k * N + j];
                            }
                            C[i * N + j] += sum;
                        }
                    }
                }
            }
        }
    }
}

/*
void blocked_dgemm(const double* A, const double* B, double* C, int M, int N, int K) {
    #pragma acc kernels copyin(A[0:M*K], B[0:K*N]) copyout(C[0:M*N])
    {
        // Iterate over blocks
        #pragma acc loop independent collapse(3)
        for (int bi = 0; bi < M; bi += BLOCK_SIZE) {
            for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
                for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
                    // Compute block-wise multiplication
                    #pragma acc loop independent collapse(2)
                    for (int i = bi; i < bi + BLOCK_SIZE && i < M; i++) {
                        for (int j = bj; j < bj + BLOCK_SIZE && j < N; j++) {
                            double sum = 0.0;
                            #pragma acc loop reduction(+:sum)
                            for (int k = bk; k < bk + BLOCK_SIZE && k < K; k++) {
                                sum += A[i * K + k] * B[k * N + j];
                            }
                            C[i * N + j] += sum;
                        }
                    }
                }
            }
        }
    }
}


void tiled_dgemm(int n, const double* A, const double* B, double* C) {
    #pragma acc kernels copyin(A[0:n*n], B[0:n*n]) copyout(C[0:n*n])
    {
        #pragma acc loop independent collapse(2)
        for (int i = 0; i < n; i += TILE_SIZE) {
            for (int j = 0; j < n; j += TILE_SIZE) {
                #pragma acc loop independent collapse(2)
                for (int ii = i; ii < i + TILE_SIZE && ii < n; ii++) {
                    for (int jj = j; jj < j + TILE_SIZE && jj < n; jj++) {
                        double sum = 0.0;
                        #pragma acc loop reduction(+:sum)
                        for (int k = 0; k < n; k++) {
                            sum += A[ii * n + k] * B[k * n + jj];
                        }
                        C[ii * n + jj] = sum;
                    }
                }
            }
        }
    }
}

*/
