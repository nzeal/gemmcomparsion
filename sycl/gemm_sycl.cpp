#include <iostream>
#include <chrono>
#include <algorithm>
#include "../include/gemm.hpp"

#ifdef SYCL_VERSION
#include <CL/sycl.hpp>
#endif

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

#ifdef SYCL_VERSION
void benchmark_gemm_sycl(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    // Create a SYCL queue
    sycl::queue q{sycl::default_selector_v};

    // Create SYCL buffers for input and output matrices
    sycl::buffer<float, 1> buf_A(A, sycl::range<1>(M * K));
    sycl::buffer<float, 1> buf_B(B, sycl::range<1>(K * N));
    sycl::buffer<float, 1> buf_C(C, sycl::range<1>(M * N));

    // Define work-group size and compute grid dimensions
    constexpr int BLOCK_SIZE = 16;  // Adjust based on your hardware

    // Submit the computation to the queue
    q.submit([&](sycl::handler& h) {
        // Get accessors for the buffers
        auto a_A = buf_A.get_access<sycl::access::mode::read>(h);
        auto a_B = buf_B.get_access<sycl::access::mode::read>(h);
        auto a_C = buf_C.get_access<sycl::access::mode::write>(h);

        // Define the computation range
        sycl::range<2> global_range{
            static_cast<size_t>(((M + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE),
            static_cast<size_t>(((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE)
        };
        sycl::range<2> local_range{BLOCK_SIZE, BLOCK_SIZE};

        // Launch the kernel
        h.parallel_for(
            sycl::nd_range<2>{global_range, local_range},
            [=](sycl::nd_item<2> item) {
                const int row = item.get_global_id(0);
                const int col = item.get_global_id(1);

                // Check if we're within matrix bounds
                if (row < M && col < N) {
                    float sum = 0.0f;

                    // Compute matrix multiplication for this element
                    for (int k = 0; k < K; ++k) {
                        sum += a_A[row * K + k] * a_B[k * N + col];
                    }

                    a_C[row * N + col] = sum;
                }
            });
    });

    // Queue will automatically synchronize at the end of scope
}
#endif
