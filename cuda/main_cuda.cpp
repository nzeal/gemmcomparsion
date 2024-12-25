#include <iostream>
#include <chrono>
#include <algorithm>
#include "include/utility.h"
#include "include/timing_utils.hpp"
#include "include/gemm.hpp"

// Forward declaration
void benchmark_gemm_cuda(const float* A, const float* B, float* C, int M, int N, int K);

int main() {
    // Test different matrix sizes
    int sizes[] = {128, 256, 512, 1024, 2048, 4096};

    for (int size : sizes) {
        std::cout << "Matrix size: " << size << "x" << size << std::endl;
        size_t dataSize = 3 * size * size * sizeof(float);
        size_t numOperations = 2 * static_cast<size_t>(size) * size * size; // Correct ops for GEMM

        float *h_A, *h_B, *h_C;
        allocateMemory(&h_A, &h_B, &h_C, size * size);

        std::fill_n(h_A, size * size, 1.0f);  // Initialize A with 1.0
        std::fill_n(h_B, size * size, 2.0f);  // Initialize B with 2.0
        std::fill_n(h_C, size * size, 0.0f);  // Initialize C with 0.0

        // Start time
        auto start = getCurrentTime();
        benchmark_gemm_cuda(h_A, h_B, h_C, size, size, size);
        auto end = getCurrentTime();

        // Compute and print metrics
        std::cout << "CUDA version" << std::endl;
        computeAndPrintMetrics(start, end, dataSize, numOperations);

        // Optionally print the result (for debugging)
        printArray(h_C, size);

        // Free allocated memory
        freeMemory(h_A, h_B, h_C);
        std::cout << "-------------------\n";
    }
    return 0;
}

