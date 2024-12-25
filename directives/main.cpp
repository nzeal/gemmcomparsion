#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include "../include/utility.h"
#include "../include/timing_utils.hpp"
#include "../include/gemm.hpp"

// Forward declarations
#if defined(SERIAL_VERSION)
void benchmark_gemm_serial(const float* A, const float* B, float* C, int M, int N, int K);
#elif defined(OMP_VERSION)
void benchmark_gemm_openmp(const float* A, const float* B, float* C, int M, int N, int K);
#elif defined(ACC_VERSION)
void benchmark_gemm_openacc(const float* A, const float* B, float* C, int M, int N, int K);
#endif

void run_benchmark(float* h_A, float* h_B, float* h_C, int size) {
    std::chrono::high_resolution_clock::time_point start, end;

    #if defined(OMP_VERSION)
        start = getCurrentTime();
        benchmark_gemm_openmp(h_A, h_B, h_C, size, size, size);
        end = getCurrentTime();
        std::cout << "OpenMP version" << std::endl;
    #elif defined(ACC_VERSION)
        start = getCurrentTime();
        benchmark_gemm_openacc(h_A, h_B, h_C, size, size, size);
        end = getCurrentTime();
        std::cout << "OpenACC version" << std::endl;
    #elif defined(SERIAL_VERSION)
        start = getCurrentTime();
        benchmark_gemm_serial(h_A, h_B, h_C, size, size, size);
        end = getCurrentTime();
        std::cout << "Serial version" << std::endl;
    #else
        #error "No version defined. Please define SERIAL_VERSION, OMP_VERSION, ACC_VERSION"
    #endif

    size_t dataSize = 3 * size * size * sizeof(float);
    size_t numOperations = 2 * static_cast<size_t>(size) * size * size;
    computeAndPrintMetrics(start, end, dataSize, numOperations);
}

int main() {
    // Test different matrix sizes
    int sizes[] = {128, 256, 512, 1024, 2048, 4096};

    for (int size : sizes) {
        std::cout << "Matrix size: " << size << "x" << size << std::endl;

        float *h_A, *h_B, *h_C;
        // Allocate CPU memory
        allocateMemory(&h_A, &h_B, &h_C, size * size);

        // Initialize CPU arrays
        std::fill_n(h_A, size * size, 1.0f);
        std::fill_n(h_B, size * size, 2.0f);
        std::fill_n(h_C, size * size, 0.0f);

        run_benchmark(h_A, h_B, h_C, size);

        // Print the array
        printArray(h_C, size);

        // Free CPU memory
        freeMemory(h_A, h_B, h_C);
        std::cout << "-------------------\n";
    }
    return 0;
}
