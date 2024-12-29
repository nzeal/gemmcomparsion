#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include "../include/utility.h"
#include "../include/timing_utils.hpp"
#include "../include/gemm.hpp"

// Forward declaration
void benchmark_gemm_cuda(const float* A, const float* B, float* C, int M, int N, int K);

// New function to store benchmark results
void storeBenchmarkResults(const std::string& filename, int size, double time, double gflops, double bandwidth) {
    std::ofstream outFile(filename, std::ios::app);
    if (outFile.is_open()) {
        outFile << std::setw(10) << size << ","
                << std::setw(15) << std::fixed << std::setprecision(6) << time << ","
                << std::setw(15) << std::fixed << std::setprecision(6) << gflops << ","
                << std::setw(15) << std::fixed << std::setprecision(6) << bandwidth << "\n";
        outFile.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

int main() {
    // Test different matrix sizes
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    // Create a filename for the results
    std::string filename = "benchmark_results_cuda.txt";

    // Write the header to the file
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << "Matrix_Size,Time_Seconds,GFLOPS,Bandwidth_GB_per_S\n";
        outFile.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return 1;
    }

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

        double time, gflops, bandwidth;
        computeAndPrintMetrics(start, end, dataSize, numOperations, time, gflops, bandwidth);

        // Store the results in the file
        storeBenchmarkResults(filename, size, time, gflops, bandwidth);

        // Optionally print the result (for debugging)
        printArray(h_C, size);

        // Free allocated memory
        freeMemory(h_A, h_B, h_C);
        std::cout << "-------------------\n";
    }
    return 0;
}
