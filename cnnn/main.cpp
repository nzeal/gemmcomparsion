#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include "include/utility.h"
#include "include/timing_utils.hpp"
#include "include/gemm.hpp"

#ifdef SYCL_VERSION
#include <CL/sycl.hpp>
#endif

// Forward declarations
#if defined(SERIAL_VERSION)
void benchmark_gemm_serial(const float* A, const float* B, float* C, int M, int N, int K);
#elif defined(SYCL_VERSION)
void benchmark_gemm_sycl(const float* A, const float* B, float* C, int M, int N, int K);
#endif

void run_benchmark(float* h_A, float* h_B, float* h_C, int size) {
    std::chrono::high_resolution_clock::time_point start, end;

    try {
        #if defined(SERIAL_VERSION)
            start = getCurrentTime();
            benchmark_gemm_serial(h_A, h_B, h_C, size, size, size);
            end = getCurrentTime();
            std::cout << "Serial version" << std::endl;
        #elif defined(SYCL_VERSION)
            // Print available devices
            std::cout << "Available SYCL devices:" << std::endl;
            for (const auto& device : cl::sycl::device::get_devices()) {
                std::cout << "  - " << device.get_info<cl::sycl::info::device::name>() << std::endl;
            }

            start = getCurrentTime();
            benchmark_gemm_sycl(h_A, h_B, h_C, size, size, size);
            end = getCurrentTime();
            std::cout << "SYCL version" << std::endl;
        #else
            #error "No version defined. Please define SERIAL_VERSION or SYCL_VERSION"
        #endif

        size_t dataSize = 3 * size * size * sizeof(float);
        size_t numOperations = 2 * static_cast<size_t>(size) * size * size;
        computeAndPrintMetrics(start, end, dataSize, numOperations);
    }
    #ifdef SYCL_VERSION
    catch (const cl::sycl::exception& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        exit(1);
    }
    #endif
    catch (const std::exception& e) {
        std::cerr << "Standard exception caught: " << e.what() << std::endl;
        exit(1);
    }
}

bool verifyResults(const float* C, int size) {
    float expected = 2.0f * size;  // Each element should be 1.0 * 2.0 * size
    for (int i = 0; i < size * size; ++i) {
        if (std::abs(C[i] - expected) > 1e-5) {
            return false;
        }
    }
    return true;
}

int main() {
    try {
#ifdef SYCL_VERSION
        // Create SYCL queue with default selector
        sycl::queue queue{sycl::default_selector_v};
        std::cout << "Running on device: "
                  << queue.get_device().get_info<sycl::info::device::name>()
                  << std::endl;
 #endif
        // Test different matrix sizes
        int sizes[] = {128, 256, 512, 1024, 2048, 4096};

        for (int size : sizes) {
            std::cout << "\nMatrix size: " << size << "x" << size << std::endl;
            std::cout << "----------------------------------------" << std::endl;

            float *h_A, *h_B, *h_C;

            // Allocate CPU memory
            allocateMemory(&h_A, &h_B, &h_C, size * size);

            // Initialize CPU arrays
            std::fill_n(h_A, size * size, 1.0f);
            std::fill_n(h_B, size * size, 2.0f);
            std::fill_n(h_C, size * size, 0.0f);

            // Run the benchmark
            run_benchmark(h_A, h_B, h_C, size);

            // Verify results
            if (verifyResults(h_C, size)) {
                std::cout << "Results are correct!" << std::endl;
            } else {
                std::cout << "Results are incorrect!" << std::endl;
            }

            // Print the array
            printArray(h_C, size);

            // Free CPU memory
            freeMemory(h_A, h_B, h_C);

            std::cout << "----------------------------------------\n";
        }
    }
    #ifdef SYCL_VERSION
    catch (const cl::sycl::exception& e) {
        std::cerr << "SYCL exception caught in main: " << e.what() << std::endl;
        return 1;
    }
    #endif
    catch (const std::exception& e) {
        std::cerr << "Standard exception caught in main: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
