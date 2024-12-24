#ifndef UTILITY_CUH
#define UTILITY_CUH

#ifdef __CUDACC__  // Check if CUDA is available
#include <cuda_runtime.h>
#endif

void checkCudaErrors(cudaError_t err) {
    // After kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    // After kernel synchronization
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronization failed: " << cudaGetErrorString(err) << std::endl;
    }
}

#endif // UTILITY_CUH

