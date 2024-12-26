// aid-cuda.h
#ifndef AID_CUDA_H
#define AID_CUDA_H

#include <iostream>
#include <cuda_runtime.h>

// Function to check CUDA error status and output details if an error occurs
inline void checkError(cudaError_t code, const char* func, const char* file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << "\n"
                  << "Function: " << func << "\n"
                  << "File: " << file << "\n"
                  << "Line: " << line << std::endl;

        if (abort) {
            exit(1);  // Exit on error
        }
    }
}

// Macro to simplify error checking with file and line info
#define chkErr(val) checkError((val), #val, __FILE__, __LINE__, true)

#endif // AID_CUDA_H

