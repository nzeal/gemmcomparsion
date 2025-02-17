cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(cuda_dgemm LANGUAGES CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit and cuBLAS
find_package(CUDAToolkit REQUIRED)

# Add executable
add_executable(dgemm_cublas dgemm_cublas.cpp)

# Link libraries using modern CMake targets
target_link_libraries(dgemm_cublas PRIVATE
    CUDA::cublas
    CUDA::cudart
)

# Optional: Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "80")  # Adjust for your GPU architecture
