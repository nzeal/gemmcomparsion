cmake_minimum_required(VERSION 3.18)
project(GEMM_Accelerated_Compute CUDA CXX)

# Set CUDA architectures policy
cmake_policy(SET CMP0104 NEW)

# Check for NVIDIA HPC compiler
if(NOT CMAKE_CXX_COMPILER MATCHES "nvc++")
    message(FATAL_ERROR "This project requires NVIDIA HPC Compiler (nvc++). Current compiler: ${CMAKE_CXX_COMPILER}")
endif()

# Output compiler information
message(STATUS "Using CXX compiler: ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")

# Set paths and flags for A100
set(GPU_ARCH "80")  # Ampere architecture for A100
set(COMMON_FLAGS "-O3")  # Assume AMD EPYC CPU. For Intel CPUs, consider adding -march=native or -xHost. For other AMD CPUs, adjust optimization flags accordingly.
set(ACC_FLAGS "-acc=gpu -Minfo=accel")
set(GPU_MANAGED_FLAGS "-gpu=sm${GPU_ARCH},managed")
set(OMP_FLAGS "-mp=gpu -Minfo=mp")

# Find CUDA
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Set CUDA compiler
find_program(CUDAToolkit_NVCC_EXECUTABLE NAMES nvcc PATHS ${CUDAToolkit_BIN_DIR} NO_DEFAULT_PATH)
if(NOT CUDAToolkit_NVCC_EXECUTABLE)
    message(FATAL_ERROR "CUDA compiler not found. Please ensure CUDA is installed and nvcc is in your PATH.")
endif()

# Project structure
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(PROJECT_CUDA_DIR ${CMAKE_SOURCE_DIR}/cuda)
set(PROJECT_SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(PROJECT_DIRECTIVES_DIR ${CMAKE_SOURCE_DIR}/directives)

# Include directories
include_directories(
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_CUDA_DIR}
)

# Source files
set(COMMON_SRCS ${PROJECT_DIRECTIVES_DIR}/main.cpp)

# Create the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Serial version
add_executable(serial
    ${COMMON_SRCS}
    ${PROJECT_SRC_DIR}/gemm_serial.cpp
)
target_compile_options(serial PRIVATE ${COMMON_FLAGS})
target_compile_definitions(serial PRIVATE SERIAL_VERSION)

# OpenACC version
add_executable(acc
    ${COMMON_SRC}
    ${PROJECT_SRC_DIR}/gemm_acc.cpp
)
target_compile_options(acc PRIVATE ${COMMON_FLAGS} ${ACC_FLAGS} ${GPU_MANAGED_FLAGS})
target_link_options(acc PRIVATE ${ACC_FLAGS} ${GPU_MANAGED_FLAGS})
target_compile_definitions(acc PRIVATE ACC_VERSION)

# OpenMP version
add_executable(omp
    ${COMMON_SRC}
    ${PROJECT_SRC_DIR}/gemm_omp.cpp
)
target_compile_options(omp PRIVATE ${COMMON_FLAGS} ${OMP_FLAGS} ${GPU_MANAGED_FLAGS})
target_link_options(omp PRIVATE ${OMP_FLAGS} ${GPU_MANAGED_FLAGS})
target_compile_definitions(omp PRIVATE OMP_VERSION)

# CUDA version
add_executable(cuda
    ${COMMON_SRC}
    ${PROJECT_CUDA_DIR}/main_cuda.cpp
    ${PROJECT_CUDA_DIR}/gemm_cuda.cpp
    ${PROJECT_SRC_DIR}/gemm_cuda_kernel.cu
)
target_compile_options(cuda PRIVATE ${COMMON_FLAGS} $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_${GPU_ARCH}>)
target_compile_options(cuda PRIVATE ${COMMON_FLAGS} $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_${GPU_ARCH}>)
target_compile_definitions(cuda PRIVATE CUDA_VERSION)

# Add custom target for environment setup
# Add custom target for environment setup
# This target prints the CXX and CUDA compiler information and their versions.
# Usage: Run `make env` or `ninja env` to display the compiler information.
add_custom_target(env
    COMMAND ${CMAKE_COMMAND} -E echo "CXX compiler: ${CMAKE_CXX_COMPILER}"
    COMMAND ${CMAKE_COMMAND} -E echo "CUDA compiler: ${CUDAToolkit_NVCC_EXECUTABLE}"
    COMMAND ${CMAKE_CXX_COMPILER} --version
# Clean target
# This target removes all generated files and directories from the build.
# Usage: Run `make clean_all` or `ninja clean_all` depending on your build system.
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/Makefile
    COMMENT "Cleaning all built files"
)   COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/Makefile
    COMMENT "Cleaning all built files"
)
