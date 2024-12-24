cmake_minimum_required(VERSION 3.10)
project(GEMM VERSION 1.0 LANGUAGES CXX CUDA)

# Set up the CUDA toolchain
if(NOT DEFINED NVHPC)
    set(CXX nvc++)
    set(NVCC nvcc)
    set(ACCFLAGS "-acc=gpu -Minfo=accel")
    set(OMPFLAGS "-mp=gpu -gpu=cc80 -Minfo=accel")
    set(CUDAFLAGS "-O3")
else()
    set(CXX g++)
    set(NVCC nvcc)
    set(ACCFLAGS "")
    set(OMPFLAGS "-fopenmp")
    set(CUDAFLAGS "-O3")
endif()

# Specify paths for CUDA
set(CUDA_PATH "/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/cuda/12.3/")
set(CUDA_INCLUDES "${CUDA_PATH}/include")
set(CUDA_LIBS "${CUDA_PATH}/lib64")

# Ensure that CUDA path exists
if(NOT EXISTS "${CUDA_PATH}")
    message(FATAL_ERROR "CUDA path ${CUDA_PATH} does not exist. Please check your CUDA installation.")
endif()

# Include directories for headers
include_directories(${CUDA_INCLUDES} include)

# Set the common source files
set(COMMON_SRC main.cpp)

# Define source files for different configurations
set(SERIAL_SRC src/gemm_serial.cpp)
set(ACC_SRC src/gemm_acc.cpp)
set(OMP_SRC src/gemm_omp.cpp)
set(CUDA_SRC main_cuda.cpp src/gemm_cuda.cpp)
set(CUDA_KERNEL src/gemm_cuda_kernel.cu)

# Define the target output directory
set(BIN_DIR ${CMAKE_BINARY_DIR}/bin)

# Create output directory if not exists
file(MAKE_DIRECTORY ${BIN_DIR})

# Object files for CUDA
set(CUDA_OBJ gemm_cuda_kernel.o)

# Serial version
add_executable(serial ${COMMON_SRC} ${SERIAL_SRC})
target_compile_definitions(serial PRIVATE SERIAL_VERSION)
target_compile_options(serial PRIVATE ${CXXFLAGS})

# OpenACC version
add_executable(acc ${COMMON_SRC} ${ACC_SRC})
target_compile_definitions(acc PRIVATE ACC_VERSION)
target_compile_options(acc PRIVATE ${CXXFLAGS} ${ACCFLAGS})

# OpenMP version
add_executable(omp ${COMMON_SRC} ${OMP_SRC})
target_compile_definitions(omp PRIVATE OMP_VERSION)
target_compile_options(omp PRIVATE ${CXXFLAGS} ${OMPFLAGS})

# CUDA kernel compilation
add_custom_command(
    OUTPUT ${CUDA_OBJ}
    COMMAND ${NVCC} ${CUDAFLAGS} ${CUDA_INCLUDES} -c ${CUDA_KERNEL} -o ${CUDA_OBJ}
    DEPENDS ${CUDA_KERNEL}
    COMMENT "Compiling CUDA kernel"
)

# CUDA version compilation (using add_executable)
add_executable(cuda ${COMMON_SRC} ${CUDA_SRC} ${CUDA_OBJ})
target_compile_options(cuda PRIVATE ${CUDAFLAGS})
target_link_libraries(cuda PRIVATE ${CUDA_LIBS})

# Ensure that the CUDA object files are built properly
add_dependencies(cuda ${CUDA_OBJ})

# Set target output directory for all executables
set_target_properties(serial acc omp cuda PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
)

# Print version information for the compilers
add_custom_target(versions
    COMMAND ${CXX} --version
    COMMAND ${NVCC} --version
    COMMENT "Print compiler versions"
)

# Clean target
add_custom_target(clean
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/CMakeFiles/clean.cmake
    COMMENT "Cleaning up generated binaries and object files"
)

