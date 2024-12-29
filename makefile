# ===================================================
# This makefile is used to compile and
# link the source files for different configurations
# (Serial, OpenACC, OpenMP, CUDA) and
# store the binaries in the bin folder.
# ===================================================

# Compiler selection
ifeq ($(strip $(NVHPC)),)
    CXX = nvc++
    NVCC = nvcc
    ACCFLAGS = -acc=gpu -Minfo=accel
    OMPFLAGS = -mp=gpu -gpu=cc80,managed -Minfo=all
    CUDAFLAGS =  -arch=sm_80 -O3
else
    CXX = g++
    NVCC = nvcc
    ACCFLAGS = # No flags for ACC when not using NVIDIA HPC SDK
    OMPFLAGS = 
    CUDAFLAGS = -O3
endif

# SYCL compiler and flags
CXX_SYCL = icpx
SYCL_FLAGS = -fsycl -std=c++17

# CUDA paths - adjust these for your system
NVHPC_INCLUDE_PATH := /leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3
CUDA_PATH ?=  $(NVHPC_INCLUDE_PATH)/cuda/12.3/
CUDA_INCLUDES = -I$(CUDA_PATH)/include

# Check if CUDA path exists
ifeq ($(wildcard $(CUDA_PATH)),)
    $(error CUDA path $(CUDA_PATH) does not exist. Please check your CUDA installation.)
endif

# Compiler flags
CXXFLAGS = -O3 -I./include

# Library paths
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart

# Source files
COMMON_SRC = directives/main.cpp
SERIAL_SRC = src/gemm_serial.cpp
ACC_SRC = src/gemm_acc.cpp
OMP_SRC = src/gemm_omp.cpp
CUDA_SRC = cuda/main_cuda.cpp cuda/gemm_cuda.cpp
CUDA_KERNEL = src/gemm_cuda_kernel.cu
SYCL_SRC = sycl/main.cpp sycl/gemm_sycl.cpp

# Object files for CUDA
CUDA_OBJ = gemm_cuda_kernel.o

# Binary directory
BINDIR = bin

# Header files
HEADERS = $(wildcard include/*.hpp) $(wildcard include/*.cuh) $(wildcard sycl/include/*.hpp)

# Targets
all: setup serial acc omp cuda sycl

# Setup target: create bin directory
setup:
	@echo "Checking if bin directory exists..."
	@mkdir -p $(BINDIR)
	@echo "Directory $(BINDIR) created successfully."

# Serial build target
serial: setup $(COMMON_SRC) $(SERIAL_SRC) $(HEADERS)
	@echo "Building serial version..."
	$(CXX) $(CXXFLAGS) -DSERIAL_VERSION $(COMMON_SRC) $(SERIAL_SRC) -o $(BINDIR)/$@
	@echo "Serial build complete. Output: $(BINDIR)/$@"

# OpenACC build target
acc: setup $(COMMON_SRC) $(ACC_SRC) $(HEADERS)
	@echo "Building OpenACC version..."
	$(CXX) $(CXXFLAGS) $(ACCFLAGS) -DACC_VERSION $(COMMON_SRC) $(ACC_SRC) -o $(BINDIR)/$@
	@echo "OpenACC build complete. Output: $(BINDIR)/$@"

# OpenMP build target
omp: setup $(COMMON_SRC) $(OMP_SRC) $(HEADERS)
	@echo "Building OpenMP version..."
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DOMP_VERSION $(COMMON_SRC) $(OMP_SRC) -o $(BINDIR)/$@
	@echo "OpenMP build complete. Output: $(BINDIR)/$@"

# CUDA kernel compilation
$(CUDA_OBJ): $(CUDA_KERNEL)
	@echo "Compiling CUDA kernel..."
	$(NVCC) $(CUDAFLAGS) $(CUDA_INCLUDES) -c $< -o $@

# CUDA build target
cuda: setup $(CUDA_OBJ) $(CUDA_SRC) $(HEADERS)
	@echo "Building CUDA version..."
	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDES) $(CUDA_SRC) $(CUDA_OBJ) -o $(BINDIR)/$@ $(CUDA_LIBS)
	@echo "CUDA build complete. Output: $(BINDIR)/$@"

sycl: setup $(SYCL_SRC) $(HEADERS)
	@echo "Building SYCL version..."
	$(CXX_SYCL) $(CXXFLAGS) $(SYCL_FLAGS) -DSYCL_VERSION $(SYCL_SRC) -o $(BINDIR)/$@
	@echo "SYCL build complete. Output: $(BINDIR)/$@"

# Clean up generated binaries and object files
clean:
	@echo "Cleaning up the bin directory and object files..."
	@rm -rf $(BINDIR)
	@rm -f *.o
	@echo "Clean up complete."

# Print version information
versions:
	@echo "Compiler versions:"
	@$(CXX) --version
	@$(NVCC) --version

# Declare phony targets
.PHONY: all setup clean versions serial acc omp cuda sycl
