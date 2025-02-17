# Compiler selection
CXX = g++
NVCC = nvcc

# CUDA paths
NVHPC_INCLUDE_PATH := /leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3
CUDA_PATH ?= $(NVHPC_INCLUDE_PATH)/cuda/12.3
CUDA_INCLUDES = -I$(CUDA_PATH)/include -I./include

# Architecture flags for different GPU generations
# Adjust these based on your GPU
ARCH_FLAGS = -gencode arch=compute_60,code=sm_60 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86

# Compiler flags
NVCCFLAGS = -O3 $(CUDA_INCLUDES) $(ARCH_FLAGS)
CXXFLAGS = -O3 $(CUDA_INCLUDES) -std=c++11

# Add necessary compilation flags
NVCCFLAGS += --compiler-options -fPIC
NVCCFLAGS += -std=c++11

# Library paths and libs
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lstdc++

# Source files
CPP_SRC = main_cuda.cpp gemm_cuda.cpp
CUDA_SRC = gemm_cuda_kernel.cu

# Object files
BINDIR = bin
CPP_OBJ = $(addprefix $(BINDIR)/, $(CPP_SRC:.cpp=.o))
CUDA_OBJ = $(addprefix $(BINDIR)/, $(CUDA_SRC:.cu=.o))

# Target executable
TARGET = $(BINDIR)/cuda_exec

# Header files
HEADERS = $(wildcard include/*.hpp) $(wildcard include/*.cuh) $(wildcard include/*.h)

# Default target
all: $(TARGET)

# Create bin directory
$(BINDIR):
	@mkdir -p $(BINDIR)

# Compilation rules
$(BINDIR)/%.o: %.cpp $(HEADERS) | $(BINDIR)
	@echo "Compiling C++ file $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BINDIR)/%.o: %.cu $(HEADERS) | $(BINDIR)
	@echo "Compiling CUDA file $<..."
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Linking
$(TARGET): $(CPP_OBJ) $(CUDA_OBJ)
	@echo "Linking..."
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(CUDA_LIBS)
	@echo "Build complete!"

clean:
	@echo "Cleaning..."
	@rm -rf $(BINDIR)
	@echo "Clean complete!"

.PHONY: all clean