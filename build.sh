#!/bin/bash

# Create build directory
mkdir -p build
cd build

# Configure with NVIDIA HPC compiler
cmake -DCMAKE_CXX_COMPILER=nvc++ ..

# Build
make -j
