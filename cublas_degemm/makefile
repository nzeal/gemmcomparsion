# Compiler and flags
NVCC = nvcc
CFLAGS = -lcublas

# Source and target
SRC = dgemm_cublas.cpp
OBJ = dgemm_cublas.o
OUT = dgemm_cublas

# Default target
all: $(OUT)

# Rule to build the executable
$(OUT): $(OBJ)
	$(NVCC) $(OBJ) $(CFLAGS) -o $(OUT)

# Rule to compile the source file
$(OBJ): $(SRC)
	$(NVCC) -c $(SRC)

# Clean up object and executable files
clean:
	rm -f $(OBJ) $(OUT)

