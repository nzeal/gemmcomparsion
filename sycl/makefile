# Compiler settings
CXX_SERIAL := icpx
CXX_SYCL := icpx

# Compiler flags
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra
SYCL_FLAGS := -fsycl

# Include directory
INCLUDES := -I./include

# Source files
SOURCES := main.cpp gemm_sycl.cpp

# Output executables
SERIAL_OUT := gemm_serial
SYCL_OUT := gemm_sycl

# Targets
all: $(SERIAL_OUT) $(SYCL_OUT)

$(SERIAL_OUT): $(SOURCES)
	$(CXX_SERIAL) $(CXXFLAGS) $(INCLUDES) -DSERIAL_VERSION $^ -o $@

$(SYCL_OUT): $(SOURCES)
	$(CXX_SYCL) $(CXXFLAGS) $(SYCL_FLAGS) $(INCLUDES) -DSYCL_VERSION $^ -o $@

clean:
	rm -f $(SERIAL_OUT) $(SYCL_OUT)

.PHONY: all clean
