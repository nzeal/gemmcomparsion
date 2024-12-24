# GEMM Comparison

This repository contains code for comparing different General Matrix Multiply (GEMM) implementations.

## Folder Structure

```
gemmcomparison/
├── include/
  ├── gemm.hpp
  ├── gemm_cuda.hpp
  ├── timing_utils.hpp
  ├── utlity.cuh
  ├── utlity.h                               
├── main.cpp             
├── gemm_serial.cpp
├── gemm_acc.cpp
├── gemm_omp.cpp
├── gemm_cuda.cpp
├── makefile
├── bin/              
└── README.md           
```

## Getting Started

### Prerequisites

- gnu compiler
- NVHPC compiler

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/gemmcomparison.git
cd gemmcomparison
```

Install:

```bash
- make serial 
- make acc
- make omp
- make cuda
```

## Usage

To run the GEMM comparison, execute the following command:

```bash
python src/compare_gemm.py 
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For any questions, please contact [your email].
