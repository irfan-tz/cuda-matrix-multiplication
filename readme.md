# CUDA Matrix Multiplication

This project demonstrates matrix multiplication using CUDA, comparing the performance and accuracy between GPU and CPU implementations.

## Description

The program performs matrix multiplication (C = A * B) where A, B, and C are NxN matrices. It uses CUDA for GPU computation and a standard nested loop approach for CPU computation. The program also measures and compares the execution time for both methods.

## Requirements

- CUDA-capable GPU
- CUDA Toolkit (version 10.0 or later recommended)
- C++ compiler with C++11 support

## Building the Project

1. Clone the repository:
`>>git clone https://github.com/irfan-tz/cuda-matrix-multiplication/`

2. Create a build directory and navigate to it:

3. Run make and build the project:
`>>make`

## Running the Program

After building, run the executable:
`make run`

## Output

The program will output:
- The maximum error between GPU and CPU computations
- Execution time for the GPU computation
- Execution time for the CPU computation

## File Structure

```bin/```
This contains all the binary files created after the compilation of the program.

```src/```
The contains the source code for the program.
- `main.cpp`: Contains the main function and host code
- `kernel.cu`: Contains the CUDA kernel for matrix multiplication
- `kernel.h`: Header file for the CUDA kernel
- `dev_array.h`: Header file for a wrapper class to manage device memory

## Notes

- The GPU timing includes only the kernel execution time, not memory transfer times.
- For small matrices, the CPU might outperform the GPU due to kernel launch and data transfer overheads.
- Performance can vary significantly based on hardware specifications.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/irfan-tz/cuda-matrix-multiplication/issues) if you want to contribute.