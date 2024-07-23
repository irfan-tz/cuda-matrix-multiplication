#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.h"

using namespace std;

__global__ void matrixMultiplicationKernel(double* A, double* B, double* C, int N) {
    __shared__ double As[32][32];
    __shared__ double Bs[32][32];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    double sum = 0.0;

    for (int p = 0; p < (N + 31) / 32; ++p) {
        if (row < N && p * 32 + tx < N)
            As[ty][tx] = A[row * N + p * 32 + tx];
        else
            As[ty][tx] = 0.0;

        if (p * 32 + ty < N && col < N)
            Bs[ty][tx] = B[(p * 32 + ty) * N + col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < 32; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

void matrixMultiplication(double *A, double *B, double *C, int N) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}
