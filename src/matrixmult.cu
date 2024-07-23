#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <cuda_runtime.h>
#include "kernel.h"
#include "dev_array.h"
#include <math.h>

using namespace std;

int main()
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices

    int N;
    cout << "Enter the matrix size: ";
    cin >> N;
    int SIZE = N*N;
    cout << "Running for " << N <<'x'<<N << " matrix.\n";
    vector<double> h_A(SIZE);
    vector<double> h_B(SIZE);
    vector<double> h_C(SIZE);

    // Initialize matrices
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = sin(static_cast<double>(i));
            h_B[i*N+j] = cos(static_cast<double>(j));
        }
    }

    // Update device arrays to use double
    dev_array<double> d_A(SIZE);
    dev_array<double> d_B(SIZE);
    dev_array<double> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);

    // GPU Timing
    cudaEvent_t start, stop;
    cout << "Starting GPU computiation.\n";
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start GPU timer
    cudaEventRecord(start);

    // GPU computation
    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
    cudaDeviceSynchronize();

    // Stop GPU timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cout << "GPU computiation completed.\n";
    float gpuMilliseconds = 0;
    cudaEventElapsedTime(&gpuMilliseconds, start, stop);

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();

    // CPU Timing
    cout << "Starting CPU computiation.\n";
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // CPU computation
    double *cpu_C = new double[SIZE];
    double sum;
    for (int row=0; row<N; row++){
        for (int col=0; col<N; col++){
            sum = 0.0;
            for (int n=0; n<N; n++){
                sum += h_A[row*N+n]*h_B[n*N+col];
            }
            cpu_C[row*N+col] = sum;
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    cout << "CPU computiation completed.\n";
    std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;

    // Error checking
    double max_error = 0.0;
    for (int ROW=0; ROW < N; ROW++){
        for (int COL=0; COL < N; COL++){
            double error = abs(cpu_C[ROW * N + COL] - h_C[ROW * N + COL]);
            if (error > max_error) {
                max_error = error;
            }
        }
    }

    // Print results
    cout << "Max Error: " << max_error << endl;
    cout << "GPU time: " << gpuMilliseconds << " ms" << endl;
    cout << "CPU time: " << cpu_ms.count() << " ms" << endl;

    // Clean up
    delete[] cpu_C;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
