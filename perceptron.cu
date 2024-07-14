#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000 // Number of input features
#define M 100  // Number of training examples

// CUDA kernel for Perceptron training
__global__ void perceptronTrain(float* X, float* y, float* w, float learning_rate, int* errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        float prediction = 0.0f;
        for (int j = 0; j < N; j++) {
            prediction += X[idx * N + j] * w[j];
        }
        
        if ((prediction > 0 && y[idx] <= 0) || (prediction <= 0 && y[idx] > 0)) {
            atomicAdd(errors, 1);
            for (int j = 0; j < N; j++) {
                atomicAdd(&w[j], learning_rate * y[idx] * X[idx * N + j]);
            }
        }
    }
}

// Host function to initialize data
void initializeData(float* X, float* y, float* w) {
    for (int i = 0; i < M * N; i++) {
        X[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < M; i++) {
        y[i] = (rand() % 2) * 2 - 1; // -1 or 1
    }
    for (int i = 0; i < N; i++) {
        w[i] = 0.0f;
    }
}

int main() {
    float *X, *y, *w;
    float *d_X, *d_y, *d_w;
    int *d_errors;
    
    // Allocate host memory
    X = (float*)malloc(M * N * sizeof(float));
    y = (float*)malloc(M * sizeof(float));
    w = (float*)malloc(N * sizeof(float));
    
    // Initialize data
    initializeData(X, y, w);
    
    // Allocate device memory
    cudaMalloc(&d_X, M * N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));
    cudaMalloc(&d_w, N * sizeof(float));
    cudaMalloc(&d_errors, sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_X, X, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Training parameters
    float learning_rate = 0.01f;
    int max_epochs = 100;
    int errors;
    
    // Training loop
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        errors = 0;
        cudaMemcpy(d_errors, &errors, sizeof(int), cudaMemcpyHostToDevice);
        
        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;
        perceptronTrain<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_y, d_w, learning_rate, d_errors);
        
        // Copy errors back to host
        cudaMemcpy(&errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("Epoch %d: errors = %d\n", epoch, errors);
        if (errors == 0) break;
    }
    
    // Copy final weights back to host
    cudaMemcpy(w, d_w, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free memory
    free(X); free(y); free(w);
    cudaFree(d_X); cudaFree(d_y); cudaFree(d_w); cudaFree(d_errors);
    
    return 0;
}