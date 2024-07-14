#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Kernel for sigmoid function
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Kernel for computing logistic regression predictions and gradient
__global__ void logisticRegressionKernel(float *X, float *y, float *weights, float *grad, int n, int d) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float prediction = 0.0f;
        for (int j = 0; j < d; j++) {
            prediction += X[tid * d + j] * weights[j];
        }
        prediction = sigmoid(prediction);
        
        float error = prediction - y[tid];
        
        for (int j = 0; j < d; j++) {
            atomicAdd(&grad[j], error * X[tid * d + j]);
        }
    }
}

// Function to train logistic regression model
void trainLogisticRegression(float *X, float *y, float *weights, int n, int d, float learning_rate, int max_iterations) {
    float *d_X, *d_y, *d_weights, *d_grad;
    
    // Allocate device memory
    cudaMalloc((void**)&d_X, n * d * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_weights, d * sizeof(float));
    cudaMalloc((void**)&d_grad, d * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_X, X, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, d * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set up grid and block dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Gradient descent loop
    for (int iter = 0; iter < max_iterations; iter++) {
        // Reset gradient
        cudaMemset(d_grad, 0, d * sizeof(float));
        
        // Compute predictions and gradient
        logisticRegressionKernel<<<gridDim, blockDim>>>(d_X, d_y, d_weights, d_grad, n, d);
        
        // Update weights
        float *grad = (float*)malloc(d * sizeof(float));
        cudaMemcpy(grad, d_grad, d * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int j = 0; j < d; j++) {
            weights[j] -= learning_rate * grad[j] / n;
        }
        
        cudaMemcpy(d_weights, weights, d * sizeof(float), cudaMemcpyHostToDevice);
        
        free(grad);
    }
    
    // Copy final weights back to host
    cudaMemcpy(weights, d_weights, d * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_weights);
    cudaFree(d_grad);
}

// Main function for demonstration
int main() {
    // Example usage
    int n = 1000; // number of samples
    int d = 5;    // number of features
    
    float *X = (float*)malloc(n * d * sizeof(float));
    float *y = (float*)malloc(n * sizeof(float));
    float *weights = (float*)malloc(d * sizeof(float));
    
    // Initialize X, y, and weights (you should replace this with your actual data)
    for (int i = 0; i < n * d; i++) {
        X[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < n; i++) {
        y[i] = rand() % 2;
    }
    for (int i = 0; i < d; i++) {
        weights[i] = 0.0f;
    }
    
    float learning_rate = 0.1f;
    int max_iterations = 100;
    
    trainLogisticRegression(X, y, weights, n, d, learning_rate, max_iterations);
    
    // Print final weights
    printf("Final weights:\n");
    for (int i = 0; i < d; i++) {
        printf("%f ", weights[i]);
    }
    printf("\n");
    
    free(X);
    free(y);
    free(weights);
    
    return 0;
}