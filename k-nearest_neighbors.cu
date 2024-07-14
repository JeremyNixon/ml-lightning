#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_POINTS 1000000
#define MAX_K 100

// CUDA kernel to calculate distances
__global__ void calculateDistances(float* trainData, float* testPoint, float* distances, int numTrainPoints, int numFeatures) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numTrainPoints) {
        float sum = 0.0f;
        for (int i = 0; i < numFeatures; i++) {
            float diff = trainData[tid * numFeatures + i] - testPoint[i];
            sum += diff * diff;
        }
        distances[tid] = sqrtf(sum);
    }
}

// CUDA kernel to find K nearest neighbors
__global__ void findKNearest(float* distances, int* indices, int numTrainPoints, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numTrainPoints) {
        for (int i = 0; i < k; i++) {
            if (distances[tid] < distances[indices[i]]) {
                for (int j = k - 1; j > i; j--) {
                    indices[j] = indices[j-1];
                }
                indices[i] = tid;
                break;
            }
        }
    }
}

// Host function to perform KNN
void knn(float* trainData, float* testPoint, int numTrainPoints, int numFeatures, int k) {
    float *d_trainData, *d_testPoint, *d_distances;
    int *d_indices;
    
    // Allocate device memory
    cudaMalloc(&d_trainData, numTrainPoints * numFeatures * sizeof(float));
    cudaMalloc(&d_testPoint, numFeatures * sizeof(float));
    cudaMalloc(&d_distances, numTrainPoints * sizeof(float));
    cudaMalloc(&d_indices, k * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_trainData, trainData, numTrainPoints * numFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_testPoint, testPoint, numFeatures * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize indices
    int* h_indices = (int*)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) h_indices[i] = i;
    cudaMemcpy(d_indices, h_indices, k * sizeof(int), cudaMemcpyHostToDevice);
    
    // Calculate distances
    int blockSize = 256;
    int gridSize = (numTrainPoints + blockSize - 1) / blockSize;
    calculateDistances<<<gridSize, blockSize>>>(d_trainData, d_testPoint, d_distances, numTrainPoints, numFeatures);
    
    // Find K nearest neighbors
    findKNearest<<<gridSize, blockSize>>>(d_distances, d_indices, numTrainPoints, k);
    
    // Copy results back to host
    cudaMemcpy(h_indices, d_indices, k * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("The indices of the %d nearest neighbors are:\n", k);
    for (int i = 0; i < k; i++) {
        printf("%d ", h_indices[i]);
    }
    printf("\n");
    
    // Free memory
    cudaFree(d_trainData);
    cudaFree(d_testPoint);
    cudaFree(d_distances);
    cudaFree(d_indices);
    free(h_indices);
}

int main() {
    // Example usage
    int numTrainPoints = 1000;
    int numFeatures = 3;
    int k = 5;
    
    float* trainData = (float*)malloc(numTrainPoints * numFeatures * sizeof(float));
    float* testPoint = (float*)malloc(numFeatures * sizeof(float));
    
    // Initialize trainData and testPoint with your data
    // ...
    
    knn(trainData, testPoint, numTrainPoints, numFeatures, k);
    
    free(trainData);
    free(testPoint);
    
    return 0;
}