#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

// Structure to represent a node in the decision tree
struct Node {
    int feature_index;
    float threshold;
    int left_child;
    int right_child;
    int label;
};

// CUDA kernel for classification
__global__ void classifyKernel(float* data, int* results, Node* tree, int num_samples, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_samples) {
        int current_node = 0;
        
        while (tree[current_node].feature_index != -1) {
            float feature_value = data[idx * num_features + tree[current_node].feature_index];
            
            if (feature_value <= tree[current_node].threshold) {
                current_node = tree[current_node].left_child;
            } else {
                current_node = tree[current_node].right_child;
            }
        }
        
        results[idx] = tree[current_node].label;
    }
}

// Host function to set up and launch the CUDA kernel
void classifyDecisionTree(float* h_data, int* h_results, Node* h_tree, int num_samples, int num_features, int num_nodes) {
    float* d_data;
    int* d_results;
    Node* d_tree;
    
    // Allocate device memory
    cudaMalloc((void**)&d_data, num_samples * num_features * sizeof(float));
    cudaMalloc((void**)&d_results, num_samples * sizeof(int));
    cudaMalloc((void**)&d_tree, num_nodes * sizeof(Node));
    
    // Copy data from host to device
    cudaMemcpy(d_data, h_data, num_samples * num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tree, h_tree, num_nodes * sizeof(Node), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (num_samples + block_size - 1) / block_size;
    classifyKernel<<<grid_size, block_size>>>(d_data, d_results, d_tree, num_samples, num_features);
    
    // Copy results back to host
    cudaMemcpy(h_results, d_results, num_samples * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_data);
    cudaFree(d_results);
    cudaFree(d_tree);
}

// Example usage
int main() {
    // Example data (2 samples, 3 features each)
    float h_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int num_samples = 2;
    int num_features = 3;
    
    // Example decision tree (3 nodes)
    Node h_tree[] = {
        {0, 2.5f, 1, 2, -1},  // Root node
        {-1, 0.0f, -1, -1, 0},  // Leaf node (class 0)
        {-1, 0.0f, -1, -1, 1}   // Leaf node (class 1)
    };
    int num_nodes = 3;
    
    // Allocate memory for results
    int* h_results = (int*)malloc(num_samples * sizeof(int));
    
    // Classify using CUDA
    classifyDecisionTree(h_data, h_results, h_tree, num_samples, num_features, num_nodes);
    
    // Print results
    for (int i = 0; i < num_samples; i++) {
        printf("Sample %d classified as: %d\n", i, h_results[i]);
    }
    
    // Free host memory
    free(h_results);
    
    return 0;
}