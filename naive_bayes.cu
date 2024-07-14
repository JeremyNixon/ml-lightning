#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAX_FEATURES 100
#define MAX_SAMPLES 10000

// Struct to hold the Naive Bayes model parameters
struct NBModel {
    float priors[2];
    float means[2][MAX_FEATURES];
    float variances[2][MAX_FEATURES];
};

// CUDA kernel for calculating means
__global__ void calculateMeans(float* data, int* labels, float* means, int n_samples, int n_features) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_features) {
        float sum_class0 = 0.0f, sum_class1 = 0.0f;
        int count_class0 = 0, count_class1 = 0;
        
        for (int i = 0; i < n_samples; i++) {
            if (labels[i] == 0) {
                sum_class0 += data[i * n_features + tid];
                count_class0++;
            } else {
                sum_class1 += data[i * n_features + tid];
                count_class1++;
            }
        }
        
        means[tid] = sum_class0 / count_class0;
        means[n_features + tid] = sum_class1 / count_class1;
    }
}

// CUDA kernel for calculating variances
__global__ void calculateVariances(float* data, int* labels, float* means, float* variances, int n_samples, int n_features) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_features) {
        float sum_sq_class0 = 0.0f, sum_sq_class1 = 0.0f;
        int count_class0 = 0, count_class1 = 0;
        
        for (int i = 0; i < n_samples; i++) {
            if (labels[i] == 0) {
                float diff = data[i * n_features + tid] - means[tid];
                sum_sq_class0 += diff * diff;
                count_class0++;
            } else {
                float diff = data[i * n_features + tid] - means[n_features + tid];
                sum_sq_class1 += diff * diff;
                count_class1++;
            }
        }
        
        variances[tid] = sum_sq_class0 / (count_class0 - 1);
        variances[n_features + tid] = sum_sq_class1 / (count_class1 - 1);
    }
}

// CUDA kernel for Naive Bayes prediction
__global__ void naiveBayesPredictKernel(float* data, NBModel* model, int* predictions, int n_samples, int n_features) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_samples) {
        float prob_class0 = logf(model->priors[0]);
        float prob_class1 = logf(model->priors[1]);
        
        for (int j = 0; j < n_features; j++) {
            float x = data[tid * n_features + j];
            float mean0 = model->means[0][j];
            float var0 = model->variances[0][j];
            float mean1 = model->means[1][j];
            float var1 = model->variances[1][j];
            
            prob_class0 += -0.5f * logf(2 * M_PI * var0) - 0.5f * ((x - mean0) * (x - mean0)) / var0;
            prob_class1 += -0.5f * logf(2 * M_PI * var1) - 0.5f * ((x - mean1) * (x - mean1)) / var1;
        }
        
        predictions[tid] = (prob_class1 > prob_class0) ? 1 : 0;
    }
}

// Function to train the Naive Bayes model
void trainNaiveBayes(float* data, int* labels, NBModel* model, int n_samples, int n_features) {
    float* d_data, *d_means, *d_variances;
    int* d_labels;
    
    // Allocate device memory
    cudaMalloc(&d_data, n_samples * n_features * sizeof(float));
    cudaMalloc(&d_labels, n_samples * sizeof(int));
    cudaMalloc(&d_means, 2 * n_features * sizeof(float));
    cudaMalloc(&d_variances, 2 * n_features * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_data, data, n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, n_samples * sizeof(int), cudaMemcpyHostToDevice);
    
    // Calculate means
    int block_size = 256;
    int grid_size = (n_features + block_size - 1) / block_size;
    calculateMeans<<<grid_size, block_size>>>(d_data, d_labels, d_means, n_samples, n_features);
    
    // Calculate variances
    calculateVariances<<<grid_size, block_size>>>(d_data, d_labels, d_means, d_variances, n_samples, n_features);
    
    // Copy results back to host
    cudaMemcpy(model->means, d_means, 2 * n_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(model->variances, d_variances, 2 * n_features * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate priors
    int count_class0 = 0, count_class1 = 0;
    for (int i = 0; i < n_samples; i++) {
        if (labels[i] == 0) count_class0++;
        else count_class1++;
    }
    model->priors[0] = (float)count_class0 / n_samples;
    model->priors[1] = (float)count_class1 / n_samples;
    
    // Free device memory
    cudaFree(d_data);
    cudaFree(d_labels);
    cudaFree(d_means);
    cudaFree(d_variances);
}

// Function to make predictions using the trained model
void predict(float* data, NBModel* model, int* predictions, int n_samples, int n_features) {
    float* d_data;
    int* d_predictions;
    NBModel* d_model;
    
    // Allocate device memory
    cudaMalloc(&d_data, n_samples * n_features * sizeof(float));
    cudaMalloc(&d_predictions, n_samples * sizeof(int));
    cudaMalloc(&d_model, sizeof(NBModel));
    
    // Copy data to device
    cudaMemcpy(d_data, data, n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model, model, sizeof(NBModel), cudaMemcpyHostToDevice);
    
    // Make predictions
    int block_size = 256;
    int grid_size = (n_samples + block_size - 1) / block_size;
    naiveBayesPredictKernel<<<grid_size, block_size>>>(d_data, d_model, d_predictions, n_samples, n_features);
    
    // Copy results back to host
    cudaMemcpy(predictions, d_predictions, n_samples * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_data);
    cudaFree(d_predictions);
    cudaFree(d_model);
}

int main() {
    // Example usage
    int n_samples = 1000;
    int n_features = 10;
    
    float* data = (float*)malloc(n_samples * n_features * sizeof(float));
    int* labels = (int*)malloc(n_samples * sizeof(int));
    
    // Initialize data and labels (you should load your actual data here)
    for (int i = 0; i < n_samples * n_features; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 10;
    }
    for (int i = 0; i < n_samples; i++) {
        labels[i] = rand() % 2;
    }
    
    // Train the model
    NBModel model;
    trainNaiveBayes(data, labels, &model, n_samples, n_features);
    
    // Make predictions
    int* predictions = (int*)malloc(n_samples * sizeof(int));
    predict(data, &model, predictions, n_samples, n_features);
    
    // Print some predictions
    printf("Some predictions:\n");
    for (int i = 0; i < 10; i++) {
        printf("Sample %d: Predicted %d, Actual %d\n", i, predictions[i], labels[i]);
    }
    
    // Free memory
    free(data);
    free(labels);
    free(predictions);
    
    return 0;
}