#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_ITERATIONS 100
#define THREADS_PER_BLOCK 256

// CUDA kernel to assign points to clusters
__global__ void assignClusters(float* d_points, float* d_centroids, int* d_assignments, int n_points, int n_clusters, int dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_points) {
        float min_dist = 1e10;
        int closest_centroid = 0;
        for (int c = 0; c < n_clusters; c++) {
            float dist = 0;
            for (int d = 0; d < dim; d++) {
                float diff = d_points[tid * dim + d] - d_centroids[c * dim + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid = c;
            }
        }
        d_assignments[tid] = closest_centroid;
    }
}

// CUDA kernel to update centroids
__global__ void updateCentroids(float* d_points, float* d_centroids, int* d_assignments, int* d_cluster_sizes, int n_points, int n_clusters, int dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_clusters * dim) {
        int c = tid / dim;
        int d = tid % dim;
        float sum = 0;
        for (int i = 0; i < n_points; i++) {
            if (d_assignments[i] == c) {
                sum += d_points[i * dim + d];
            }
        }
        d_centroids[tid] = sum / d_cluster_sizes[c];
    }
}

// Host function to run K-Means
void kMeans(float* h_points, float* h_centroids, int* h_assignments, int n_points, int n_clusters, int dim) {
    // Allocate device memory
    float *d_points, *d_centroids;
    int *d_assignments, *d_cluster_sizes;
    cudaMalloc(&d_points, n_points * dim * sizeof(float));
    cudaMalloc(&d_centroids, n_clusters * dim * sizeof(float));
    cudaMalloc(&d_assignments, n_points * sizeof(int));
    cudaMalloc(&d_cluster_sizes, n_clusters * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_points, h_points, n_points * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, n_clusters * dim * sizeof(float), cudaMemcpyHostToDevice);

    // Main K-Means loop
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Assign clusters
        int blocks = (n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        assignClusters<<<blocks, THREADS_PER_BLOCK>>>(d_points, d_centroids, d_assignments, n_points, n_clusters, dim);

        // Count cluster sizes
        cudaMemset(d_cluster_sizes, 0, n_clusters * sizeof(int));
        for (int i = 0; i < n_points; i++) {
            int cluster = d_assignments[i];
            atomicAdd(&d_cluster_sizes[cluster], 1);
        }

        // Update centroids
        blocks = (n_clusters * dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        updateCentroids<<<blocks, THREADS_PER_BLOCK>>>(d_points, d_centroids, d_assignments, d_cluster_sizes, n_points, n_clusters, dim);
    }

    // Copy results back to host
    cudaMemcpy(h_assignments, d_assignments, n_points * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids, d_centroids, n_clusters * dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaFree(d_cluster_sizes);
}

int main() {
    // Example usage
    const int n_points = 1000000;
    const int n_clusters = 10;
    const int dim = 3;

    float* h_points = (float*)malloc(n_points * dim * sizeof(float));
    float* h_centroids = (float*)malloc(n_clusters * dim * sizeof(float));
    int* h_assignments = (int*)malloc(n_points * sizeof(int));

    // Initialize points and centroids (not shown for brevity)

    kMeans(h_points, h_centroids, h_assignments, n_points, n_clusters, dim);

    // Process results (not shown for brevity)

    free(h_points);
    free(h_centroids);
    free(h_assignments);

    return 0;
}