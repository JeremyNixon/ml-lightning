#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for computing the sum of x, y, x^2, and xy
__global__ void computeSums(float *x, float *y, float *sum_x, float *sum_y, float *sum_xy, float *sum_x2, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ float s_sum_x[256];
    __shared__ float s_sum_y[256];
    __shared__ float s_sum_xy[256];
    __shared__ float s_sum_x2[256];

    float local_sum_x = 0, local_sum_y = 0, local_sum_xy = 0, local_sum_x2 = 0;

    for (int i = tid; i < n; i += stride) {
        local_sum_x += x[i];
        local_sum_y += y[i];
        local_sum_xy += x[i] * y[i];
        local_sum_x2 += x[i] * x[i];
    }

    s_sum_x[threadIdx.x] = local_sum_x;
    s_sum_y[threadIdx.x] = local_sum_y;
    s_sum_xy[threadIdx.x] = local_sum_xy;
    s_sum_x2[threadIdx.x] = local_sum_x2;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum_x[threadIdx.x] += s_sum_x[threadIdx.x + s];
            s_sum_y[threadIdx.x] += s_sum_y[threadIdx.x + s];
            s_sum_xy[threadIdx.x] += s_sum_xy[threadIdx.x + s];
            s_sum_x2[threadIdx.x] += s_sum_x2[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum_x, s_sum_x[0]);
        atomicAdd(sum_y, s_sum_y[0]);
        atomicAdd(sum_xy, s_sum_xy[0]);
        atomicAdd(sum_x2, s_sum_x2[0]);
    }
}

// Host function to perform linear regression
void linearRegression(float *h_x, float *h_y, int n, float *slope, float *intercept) {
    float *d_x, *d_y;
    float *d_sum_x, *d_sum_y, *d_sum_xy, *d_sum_x2;
    float h_sum_x = 0, h_sum_y = 0, h_sum_xy = 0, h_sum_x2 = 0;

    // Allocate device memory
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_sum_x, sizeof(float));
    cudaMalloc((void**)&d_sum_y, sizeof(float));
    cudaMalloc((void**)&d_sum_xy, sizeof(float));
    cudaMalloc((void**)&d_sum_x2, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize sum variables on device
    cudaMemset(d_sum_x, 0, sizeof(float));
    cudaMemset(d_sum_y, 0, sizeof(float));
    cudaMemset(d_sum_xy, 0, sizeof(float));
    cudaMemset(d_sum_x2, 0, sizeof(float));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    computeSums<<<gridSize, blockSize>>>(d_x, d_y, d_sum_x, d_sum_y, d_sum_xy, d_sum_x2, n);

    // Copy results back to host
    cudaMemcpy(&h_sum_x, d_sum_x, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum_y, d_sum_y, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum_xy, d_sum_xy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum_x2, d_sum_x2, sizeof(float), cudaMemcpyDeviceToHost);

    // Compute slope and intercept
    *slope = (n * h_sum_xy - h_sum_x * h_sum_y) / (n * h_sum_x2 - h_sum_x * h_sum_x);
    *intercept = (h_sum_y - *slope * h_sum_x) / n;

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_sum_x);
    cudaFree(d_sum_y);
    cudaFree(d_sum_xy);
    cudaFree(d_sum_x2);
}

int main() {
    const int N = 1000000;
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));

    // Initialize data (you should replace this with your actual data)
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)i / N;
        h_y[i] = 2 * h_x[i] + 1 + ((float)rand() / RAND_MAX - 0.5) * 0.1;  // y = 2x + 1 + some noise
    }

    float slope, intercept;
    linearRegression(h_x, h_y, N, &slope, &intercept);

    printf("Linear Regression Results:\n");
    printf("Slope: %f\n", slope);
    printf("Intercept: %f\n", intercept);

    free(h_x);
    free(h_y);

    return 0;
}