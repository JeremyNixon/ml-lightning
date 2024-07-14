#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

// Sigmoid activation function
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Sigmoid derivative
__device__ float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

// Initialize weights with random values
__global__ void init_weights(float* weights, int size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = curand_uniform(&state) - 0.5f;
    }
}

// Forward propagation kernel
__global__ void forward_propagation(float* input, float* weights, float* output, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[idx * input_size + i];
        }
        output[idx] = sigmoid(sum);
    }
}

// Backward propagation kernel
__global__ void backward_propagation(float* input, float* weights, float* output, float* error, float* delta, int input_size, int output_size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        delta[idx] = error[idx] * sigmoid_derivative(output[idx]);
        for (int i = 0; i < input_size; i++) {
            atomicAdd(&weights[idx * input_size + i], learning_rate * delta[idx] * input[i]);
        }
    }
}

// MLP structure
struct MLP {
    int input_size;
    int hidden_size;
    int output_size;
    float *weights1, *weights2;
    float *hidden, *output;
    float *error, *delta_output, *delta_hidden;
};

// Initialize MLP
void init_mlp(MLP* mlp, int input_size, int hidden_size, int output_size) {
    mlp->input_size = input_size;
    mlp->hidden_size = hidden_size;
    mlp->output_size = output_size;

    // Allocate memory for weights and intermediate results
    CHECK_CUDA_ERROR(cudaMalloc(&mlp->weights1, input_size * hidden_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&mlp->weights2, hidden_size * output_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&mlp->hidden, hidden_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&mlp->output, output_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&mlp->error, output_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&mlp->delta_output, output_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&mlp->delta_hidden, hidden_size * sizeof(float)));

    // Initialize weights
    int threads_per_block = 256;
    int blocks = (input_size * hidden_size + threads_per_block - 1) / threads_per_block;
    init_weights<<<blocks, threads_per_block>>>(mlp->weights1, input_size * hidden_size, time(NULL));
    blocks = (hidden_size * output_size + threads_per_block - 1) / threads_per_block;
    init_weights<<<blocks, threads_per_block>>>(mlp->weights2, hidden_size * output_size, time(NULL));
}

// Forward pass
void forward_pass(MLP* mlp, float* input) {
    int threads_per_block = 256;
    int blocks;

    // Input to hidden layer
    blocks = (mlp->hidden_size + threads_per_block - 1) / threads_per_block;
    forward_propagation<<<blocks, threads_per_block>>>(input, mlp->weights1, mlp->hidden, mlp->input_size, mlp->hidden_size);

    // Hidden to output layer
    blocks = (mlp->output_size + threads_per_block - 1) / threads_per_block;
    forward_propagation<<<blocks, threads_per_block>>>(mlp->hidden, mlp->weights2, mlp->output, mlp->hidden_size, mlp->output_size);
}

// Backward pass
void backward_pass(MLP* mlp, float* input, float* target, float learning_rate) {
    int threads_per_block = 256;
    int blocks;

    // Calculate output error
    blocks = (mlp->output_size + threads_per_block - 1) / threads_per_block;
    // Note: You need to implement a kernel to calculate the error (target - output)

    // Output to hidden layer
    backward_propagation<<<blocks, threads_per_block>>>(mlp->hidden, mlp->weights2, mlp->output, mlp->error, mlp->delta_output, mlp->hidden_size, mlp->output_size, learning_rate);

    // Hidden to input layer
    blocks = (mlp->hidden_size + threads_per_block - 1) / threads_per_block;
    // Note: You need to implement a kernel to calculate hidden layer error and update weights1
}

// Train the MLP
void train_mlp(MLP* mlp, float* input, float* target, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        forward_pass(mlp, input);
        backward_pass(mlp, input, target, learning_rate);
    }
}

// Free MLP memory
void free_mlp(MLP* mlp) {
    cudaFree(mlp->weights1);
    cudaFree(mlp->weights2);
    cudaFree(mlp->hidden);
    cudaFree(mlp->output);
    cudaFree(mlp->error);
    cudaFree(mlp->delta_output);
    cudaFree(mlp->delta_hidden);
}

// Main function (for demonstration)
int main() {
    MLP mlp;
    int input_size = 784;  // e.g., for MNIST
    int hidden_size = 128;
    int output_size = 10;

    init_mlp(&mlp, input_size, hidden_size, output_size);

    // Here you would load your training data and labels
    float *input, *target;
    // Allocate and initialize input and target...

    train_mlp(&mlp, input, target, 100, 0.01f);

    free_mlp(&mlp);
    return 0;
}