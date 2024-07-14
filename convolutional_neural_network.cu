#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Constants
const int MNIST_IMAGE_SIZE = 28;
const int MNIST_PIXELS = MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE;
const int NUM_CLASSES = 10;
const int BATCH_SIZE = 32;
const int NUM_FILTERS = 8;
const int FILTER_SIZE = 3;
const int CONV_OUTPUT_SIZE = MNIST_IMAGE_SIZE - FILTER_SIZE + 1;
const int CONV_OUTPUT_PIXELS = CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE;
const float LEARNING_RATE = 0.01f;
const int NUM_ITERATIONS = 1000;

// Kernel functions

__global__ void im2col_kernel(const float* data_im, int channels, int height, int width, 
                              int kernel_h, int kernel_w, int pad_h, int pad_w, 
                              int stride_h, int stride_w, int dilation_h, int dilation_w, 
                              int height_col, int width_col, float* data_col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int height_col_size = height_col * width_col;
    int channel_col_size = height_col_size * channels * kernel_h * kernel_w;

    if (index < channel_col_size) {
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in % channels;
        int h_in = h_out * stride_h - pad_h;
        int w_in = w_out * stride_w - pad_w;
        int kernel_index = channel_in / channels;
        int h_kernel = kernel_index / kernel_w;
        int w_kernel = kernel_index % kernel_w;
        h_in += dilation_h * h_kernel;
        w_in += dilation_w * w_kernel;

        if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
            data_col[index] = data_im[(channel_out * height + h_in) * width + w_in];
        } else {
            data_col[index] = 0;
        }
    }
}

__global__ void col2im_kernel(const float* data_col, int channels, int height, int width,
                              int kernel_h, int kernel_w, int pad_h, int pad_w,
                              int stride_h, int stride_w, int dilation_h, int dilation_w,
                              int height_col, int width_col, float* data_im) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int height_col_size = height_col * width_col;
    int channel_col_size = height_col_size * channels * kernel_h * kernel_w;

    if (index < channel_col_size) {
        float val = 0;
        int w = index % width;
        int h = (index / width) % height;
        int c = index / (width * height);

        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int h_im = h - kernel_row * dilation_h + pad_h;
                int w_im = w - kernel_col * dilation_w + pad_w;
                if (h_im % stride_h == 0 && w_im % stride_w == 0) {
                    h_im /= stride_h;
                    w_im /= stride_w;
                    if (h_im >= 0 && h_im < height_col && w_im >= 0 && w_im < width_col) {
                        int col_index = (((c * kernel_h + kernel_row) * kernel_w + kernel_col) * height_col + h_im) * width_col + w_im;
                        val += data_col[col_index];
                    }
                }
            }
        }
        data_im[index] = val;
    }
}

__global__ void convolutionForward(float* input, float* filters, float* output, int inputSize, int filterSize, int numFilters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE * NUM_FILTERS * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE) {
        int n = idx / (NUM_FILTERS * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
        int f = (idx / (CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE)) % NUM_FILTERS;
        int h = (idx / CONV_OUTPUT_SIZE) % CONV_OUTPUT_SIZE;
        int w = idx % CONV_OUTPUT_SIZE;
        
        float sum = 0.0f;
        for (int c = 0; c < 1; c++) {  // Assuming single-channel input for MNIST
            for (int kh = 0; kh < FILTER_SIZE; kh++) {
                for (int kw = 0; kw < FILTER_SIZE; kw++) {
                    int im_row = h + kh;
                    int im_col = w + kw;
                    int im_idx = ((n * 1 + c) * FILTER_SIZE * FILTER_SIZE + kh * FILTER_SIZE + kw) * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE + h * CONV_OUTPUT_SIZE + w;
                    int filter_idx = ((f * 1 + c) * FILTER_SIZE + kh) * FILTER_SIZE + kw;
                    sum += input[im_idx] * filters[filter_idx];
                }
            }
        }
        output[idx] = sum;
    }
}

__global__ void reluActivation(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void denseForward(float* input, float* weights, float* output, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE * outputSize) {
        int sample = idx / outputSize;
        int neuron = idx % outputSize;
        float sum = 0.0f;
        for (int i = 0; i < inputSize; i++) {
            sum += input[sample * inputSize + i] * weights[i * outputSize + neuron];
        }
        output[idx] = sum;
    }
}

__global__ void softmaxActivation(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE) {
        float max_val = -INFINITY;
        for (int i = 0; i < NUM_CLASSES; i++) {
            max_val = fmaxf(max_val, input[idx * NUM_CLASSES + i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < NUM_CLASSES; i++) {
            sum += expf(input[idx * NUM_CLASSES + i] - max_val);
        }
        for (int i = 0; i < NUM_CLASSES; i++) {
            output[idx * NUM_CLASSES + i] = expf(input[idx * NUM_CLASSES + i] - max_val) / sum;
        }
    }
}

__global__ void computeLoss(float* predictions, int* labels, float* loss, int batchSize, int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        int label = labels[idx];
        float pred = predictions[idx * numClasses + label];
        atomicAdd(loss, -logf(fmaxf(pred, 1e-15f)));
    }
}

__global__ void softmaxGradient(float* softmax_output, int* labels, float* gradient, int batchSize, int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize * numClasses) {
        int sample = idx / numClasses;
        int class_idx = idx % numClasses;
        float y = (class_idx == labels[sample]) ? 1.0f : 0.0f;
        gradient[idx] = (softmax_output[idx] - y) / batchSize;
    }
}

__global__ void denseBackward(float* input, float* weights, float* output_gradient, float* input_gradient, float* weight_gradient, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE * inputSize) {
        int sample = idx / inputSize;
        int input_idx = idx % inputSize;
        float sum = 0.0f;
        for (int j = 0; j < outputSize; j++) {
            float out_grad = output_gradient[sample * outputSize + j];
            sum += weights[input_idx * outputSize + j] * out_grad;
            atomicAdd(&weight_gradient[input_idx * outputSize + j], input[idx] * out_grad);
        }
        input_gradient[idx] = sum;
    }
}

__global__ void reluGradient(float* input, float* gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradient[idx] *= (input[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

__global__ void convolutionBackward(float* input, float* filters, float* output_gradient, float* input_gradient, float* filter_gradient, int inputSize, int filterSize, int numFilters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE * NUM_FILTERS * FILTER_SIZE * FILTER_SIZE) {
        int n = idx / (NUM_FILTERS * FILTER_SIZE * FILTER_SIZE);
        int f = (idx / (FILTER_SIZE * FILTER_SIZE)) % NUM_FILTERS;
        int kh = (idx / FILTER_SIZE) % FILTER_SIZE;
        int kw = idx % FILTER_SIZE;
        
        float sum = 0.0f;
        for (int h = 0; h < CONV_OUTPUT_SIZE; h++) {
            for (int w = 0; w < CONV_OUTPUT_SIZE; w++) {
                int im_idx = ((n * 1 + 0) * FILTER_SIZE * FILTER_SIZE + kh * FILTER_SIZE + kw) * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE + h * CONV_OUTPUT_SIZE + w;
                int out_idx = ((n * NUM_FILTERS + f) * CONV_OUTPUT_SIZE + h) * CONV_OUTPUT_SIZE + w;
                sum += input[im_idx] * output_gradient[out_idx];
            }
        }
        filter_gradient[f * FILTER_SIZE * FILTER_SIZE + kh * FILTER_SIZE + kw] += sum;
    }
}

__global__ void updateParameters(float* params, float* gradients, int size, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= learningRate * gradients[idx];
    }
}

// Helper functions

void im2col_gpu(const float* data_im, int channels, int height, int width,
                int kernel_h, int kernel_w, int pad_h, int pad_w,
                int stride_h, int stride_w, int dilation_h, int dilation_w,
                float* data_col) {
    int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;
    
    im2col_kernel<<<(num_kernels + 255) / 256, 256>>>(
        data_im, channels, height, width, kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
        height_col, width_col, data_col);
}

void col2im_gpu(const float* data_col, int channels, int height, int width,
                int kernel_h, int kernel_w, int pad_h, int pad_w,
                int stride_h, int stride_w, int dilation_h, int dilation_w,
                float* data_im) {
    int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height * width;
    
    // Initialize data_im to all zeros
    cudaMemset(data_im, 0, sizeof(float) * num_kernels);
    
    col2im_kernel<<<(num_kernels + 255) / 256, 256>>>(
        data_col, channels, height, width, kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
        height_col, width_col, data_im);
}

void loadMNISTData(const char* filename, float** images, int** labels, int* numImages) {
    // This function should load MNIST data from a file
    // For brevity, we'll just allocate random data here
    *numImages = 10000;  // Using a smaller dataset for demonstration
    *images = (float*)malloc(*numImages * MNIST_PIXELS * sizeof(float));
    *labels = (int*)malloc(*numImages * sizeof(int));

    for (int i = 0; i < *numImages * MNIST_PIXELS; i++) {
            (*images)[i] = (float)rand() / RAND_MAX;
        }
        for (int i = 0; i < *numImages; i++) {
            (*labels)[i] = rand() % NUM_CLASSES;
        }
    }

// Main function
int main() {
    // Load MNIST data
    float* h_train_images;
    int* h_train_labels;
    int numTrainImages;
    loadMNISTData("train-images-idx3-ubyte", &h_train_images, &h_train_labels, &numTrainImages);

    // Allocate device memory
    float *d_images, *d_conv_filters, *d_conv_output, *d_relu_output, *d_dense_weights, *d_dense_output, *d_softmax_output;
    int *d_labels;
    float *d_loss, *d_softmax_gradient, *d_dense_gradient, *d_relu_gradient, *d_conv_gradient;
    float *d_conv_filter_gradient, *d_dense_weight_gradient;
    float *d_col_data, *d_col_grad;

    CHECK_CUDA(cudaMalloc((void**)&d_images, BATCH_SIZE * MNIST_PIXELS * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_labels, BATCH_SIZE * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_conv_filters, NUM_FILTERS * FILTER_SIZE * FILTER_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_conv_output, BATCH_SIZE * NUM_FILTERS * CONV_OUTPUT_PIXELS * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_relu_output, BATCH_SIZE * NUM_FILTERS * CONV_OUTPUT_PIXELS * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_dense_weights, (NUM_FILTERS * CONV_OUTPUT_PIXELS * NUM_CLASSES) * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_dense_output, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_softmax_output, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_loss, sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_softmax_gradient, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_dense_gradient, BATCH_SIZE * NUM_FILTERS * CONV_OUTPUT_PIXELS * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_relu_gradient, BATCH_SIZE * NUM_FILTERS * CONV_OUTPUT_PIXELS * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_conv_gradient, BATCH_SIZE * MNIST_PIXELS * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_conv_filter_gradient, NUM_FILTERS * FILTER_SIZE * FILTER_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_dense_weight_gradient, (NUM_FILTERS * CONV_OUTPUT_PIXELS * NUM_CLASSES) * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_col_data, BATCH_SIZE * NUM_FILTERS * FILTER_SIZE * FILTER_SIZE * CONV_OUTPUT_PIXELS * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_col_grad, BATCH_SIZE * NUM_FILTERS * FILTER_SIZE * FILTER_SIZE * CONV_OUTPUT_PIXELS * sizeof(float)));

    // Initialize weights
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, 1234ULL);
    curandGenerateUniform(prng, d_conv_filters, NUM_FILTERS * FILTER_SIZE * FILTER_SIZE);
    curandGenerateUniform(prng, d_dense_weights, NUM_FILTERS * CONV_OUTPUT_PIXELS * NUM_CLASSES);
    curandDestroyGenerator(prng);

    // Training loop
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        // Select random batch
        int batchStart = rand() % (numTrainImages - BATCH_SIZE);
        CHECK_CUDA(cudaMemcpy(d_images, h_train_images + batchStart * MNIST_PIXELS, BATCH_SIZE * MNIST_PIXELS * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_labels, h_train_labels + batchStart, BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));

        // Forward pass
        im2col_gpu(d_images, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, FILTER_SIZE, FILTER_SIZE, 0, 0, 1, 1, 1, 1, d_col_data);
        convolutionForward<<<(BATCH_SIZE * NUM_FILTERS * CONV_OUTPUT_PIXELS + 255) / 256, 256>>>(d_col_data, d_conv_filters, d_conv_output, MNIST_IMAGE_SIZE, FILTER_SIZE, NUM_FILTERS);

        int reluSize = BATCH_SIZE * NUM_FILTERS * CONV_OUTPUT_PIXELS;
        reluActivation<<<(reluSize + 255) / 256, 256>>>(d_conv_output, d_relu_output, reluSize);

        int denseInputSize = NUM_FILTERS * CONV_OUTPUT_PIXELS;
        denseForward<<<(BATCH_SIZE * NUM_CLASSES + 255) / 256, 256>>>(d_relu_output, d_dense_weights, d_dense_output, denseInputSize, NUM_CLASSES);

        softmaxActivation<<<BATCH_SIZE, 1>>>(d_dense_output, d_softmax_output, NUM_CLASSES);

        // Compute loss
        CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));
        computeLoss<<<(BATCH_SIZE + 255) / 256, 256>>>(d_softmax_output, d_labels, d_loss, BATCH_SIZE, NUM_CLASSES);

        float h_loss;
        CHECK_CUDA(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        h_loss /= BATCH_SIZE;

        if (iter % 10 == 0) {
            printf("Iteration %d, Loss: %f\n", iter, h_loss);
        }

        // Backward pass
        softmaxGradient<<<(BATCH_SIZE * NUM_CLASSES + 255) / 256, 256>>>(d_softmax_output, d_labels, d_softmax_gradient, BATCH_SIZE, NUM_CLASSES);

        denseBackward<<<(BATCH_SIZE * denseInputSize + 255) / 256, 256>>>(d_relu_output, d_dense_weights, d_softmax_gradient, d_dense_gradient, d_dense_weight_gradient, denseInputSize, NUM_CLASSES);

        reluGradient<<<(reluSize + 255) / 256, 256>>>(d_conv_output, d_dense_gradient, reluSize);

        convolutionBackward<<<(BATCH_SIZE * NUM_FILTERS * FILTER_SIZE * FILTER_SIZE + 255) / 256, 256>>>(d_col_data, d_conv_filters, d_dense_gradient, d_col_grad, d_conv_filter_gradient, MNIST_IMAGE_SIZE, FILTER_SIZE, NUM_FILTERS);

        col2im_gpu(d_col_grad, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, FILTER_SIZE, FILTER_SIZE, 0, 0, 1, 1, 1, 1, d_conv_gradient);

        // Update parameters
        updateParameters<<<(NUM_FILTERS * FILTER_SIZE * FILTER_SIZE + 255) / 256, 256>>>(d_conv_filters, d_conv_filter_gradient, NUM_FILTERS * FILTER_SIZE * FILTER_SIZE, LEARNING_RATE);
        updateParameters<<<(NUM_FILTERS * CONV_OUTPUT_PIXELS * NUM_CLASSES + 255) / 256, 256>>>(d_dense_weights, d_dense_weight_gradient, NUM_FILTERS * CONV_OUTPUT_PIXELS * NUM_CLASSES, LEARNING_RATE);

        // Reset gradients
        cudaMemset(d_conv_filter_gradient, 0, NUM_FILTERS * FILTER_SIZE * FILTER_SIZE * sizeof(float));
        cudaMemset(d_dense_weight_gradient, 0, NUM_FILTERS * CONV_OUTPUT_PIXELS * NUM_CLASSES * sizeof(float));
    }

    // Evaluate the model (you can add your own evaluation code here)

    // Free device memory
    cudaFree(d_images);
    cudaFree(d_labels);
    cudaFree(d_conv_filters);
    cudaFree(d_conv_output);
    cudaFree(d_relu_output);
    cudaFree(d_dense_weights);
    cudaFree(d_dense_output);
    cudaFree(d_softmax_output);
    cudaFree(d_loss);
    cudaFree(d_softmax_gradient);
    cudaFree(d_dense_gradient);
    cudaFree(d_relu_gradient);
    cudaFree(d_conv_gradient);
    cudaFree(d_conv_filter_gradient);
    cudaFree(d_dense_weight_gradient);
    cudaFree(d_col_data);
    cudaFree(d_col_grad);

    // Free host memory
    free(h_train_images);
    free(h_train_labels);

    return 0;
}

