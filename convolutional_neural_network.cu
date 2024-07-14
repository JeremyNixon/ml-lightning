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
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, 
                static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

// Hyperparameters
const int INPUT_SIZE = 784;  // 28x28
const int CONV_FILTERS = 8;
const int CONV_FILTER_SIZE = 3;
const int CONV_OUTPUT_SIZE = 26 * 26 * CONV_FILTERS;
const int NUM_CLASSES = 10;
const float LEARNING_RATE = 0.01f;
const int BATCH_SIZE = 32;
const int NUM_EPOCHS = 10;

// ReLU activation kernel
__global__ void relu_activation(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLU derivative kernel
__global__ void relu_derivative(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? 1.0f : 0.0f;
    }
}

// Softmax derivative kernel
__global__ void softmax_derivative(float* softmax_output, float* target, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = softmax_output[idx] - target[idx];
    }
}


// Convolution layer
__global__ void convolution_forward(float* input, float* filters, float* output, int input_size, int filter_size, int num_filters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = input_size - filter_size + 1;
    
    if (idx < stride * stride * num_filters) {
        int f = idx / (stride * stride);
        int y = (idx % (stride * stride)) / stride;
        int x = idx % stride;
        
        float sum = 0.0f;
        for (int fy = 0; fy < filter_size; fy++) {
            for (int fx = 0; fx < filter_size; fx++) {
                int input_idx = (y + fy) * input_size + (x + fx);
                int filter_idx = f * filter_size * filter_size + fy * filter_size + fx;
                sum += input[input_idx] * filters[filter_idx];
            }
        }
        output[idx] = sum;
    }
}

// Dense layer
__global__ void dense_forward(float* input, float* weights, float* output, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[i * output_size + idx];
        }
        output[idx] = sum;
    }
}

// Softmax
__global__ void softmax(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float max_val = -INFINITY;
        for (int i = 0; i < size; i++) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }
        
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += expf(input[i] - max_val);
        }
        
        output[idx] = expf(input[idx] - max_val) / sum;
    }
}

// Backpropagation kernels
__global__ void conv_backward(float* input, float* filters, float* output_grad, float* input_grad, float* filter_grad, int input_size, int filter_size, int num_filters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = input_size - filter_size + 1;
    
    if (idx < input_size * input_size) {
        int y = idx / input_size;
        int x = idx % input_size;
        
        float grad = 0.0f;
        for (int f = 0; f < num_filters; f++) {
            for (int fy = 0; fy < filter_size; fy++) {
                for (int fx = 0; fx < filter_size; fx++) {
                    if (y - fy >= 0 && y - fy < stride && x - fx >= 0 && x - fx < stride) {
                        int output_idx = f * stride * stride + (y - fy) * stride + (x - fx);
                        int filter_idx = f * filter_size * filter_size + fy * filter_size + fx;
                        grad += output_grad[output_idx] * filters[filter_idx];
                    }
                }
            }
        }
        input_grad[idx] = grad;
    }
}

__global__ void dense_backward(float* input, float* weights, float* output_grad, float* input_grad, float* weight_grad, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < input_size * output_size) {
        int i = idx / output_size;
        int j = idx % output_size;
        
        weight_grad[idx] = input[i] * output_grad[j];
        atomicAdd(&input_grad[i], weights[idx] * output_grad[j]);
    }
}

// Adam optimizer
struct AdamOptimizer {
    float* m;
    float* v;
    float beta1;
    float beta2;
    float epsilon;
    int size;
    int t;
    
    AdamOptimizer(int size) : size(size), beta1(0.9f), beta2(0.999f), epsilon(1e-8f), t(0) {
        cudaMalloc(&m, size * sizeof(float));
        cudaMalloc(&v, size * sizeof(float));
        cudaMemset(m, 0, size * sizeof(float));
        cudaMemset(v, 0, size * sizeof(float));
    }
    
    ~AdamOptimizer() {
        cudaFree(m);
        cudaFree(v);
    }
    
    __device__ void update(float* params, float* grads, float lr) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx < size) {
            t++;
            float m_t = beta1 * m[idx] + (1 - beta1) * grads[idx];
            float v_t = beta2 * v[idx] + (1 - beta2) * grads[idx] * grads[idx];
            float m_hat = m_t / (1 - powf(beta1, t));
            float v_hat = v_t / (1 - powf(beta2, t));
            params[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
            m[idx] = m_t;
            v[idx] = v_t;
        }
    }
};

// Main CNN class
class CNN {
public:
    float *conv_filters, *dense_weights;
    float *conv_output, *relu_output, *dense_output, *softmax_output;
    float *conv_grad, *dense_grad;
    AdamOptimizer *conv_optimizer, *dense_optimizer;
    
    CNN() {
        // Allocate memory and initialize weights
        cudaMalloc(&conv_filters, CONV_FILTERS * CONV_FILTER_SIZE * CONV_FILTER_SIZE * sizeof(float));
        cudaMalloc(&dense_weights, CONV_OUTPUT_SIZE * NUM_CLASSES * sizeof(float));
        cudaMalloc(&conv_output, CONV_OUTPUT_SIZE * sizeof(float));
        cudaMalloc(&relu_output, CONV_OUTPUT_SIZE * sizeof(float));
        cudaMalloc(&dense_output, NUM_CLASSES * sizeof(float));
        cudaMalloc(&softmax_output, NUM_CLASSES * sizeof(float));
        cudaMalloc(&conv_grad, CONV_FILTERS * CONV_FILTER_SIZE * CONV_FILTER_SIZE * sizeof(float));
        cudaMalloc(&dense_grad, CONV_OUTPUT_SIZE * NUM_CLASSES * sizeof(float));
        
        // Initialize optimizers
        conv_optimizer = new AdamOptimizer(CONV_FILTERS * CONV_FILTER_SIZE * CONV_FILTER_SIZE);
        dense_optimizer = new AdamOptimizer(CONV_OUTPUT_SIZE * NUM_CLASSES);
        
        // Initialize weights with random values
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateNormal(gen, conv_filters, CONV_FILTERS * CONV_FILTER_SIZE * CONV_FILTER_SIZE, 0.0f, 0.1f);
        curandGenerateNormal(gen, dense_weights, CONV_OUTPUT_SIZE * NUM_CLASSES, 0.0f, 0.1f);
        curandDestroyGenerator(gen);
    }
    
    ~CNN() {
        cudaFree(conv_filters);
        cudaFree(dense_weights);
        cudaFree(conv_output);
        cudaFree(relu_output);
        cudaFree(dense_output);
        cudaFree(softmax_output);
        cudaFree(conv_grad);
        cudaFree(dense_grad);
        delete conv_optimizer;
        delete dense_optimizer;
    }
    
    void forward(float* input) {
        convolution_forward<<<(CONV_OUTPUT_SIZE + 255) / 256, 256>>>(input, conv_filters, conv_output, 28, CONV_FILTER_SIZE, CONV_FILTERS);
        relu_activation<<<(CONV_OUTPUT_SIZE + 255) / 256, 256>>>(conv_output, relu_output, CONV_OUTPUT_SIZE);
        dense_forward<<<(NUM_CLASSES + 255) / 256, 256>>>(relu_output, dense_weights, dense_output, CONV_OUTPUT_SIZE, NUM_CLASSES);
        softmax<<<1, NUM_CLASSES>>>(dense_output, softmax_output, NUM_CLASSES);
    }

    void backward(float* input, float* target) {
        // Softmax derivative
        softmax_derivative<<<1, NUM_CLASSES>>>(softmax_output, target, dense_grad, NUM_CLASSES);

        // Dense layer backward
        cudaMemset(input_grad, 0, INPUT_SIZE * sizeof(float));
        dense_backward<<<(CONV_OUTPUT_SIZE * NUM_CLASSES + 255) / 256, 256>>>(
            relu_output, dense_weights, dense_grad, input_grad, dense_grad, CONV_OUTPUT_SIZE, NUM_CLASSES);

        // ReLU derivative
        float* relu_grad;
        cudaMalloc(&relu_grad, CONV_OUTPUT_SIZE * sizeof(float));
        relu_derivative<<<(CONV_OUTPUT_SIZE + 255) / 256, 256>>>(conv_output, relu_grad, CONV_OUTPUT_SIZE);

        // Element-wise multiplication of dense input grad and relu derivative
        // ... (implementation omitted for brevity)

        // Convolution layer backward
        conv_backward<<<(INPUT_SIZE + 255) / 256, 256>>>(
            input, conv_filters, input_grad, input_grad, conv_grad, 28, CONV_FILTER_SIZE, CONV_FILTERS);

        // Update weights using Adam optimizer
        conv_optimizer->update(conv_filters, conv_grad, LEARNING_RATE);
        dense_optimizer->update(dense_weights, dense_grad, LEARNING_RATE);

        cudaFree(relu_grad);
    }
    

    void train(float* train_data, float* train_labels, int num_samples, int num_epochs) {
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            float total_loss = 0.0f;
            for (int i = 0; i < num_samples; i += BATCH_SIZE) {
                float* batch_data = train_data + i * INPUT_SIZE;
                float* batch_labels = train_labels + i * NUM_CLASSES;

                forward(batch_data);
                backward(batch_data, batch_labels);

                // Compute loss
                float batch_loss;
                compute_loss<<<1, 1>>>(softmax_output, batch_labels, &batch_loss, BATCH_SIZE, NUM_CLASSES);
                cudaMemcpy(&total_loss, &batch_loss, sizeof(float), cudaMemcpyDeviceToHost);
                total_loss += batch_loss;
            }
            printf("Epoch %d completed, Average Loss: %f\n", epoch + 1, total_loss / num_samples);
        }
    }

    float* predict(float* test_data, int num_samples) {
        float* predictions;
        cudaMalloc(&predictions, num_samples * sizeof(float));

        for (int i = 0; i < num_samples; i++) {
            forward(test_data + i * INPUT_SIZE);
            int prediction;
            cudaMemcpy(&prediction, thrust::max_element(thrust::device, softmax_output, softmax_output + NUM_CLASSES) - softmax_output, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(predictions + i, &prediction, sizeof(float), cudaMemcpyHostToDevice);
        }

        return predictions;
    }
};

// MNIST data loading function
void load_mnist(const char* image_filename, const char* label_filename, 
                float** images, float** labels, int* num_images) {
    FILE* f = fopen(image_filename, "rb");
    if (f == NULL) {
        printf("Failed to open image file\n");
        exit(1);
    }

    int magic_number, n_rows, n_cols;
    fread(&magic_number, sizeof(int), 1, f);
    fread(num_images, sizeof(int), 1, f);
    fread(&n_rows, sizeof(int), 1, f);
    fread(&n_cols, sizeof(int), 1, f);

    magic_number = __builtin_bswap32(magic_number);
    *num_images = __builtin_bswap32(*num_images);
    n_rows = __builtin_bswap32(n_rows);
    n_cols = __builtin_bswap32(n_cols);

    *images = (float*)malloc(*num_images * n_rows * n_cols * sizeof(float));
    unsigned char* temp = (unsigned char*)malloc(*num_images * n_rows * n_cols);
    fread(temp, 1, *num_images * n_rows * n_cols, f);
    for (int i = 0; i < *num_images * n_rows * n_cols; ++i) {
        (*images)[i] = temp[i] / 255.0f;
    }
    free(temp);
    fclose(f);

    f = fopen(label_filename, "rb");
    if (f == NULL) {
        printf("Failed to open label file\n");
        exit(1);
    }

    fread(&magic_number, sizeof(int), 1, f);
    fread(num_images, sizeof(int), 1, f);
    magic_number = __builtin_bswap32(magic_number);
    *num_images = __builtin_bswap32(*num_images);

    *labels = (float*)malloc(*num_images * NUM_CLASSES * sizeof(float));
    temp = (unsigned char*)malloc(*num_images);
    fread(temp, 1, *num_images, f);
    for (int i = 0; i < *num_images; ++i) {
        for (int j = 0; j < NUM_CLASSES; ++j) {
            (*labels)[i * NUM_CLASSES + j] = (temp[i] == j) ? 1.0f : 0.0f;
        }
    }
    free(temp);
    fclose(f);
}

int main() {
    float *train_images, *train_labels, *test_images, *test_labels;
    int num_train_images, num_test_images;

    load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 
               &train_images, &train_labels, &num_train_images);
    load_mnist("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 
               &test_images, &test_labels, &num_test_images);

    float *d_train_images, *d_train_labels, *d_test_images, *d_test_labels;
    cudaMalloc(&d_train_images, num_train_images * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_train_labels, num_train_images * NUM_CLASSES * sizeof(float));
    cudaMalloc(&d_test_images, num_test_images * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_test_labels, num_test_images * NUM_CLASSES * sizeof(float));

    cudaMemcpy(d_train_images, train_images, num_train_images * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_labels, train_labels, num_train_images * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_images, test_images, num_test_images * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_labels, test_labels, num_test_images * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);

    CNN cnn;
    cnn.train(d_train_images, d_train_labels, num_train_images, NUM_EPOCHS);

    float* predictions = cnn.predict(d_test_images, num_test_images);
    
    // Compute accuracy
    int correct = 0;
    for (int i = 0; i < num_test_images; i++) {
        int true_label = 0;
        for (int j = 1; j < NUM_CLASSES; j++) {
            if (test_labels[i * NUM_CLASSES + j] > test_labels[i * NUM_CLASSES + true_label]) {
                true_label = j;
            }
        }
        if (predictions[i] == true_label) {
            correct++;
        }
    }
    float accuracy = (float)correct / num_test_images;
    printf("Test Accuracy: %.2f%%\n", accuracy * 100);

    // Free memory
    cudaFree(d_train_images);
    cudaFree(d_train_labels);
    cudaFree(d_test_images);
    cudaFree(d_test_labels);
    cudaFree(predictions);
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);

    return 0;
}