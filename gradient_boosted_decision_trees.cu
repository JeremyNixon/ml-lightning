#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <iostream>
#include <vector>

// Constants
const int MAX_DEPTH = 5;
const int NUM_TREES = 100;
const float LEARNING_RATE = 0.1f;

// Structure to represent a node in the decision tree
struct Node {
    float split_value;
    int feature_index;
    bool is_leaf;
    float leaf_value;
    int left_child;
    int right_child;
};

// Kernel to compute gradients
__global__ void computeGradientsKernel(float* y, float* y_pred, float* gradients, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        gradients[idx] = y_pred[idx] - y[idx];
    }
}

// Kernel to find the best split
__global__ void findBestSplitKernel(float* features, float* gradients, int n, int num_features, 
                                    float* split_values, float* split_gains) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature_idx < num_features) {
        float best_gain = 0.0f;
        float best_split = 0.0f;

        for (int i = 0; i < n; ++i) {
            float split = features[feature_idx * n + i];
            float left_sum = 0.0f, right_sum = 0.0f;
            int left_count = 0, right_count = 0;

            for (int j = 0; j < n; ++j) {
                if (features[feature_idx * n + j] <= split) {
                    left_sum += gradients[j];
                    left_count++;
                } else {
                    right_sum += gradients[j];
                    right_count++;
                }
            }

            float gain = (left_sum * left_sum / left_count) + (right_sum * right_sum / right_count);
            if (gain > best_gain) {
                best_gain = gain;
                best_split = split;
            }
        }

        split_values[feature_idx] = best_split;
        split_gains[feature_idx] = best_gain;
    }
}

// Function to build a single tree
Node* buildTree(thrust::device_vector<float>& features, thrust::device_vector<float>& gradients, 
                int n, int num_features, int max_depth) {
    std::vector<Node> tree(std::pow(2, max_depth + 1) - 1);
    int node_count = 0;

    // Allocate memory for split values and gains
    thrust::device_vector<float> d_split_values(num_features);
    thrust::device_vector<float> d_split_gains(num_features);

    // Build the tree recursively
    std::function<void(int, int, int, int)> buildTreeRecursive = [&](int node_id, int start, int end, int depth) {
        if (depth >= max_depth || end - start <= 1) {
            // Create a leaf node
            tree[node_id].is_leaf = true;
            tree[node_id].leaf_value = thrust::reduce(gradients.begin() + start, gradients.begin() + end) / (end - start);
            return;
        }

        // Find the best split
        int num_threads = 256;
        int num_blocks = (num_features + num_threads - 1) / num_threads;
        findBestSplitKernel<<<num_blocks, num_threads>>>(
            thrust::raw_pointer_cast(features.data()), 
            thrust::raw_pointer_cast(gradients.data()), 
            end - start, num_features,
            thrust::raw_pointer_cast(d_split_values.data()),
            thrust::raw_pointer_cast(d_split_gains.data())
        );
        cudaDeviceSynchronize();

        // Find the feature with the best gain
        thrust::device_vector<float>::iterator max_it = thrust::max_element(d_split_gains.begin(), d_split_gains.end());
        int best_feature = max_it - d_split_gains.begin();
        float best_split = d_split_values[best_feature];

        // Create an internal node
        tree[node_id].is_leaf = false;
        tree[node_id].feature_index = best_feature;
        tree[node_id].split_value = best_split;
        tree[node_id].left_child = ++node_count;
        tree[node_id].right_child = ++node_count;

        // Partition the data
        int mid = thrust::partition(thrust::device, 
            features.begin() + start, features.begin() + end,
            [best_feature, best_split, n] __device__ (float val) {
                return val <= best_split;
            }) - features.begin();

        // Recursively build left and right subtrees
        buildTreeRecursive(tree[node_id].left_child, start, mid, depth + 1);
        buildTreeRecursive(tree[node_id].right_child, mid, end, depth + 1);
    };

    buildTreeRecursive(0, 0, n, 0);
    return tree.data();
}

// Main GBDT class
class GradientBoostedDecisionTrees {
private:
    std::vector<Node*> trees;
    int num_trees;
    float learning_rate;

public:
    GradientBoostedDecisionTrees(int num_trees = NUM_TREES, float learning_rate = LEARNING_RATE) 
        : num_trees(num_trees), learning_rate(learning_rate) {}

    void fit(thrust::device_vector<float>& X, thrust::device_vector<float>& y, int n, int num_features) {
        thrust::device_vector<float> y_pred(n, 0.0f);
        thrust::device_vector<float> gradients(n);

        for (int i = 0; i < num_trees; ++i) {
            // Compute gradients
            int num_threads = 256;
            int num_blocks = (n + num_threads - 1) / num_threads;
            computeGradientsKernel<<<num_blocks, num_threads>>>(
                thrust::raw_pointer_cast(y.data()),
                thrust::raw_pointer_cast(y_pred.data()),
                thrust::raw_pointer_cast(gradients.data()),
                n
            );
            cudaDeviceSynchronize();

            // Build a tree
            Node* tree = buildTree(X, gradients, n, num_features, MAX_DEPTH);
            trees.push_back(tree);

            // Update predictions
            for (int j = 0; j < n; ++j) {
                y_pred[j] += learning_rate * predict(X, j, num_features, tree);
            }
        }
    }

    float predict(thrust::device_vector<float>& X, int sample_idx, int num_features, Node* tree) {
        Node* node = tree;
        while (!node->is_leaf) {
            if (X[node->feature_index * num_features + sample_idx] <= node->split_value) {
                node = tree + node->left_child;
            } else {
                node = tree + node->right_child;
            }
        }
        return node->leaf_value;
    }

    thrust::device_vector<float> predict(thrust::device_vector<float>& X, int n, int num_features) {
        thrust::device_vector<float> predictions(n, 0.0f);
        for (Node* tree : trees) {
            for (int i = 0; i < n; ++i) {
                predictions[i] += learning_rate * predict(X, i, num_features, tree);
            }
        }
        return predictions;
    }
};

int main() {
    // Example usage
    int n = 1000;
    int num_features = 10;

    // Generate random data
    thrust::host_vector<float> h_X(n * num_features);
    thrust::host_vector<float> h_y(n);
    for (int i = 0; i < n * num_features; ++i) {
        h_X[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < n; ++i) {
        h_y[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Copy data to device
    thrust::device_vector<float> d_X = h_X;
    thrust::device_vector<float> d_y = h_y;

    // Create and train the GBDT model
    GradientBoostedDecisionTrees gbdt;
    gbdt.fit(d_X, d_y, n, num_features);

    // Make predictions
    thrust::device_vector<float> predictions = gbdt.predict(d_X, n, num_features);

    // Copy predictions back to host
    thrust::host_vector<float> h_predictions = predictions;

    // Print some predictions
    std::cout << "First 10 predictions:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_predictions[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}