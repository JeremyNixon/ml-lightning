# ML Lightning

ml-lightning is a powerful machine learning library that provides CUDA-accelerated implementations of popular machine learning algorithms. It's designed to leverage the power of GPUs to significantly speed up training and inference for a wide range of machine learning tasks.

## Features

ml-lightning includes CUDA-accelerated versions of the following algorithms:

1. Random Forests
2. Neural Network Classifier
3. Convolutional Neural Network
4. Bagged Decision Trees
5. Decision Trees for Classification
6. Linear Regression (with Lasso and Ridge penalties)
7. K-Nearest Neighbors
8. K-Means Clustering (Lloyd's Algorithm)
9. K-Means++
10. Perceptron
11. Naive Bayes
12. Logistic Regression

## Installation

```bash
pip install ml-lightning
```

## Requirements

- Python 3.7+
- CUDA-compatible GPU
- CUDA Toolkit 11.0+

## Quick Start

Here's a simple example of how to use ml-lightning:

```python
import ml_lightning as mll

# Load your data
X, y = mll.load_data("your_dataset.csv")

# Create and train a Random Forest model
rf_model = mll.RandomForest(n_estimators=100)
rf_model.fit(X, y)

# Make predictions
predictions = rf_model.predict(X_test)
```

## Documentation

For detailed documentation on each algorithm and its usage, please visit our [official documentation](https://ml-lightning.readthedocs.io).

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more information on how to get started.

## License

ml-lightning is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Citation

If you use ml-lightning in your research, please cite it as follows:

```
@software{ml_lightning,
  title = {ml-lightning: CUDA-accelerated Machine Learning Algorithms},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/ml-lightning}
}
```

## Support

For questions, issues, or feature requests, please open an issue on our [GitHub repository](https://github.com/yourusername/ml-lightning/issues).

