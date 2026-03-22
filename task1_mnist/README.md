# Task 1: MNIST Classification with OOP Design

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21+-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-green.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![pytest](https://img.shields.io/badge/pytest-9.0+-blue.svg)](https://pytest.org/)

A comprehensive implementation of MNIST digit classification using Object-Oriented Programming principles. This project provides three different machine learning approaches (Random Forest, Feed-Forward Neural Network, and Convolutional Neural Network) wrapped in a unified interface.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Demo](#demo)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Unified Interface**: Single `MnistClassifier` class supporting multiple algorithms
- **Three ML Approaches**:
  - Random Forest (traditional ML)
  - Feed-Forward Neural Network (deep learning)
  - Convolutional Neural Network (computer vision)
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix
- **Data Preprocessing**: Automatic data transformation for different model requirements
- **Extensive Testing**: Unit tests covering models, data processing, and evaluation
- **Interactive Demo**: Jupyter notebook with benchmarking and visualization
- **Logging**: Detailed logging for debugging and monitoring

## Project Structure

```
task1_mnist/
├── src/
│   ├── classifier/
│   │   └── mnist_classifier.py      # Unified classifier interface
│   ├── models/
│   │   ├── rf_classifier.py         # Random Forest implementation
│   │   ├── nn_classifier.py         # Neural Network implementation
│   │   └── cnn_classifier.py        # CNN implementation
│   ├── loaders/
│   │   └── mnist_loader.py          # Data loading utilities
│   ├── services/
│   │   └── evaluation.py            # Evaluation and metrics
│   ├── interfaces/
│   │   └── mnist_classifier_interface.py  # Abstract interface
│   └── utils/
│       └── logger.py                 # Logging utilities
├── tests/
│   ├── conftest.py                   # Test fixtures
│   ├── test_models.py               # Model tests
│   ├── test_loader.py               # Data loader tests
│   ├── test_evaluation.py           # Evaluation tests
│   └── test_mnist_classifier.py     # Integration tests
├── demo.ipynb                       # Interactive demonstration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/andriichernovv/winstars-ai-ds-test.git
   cd winstars-ai-ds-test
   ```

2. **Install the project in development mode**:
   ```bash
   pip install -e .[task1]
   ```

   Or install all dependencies via pyproject.toml:
   ```bash
   pip install -e .[task1,dev]
   ```

3. **Install additional development dependencies** (optional, for testing):
   ```bash
   pip install -e .[dev]
   ```

### Dependencies

- **Core**: `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- **Deep Learning**: `tensorflow`, `torch`, `torchvision`
- **Development**: `pytest`, `jupyter`, `black`, `ruff`

## Usage

### Basic Example

```python
import numpy as np
from src.classifier.mnist_classifier import MnistClassifier
from src.loaders.mnist_loader import load_mnist

# Load MNIST data
x_train, y_train, x_test, y_test = load_mnist()

# Create classifier (choose algorithm: 'rf', 'nn', or 'cnn')
classifier = MnistClassifier(algorithm='rf')

# Train the model
classifier.train(x_train, y_train)

# Make predictions
predictions = classifier.predict(x_test)

print(f"Predictions shape: {predictions.shape}")
print(f"Sample predictions: {predictions[:10]}")
```

### Advanced Usage with Evaluation

```python
from src.services.evaluation import EvaluationService

# Train and predict
classifier = MnistClassifier(algorithm='cnn')
classifier.train(x_train, y_train)
predictions = classifier.predict(x_test)

# Evaluate performance
evaluator = EvaluationService()
result = evaluator.evaluate(y_test, predictions)

print(f"Accuracy: {result.metrics_df.loc['accuracy', 'value']:.4f}")
print(f"F1 Score: {result.metrics_df.loc['f1_macro', 'value']:.4f}")

# Display confusion matrix
print(result.confusion_matrix_df)
```

## Models

### Random Forest Classifier (`rf`)
- **Input**: Flattened images (784 features)
- **Algorithm**: Ensemble of decision trees
- **Pros**: Fast training, interpretable, no hyperparameters to tune
- **Best for**: Baseline performance, quick prototyping

### Feed-Forward Neural Network (`nn`)
- **Input**: Flattened images (784 features)
- **Architecture**: 2 hidden layers (128, 64 neurons)
- **Pros**: Better performance than RF, handles non-linear patterns
- **Best for**: Balanced performance and computational cost

### Convolutional Neural Network (`cnn`)
- **Input**: 2D images with channel dimension (28×28×1)
- **Architecture**: Conv layers + MaxPooling + Dense layers
- **Pros**: State-of-the-art performance, leverages spatial structure
- **Best for**: Maximum accuracy, production use

## Evaluation

The evaluation service provides comprehensive metrics:

- **Global Metrics**: Accuracy, macro/micro averages
- **Per-Class Metrics**: Precision, recall, F1-score for each digit
- **Confusion Matrix**: Detailed error analysis
- **Probability Metrics**: Log loss, top-k accuracy (when probabilities available)
- **Error Analysis**: Most confused digit pairs

### Sample Output

```
Accuracy: 0.9876
F1 Score: 0.9875

Confusion Matrix:
   0     1     2     3     4     5     6     7     8     9
0  980   0     1     0     0     1     2     0     1     0
1    0  1134  1     0     0     0     1     1     0     0
...
```

## Testing

Run the test suite to ensure everything works correctly:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_mnist_classifier.py::TestMnistClassifier::test_init_valid_algorithm -v
```

### Test Coverage

- **Model Tests**: Individual classifier functionality
- **Data Loader Tests**: MNIST loading and preprocessing
- **Integration Tests**: End-to-end classifier workflow
- **Evaluation Tests**: Metrics calculation and edge cases
- **Error Handling**: Invalid inputs and edge cases

## Demo

Explore the interactive demonstration in `demo.ipynb`:

```bash
jupyter notebook demo.ipynb
```

The demo includes:
- **Benchmarking**: Compare all three models side-by-side
- **Visualization**: Training curves, confusion matrices, error analysis
- **Edge Cases**: Testing with various input scenarios
- **Performance Metrics**: Detailed evaluation reports

## API Reference

### MnistClassifier

```python
class MnistClassifier:
    def __init__(self, algorithm: str) -> None
    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None
    def predict(self, x: np.ndarray) -> np.ndarray
```

**Parameters:**
- `algorithm`: One of `'rf'`, `'nn'`, `'cnn'`

### EvaluationService

```python
class EvaluationService:
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_proba: np.ndarray = None, labels: list = None) -> EvaluationResult
```

**Returns:** `EvaluationResult` dataclass with metrics, confusion matrix, and analysis.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-model`
3. Write tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Follow code style: `black src/ tests/`
6. Commit changes: `git commit -m "Add new feature"`
7. Push to branch: `git push origin feature/new-model`
8. Create a Pull Request

### Code Style

- Use `black` for code formatting
- Use `ruff` for linting
- Follow PEP 8 conventions
- Add type hints for function parameters and return values
- Write comprehensive docstrings

## License

This project is part of the Winstars AI DS Test assignment. See the main repository for licensing information.

## Authors

- **Andrii Chernov** - *Initial implementation*

---

*This implementation demonstrates best practices in machine learning engineering, including modular design, comprehensive testing, and clear documentation.*
