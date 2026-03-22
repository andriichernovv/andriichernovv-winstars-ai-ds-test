import numpy as np
import pytest


@pytest.fixture
def small_mnist_data():
    """Fixture providing small synthetic MNIST-like data for testing."""
    np.random.seed(42)
    # 100 samples of 28x28 images
    x = np.random.rand(100, 28, 28).astype(np.float32)
    # Random labels 0-9
    y = np.random.randint(0, 10, 100)
    return x, y


@pytest.fixture
def flattened_mnist_data(small_mnist_data):
    """Fixture providing flattened data for RF and NN models."""
    x, y = small_mnist_data
    x_flat = x.reshape(x.shape[0], -1)  # (100, 784)
    return x_flat, y


@pytest.fixture
def cnn_mnist_data(small_mnist_data):
    """Fixture providing data with channel dimension for CNN."""
    x, y = small_mnist_data
    x_cnn = x[..., np.newaxis]  # (100, 28, 28, 1)
    return x_cnn, y