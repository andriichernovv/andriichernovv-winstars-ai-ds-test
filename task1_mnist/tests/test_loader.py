import numpy as np
import pytest

from src.loaders.mnist_loader import load_mnist, flatten_images, add_channel_dimension


class TestMnistLoader:
    """Test the MNIST data loader."""

    def test_load_mnist_structure(self):
        """Test that load_mnist returns data with correct structure."""
        x_train, y_train, x_test, y_test = load_mnist()

        # Check shapes
        assert x_train.shape == (60000, 28, 28)
        assert y_train.shape == (60000,)
        assert x_test.shape == (10000, 28, 28)
        assert y_test.shape == (10000,)

        # Check data types
        assert x_train.dtype == np.float32
        assert y_train.dtype == np.uint8

        # Check label range
        assert y_train.min() == 0
        assert y_train.max() == 9

    def test_load_mnist_subset(self):
        """Test loading a subset of MNIST data."""
        subset_size = 1000
        x_train, y_train, x_test, y_test = load_mnist(train_size=subset_size, test_size=subset_size)

        assert x_train.shape == (subset_size, 28, 28)
        assert y_train.shape == (subset_size,)
        assert x_test.shape == (subset_size, 28, 28)
        assert y_test.shape == (subset_size,)

    def test_flatten_images(self, small_mnist_data):
        """Test flattening images."""
        x, y = small_mnist_data
        x_flat = flatten_images(x)

        assert x_flat.shape == (x.shape[0], 28 * 28)
        assert x_flat.dtype == x.dtype

    def test_add_channel_dimension(self, small_mnist_data):
        """Test adding channel dimension for CNN."""
        x, y = small_mnist_data
        x_cnn = add_channel_dimension(x)

        assert x_cnn.shape == (x.shape[0], 28, 28, 1)
        assert x_cnn.dtype == x.dtype

    def test_load_mnist_no_normalization(self):
        """Test loading MNIST without normalization."""
        x_train, y_train, x_test, y_test = load_mnist(normalize=False)

        # Check that values are in [0, 255] range
        assert x_train.min() >= 0
        assert x_train.max() <= 255
        assert x_train.dtype == np.uint8

    def test_flatten_images_single_image(self):
        """Test flattening a single image."""
        single_image = np.random.rand(28, 28).astype(np.float32)
        flattened = flatten_images(single_image[np.newaxis, ...])  # Add batch dimension

        assert flattened.shape == (1, 784)
        assert flattened.dtype == np.float32

    def test_add_channel_dimension_single_image(self):
        """Test adding channel dimension to a single image."""
        single_image = np.random.rand(28, 28).astype(np.float32)
        with_channel = add_channel_dimension(single_image[np.newaxis, ...])

        assert with_channel.shape == (1, 28, 28, 1)
        assert with_channel.dtype == np.float32

    def test_flatten_images_invalid_shape(self):
        """Test error handling for invalid image shapes."""
        # Test with wrong dimensions
        invalid_images = np.random.rand(10, 32, 32)  # Not 28x28

        with pytest.raises(ValueError):
            flatten_images(invalid_images)

    def test_add_channel_dimension_invalid_shape(self):
        """Test error handling for invalid image shapes in channel addition."""
        invalid_images = np.random.rand(10, 32, 32)  # Not 28x28

        with pytest.raises(ValueError):
            add_channel_dimension(invalid_images)