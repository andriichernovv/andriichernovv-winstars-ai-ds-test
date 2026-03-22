import numpy as np
import pytest

from src.classifier.mnist_classifier import MnistClassifier


class TestMnistClassifier:
    """Test the MnistClassifier wrapper class."""

    @pytest.mark.parametrize("algorithm", ["rf", "nn", "cnn"])
    def test_init_valid_algorithm(self, algorithm):
        """Test initialization with valid algorithms."""
        classifier = MnistClassifier(algorithm)
        assert classifier.algorithm == algorithm
        assert hasattr(classifier, 'model')
        assert classifier.model is not None

    def test_init_invalid_algorithm(self):
        """Test initialization with invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="Algorithm must be one of"):
            MnistClassifier("invalid")

    @pytest.mark.parametrize("algorithm", ["rf", "nn", "cnn"])
    def test_train_and_predict(self, algorithm, small_mnist_data):
        """Test that train and predict work without errors and return correct shapes."""
        x, y = small_mnist_data
        classifier = MnistClassifier(algorithm)

        # Train should not raise
        classifier.train(x, y)

        # Predict should return array of same length as input
        predictions = classifier.predict(x)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (x.shape[0],)
        assert predictions.dtype in [np.int32, np.int64]  # Labels are integers

    def test_predict_untrained_model(self, small_mnist_data):
        """Test that predict works even without training (though results may be random)."""
        x, y = small_mnist_data
        classifier = MnistClassifier("rf")  # Use RF as it's fast

        predictions = classifier.predict(x)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (x.shape[0],)

    def test_mnist_classifier_invalid_input_shape(self):
        """Test MnistClassifier with invalid input shapes."""
        classifier = MnistClassifier("rf")

        # Test with 1D input
        x_invalid = np.random.rand(10)  # 1D array
        with pytest.raises(ValueError):
            classifier.predict(x_invalid)

        # Test with wrong 2D shape
        x_invalid_2d = np.random.rand(10, 100)  # Not 784 features
        with pytest.raises(ValueError):
            classifier.predict(x_invalid_2d)

    def test_mnist_classifier_data_transformation_rf(self):
        """Test that RF classifier properly transforms 2D input to 1D."""
        classifier = MnistClassifier("rf")

        # Create 2D input like images
        x_2d = np.random.rand(5, 28, 28).astype(np.float32)
        y = np.random.randint(0, 10, 5)

        # Should work without errors (internal transformation)
        classifier.train(x_2d, y)
        predictions = classifier.predict(x_2d)
        assert predictions.shape == (5,)

    def test_mnist_classifier_data_transformation_cnn(self):
        """Test that CNN classifier properly transforms 2D input to 4D."""
        classifier = MnistClassifier("cnn")

        x_2d = np.random.rand(5, 28, 28).astype(np.float32)
        y = np.random.randint(0, 10, 5)

        # Should work without errors
        classifier.train(x_2d, y)
        predictions = classifier.predict(x_2d)
        assert predictions.shape == (5,)