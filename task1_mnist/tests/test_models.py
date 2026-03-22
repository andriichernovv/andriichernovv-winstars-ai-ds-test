import numpy as np
import pytest

from src.models.rf_classifier import RandomForestMnistClassifier
from src.models.nn_classifier import FeedForwardMnistClassifier
from src.models.cnn_classifier import CnnMnistClassifier


class TestRandomForestMnistClassifier:
    """Test the Random Forest classifier."""

    def test_init(self):
        """Test initialization."""
        model = RandomForestMnistClassifier()
        assert hasattr(model, 'model')

    def test_train_and_predict(self, flattened_mnist_data):
        """Test train and predict with flattened data."""
        x, y = flattened_mnist_data
        model = RandomForestMnistClassifier()

        model.train(x, y)
        predictions = model.predict(x)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (x.shape[0],)
        assert all(pred in range(10) for pred in predictions)


class TestFeedForwardMnistClassifier:
    """Test the Feed-Forward Neural Network classifier."""

    def test_init(self):
        """Test initialization."""
        model = FeedForwardMnistClassifier()
        assert hasattr(model, 'epochs')

    def test_train_and_predict(self, flattened_mnist_data):
        """Test train and predict with flattened data."""
        x, y = flattened_mnist_data
        model = FeedForwardMnistClassifier(epochs=1)  # Fast training for tests

        model.train(x, y)
        predictions = model.predict(x)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (x.shape[0],)
        assert all(pred in range(10) for pred in predictions)


class TestCnnMnistClassifier:
    """Test the Convolutional Neural Network classifier."""

    def test_init(self):
        """Test initialization."""
        model = CnnMnistClassifier()
        assert hasattr(model, 'epochs')

    def test_train_and_predict(self, cnn_mnist_data):
        """Test train and predict with channel dimension data."""
        x, y = cnn_mnist_data
        model = CnnMnistClassifier(epochs=1)  # Fast training for tests

        model.train(x, y)
        predictions = model.predict(x)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (x.shape[0],)
        assert all(pred in range(10) for pred in predictions)


class TestModelErrorHandling:
    """Test error handling and edge cases for all models."""

    def test_rf_train_invalid_shape_features(self):
        """Test RandomForest with invalid feature dimensions."""
        model = RandomForestMnistClassifier()
        # Wrong shape: should be (n, 784), not (n, 100)
        x_invalid = np.random.rand(10, 100)
        y = np.random.randint(0, 10, 10)

        with pytest.raises(ValueError):
            model.train(x_invalid, y)

    def test_nn_train_invalid_shape_features(self):
        """Test FeedForward NN with invalid feature dimensions."""
        model = FeedForwardMnistClassifier()
        x_invalid = np.random.rand(10, 100)  # Not 784
        y = np.random.randint(0, 10, 10)

        with pytest.raises(ValueError):
            model.train(x_invalid, y)

    def test_cnn_train_invalid_shape_4d(self):
        """Test CNN with invalid 4D input shape."""
        model = CnnMnistClassifier()
        x_invalid = np.random.rand(10, 28, 28, 3)  # Wrong channel count
        y = np.random.randint(0, 10, 10)

        with pytest.raises(ValueError):
            model.train(x_invalid, y)

    def test_models_train_empty_data(self):
        """Test training with empty data arrays."""
        models = [RandomForestMnistClassifier(), FeedForwardMnistClassifier(), CnnMnistClassifier()]

        for model in models:
            x_empty = np.array([]).reshape(0, 784 if not isinstance(model, CnnMnistClassifier) else 28*28)
            y_empty = np.array([])

            with pytest.raises(ValueError):
                model.train(x_empty, y_empty)

    def test_models_predict_before_training(self):
        """Test prediction on untrained models."""
        models = [RandomForestMnistClassifier(), FeedForwardMnistClassifier(), CnnMnistClassifier()]

        for model in models:
            x_test = np.random.rand(5, 784 if not isinstance(model, CnnMnistClassifier) else 28*28*1)
            if isinstance(model, CnnMnistClassifier):
                x_test = x_test.reshape(5, 28, 28, 1)

            # Should raise error or return random predictions
            with pytest.raises(Exception):  # Could be ValueError, AttributeError, etc.
                model.predict(x_test)

    def test_models_single_sample_prediction(self, flattened_mnist_data, cnn_mnist_data):
        """Test prediction on single sample."""
        x_flat, y = flattened_mnist_data
        x_cnn, _ = cnn_mnist_data

        # Test RF and NN with single flattened sample
        rf_model = RandomForestMnistClassifier()
        rf_model.train(x_flat, y)
        pred_rf = rf_model.predict(x_flat[:1])
        assert pred_rf.shape == (1,)

        nn_model = FeedForwardMnistClassifier(epochs=1)
        nn_model.train(x_flat, y)
        pred_nn = nn_model.predict(x_flat[:1])
        assert pred_nn.shape == (1,)

        # Test CNN with single sample
        cnn_model = CnnMnistClassifier(epochs=1)
        cnn_model.train(x_cnn, y)
        pred_cnn = cnn_model.predict(x_cnn[:1])
        assert pred_cnn.shape == (1,)