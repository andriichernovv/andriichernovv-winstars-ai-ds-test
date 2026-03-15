import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from src.interfaces.mnist_classifier_interface import MnistClassifierInterface
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FeedForwardMnistClassifier(MnistClassifierInterface):
    """
    Feed-Forward Neural Network classifier for MNIST.
    Expects flattened images of shape (n_samples, 784).
    """
    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 5,
        batch_size: int = 32,
        
    ):
        logger.info(
            "Initializing FeedForwardMnistClassifier "
            f"(lr={learning_rate}, epochs={epochs}, batch_size={batch_size})"
        )

        self.epochs = epochs
        self.batch_size = batch_size

        self.model = Sequential([
            Dense(128, activation="relu", input_shape=(784,)),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax"),
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the neural network.
        """
        logger.info(
            f"Starting NN training. x_train shape={x_train.shape}, "
            f"y_train shape={y_train.shape}"
        )

        if x_train.ndim != 2 or x_train.shape[1] != 784:
            logger.error(f"Invalid input shape for NN: {x_train.shape}")
            raise ValueError(
                f"Expected shape (n_samples, 784), got {x_train.shape}"
            )

        self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
        )
        logger.info("Neural network training completed.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict digit labels.
        """
        logger.info(f"Running NN inference. Input shape: {x.shape}")

        if x.ndim != 2 or x.shape[1] != 784:
            logger.error(f"Invalid input shape for NN prediction: {x.shape}")
            raise ValueError(
                f"Expected shape (n_samples, 784), got {x.shape}"
            )

        probabilities = self.model.predict(x)
        predictions = np.argmax(probabilities, axis=1)
        logger.info(f"Inference completed. Output shape: {predictions.shape}")

        return predictions