import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from src.interfaces.mnist_classifier_interface import MnistClassifierInterface
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CnnMnistClassifier(MnistClassifierInterface):
    """
    Convolutional Neural Network classifier for MNIST.
    Expects input images of shape (n_samples, 28, 28, 1).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 5,
        batch_size: int = 32,
    ) -> None:
        logger.info(
            "Initializing CnnMnistClassifier "
            f"with learning_rate={learning_rate}, epochs={epochs}, batch_size={batch_size}"
        )

        self.epochs = epochs
        self.batch_size = batch_size

        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the CNN classifier.

        Parameters
        ----------
        x_train : np.ndarray
            Training data of shape (n_samples, 28, 28, 1).
        y_train : np.ndarray
            Training labels of shape (n_samples,).
        """
        logger.info(
            f"Starting CNN training. "
            f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}"
        )

        if x_train.ndim != 4 or x_train.shape[1:] != (28, 28, 1):
            logger.error(f"Invalid x_train shape for CNN: {x_train.shape}")
            raise ValueError(
                f"Expected x_train shape (n_samples, 28, 28, 1), got {x_train.shape}."
            )

        if y_train.ndim != 1:
            logger.error(f"Invalid y_train shape for CNN: {y_train.shape}")
            raise ValueError(
                f"Expected y_train shape (n_samples,), got {y_train.shape}."
            )

        self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
        )

        logger.info("CNN training completed successfully.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels using the trained CNN classifier.

        Parameters
        ----------
        x : np.ndarray
            Input data of shape (n_samples, 28, 28, 1).

        Returns
        -------
        np.ndarray
            Predicted labels of shape (n_samples,).
        """
        logger.info(f"Starting CNN inference. Input shape: {x.shape}")

        if x.ndim != 4 or x.shape[1:] != (28, 28, 1):
            logger.error(f"Invalid input shape for CNN prediction: {x.shape}")
            raise ValueError(
                f"Expected input shape (n_samples, 28, 28, 1), got {x.shape}."
            )

        probabilities = self.model.predict(x)
        predictions = np.argmax(probabilities, axis=1)

        logger.info(
            f"CNN inference completed successfully. Output shape: {predictions.shape}"
        )

        return predictions