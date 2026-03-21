import numpy as np

from src.utils.logger import setup_logger

from src.models.rf_classifier import RandomForestMnistClassifier
from src.models.nn_classifier import FeedForwardMnistClassifier
from src.models.cnn_classifier import CnnMnistClassifier

from src.data.mnist_loader import flatten_images, add_channel_dimension


logger = setup_logger(__name__)


class MnistClassifier:
    """
    Unified MNIST classifier that supports multiple algorithms.

    Supported algorithms:
    - "rf"  : Random Forest
    - "nn"  : Feed-Forward Neural Network
    - "cnn" : Convolutional Neural Network
    """
    def __init__(self, algorithm: str) -> None:
        logger.info(f"Initializing MnistClassifier with algorithm='{algorithm}'")

        self.algorithm = algorithm

        if algorithm == "rf":
            self.model = RandomForestMnistClassifier()
        elif algorithm == "nn":
            self.model = FeedForwardMnistClassifier()
        elif algorithm == "cnn":
            self.model = CnnMnistClassifier()
        else:
            logger.error(f"Unsupported algorithm: {algorithm}")
            raise ValueError(
                "Algorithm must be one of: 'rf', 'nn', 'cnn'"
            )

        logger.info(f"Model {self.model.__class__.__name__} created.")

    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """
        Prepare input data depending on the algorithm.
        """
        logger.info(f"Preparing input data. Original shape: {x.shape}")

        if self.algorithm in ["rf", "nn"]:
            x = flatten_images(x)
            logger.info(f"Data flattened. New shape: {x.shape}")

        elif self.algorithm == "cnn":
            x = add_channel_dimension(x)
            logger.info(f"Channel dimension added. New shape: {x.shape}")

        return x

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the selected model.
        """
        logger.info("Starting training pipeline.")

        x_train = self._prepare_input(x_train)
        self.model.train(x_train, y_train)

        logger.info("Training pipeline completed.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict labels using the selected model.
        """
        logger.info("Starting prediction pipeline.")

        x = self._prepare_input(x)
        predictions = self.model.predict(x)
        
        logger.info("Prediction pipeline completed.")

        return predictions