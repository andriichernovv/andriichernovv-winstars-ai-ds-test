import numpy as np
from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    """
    Abstract interface for MNIST classifiers.
    All classifier implementations must provide
    train and predict methods.
    """

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the classifier on the provided training data.
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the provided input data.
        """
        pass