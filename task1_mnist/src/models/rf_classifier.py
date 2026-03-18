import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.utils.logger import setup_logger
from src.models.interface import MnistClassifierInterface

logger = setup_logger(__name__)


class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    Random Forest classifier for MNIST.
    Expects flattened input images of shape (n_samples, 784)
    """
    def __init__(self, 
                 n_estimators: int = 100, 
                 random_state: int = 42,
                 max_depth: int | None = None
                ) -> None:
        logger.info(
            "Initializing RandomForestMnistClassifier "
            f"with n_estimators={n_estimators}, "
            f"random_state={random_state}, max_depth={max_depth}"
        )
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            # Uses all cores for fast learning.
            n_jobs=-1
        )

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the Random Forest classifier.

        Parameters
        ----------
        x_train : np.ndarray
            Training data of shape (n_samples, 784).
        y_train : np.ndarray
            Training labels of shape (n_samples,).
        """
        logger.info(
            f"Starting Random Forest training. "
            f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}"
        )

        if x_train.ndim != 2 or x_train.shape[1] != 784:
            logger.error(f"Invalid x_train shape for Random Forest: {x_train.shape}")
            raise ValueError(
                f"Expected x_train shape (n_samples, 784), got {x_train.shape}."
            )

        if y_train.ndim != 1:
            logger.error(f"Invalid y_train shape for Random Forest: {y_train.shape}")
            raise ValueError(
                f"Expected y_train shape (n_samples,), got {y_train.shape}."
            )

        self.model.fit(x_train, y_train)

        logger.info("Random Forest training completed successfully.")
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels using the trained Random Forest classifier.

        Parameters
        ----------
        x : np.ndarray
            Input data of shape (n_samples, 784).

        Returns
        -------
        np.ndarray
            Predicted labels of shape (n_samples,).
        """
        logger.info(f"Starting Random Forest inference. Input shape: {x.shape}")

        if x.ndim != 2 or x.shape[1] != 784:
            logger.error(f"Invalid input shape for Random Forest prediction: {x.shape}")
            raise ValueError(
                f"Expected input shape (n_samples, 784), got {x.shape}."
            )
        
        predictions = self.model.predict(x)
        logger.info(
            f"Random Forest inference completed successfully. "
            f"Output shape: {predictions.shape}"
        )    
        
        return predictions