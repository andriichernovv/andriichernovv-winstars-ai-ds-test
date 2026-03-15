import logging
import numpy as np


def setup_logger(name: str = "mnist_project") -> logging.Logger:
    """
    Configure and return a logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger

def log_shape(logger: logging.Logger, name: str, array: np.ndarray) -> None:
    """
    Log the shape of a NumPy array using the provided logger.
    """
    logger.info("%s shape: %s", name, array.shape)