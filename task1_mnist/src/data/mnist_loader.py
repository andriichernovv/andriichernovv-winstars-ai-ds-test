import numpy as np
from tensorflow.keras.datasets import mnist
from typing import Tuple

from src.utils.logger import setup_logger, log_shape

logger = setup_logger(__name__)


def load_mnist(
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the MNIST dataset.

    Parameters
    ----------
    normalize : bool, default=True
        If True, pixel values are scaled to the [0, 1] range.

    Returns
    -------
    x_train : np.ndarray
        Training images with shape (60000, 28, 28).
    y_train : np.ndarray
        Training labels with shape (60000,).
    x_test : np.ndarray
        Test images with shape (10000, 28, 28).
    y_test : np.ndarray
        Test labels with shape (10000,).
    """
    logger.info("Starting MNIST dataset loading.")

    try:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    except Exception as e:
        logger.error(f"Failed to load MNIST dataset: {e}")
        raise

    logger.info("MNIST dataset loaded successfully.")
    log_shape(logger=logger, "x_train", x_train)
    log_shape(logger=logger, "y_train", y_train)
    log_shape(logger=logger, "x_test", x_test)
    log_shape(logger=logger, "y_test", y_test)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    logger.info("Input images converted to float32.")

    if normalize:
        x_train /= 255.0
        x_test /= 255.0
        logger.info("Input images normalized to the [0, 1] range.")
    else:
        logger.info("Normalization skipped.")

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    logger.info("Labels converted to int64.")

    logger.info("MNIST preprocessing completed successfully.")

    return x_train, y_train, x_test, y_test


def flatten_images(x: np.ndarray) -> np.ndarray:
    """
    Convert images from shape (n_samples, 28, 28) to (n_samples, 784).

    Also supports a single image with shape (28, 28).
    """
    logger.info(f"Flattening images. Input shape: {x.shape}")

    if x.ndim == 2:
        logger.info("Single image detected. Expanding batch dimension.")
        x = np.expand_dims(x, axis=0)

    if x.ndim != 3 or x.shape[1:] != (28, 28):
        logger.error(f"Invalid input shape for flatten_images: {x.shape}")
        raise ValueError(
            f"Expected input shape (n_samples, 28, 28) or (28, 28), got {x.shape}."
        )

    x_flat = x.reshape(x.shape[0], -1)
    logger.info(f"Flattening completed. Output shape: {x_flat.shape}")

    return x_flat


def add_channel_dimension(x: np.ndarray) -> np.ndarray:
    """
    Convert images from shape (n_samples, 28, 28) to (n_samples, 28, 28, 1).

    Also supports a single image with shape (28, 28).
    """
    logger.info(f"Adding channel dimension. Input shape: {x.shape}")

    if x.ndim == 2:
        logger.info("Single image detected. Expanding batch dimension.")
        x = np.expand_dims(x, axis=0)

    if x.ndim != 3 or x.shape[1:] != (28, 28):
        logger.error(f"Invalid input shape for add_channel_dimension: {x.shape}")
        raise ValueError(
            f"Expected input shape (n_samples, 28, 28) or (28, 28), got {x.shape}."
        )

    x_with_channel = np.expand_dims(x, axis=-1)
    logger.info(f"Channel dimension added. Output shape: {x_with_channel.shape}")

    return x_with_channel


def validate_labels(y: np.ndarray) -> np.ndarray:
    """
    Validate labels and convert them to integer dtype.
    """
    logger.info("Validating labels.")

    y = np.asarray(y)

    if y.ndim != 1:
        logger.error(f"Invalid label shape: {y.shape}")
        raise ValueError(f"Expected labels with shape (n_samples,), got {y.shape}.")

    y = y.astype(np.int64)
    logger.info(f"Labels validated successfully. Shape: {y.shape}")

    return y