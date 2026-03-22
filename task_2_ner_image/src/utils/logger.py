import logging
from pathlib import Path


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)

    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def log_shape(logger: logging.Logger, name: str, array) -> None:
    """
    Log array shape information.

    Args:
        logger: Logger instance
        name: Name of the array
        array: Array-like object with shape attribute
    """
    try:
        shape = getattr(array, 'shape', 'no shape')
        logger.info(f"{name} shape: {shape}")
    except Exception as e:
        logger.warning(f"Could not log shape for {name}: {e}")


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path