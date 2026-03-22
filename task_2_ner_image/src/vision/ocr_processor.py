import logging
from pathlib import Path
from typing import List, Tuple, Optional

import easyocr
import numpy as np
from PIL import Image

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OCRProcessor:
    """
    OCR processor for extracting text from images using EasyOCR.
    """

    def __init__(self, languages: List[str] = ['en'], gpu: bool = False):
        """
        Initialize OCR processor.

        Args:
            languages: List of languages for OCR (default: English)
            gpu: Whether to use GPU for OCR
        """
        logger.info(f"Initializing OCR processor with languages: {languages}, GPU: {gpu}")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        logger.info("OCR processor initialized successfully")

    def extract_text(self, image_path: str | Path) -> List[Tuple[str, List[List[int]], float]]:
        """
        Extract text from image.

        Args:
            image_path: Path to image file

        Returns:
            List of tuples: (text, bounding_box, confidence)
        """
        image_path = Path(image_path)
        logger.info(f"Extracting text from image: {image_path}")

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Read image
            image = Image.open(image_path)
            image_array = np.array(image)

            # Extract text
            results = self.reader.readtext(image_array)

            logger.info(f"Extracted {len(results)} text regions from {image_path}")
            return results

        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            raise

    def extract_text_from_image_array(self, image_array: np.ndarray) -> List[Tuple[str, List[List[int]], float]]:
        """
        Extract text from image array.

        Args:
            image_array: Image as numpy array

        Returns:
            List of tuples: (text, bounding_box, confidence)
        """
        logger.info("Extracting text from image array")

        try:
            results = self.reader.readtext(image_array)
            logger.info(f"Extracted {len(results)} text regions")
            return results

        except Exception as e:
            logger.error(f"Error extracting text from image array: {e}")
            raise

    def get_text_only(self, image_path: str | Path) -> str:
        """
        Extract only text content from image (without bounding boxes).

        Args:
            image_path: Path to image file

        Returns:
            Extracted text as string
        """
        results = self.extract_text(image_path)
        texts = [text for text, _, _ in results]
        full_text = ' '.join(texts)
        logger.info(f"Extracted text: {full_text[:100]}...")
        return full_text

    def get_text_with_confidence(self, image_path: str | Path,
                               confidence_threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Extract text with confidence scores, filtered by threshold.

        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence score

        Returns:
            List of tuples: (text, confidence)
        """
        results = self.extract_text(image_path)
        filtered_results = [(text, confidence) for text, _, confidence in results
                          if confidence >= confidence_threshold]

        logger.info(f"Filtered {len(filtered_results)} text regions with confidence >= {confidence_threshold}")
        return filtered_results