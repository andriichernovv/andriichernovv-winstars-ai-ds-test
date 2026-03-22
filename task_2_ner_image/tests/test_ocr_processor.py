import numpy as np
import pytest
from PIL import Image

from src.vision.ocr_processor import OCRProcessor


class TestOCRProcessor:
    """Test OCR processor functionality."""

    def test_init(self):
        """Test OCR processor initialization."""
        processor = OCRProcessor(languages=['en'], gpu=False)
        assert hasattr(processor, 'reader')
        assert processor.reader is not None

    def test_extract_text_from_image_array(self, sample_image_array):
        """Test text extraction from image array."""
        processor = OCRProcessor(languages=['en'], gpu=False)

        results = processor.extract_text_from_image_array(sample_image_array)

        assert isinstance(results, list)
        if results:  # If OCR found text
            for result in results:
                assert len(result) == 3  # text, bbox, confidence
                assert isinstance(result[0], str)  # text
                assert isinstance(result[1], list)  # bbox
                assert isinstance(result[2], (int, float))  # confidence

    def test_get_text_only(self, sample_image_array):
        """Test extracting text only from image array."""
        processor = OCRProcessor(languages=['en'], gpu=False)

        # Create a temporary image file
        img = Image.fromarray(sample_image_array)
        temp_path = "temp_test_image.png"
        img.save(temp_path)

        try:
            text = processor.get_text_only(temp_path)
            assert isinstance(text, str)
        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_get_text_with_confidence(self, sample_image_array):
        """Test extracting text with confidence filtering."""
        processor = OCRProcessor(languages=['en'], gpu=False)

        # Create a temporary image file
        img = Image.fromarray(sample_image_array)
        temp_path = "temp_test_image.png"
        img.save(temp_path)

        try:
            results = processor.get_text_with_confidence(temp_path, confidence_threshold=0.5)
            assert isinstance(results, list)

            for text, confidence in results:
                assert isinstance(text, str)
                assert isinstance(confidence, (int, float))
                assert confidence >= 0.5
        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_invalid_image_path(self):
        """Test error handling for invalid image path."""
        processor = OCRProcessor()

        with pytest.raises(FileNotFoundError):
            processor.extract_text("nonexistent_image.png")