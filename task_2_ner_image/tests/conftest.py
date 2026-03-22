import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont


@pytest.fixture
def sample_image_array():
    """Create a sample image array with text for testing."""
    # Create a white image
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)

    # Add some text
    try:
        # Try to use a system font
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        # Fallback to default font
        font = ImageFont.load_default()

    # Draw text
    draw.text((50, 50), "John Smith works at Google Inc.", fill='black', font=font)
    draw.text((50, 100), "Contact: john.smith@gmail.com", fill='black', font=font)

    return np.array(img)


@pytest.fixture
def sample_text():
    """Sample text for NER testing."""
    return "John Smith works at Google Inc. in New York. Contact him at john.smith@gmail.com."


@pytest.fixture
def mock_ocr_results():
    """Mock OCR results for testing."""
    return [
        ("John Smith works at Google Inc.", [[50, 50], [300, 50], [300, 70], [50, 70]], 0.95),
        ("Contact: john.smith@gmail.com", [[50, 100], [280, 100], [280, 120], [50, 120]], 0.92)
    ]


@pytest.fixture
def mock_ner_entities():
    """Mock NER entities for testing."""
    return [
        {'entity': 'John Smith', 'label': 'PERSON', 'confidence': 0.99, 'start': 0, 'end': 10},
        {'entity': 'Google Inc', 'label': 'ORG', 'confidence': 0.98, 'start': 20, 'end': 30},
        {'entity': 'New York', 'label': 'LOC', 'confidence': 0.97, 'start': 34, 'end': 42}
    ]