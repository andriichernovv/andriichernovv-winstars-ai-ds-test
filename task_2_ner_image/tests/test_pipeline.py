import numpy as np
import pytest
from PIL import Image

from src.pipeline.ner_image_pipeline import NERImagePipeline


class TestNERImagePipeline:
    """Test NER Image Pipeline functionality."""

    def test_init(self):
        """Test pipeline initialization."""
        pipeline = NERImagePipeline(use_gpu=False)
        assert hasattr(pipeline, 'ocr_processor')
        assert hasattr(pipeline, 'ner_processor')

    def test_process_image_array(self, sample_image_array):
        """Test processing image array."""
        pipeline = NERImagePipeline(use_gpu=False)

        result = pipeline.process_image_array(sample_image_array)

        assert isinstance(result, dict)
        assert 'extracted_text' in result
        assert 'ocr_results' in result
        assert 'entities' in result
        assert 'entity_summary' in result
        assert 'processing_stats' in result

        assert isinstance(result['extracted_text'], str)
        assert isinstance(result['ocr_results'], list)
        assert isinstance(result['entities'], list)
        assert isinstance(result['entity_summary'], dict)

    def test_process_image_file(self, sample_image_array):
        """Test processing image file."""
        pipeline = NERImagePipeline(use_gpu=False)

        # Create temporary image file
        img = Image.fromarray(sample_image_array)
        temp_path = "temp_test_image.png"
        img.save(temp_path)

        try:
            result = pipeline.process_image(temp_path)

            assert isinstance(result, dict)
            assert result['image_path'] == temp_path
            assert 'extracted_text' in result
            assert 'entities' in result
        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_process_batch(self, sample_image_array):
        """Test batch processing."""
        pipeline = NERImagePipeline(use_gpu=False)

        # Create multiple temporary images
        temp_paths = []
        for i in range(2):
            img = Image.fromarray(sample_image_array)
            temp_path = f"temp_test_image_{i}.png"
            img.save(temp_path)
            temp_paths.append(temp_path)

        try:
            results = pipeline.process_batch(temp_paths)

            assert isinstance(results, list)
            assert len(results) == 2

            for result in results:
                assert isinstance(result, dict)
                assert 'extracted_text' in result
                assert 'entities' in result
        finally:
            import os
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)

    def test_get_entities_by_type(self, sample_image_array):
        """Test getting entities grouped by type."""
        pipeline = NERImagePipeline(use_gpu=False)

        # Create temporary image file
        img = Image.fromarray(sample_image_array)
        temp_path = "temp_test_image.png"
        img.save(temp_path)

        try:
            entities_by_type = pipeline.get_entities_by_type(temp_path)

            assert isinstance(entities_by_type, dict)
            for entity_type, entities in entities_by_type.items():
                assert isinstance(entity_type, str)
                assert isinstance(entities, list)
        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_invalid_image_path(self):
        """Test error handling for invalid image path."""
        pipeline = NERImagePipeline()

        with pytest.raises(FileNotFoundError):
            pipeline.process_image("nonexistent_image.png")

    def test_empty_image_array(self):
        """Test processing empty image array."""
        pipeline = NERImagePipeline()

        empty_array = np.array([]).reshape(0, 0, 3)

        # Should handle gracefully or raise appropriate error
        with pytest.raises(Exception):
            pipeline.process_image_array(empty_array)