import pytest

from src.ner.ner_processor import NERProcessor


class TestNERProcessor:
    """Test NER processor functionality."""

    def test_init(self):
        """Test NER processor initialization."""
        processor = NERProcessor(use_gpu=False)
        assert hasattr(processor, 'ner_pipeline')
        assert processor.ner_pipeline is not None

    def test_extract_entities(self, sample_text):
        """Test entity extraction from text."""
        processor = NERProcessor(use_gpu=False)

        entities = processor.extract_entities(sample_text)

        assert isinstance(entities, list)
        for entity in entities:
            assert isinstance(entity, dict)
            assert 'entity' in entity
            assert 'label' in entity
            assert 'confidence' in entity
            assert 'start' in entity
            assert 'end' in entity
            assert isinstance(entity['confidence'], float)
            assert 0 <= entity['confidence'] <= 1

    def test_extract_entities_empty_text(self):
        """Test entity extraction with empty text."""
        processor = NERProcessor(use_gpu=False)

        entities = processor.extract_entities("")
        assert entities == []

        entities = processor.extract_entities("   ")
        assert entities == []

    def test_extract_entities_by_type(self, sample_text):
        """Test entity extraction grouped by type."""
        processor = NERProcessor(use_gpu=False)

        grouped_entities = processor.extract_entities_by_type(sample_text)

        assert isinstance(grouped_entities, dict)
        for entity_type, entities in grouped_entities.items():
            assert isinstance(entity_type, str)
            assert isinstance(entities, list)
            for entity in entities:
                assert entity['label'] == entity_type

    def test_filter_entities_by_confidence(self, sample_text):
        """Test entity filtering by confidence."""
        processor = NERProcessor(use_gpu=False)

        entities = processor.extract_entities(sample_text)
        filtered_entities = processor.filter_entities_by_confidence(entities, confidence_threshold=0.9)

        assert len(filtered_entities) <= len(entities)
        for entity in filtered_entities:
            assert entity['confidence'] >= 0.9

    def test_get_entity_summary(self, sample_text):
        """Test entity summary generation."""
        processor = NERProcessor(use_gpu=False)

        summary = processor.get_entity_summary(sample_text)

        assert isinstance(summary, dict)
        for entity_type, count in summary.items():
            assert isinstance(entity_type, str)
            assert isinstance(count, int)
            assert count > 0