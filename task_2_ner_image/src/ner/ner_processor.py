import logging
from typing import List, Dict, Tuple, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class NERProcessor:
    """
    Named Entity Recognition processor using Hugging Face transformers.
    """

    def __init__(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
                 use_gpu: bool = False):
        """
        Initialize NER processor.

        Args:
            model_name: Hugging Face model name for NER
            use_gpu: Whether to use GPU
        """
        logger.info(f"Initializing NER processor with model: {model_name}, GPU: {use_gpu}")

        self.device = 0 if use_gpu and torch.cuda.is_available() else -1
        self.model_name = model_name

        # Initialize NER pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            aggregation_strategy="simple"  # Group sub-word tokens
        )

        logger.info("NER processor initialized successfully")

    def extract_entities(self, text: str) -> List[Dict[str, Union[str, float, int]]]:
        """
        Extract named entities from text.

        Args:
            text: Input text

        Returns:
            List of entity dictionaries with keys: entity, label, confidence, start, end
        """
        logger.info(f"Extracting entities from text: {text[:100]}...")

        if not text or not text.strip():
            logger.warning("Empty text provided to NER processor")
            return []

        try:
            # Run NER
            entities = self.ner_pipeline(text)

            # Convert to standardized format
            standardized_entities = []
            for entity in entities:
                standardized_entity = {
                    'entity': entity['word'],
                    'label': entity['entity_group'],
                    'confidence': float(entity['score']),
                    'start': int(entity['start']),
                    'end': int(entity['end'])
                }
                standardized_entities.append(standardized_entity)

            logger.info(f"Extracted {len(standardized_entities)} entities")
            return standardized_entities

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            raise

    def extract_entities_by_type(self, text: str) -> Dict[str, List[Dict[str, Union[str, float, int]]]]:
        """
        Extract entities grouped by type.

        Args:
            text: Input text

        Returns:
            Dictionary with entity types as keys and lists of entities as values
        """
        entities = self.extract_entities(text)

        # Group by label
        grouped_entities = {}
        for entity in entities:
            label = entity['label']
            if label not in grouped_entities:
                grouped_entities[label] = []
            grouped_entities[label].append(entity)

        logger.info(f"Grouped entities by type: {list(grouped_entities.keys())}")
        return grouped_entities

    def filter_entities_by_confidence(self, entities: List[Dict[str, Union[str, float, int]]],
                                    confidence_threshold: float = 0.8) -> List[Dict[str, Union[str, float, int]]]:
        """
        Filter entities by confidence score.

        Args:
            entities: List of entity dictionaries
            confidence_threshold: Minimum confidence score

        Returns:
            Filtered list of entities
        """
        filtered = [entity for entity in entities if entity['confidence'] >= confidence_threshold]
        logger.info(f"Filtered entities: {len(entities)} -> {len(filtered)} (threshold: {confidence_threshold})")
        return filtered

    def get_entity_summary(self, text: str) -> Dict[str, int]:
        """
        Get summary of entity types found in text.

        Args:
            text: Input text

        Returns:
            Dictionary with entity types and their counts
        """
        entities = self.extract_entities(text)
        summary = {}

        for entity in entities:
            label = entity['label']
            summary[label] = summary.get(label, 0) + 1

        logger.info(f"Entity summary: {summary}")
        return summary