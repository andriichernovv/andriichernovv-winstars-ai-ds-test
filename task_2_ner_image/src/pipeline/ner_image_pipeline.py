import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
from PIL import Image

from src.vision.ocr_processor import OCRProcessor
from src.ner.ner_processor import NERProcessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class NERImagePipeline:
    """
    Pipeline for Named Entity Recognition on images.
    Combines OCR for text extraction and NER for entity recognition.
    """

    def __init__(self,
                 ocr_languages: List[str] = ['en'],
                 ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
                 use_gpu: bool = False):
        """
        Initialize the NER image pipeline.

        Args:
            ocr_languages: Languages for OCR
            ner_model: NER model name
            use_gpu: Whether to use GPU
        """
        logger.info("Initializing NER Image Pipeline")

        self.ocr_processor = OCRProcessor(languages=ocr_languages, gpu=use_gpu)
        self.ner_processor = NERProcessor(model_name=ner_model, use_gpu=use_gpu)

        logger.info("NER Image Pipeline initialized successfully")

    def process_image(self, image_path: str | Path,
                     confidence_threshold: float = 0.5) -> Dict[str, Union[str, List, Dict]]:
        """
        Process image: extract text and recognize entities.

        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence for OCR text

        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing image: {image_path}")

        # Extract text from image
        ocr_results = self.ocr_processor.extract_text(image_path)

        # Filter OCR results by confidence
        filtered_ocr = [(text, bbox, conf) for text, bbox, conf in ocr_results
                       if conf >= confidence_threshold]

        # Combine all text
        all_text = ' '.join([text for text, _, _ in filtered_ocr])

        # Extract entities from text
        entities = self.ner_processor.extract_entities(all_text)

        # Prepare result
        result = {
            'image_path': str(image_path),
            'extracted_text': all_text,
            'ocr_results': filtered_ocr,
            'entities': entities,
            'entity_summary': self.ner_processor.get_entity_summary(all_text),
            'processing_stats': {
                'total_ocr_regions': len(ocr_results),
                'filtered_ocr_regions': len(filtered_ocr),
                'total_entities': len(entities)
            }
        }

        logger.info(f"Image processing completed. Found {len(entities)} entities in {len(filtered_ocr)} text regions")
        return result

    def process_image_array(self, image_array: np.ndarray,
                           confidence_threshold: float = 0.5) -> Dict[str, Union[str, List, Dict]]:
        """
        Process image array: extract text and recognize entities.

        Args:
            image_array: Image as numpy array
            confidence_threshold: Minimum confidence for OCR text

        Returns:
            Dictionary with processing results
        """
        logger.info("Processing image array")

        # Extract text from image array
        ocr_results = self.ocr_processor.extract_text_from_image_array(image_array)

        # Filter OCR results by confidence
        filtered_ocr = [(text, bbox, conf) for text, bbox, conf in ocr_results
                       if conf >= confidence_threshold]

        # Combine all text
        all_text = ' '.join([text for text, _, _ in filtered_ocr])

        # Extract entities from text
        entities = self.ner_processor.extract_entities(all_text)

        # Prepare result
        result = {
            'image_path': None,
            'extracted_text': all_text,
            'ocr_results': filtered_ocr,
            'entities': entities,
            'entity_summary': self.ner_processor.get_entity_summary(all_text),
            'processing_stats': {
                'total_ocr_regions': len(ocr_results),
                'filtered_ocr_regions': len(filtered_ocr),
                'total_entities': len(entities)
            }
        }

        logger.info(f"Image array processing completed. Found {len(entities)} entities in {len(filtered_ocr)} text regions")
        return result

    def process_batch(self, image_paths: List[str | Path],
                     confidence_threshold: float = 0.5) -> List[Dict[str, Union[str, List, Dict]]]:
        """
        Process batch of images.

        Args:
            image_paths: List of image paths
            confidence_threshold: Minimum confidence for OCR text

        Returns:
            List of processing results
        """
        logger.info(f"Processing batch of {len(image_paths)} images")

        results = []
        for image_path in image_paths:
            try:
                result = self.process_image(image_path, confidence_threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e),
                    'extracted_text': '',
                    'ocr_results': [],
                    'entities': [],
                    'entity_summary': {},
                    'processing_stats': {'error': True}
                })

        logger.info(f"Batch processing completed. Processed {len(results)} images")
        return results

    def get_entities_by_type(self, image_path: str | Path,
                           confidence_threshold: float = 0.5) -> Dict[str, List[Dict[str, Union[str, float, int]]]]:
        """
        Get entities grouped by type from image.

        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence for OCR text

        Returns:
            Dictionary with entity types as keys
        """
        result = self.process_image(image_path, confidence_threshold)
        return self.ner_processor.extract_entities_by_type(result['extracted_text'])

    def visualize_results(self, result: Dict[str, Union[str, List, Dict]],
                         save_path: Optional[str] = None) -> Optional[Image.Image]:
        """
        Visualize OCR and NER results on image.

        Args:
            result: Processing result from process_image
            save_path: Optional path to save visualization

        Returns:
            PIL Image with visualization
        """
        logger.info("Creating visualization")

        if result['image_path'] is None:
            logger.warning("Cannot visualize results without image path")
            return None

        # Load original image
        image = Image.open(result['image_path'])
        image_array = np.array(image)

        # Create visualization (simplified - just return original for now)
        # In a full implementation, this would draw bounding boxes and entity labels

        if save_path:
            image.save(save_path)
            logger.info(f"Visualization saved to {save_path}")

        return image