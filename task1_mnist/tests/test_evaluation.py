import numpy as np
import pytest

from src.services.evaluation import EvaluationService, EvaluationResult


class TestEvaluationService:
    """Test the evaluation service."""

    def test_evaluate_predictions(self):
        """Test evaluation of predictions."""
        # Create mock data
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])  # One wrong prediction

        service = EvaluationService()
        result = service.evaluate(y_true, y_pred)

        assert isinstance(result, EvaluationResult)
        assert result.accuracy == 5/6  # 5 correct out of 6
        assert result.precision.shape == (3,)  # 3 classes
        assert result.recall.shape == (3,)
        assert result.f1_score.shape == (3,)
        assert result.confusion_matrix.shape == (3, 3)

    def test_evaluate_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        service = EvaluationService()
        result = service.evaluate(y_true, y_pred)

        assert result.accuracy == 1.0
        assert np.all(result.precision == 1.0)
        assert np.all(result.recall == 1.0)
        assert np.all(result.f1_score == 1.0)

    def test_evaluate_single_class(self):
        """Test evaluation with single class."""
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 1])

        service = EvaluationService()
        result = service.evaluate(y_true, y_pred)

        assert result.accuracy == 1.0
        # For single class, sklearn returns nan for some metrics, but we handle it
        assert not np.isnan(result.accuracy)

    def test_evaluate_empty_predictions(self):
        """Test evaluation with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])

        service = EvaluationService()
        with pytest.raises(ValueError):
            service.evaluate(y_true, y_pred)

    def test_evaluate_mismatched_lengths(self):
        """Test evaluation with mismatched prediction and true label lengths."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1])  # Different length

        service = EvaluationService()
        with pytest.raises(ValueError):
            service.evaluate(y_true, y_pred)

    def test_evaluate_single_sample(self):
        """Test evaluation with single sample."""
        y_true = np.array([5])
        y_pred = np.array([5])

        service = EvaluationService()
        result = service.evaluate(y_true, y_pred)

        assert result.accuracy == 1.0
        assert result.confusion_matrix.shape == (1, 1)

    def test_evaluate_with_probabilities(self):
        """Test evaluation with probability predictions."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])

        service = EvaluationService()
        result = service.evaluate(y_true, y_proba=y_proba)

        assert result.accuracy == 1.0
        # Should have additional probability-based metrics if implemented

    def test_evaluate_confusion_matrix_properties(self):
        """Test that confusion matrix properties are correct."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])  # Some correct, some wrong

        service = EvaluationService()
        result = service.evaluate(y_true, y_pred)

        # Confusion matrix should sum to total samples
        assert result.confusion_matrix.sum() == len(y_true)
        # Diagonal should be correct predictions
        assert result.confusion_matrix.diagonal().sum() == 3  # 3 correct predictions