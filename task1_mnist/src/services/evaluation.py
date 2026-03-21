from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)


@dataclass
class EvaluationResult:
    """
    Container for evaluation outputs.
    """

    metrics_df: pd.DataFrame
    probability_metrics_df: pd.DataFrame
    class_report_df: pd.DataFrame
    confusion_matrix_df: pd.DataFrame
    errors_df: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    labels: list[Any]


class EvaluationService:
    """
    Service for classification evaluation and visualization.

    Supports:
    - global metrics
    - probability-based metrics
    - per-class report
    - confusion matrix
    - error analysis
    - visualization
    """

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # -----
    # Main public API
    # -----
    def evaluate(
        self,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
        y_proba: np.ndarray | None = None,
        labels: list[Any] | None = None,
        top_k: int = 3,
        zero_division: int = 0,
    ) -> EvaluationResult:
        """
        Run full evaluation.

        Parameters
        ----------
        y_true : np.ndarray | list[Any]
            Ground truth labels.
        y_pred : np.ndarray | list[Any]
            Predicted labels.
        y_proba : np.ndarray | None, default=None
            Predicted probabilities with shape (n_samples, n_classes).
        labels : list[Any] | None, default=None
            Explicit class order.
        top_k : int, default=3
            K for top-k accuracy when probabilities are available.
        zero_division : int, default=0
            Value used by sklearn in zero-division cases.

        Returns
        -------
        EvaluationResult
            Full evaluation output.
        """
        y_true_arr, y_pred_arr = self._validate_label_inputs(y_true, y_pred)
        labels_resolved = self._resolve_labels(y_true_arr, y_pred_arr, labels)
        y_proba_arr = self._validate_probability_inputs(
            y_proba=y_proba,
            n_samples=len(y_true_arr),
            n_classes=len(labels_resolved),
        )

        metrics_df = self.calculate_metrics(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            zero_division=zero_division,
        )

        probability_metrics_df = self.calculate_probability_metrics(
            y_true=y_true_arr,
            y_proba=y_proba_arr,
            labels=labels_resolved,
            top_k=top_k,
        )

        class_report_df = self.build_classification_report(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            labels=labels_resolved,
            zero_division=zero_division,
        )

        confusion_matrix_df = self.build_confusion_matrix(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            labels=labels_resolved,
            normalize=None,
        )

        errors_df = self.build_error_analysis(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
        )

        return EvaluationResult(
            metrics_df=metrics_df,
            probability_metrics_df=probability_metrics_df,
            class_report_df=class_report_df,
            confusion_matrix_df=confusion_matrix_df,
            errors_df=errors_df,
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            y_proba=y_proba_arr,
            labels=labels_resolved,
        )

    # -----
    # Metrics
    # -----
    def calculate_metrics(
        self,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
        zero_division: int = 0,
    ) -> pd.DataFrame:
        """
        Calculate global metrics.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame with aggregate metrics.
        """
        y_true_arr, y_pred_arr = self._validate_label_inputs(y_true, y_pred)

        metrics = {
            "accuracy": accuracy_score(y_true_arr, y_pred_arr),
            "precision_macro": precision_score(
                y_true_arr,
                y_pred_arr,
                average="macro",
                zero_division=zero_division,
            ),
            "recall_macro": recall_score(
                y_true_arr,
                y_pred_arr,
                average="macro",
                zero_division=zero_division,
            ),
            "f1_macro": f1_score(
                y_true_arr,
                y_pred_arr,
                average="macro",
                zero_division=zero_division,
            ),
            "precision_weighted": precision_score(
                y_true_arr,
                y_pred_arr,
                average="weighted",
                zero_division=zero_division,
            ),
            "recall_weighted": recall_score(
                y_true_arr,
                y_pred_arr,
                average="weighted",
                zero_division=zero_division,
            ),
            "f1_weighted": f1_score(
                y_true_arr,
                y_pred_arr,
                average="weighted",
                zero_division=zero_division,
            ),
            "n_samples": len(y_true_arr),
        }

        return pd.DataFrame([metrics])

    def calculate_probability_metrics(
        self,
        y_true: np.ndarray | list[Any],
        y_proba: np.ndarray | None,
        labels: list[Any],
        top_k: int = 3,
    ) -> pd.DataFrame:
        """
        Calculate probability-based metrics.

        Returns empty DataFrame if probabilities are not provided.
        """
        y_true_arr = np.asarray(y_true)

        if y_proba is None:
            return pd.DataFrame()

        metrics: dict[str, float] = {}

        try:
            metrics["log_loss"] = log_loss(y_true_arr, y_proba, labels=labels)
        except ValueError:
            metrics["log_loss"] = np.nan

        safe_top_k = min(top_k, y_proba.shape[1])
        try:
            metrics[f"top_{safe_top_k}_accuracy"] = top_k_accuracy_score(
                y_true_arr,
                y_proba,
                k=safe_top_k,
                labels=labels,
            )
        except ValueError:
            metrics[f"top_{safe_top_k}_accuracy"] = np.nan

        return pd.DataFrame([metrics])

    def build_classification_report(
        self,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
        labels: list[Any] | None = None,
        zero_division: int = 0,
    ) -> pd.DataFrame:
        """
        Build classification report as DataFrame.
        """
        y_true_arr, y_pred_arr = self._validate_label_inputs(y_true, y_pred)
        labels_resolved = self._resolve_labels(y_true_arr, y_pred_arr, labels)

        report_dict = classification_report(
            y_true_arr,
            y_pred_arr,
            labels=labels_resolved,
            output_dict=True,
            zero_division=zero_division,
        )

        report_df = pd.DataFrame(report_dict).T
        report_df.index.name = "label"
        return report_df.reset_index()

    def build_confusion_matrix(
        self,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
        labels: list[Any] | None = None,
        normalize: str | None = None,
    ) -> pd.DataFrame:
        """
        Build confusion matrix as DataFrame.

        normalize:
        - None
        - "true"
        - "pred"
        - "all"
        """
        y_true_arr, y_pred_arr = self._validate_label_inputs(y_true, y_pred)
        labels_resolved = self._resolve_labels(y_true_arr, y_pred_arr, labels)

        cm = confusion_matrix(
            y_true_arr,
            y_pred_arr,
            labels=labels_resolved,
            normalize=normalize,
        )

        return pd.DataFrame(
            cm,
            index=pd.Index(labels_resolved, name="true_label"),
            columns=pd.Index(labels_resolved, name="pred_label"),
        )

    def build_error_analysis(
        self,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
    ) -> pd.DataFrame:
        """
        Return only misclassified samples.
        """
        y_true_arr, y_pred_arr = self._validate_label_inputs(y_true, y_pred)

        error_df = pd.DataFrame(
            {
                "y_true": y_true_arr,
                "y_pred": y_pred_arr,
            }
        )
        error_df["is_correct"] = error_df["y_true"] == error_df["y_pred"]

        return error_df.loc[~error_df["is_correct"]].reset_index(drop=True)

    def build_results_dataframe(
        self,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
        y_proba: np.ndarray | None = None,
        labels: list[Any] | None = None,
    ) -> pd.DataFrame:
        """
        Build a unified results DataFrame with y_true, y_pred and optional probabilities.

        Probability columns are named:
        proba_<class_label>
        """
        y_true_arr, y_pred_arr = self._validate_label_inputs(y_true, y_pred)
        labels_resolved = self._resolve_labels(y_true_arr, y_pred_arr, labels)

        result_df = pd.DataFrame(
            {
                "y_true": y_true_arr,
                "y_pred": y_pred_arr,
            }
        )

        if y_proba is not None:
            y_proba_arr = self._validate_probability_inputs(
                y_proba=y_proba,
                n_samples=len(y_true_arr),
                n_classes=len(labels_resolved),
            )

            proba_columns = [f"proba_{label}" for label in labels_resolved]
            proba_df = pd.DataFrame(y_proba_arr, columns=proba_columns)
            result_df = pd.concat([result_df, proba_df], axis=1)

        return result_df

    # -----
    # Visualization
    # -----
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
        labels: list[Any] | None = None,
        normalize: str | None = None,
        figsize: tuple[int, int] = (8, 6),
        cmap: str = "Blues",
        save_name: str | None = None,
    ) -> None:
        """
        Plot confusion matrix.

        normalize:
        - None
        - "true"
        - "pred"
        - "all"
        """
        cm_df = self.build_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            normalize=normalize,
        )

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm_df.values, cmap=cmap)

        title = "Confusion Matrix"
        if normalize is not None:
            title += f" ({normalize} normalized)"
        ax.set_title(title)

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(np.arange(len(cm_df.columns)))
        ax.set_yticks(np.arange(len(cm_df.index)))
        ax.set_xticklabels(cm_df.columns)
        ax.set_yticklabels(cm_df.index)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(cm_df.shape[0]):
            for j in range(cm_df.shape[1]):
                value = cm_df.iloc[i, j]
                text = f"{value:.2f}" if normalize is not None else f"{int(value)}"
                ax.text(j, i, text, ha="center", va="center")

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        self._save_figure(fig, save_name)
        plt.show()

    def plot_classification_report_heatmap(
        self,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
        labels: list[Any] | None = None,
        zero_division: int = 0,
        figsize: tuple[int, int] = (10, 6),
        cmap: str = "YlGnBu",
        save_name: str | None = None,
    ) -> None:
        """
        Plot per-class precision, recall and F1 as heatmap.
        """
        report_df = self.build_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            zero_division=zero_division,
        )

        rows_to_keep = report_df[
            ~report_df["label"].isin(["accuracy", "macro avg", "weighted avg"])
        ].copy()

        cols_to_plot = ["precision", "recall", "f1-score", "support"]
        plot_df = rows_to_keep.set_index("label")[cols_to_plot]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(plot_df.values, cmap=cmap, aspect="auto")

        ax.set_title("Classification Report Heatmap")
        ax.set_xticks(np.arange(len(plot_df.columns)))
        ax.set_yticks(np.arange(len(plot_df.index)))
        ax.set_xticklabels(plot_df.columns)
        ax.set_yticklabels(plot_df.index)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(plot_df.shape[0]):
            for j in range(plot_df.shape[1]):
                value = plot_df.iloc[i, j]
                if plot_df.columns[j] == "support":
                    text = f"{int(value)}"
                else:
                    text = f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center")

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        self._save_figure(fig, save_name)
        plt.show()

    def plot_sample_predictions(
        self,
        images: np.ndarray,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
        only_misclassified: bool = False,
        n_samples: int = 9,
        figsize: tuple[int, int] = (10, 10),
        save_name: str | None = None,
    ) -> None:
        """
        Plot image samples with true/pred labels.
        """
        y_true_arr, y_pred_arr = self._validate_label_inputs(y_true, y_pred)

        if len(images) != len(y_true_arr):
            raise ValueError(
                "images and y_true must have the same number of samples. "
                f"Got len(images)={len(images)} and len(y_true)={len(y_true_arr)}."
            )

        if only_misclassified:
            selected_idx = np.where(y_true_arr != y_pred_arr)[0]
        else:
            selected_idx = np.arange(len(y_true_arr))

        if len(selected_idx) == 0:
            raise ValueError("No samples available for the selected plotting condition.")

        selected_idx = selected_idx[:n_samples]
        n = len(selected_idx)
        grid_size = int(np.ceil(np.sqrt(n)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        axes = np.array(axes).reshape(-1)

        for ax in axes[n:]:
            ax.axis("off")

        for plot_ax, idx in zip(axes, selected_idx):
            image = images[idx]

            if image.ndim == 3 and image.shape[-1] == 1:
                image = image.squeeze(-1)

            if image.ndim == 2:
                plot_ax.imshow(image, cmap="gray")
            else:
                plot_ax.imshow(image)

            plot_ax.set_title(f"true={y_true_arr[idx]}, pred={y_pred_arr[idx]}")
            plot_ax.axis("off")

        title = "Misclassified Samples" if only_misclassified else "Sample Predictions"
        fig.suptitle(title)
        fig.tight_layout()

        self._save_figure(fig, save_name)
        plt.show()

    def plot_prediction_distribution(
        self,
        y_pred: np.ndarray | list[Any],
        labels: list[Any] | None = None,
        figsize: tuple[int, int] = (8, 5),
        save_name: str | None = None,
    ) -> None:
        """
        Plot predicted label distribution.
        """
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim != 1:
            raise ValueError(f"y_pred must be 1D. Got shape={y_pred_arr.shape}.")

        if labels is None:
            labels_resolved = sorted(pd.unique(y_pred_arr))
        else:
            labels_resolved = list(labels)

        counts = pd.Series(y_pred_arr).value_counts().reindex(labels_resolved, fill_value=0)

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(len(counts)), counts.values)
        ax.set_title("Prediction Distribution")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Count")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index)

        fig.tight_layout()
        self._save_figure(fig, save_name)
        plt.show()

    # -----
    # Saving
    # -----
    def save_dataframe(
        self,
        df: pd.DataFrame,
        file_name: str,
        index: bool = True,
    ) -> Path:
        """
        Save DataFrame to CSV.
        """
        path = self._build_output_path(file_name)
        df.to_csv(path, index=index)
        return path

    # -----
    # Internal helpers
    # -----
    def _validate_label_inputs(
        self,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if y_true_arr.ndim != 1:
            raise ValueError(f"y_true must be 1D. Got shape={y_true_arr.shape}.")
        if y_pred_arr.ndim != 1:
            raise ValueError(f"y_pred must be 1D. Got shape={y_pred_arr.shape}.")
        if len(y_true_arr) != len(y_pred_arr):
            raise ValueError(
                "y_true and y_pred must have the same length. "
                f"Got len(y_true)={len(y_true_arr)} and len(y_pred)={len(y_pred_arr)}."
            )

        return y_true_arr, y_pred_arr

    def _validate_probability_inputs(
        self,
        y_proba: np.ndarray | None,
        n_samples: int,
        n_classes: int,
    ) -> np.ndarray | None:
        if y_proba is None:
            return None

        y_proba_arr = np.asarray(y_proba)

        if y_proba_arr.ndim != 2:
            raise ValueError(
                f"y_proba must be 2D with shape (n_samples, n_classes). "
                f"Got shape={y_proba_arr.shape}."
            )

        if y_proba_arr.shape[0] != n_samples:
            raise ValueError(
                "y_proba must have the same number of rows as y_true. "
                f"Got y_proba.shape[0]={y_proba_arr.shape[0]} and n_samples={n_samples}."
            )

        if y_proba_arr.shape[1] != n_classes:
            raise ValueError(
                "y_proba number of columns must match number of classes. "
                f"Got y_proba.shape[1]={y_proba_arr.shape[1]} and n_classes={n_classes}."
            )

        return y_proba_arr

    def _resolve_labels(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[Any] | None,
    ) -> list[Any]:
        if labels is not None:
            return list(labels)

        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        return unique_labels.tolist()

    def _build_output_path(self, file_name: str) -> Path:
        if self.output_dir is None:
            raise ValueError(
                "output_dir is not set. "
                "Initialize EvaluationService(output_dir=...) to save files."
            )
        return self.output_dir / file_name

    def _save_figure(self, fig: plt.Figure, save_name: str | None) -> None:
        if save_name is None:
            return
        path = self._build_output_path(save_name)
        fig.savefig(path, bbox_inches="tight")