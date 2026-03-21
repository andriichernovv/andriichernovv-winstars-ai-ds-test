from __future__ import annotations

from typing import Optional, Sequence

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


REQUIRED_COLUMNS: tuple[str, str] = ("y_true", "y_pred")


def validate_results_dataframe(
    df: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
) -> None:
    """
    Validate that the input dataframe contains the required columns
    and has a valid structure for metric computation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with predictions.
    required_columns : Sequence[str], default=("y_true", "y_pred")
        Columns that must be present in the dataframe.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If dataframe is empty or required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected pandas DataFrame, got {type(df).__name__}."
        )

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}."
        )


def get_probability_columns(df: pd.DataFrame, prefix: str = "proba_") -> list[str]:
    """
    Extract probability columns from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    prefix : str, default="proba_"
        Prefix used for probability columns.

    Returns
    -------
    list[str]
        Sorted list of probability columns.
    """
    proba_columns = [col for col in df.columns if col.startswith(prefix)]

    def _extract_class_index(column_name: str) -> int:
        suffix = column_name.replace(prefix, "")
        return int(suffix)

    return sorted(proba_columns, key=_extract_class_index)


def compute_main_metrics(
    df: pd.DataFrame,
    average: str = "weighted",
    zero_division: int = 0,
) -> pd.DataFrame:
    """
    Compute the main classification metrics from a results dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'y_true' and 'y_pred'.
    average : str, default="weighted"
        Averaging strategy for precision, recall and F1.
    zero_division : int, default=0
        Value to return when there is a zero division case.

    Returns
    -------
    pd.DataFrame
        One-row dataframe with aggregate metrics.
    """
    validate_results_dataframe(df)

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average=average, zero_division=zero_division
        ),
        "recall": recall_score(
            y_true, y_pred, average=average, zero_division=zero_division
        ),
        "f1": f1_score(
            y_true, y_pred, average=average, zero_division=zero_division
        ),
        "n_samples": len(df),
    }

    return pd.DataFrame([metrics])


def compute_probability_metrics(
    df: pd.DataFrame,
    proba_prefix: str = "proba_",
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Compute probability-based metrics if probability columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'y_true' and probability columns.
    proba_prefix : str, default="proba_"
        Prefix for probability columns.
    top_k : int, default=3
        K for top-k accuracy.

    Returns
    -------
    pd.DataFrame
        One-row dataframe with probability-based metrics.
        If probability columns are absent, returns an empty dataframe.
    """
    validate_results_dataframe(df)

    proba_columns = get_probability_columns(df, prefix=proba_prefix)
    if not proba_columns:
        return pd.DataFrame()

    y_true = df["y_true"].to_numpy()
    y_proba = df[proba_columns].to_numpy()

    class_labels = [int(col.replace(proba_prefix, "")) for col in proba_columns]

    metrics: dict[str, float] = {}

    try:
        metrics["log_loss"] = log_loss(y_true, y_proba, labels=class_labels)
    except ValueError:
        metrics["log_loss"] = np.nan

    try:
        metrics[f"top_{top_k}_accuracy"] = top_k_accuracy_score(
            y_true,
            y_proba,
            k=top_k,
            labels=class_labels,
        )
    except ValueError:
        metrics[f"top_{top_k}_accuracy"] = np.nan

    return pd.DataFrame([metrics])


def compute_class_metrics(
    df: pd.DataFrame,
    zero_division: int = 0,
) -> pd.DataFrame:
    """
    Compute per-class precision, recall, F1-score and support.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'y_true' and 'y_pred'.
    zero_division : int, default=0
        Value to return when there is a zero division case.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per class and aggregate rows from
        sklearn classification_report.
    """
    validate_results_dataframe(df)

    report_dict = classification_report(
        df["y_true"],
        df["y_pred"],
        output_dict=True,
        zero_division=zero_division,
    )

    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "label"})

    return report_df


def compute_confusion_matrix_df(
    df: pd.DataFrame,
    normalize: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute confusion matrix and return it as a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'y_true' and 'y_pred'.
    normalize : Optional[str], default=None
        Normalization mode passed to sklearn.confusion_matrix.
        Supported values: None, 'true', 'pred', 'all'.

    Returns
    -------
    pd.DataFrame
        Confusion matrix as dataframe with indexed labels.
    """
    validate_results_dataframe(df)

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    labels = sorted(pd.unique(pd.concat([df["y_true"], df["y_pred"]], axis=0)))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize=normalize,
    )

    return pd.DataFrame(
        cm,
        index=pd.Index(labels, name="true_label"),
        columns=pd.Index(labels, name="pred_label"),
    )


def compute_error_analysis_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a dataframe containing only misclassified rows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'y_true' and 'y_pred'.

    Returns
    -------
    pd.DataFrame
        Subset of rows where prediction is incorrect, with an extra
        boolean column 'is_correct'.
    """
    validate_results_dataframe(df)

    error_df = df.copy()
    error_df["is_correct"] = error_df["y_true"] == error_df["y_pred"]

    return error_df.loc[~error_df["is_correct"]].reset_index(drop=True)


def compute_all_metrics(
    df: pd.DataFrame,
    average: str = "weighted",
    zero_division: int = 0,
    proba_prefix: str = "proba_",
    top_k: int = 3,
    normalize_confusion: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute all evaluation tables from a single results dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing predictions and optional probabilities.
    average : str, default="weighted"
        Averaging strategy for global precision, recall and F1.
    zero_division : int, default=0
        Value to return when there is a zero division case.
    proba_prefix : str, default="proba_"
        Prefix for probability columns.
    top_k : int, default=3
        K for top-k accuracy.
    normalize_confusion : Optional[str], default=None
        Normalization mode for confusion matrix.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with metric tables.
    """
    validate_results_dataframe(df)

    main_metrics_df = compute_main_metrics(
        df=df,
        average=average,
        zero_division=zero_division,
    )

    probability_metrics_df = compute_probability_metrics(
        df=df,
        proba_prefix=proba_prefix,
        top_k=top_k,
    )

    class_metrics_df = compute_class_metrics(
        df=df,
        zero_division=zero_division,
    )

    confusion_matrix_df = compute_confusion_matrix_df(
        df=df,
        normalize=normalize_confusion,
    )

    error_analysis_df = compute_error_analysis_df(df=df)

    return {
        "main_metrics": main_metrics_df,
        "probability_metrics": probability_metrics_df,
        "class_metrics": class_metrics_df,
        "confusion_matrix": confusion_matrix_df,
        "errors": error_analysis_df,
    }