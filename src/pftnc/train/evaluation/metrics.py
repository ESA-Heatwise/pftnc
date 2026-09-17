from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics

from pftnc.train.schemas import MetricResults

MetricFunction = Callable[[np.ndarray, np.ndarray], float]

METRIC_REGISTRY: dict[str, MetricFunction] = {
    "mse": sk_metrics.mean_squared_error,
    "mae": sk_metrics.mean_absolute_error,
    "r2": sk_metrics.r2_score,
    "rmse": sk_metrics.root_mean_squared_error,
    "explained_variance": sk_metrics.explained_variance_score,
    "max_error": sk_metrics.max_error,
    "median_absolute_error": sk_metrics.median_absolute_error,
}


def validate_metrics(metric_names: Iterable[str]) -> None:
    names = list(metric_names)

    if not names:
        raise ValueError("At least one evaluation metric must be configured.")

    unsupported = [
        metric_name for metric_name in names if metric_name not in METRIC_REGISTRY
    ]
    if unsupported:
        raise ValueError(f"Unsupported metrics: {unsupported}")


def validate_prediction_frames(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> None:
    """Validate truth and prediction alignment before evaluation."""

    if y_true.empty:
        raise ValueError("Cannot evaluate empty truth data.")

    if not y_true.index.is_unique:
        raise ValueError("y_true must have a unique index.")

    if not y_pred.index.is_unique:
        raise ValueError("y_pred must have a unique index.")

    if not y_true.index.equals(y_pred.index):
        raise ValueError("y_true and y_pred must have matching indexes.")

    if not y_true.columns.is_unique:
        raise ValueError("y_true must have unique columns.")

    if not y_pred.columns.is_unique:
        raise ValueError("y_pred must have unique columns.")

    if list(y_true.columns) != list(y_pred.columns):
        raise ValueError("y_true and y_pred must have matching columns.")


def compute_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    metric_names: Iterable[str],
) -> MetricResults:
    names = tuple(metric_names)
    validate_metrics(names)
    validate_prediction_frames(y_true, y_pred)

    metrics: MetricResults = {}
    for target in y_true.columns:
        metrics[target] = {}
        for metric_name in names:
            function = METRIC_REGISTRY[metric_name]
            metrics[target][metric_name] = float(
                function(
                    y_true[target].to_numpy(),
                    y_pred[target].to_numpy(),
                )
            )

    return metrics
