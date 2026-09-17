from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from pftnc.train.schemas import MetricResults


class PredictionExporterProtocol(Protocol):
    def export(
        self,
        run_dir: Path,
        split: str,
        X: pd.DataFrame,
        metadata: pd.DataFrame,
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
    ) -> Path: ...


class EvaluationPlotterProtocol(Protocol):
    def prediction_plots(
        self,
        run_dir: Path,
        split: str,
        metadata: pd.DataFrame,
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
    ) -> None: ...

    def metric_comparison_plots(
        self,
        metrics: MetricResults,
        split: str,
        run_dir: Path,
    ) -> None: ...


class ShapLoggerProtocol(Protocol):
    def log(
        self,
        run_dir: Path,
        model: Any,
        X: pd.DataFrame,
        targets: pd.Index,
        split: str,
    ) -> None: ...
