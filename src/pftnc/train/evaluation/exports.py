from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pftnc.train.evaluation.metrics import validate_prediction_frames
from pftnc.train.tracking import MlflowTracker

logger = logging.getLogger(__name__)


class PredictionExporter:
    """Export metadata, model inputs, truth, and predictions to CSV."""

    def __init__(self, tracker: MlflowTracker) -> None:
        self.tracker = tracker

    def export(
        self,
        run_dir: Path,
        split: str,
        X: pd.DataFrame,
        metadata: pd.DataFrame,
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
    ) -> Path:
        validate_prediction_frames(y_true, y_pred)
        self._validate_input_indexes(X, metadata, y_true)

        aligned_metadata = metadata.loc[X.index].copy()

        feature_df = X.drop(
            columns=[
                column for column in aligned_metadata.columns if column in X.columns
            ],
            errors="ignore",
        )

        frame = pd.concat(
            [
                aligned_metadata,
                feature_df,
                y_true.add_suffix("_true"),
                y_pred.add_suffix("_pred"),
            ],
            axis=1,
        )

        path = run_dir / "predictions" / f"{split}_full_predictions.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

        if not path.is_file():
            raise RuntimeError(f"Prediction CSV was not created: {path}")

        logger.debug("Saved prediction CSV: %s", path)
        self.tracker.log_artifact(
            path,
            artifact_path=f"{split}/predictions",
        )
        return path

    @staticmethod
    def _validate_input_indexes(
        X: pd.DataFrame,
        metadata: pd.DataFrame,
        y_true: pd.DataFrame,
    ) -> None:
        if not X.index.is_unique:
            raise ValueError("X must have a unique index.")

        if not metadata.index.is_unique:
            raise ValueError("Prediction metadata must have a unique index.")

        if not X.index.equals(y_true.index):
            raise ValueError("X, y_true, and y_pred must have matching indexes.")

        missing_metadata_rows = X.index.difference(metadata.index)
        if not missing_metadata_rows.empty:
            raise ValueError(
                "Prediction metadata is missing rows required by X. "
                f"Missing count: {len(missing_metadata_rows)}"
            )
