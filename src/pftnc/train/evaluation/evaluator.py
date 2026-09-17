from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pftnc.train.data import PreparedSplit
from pftnc.train.evaluation.metrics import (
    compute_metrics,
    validate_metrics,
    validate_prediction_frames,
)
from pftnc.train.evaluation.protocols import (
    EvaluationPlotterProtocol,
    PredictionExporterProtocol,
    ShapLoggerProtocol,
)
from pftnc.train.models.manager import ModelManager
from pftnc.train.schemas import MetricResults, ModelBundleReference
from pftnc.train.targets.manager import TargetTransformManager
from pftnc.train.tracking import ArtifactStore, MlflowTracker

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate model bundles and already-computed physical predictions."""

    def __init__(
        self,
        metric_names: list[str],
        models: ModelManager,
        targets: TargetTransformManager,
        tracker: MlflowTracker,
        artifacts: ArtifactStore,
        exporter: PredictionExporterProtocol,
        plotter: EvaluationPlotterProtocol,
        shap_logger: ShapLoggerProtocol,
    ) -> None:
        validate_metrics(metric_names)

        self.metric_names = tuple(metric_names)
        self.models = models
        self.targets = targets
        self.tracker = tracker
        self.artifacts = artifacts
        self.exporter = exporter
        self.plotter = plotter
        self.shap_logger = shap_logger

    def evaluate_model(
        self,
        bundle: ModelBundleReference,
        split_data: PreparedSplit,
        run_dir: Path,
        split: str,
        step: int | None = None,
    ) -> MetricResults:
        """
        Evaluate a model bundle in physical target units.

        PreparedSplit.y always contains physical labels. Target transformations
        are applied only to the separate model-training copies. Predictions are
        returned in model-output space and are converted back through the target
        manager before metrics and reporting.
        """

        logger.info(
            "Evaluating split '%s'%s",
            split,
            f" (step={step})" if step is not None else "",
        )

        model_output_columns = pd.Index(self.targets.model_outputs)

        predictions, loaded_models = self.models.predict(
            bundle,
            split_data.X,
            model_output_columns,
        )

        physical_predictions = self.targets.inverse_predictions(
            split_data.X,
            predictions,
        )
        physical_truth = self.targets.physical_truth(
            split_data.y,
        )

        split_tag = split if step is None else f"{split}/{step}"

        metrics = self._evaluate_physical_predictions(
            scope=split,
            split_tag=split_tag,
            artifact_scope=split_tag,
            metrics_filename=(
                f"{split}_metrics.json"
                if step is None
                else f"{split}_{step}_metrics.json"
            ),
            split_data=split_data,
            y_true=physical_truth,
            y_pred=physical_predictions,
            run_dir=run_dir,
            step=step,
        )

        shap_output_names = model_output_columns

        if bundle.is_multi_target:
            self.shap_logger.log(
                run_dir,
                loaded_models["__multi_target__"],
                split_data.X,
                shap_output_names,
                split_tag,
            )
        else:
            for output_column in model_output_columns:
                self.shap_logger.log(
                    run_dir,
                    loaded_models[output_column],
                    split_data.X,
                    pd.Index([output_column]),
                    split_tag,
                )

        return metrics

    def evaluate_predictions(
        self,
        name: str,
        predictions: pd.DataFrame,
        split_data: PreparedSplit,
        run_dir: Path,
        step: int | None = None,
    ) -> MetricResults:
        """
        Evaluate already-computed predictions in physical target space.

        Final-test ensembles are constructed after each fold prediction has
        already been inverse-transformed. They must not be inverse-transformed
        a second time here.
        """

        logger.info(
            "Evaluating predictions: %s%s",
            name,
            f" (step={step})" if step is not None else "",
        )

        physical_truth = self.targets.physical_truth(
            split_data.y,
        )
        physical_predictions = self._ordered_physical_predictions(predictions)

        return self._evaluate_physical_predictions(
            scope=name,
            split_tag=f"final_test/{name}",
            artifact_scope=name,
            metrics_filename=f"{name}_metrics.json",
            split_data=split_data,
            y_true=physical_truth,
            y_pred=physical_predictions,
            run_dir=run_dir,
            step=step,
        )

    def _evaluate_physical_predictions(
        self,
        *,
        scope: str,
        split_tag: str,
        artifact_scope: str,
        metrics_filename: str,
        split_data: PreparedSplit,
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
        run_dir: Path,
        step: int | None,
    ) -> MetricResults:
        self._validate_split_alignment(
            split_data=split_data,
            y_true=y_true,
            y_pred=y_pred,
        )

        metrics = compute_metrics(
            y_true,
            y_pred,
            self.metric_names,
        )
        self._log_metrics(
            metrics=metrics,
            scope=scope,
            step=step,
        )

        metrics_path = self.artifacts.write_json(
            run_dir / metrics_filename,
            metrics,
        )
        self.tracker.log_artifact(
            metrics_path,
            artifact_path=artifact_scope,
        )

        self.exporter.export(
            run_dir,
            split_tag,
            split_data.X,
            split_data.metadata,
            y_true,
            y_pred,
        )
        self.plotter.prediction_plots(
            run_dir,
            split_tag,
            split_data.metadata,
            y_true,
            y_pred,
        )
        self.plotter.metric_comparison_plots(
            metrics,
            split_tag,
            run_dir,
        )

        logger.debug(
            "Completed physical evaluation for scope '%s'",
            scope,
        )
        return metrics

    def _ordered_physical_predictions(
        self,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        expected = list(self.targets.physical_targets)
        missing = set(expected) - set(predictions.columns)
        extra = set(predictions.columns) - set(expected)

        if missing or extra:
            raise ValueError(
                "Physical prediction columns do not match physical targets. "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        return predictions[expected].copy()

    def _log_metrics(
        self,
        metrics: MetricResults,
        scope: str,
        step: int | None,
    ) -> None:
        for target, target_metrics in metrics.items():
            for metric_name, value in target_metrics.items():
                self.tracker.log_metric(
                    self._metric_key(
                        scope=scope,
                        metric_name=metric_name,
                        target=target,
                    ),
                    value,
                    step=step,
                )

    @staticmethod
    def _metric_key(
        scope: str,
        metric_name: str,
        target: str,
    ) -> str:
        return f"evaluation/{scope}/{metric_name}/{target}"

    @staticmethod
    def _validate_split_alignment(
        split_data: PreparedSplit,
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
    ) -> None:
        validate_prediction_frames(y_true, y_pred)

        if not split_data.X.index.is_unique:
            raise ValueError("split_data.X must have a unique index.")

        if not split_data.X.index.equals(y_true.index):
            raise ValueError(
                "split_data.X, physical truth, and predictions must "
                "have matching indexes."
            )

        if not split_data.metadata.index.is_unique:
            raise ValueError("split_data.metadata must have a unique index.")

        missing_metadata_rows = split_data.X.index.difference(split_data.metadata.index)
        if not missing_metadata_rows.empty:
            raise ValueError(
                "split_data.metadata is missing rows required by X. "
                f"Missing count: {len(missing_metadata_rows)}"
            )
