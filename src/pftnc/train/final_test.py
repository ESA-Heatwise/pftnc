from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pftnc.train.data import DatasetRepository
from pftnc.train.evaluation.evaluator import Evaluator
from pftnc.train.models.manager import ModelManager
from pftnc.train.schemas import BestModelResult, FoldResult, SelectedModel
from pftnc.train.targets.manager import TargetTransformManager
from pftnc.train.tracking import MlflowTracker

logger = logging.getLogger(__name__)


class ModelSelector:
    """Select overall and per-target best fold models from validation metrics."""

    def __init__(
        self,
        selection_config: Any,
        metric_names: list[str],
        targets: TargetTransformManager,
        models: ModelManager,
    ) -> None:
        self.config = selection_config
        self.metric_names = tuple(metric_names)
        self.targets = targets
        self.models = models

    def select(
        self,
        fold_results: dict[str, FoldResult],
    ) -> BestModelResult:
        if not fold_results:
            raise ValueError("No fold results were provided for model selection.")

        metric = self.config.metric
        direction = self.config.direction
        target = getattr(self.config, "target", None)
        mode = getattr(self.config, "mode", "mean")

        if metric is None:
            raise ValueError("best_model_selection.metric must be set")

        if metric not in self.metric_names:
            raise ValueError(f"{metric} must be one of training.metrics")

        if direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be 'minimize' or 'maximize'")

        if mode not in {"mean", "median", "worst"}:
            raise ValueError("mode must be one of 'mean', 'median', or 'worst'")

        logger.info(
            "Selecting best model | metric=%s | direction=%s | mode=%s | target=%s",
            metric,
            direction,
            mode,
            target,
        )

        best_fold: str | None = None
        best_score: float | None = None

        for fold_name, result in fold_results.items():
            validation = result.validation_metrics

            if target is not None:
                score = _get_metric_score(
                    validation_metrics=validation,
                    target=target,
                    metric=metric,
                )
            else:
                scores = [
                    _get_metric_score(
                        validation_metrics=validation,
                        target=physical_target,
                        metric=metric,
                    )
                    for physical_target in self.targets.physical_targets
                ]

                score = _aggregate_selection_scores(
                    scores=scores,
                    mode=mode,
                    direction=direction,
                )

            if _is_better(
                score=score,
                current_best=best_score,
                direction=direction,
            ):
                best_fold = fold_name
                best_score = score

        if best_fold is None or best_score is None:
            raise ValueError("No valid folds found for model selection.")

        best_result = fold_results[best_fold]

        overall = SelectedModel(
            model_uri=best_result.model_bundle.model_uri,
            fold=best_fold,
            metric=metric,
            score=best_score,
            target=target,
            mode=mode,
        )

        per_target: dict[str, SelectedModel] = {}

        for physical_target in self.targets.physical_targets:
            target_fold: str | None = None
            target_score: float | None = None

            for fold_name, result in fold_results.items():
                validation = result.validation_metrics

                if physical_target not in validation:
                    continue

                score = _get_metric_score(
                    validation_metrics=validation,
                    target=physical_target,
                    metric=metric,
                )

                if _is_better(
                    score=score,
                    current_best=target_score,
                    direction=direction,
                ):
                    target_fold = fold_name
                    target_score = score

            if target_fold is None or target_score is None:
                raise ValueError(f"No valid fold found for target '{physical_target}'.")

            bundle = fold_results[target_fold].model_bundle

            per_target[physical_target] = SelectedModel(
                model_uri=self.models.selected_model_uri(
                    bundle=bundle,
                    target=physical_target,
                    requires_complete_bundle=(
                        self.targets.requires_complete_bundle_for_target_selection
                    ),
                ),
                fold=target_fold,
                metric=metric,
                score=target_score,
                target=physical_target,
                mode="per_target",
            )

        logger.info(
            "Best overall model selected | fold=%s | metric=%s | score=%.4f",
            best_fold,
            metric,
            best_score,
        )

        logger.info(
            "Per-target best folds: %s",
            {name: selection.fold for name, selection in per_target.items()},
        )

        return BestModelResult(
            overall=overall,
            per_target=per_target,
        )


class FinalTestRunner:
    """Evaluate fold models and their ensembles on the final test split."""

    def __init__(
        self,
        repository: DatasetRepository,
        models: ModelManager,
        targets: TargetTransformManager,
        evaluator: Evaluator,
        tracker: MlflowTracker,
        selection_config: Any,
        metric_names: list[str],
    ) -> None:
        self.repository = repository
        self.models = models
        self.targets = targets
        self.evaluator = evaluator
        self.tracker = tracker
        self.selection_config = selection_config
        self.metric_names = tuple(metric_names)

    def run(
        self,
        fold_results: dict[str, FoldResult],
        run_dir: Path,
    ) -> dict[str, Any]:
        logger.info("Starting final test evaluation")

        if not fold_results:
            raise ValueError("No fold results were provided for final-test evaluation.")

        expected_feature_columns = self._resolve_feature_columns(fold_results)

        final_data = self.repository.load_final_test(
            model_columns=expected_feature_columns,
        )

        logger.info(
            "Final test data loaded | X_final=%s | y_final=%s",
            final_data.X.shape,
            final_data.y.shape,
        )

        weight_metric, direction = self._resolve_weighting_config()

        physical_truth = self.targets.physical_truth(final_data.y)

        physical_targets = list(self.targets.physical_targets)

        model_output_columns = pd.Index(self.targets.model_outputs)

        final_test_dir = run_dir / "final_test"
        final_test_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results: dict[str, Any] = {}

        all_predictions: list[pd.DataFrame] = []

        global_weights: list[float] = []
        globally_weighted_predictions: list[pd.DataFrame] = []

        per_target_weights: dict[
            str,
            list[float],
        ] = {target: [] for target in physical_targets}

        per_target_weighted_predictions: dict[
            str,
            list[pd.Series],
        ] = {target: [] for target in physical_targets}

        with self.tracker.nested_run(run_name="final_test"):
            self.tracker.log_params(
                {
                    "n_fold_models": len(fold_results),
                    "weight_metric": weight_metric,
                    "weight_direction": direction,
                }
            )

            for fold_name, fold_result in fold_results.items():
                logger.info(
                    "Scoring fold model on final_test: %s",
                    fold_name,
                )

                model_predictions, _ = self.models.predict(
                    fold_result.model_bundle,
                    final_data.X,
                    model_output_columns,
                )

                physical_predictions = self.targets.inverse_predictions(
                    final_data.X,
                    model_predictions,
                )

                physical_predictions = physical_predictions[physical_targets].copy()

                self._validate_predictions(
                    fold_name=fold_name,
                    predictions=physical_predictions,
                    truth=physical_truth,
                )

                all_predictions.append(physical_predictions)

                results[fold_name] = self.evaluator.evaluate_predictions(
                    name=fold_name,
                    predictions=physical_predictions,
                    split_data=final_data,
                    run_dir=final_test_dir,
                    step=self._fold_step(fold_name),
                )

                validation_scores = [
                    self._validation_score(
                        fold_result=fold_result,
                        target=target,
                        metric=weight_metric,
                    )
                    for target in physical_targets
                ]

                mean_validation_score = float(np.mean(validation_scores))

                global_weight = _score_to_weight(
                    score=mean_validation_score,
                    direction=direction,
                )

                global_weights.append(global_weight)

                globally_weighted_predictions.append(
                    physical_predictions * global_weight
                )

                for target in physical_targets:
                    target_score = self._validation_score(
                        fold_result=fold_result,
                        target=target,
                        metric=weight_metric,
                    )

                    target_weight = _score_to_weight(
                        score=target_score,
                        direction=direction,
                    )

                    per_target_weights[target].append(target_weight)

                    per_target_weighted_predictions[target].append(
                        physical_predictions[target] * target_weight
                    )

            ensemble_simple = self._simple_ensemble(all_predictions)

            ensemble_weighted = self._weighted_ensemble(
                weighted_predictions=(globally_weighted_predictions),
                weights=global_weights,
            )

            ensemble_per_target = self._per_target_weighted_ensemble(
                weighted_predictions=(per_target_weighted_predictions),
                weights=per_target_weights,
                targets=physical_targets,
                index=final_data.X.index,
            )

            results["ensemble_simple"] = self.evaluator.evaluate_predictions(
                name="ensemble_simple",
                predictions=ensemble_simple,
                split_data=final_data,
                run_dir=final_test_dir,
            )

            results["ensemble_weighted"] = self.evaluator.evaluate_predictions(
                name="ensemble_weighted",
                predictions=ensemble_weighted,
                split_data=final_data,
                run_dir=final_test_dir,
            )

            results["ensemble_weighted_per_target"] = (
                self.evaluator.evaluate_predictions(
                    name=("ensemble_weighted_per_target"),
                    predictions=(ensemble_per_target),
                    split_data=final_data,
                    run_dir=final_test_dir,
                )
            )

        self._export_per_fold_summary(
            results=results,
            run_dir=run_dir,
        )

        return results

    def _resolve_feature_columns(
        self,
        fold_results: dict[str, FoldResult],
    ) -> list[str]:
        feature_columns_by_fold = {
            fold_name: tuple(result.model_bundle.feature_columns)
            for fold_name, result in fold_results.items()
        }

        missing = [
            fold_name
            for fold_name, columns in feature_columns_by_fold.items()
            if not columns
        ]

        if missing:
            raise ValueError(
                f"The following model bundles have no feature columns: {missing}"
            )

        unique_feature_sets = set(feature_columns_by_fold.values())

        if len(unique_feature_sets) != 1:
            raise ValueError(
                "Fold models were trained with different "
                "feature columns: "
                f"{feature_columns_by_fold}"
            )

        return list(next(iter(unique_feature_sets)))

    def _resolve_weighting_config(
        self,
    ) -> tuple[str, str]:
        metric = self.selection_config.metric
        direction = self.selection_config.direction

        if metric is None:
            raise ValueError("best_model_selection.metric must be set")

        if direction not in {
            "minimize",
            "maximize",
        }:
            raise ValueError(
                "best_model_selection.direction must be 'minimize' or 'maximize'"
            )

        if not self.metric_names:
            raise ValueError("At least one training metric must be configured.")

        if metric not in self.metric_names:
            fallback = self.metric_names[0]

            logger.warning(
                "Weight metric '%s' not found in training metrics %s. "
                "Falling back to '%s'.",
                metric,
                self.metric_names,
                fallback,
            )

            metric = fallback

        return metric, direction

    def _validation_score(
        self,
        fold_result: FoldResult,
        target: str,
        metric: str,
    ) -> float:
        return _get_metric_score(
            validation_metrics=(fold_result.validation_metrics),
            target=target,
            metric=metric,
        )

    def _validate_predictions(
        self,
        fold_name: str,
        predictions: pd.DataFrame,
        truth: pd.DataFrame,
    ) -> None:
        if not predictions.index.equals(truth.index):
            raise ValueError(f"Prediction index mismatch for fold '{fold_name}'.")

        if list(predictions.columns) != list(truth.columns):
            raise ValueError(
                "Prediction columns do not match "
                "physical targets for "
                f"fold '{fold_name}'."
            )

        values = predictions.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite predictions generated by fold '{fold_name}'.")

    def _simple_ensemble(
        self,
        predictions: list[pd.DataFrame],
    ) -> pd.DataFrame:
        if not predictions:
            raise ValueError("No fold predictions were generated for final_test.")

        total = predictions[0].copy()

        for prediction in predictions[1:]:
            total = total.add(prediction)

        return total / len(predictions)

    def _weighted_ensemble(
        self,
        weighted_predictions: list[pd.DataFrame],
        weights: list[float],
    ) -> pd.DataFrame:
        if not weighted_predictions:
            raise ValueError("No globally weighted predictions were generated.")

        if len(weighted_predictions) != len(weights):
            raise ValueError("Prediction and weight counts do not match.")

        total_weight = float(np.sum(weights))

        if not np.isfinite(total_weight) or total_weight <= 0:
            raise ValueError(
                "Global ensemble weight must be "
                "positive and finite. "
                f"Received: {total_weight}"
            )

        total = weighted_predictions[0].copy()

        for prediction in weighted_predictions[1:]:
            total = total.add(prediction)

        return total / total_weight

    def _per_target_weighted_ensemble(
        self,
        weighted_predictions: dict[
            str,
            list[pd.Series],
        ],
        weights: dict[
            str,
            list[float],
        ],
        targets: list[str],
        index: pd.Index,
    ) -> pd.DataFrame:
        ensemble = pd.DataFrame(
            index=index,
            columns=targets,
            dtype=float,
        )

        for target in targets:
            target_predictions = weighted_predictions[target]
            target_weights = weights[target]

            if not target_predictions:
                raise ValueError(f"No predictions available for target '{target}'.")

            if len(target_predictions) != len(target_weights):
                raise ValueError(
                    f"Prediction and weight counts do not match for target '{target}'."
                )

            total_weight = float(np.sum(target_weights))

            if not np.isfinite(total_weight) or total_weight <= 0:
                raise ValueError(
                    "Total weight must be positive "
                    "and finite for target "
                    f"'{target}'. "
                    f"Received: {total_weight}"
                )

            total = target_predictions[0].copy()

            for prediction in target_predictions[1:]:
                total = total.add(prediction)

            ensemble[target] = total / total_weight

        return ensemble

    def _fold_step(
        self,
        fold_name: str,
    ) -> int | None:
        _, separator, suffix = fold_name.rpartition("_")

        if not separator:
            logger.warning(
                "Could not derive step from fold name: %s",
                fold_name,
            )
            return None

        try:
            return int(suffix)
        except ValueError:
            logger.warning(
                "Could not derive numeric step from fold name: %s",
                fold_name,
            )
            return None

    def _export_per_fold_summary(
        self,
        results: dict[str, Any],
        run_dir: Path,
    ) -> None:
        per_fold_rows = {
            name: metrics
            for name, metrics in results.items()
            if name.startswith("fold_")
        }

        if not per_fold_rows:
            return

        path = run_dir / "final_test_per_fold.csv"

        pd.DataFrame(per_fold_rows).T.to_csv(path)

        self.tracker.log_artifact(path)

        logger.info(
            "Saved final test per-fold summary to: %s",
            path,
        )


def _get_metric_score(
    validation_metrics: dict[
        str,
        dict[str, float],
    ],
    target: str,
    metric: str,
) -> float:
    target_metrics = validation_metrics.get(target)

    if target_metrics is None:
        raise ValueError(f"Validation metrics are missing target '{target}'.")

    if metric not in target_metrics:
        raise ValueError(
            f"Validation metric '{metric}' is missing for target '{target}'."
        )

    score = float(target_metrics[metric])

    if not np.isfinite(score):
        raise ValueError(
            "Validation score must be finite. "
            f"target='{target}', "
            f"metric='{metric}', "
            f"score={score}"
        )

    return score


def _aggregate_selection_scores(
    scores: list[float],
    mode: str,
    direction: str,
) -> float:
    if not scores:
        raise ValueError("No metric scores were provided for aggregation.")

    if mode == "worst":
        return max(scores) if direction == "minimize" else min(scores)

    if mode == "median":
        return float(np.median(scores))

    if mode == "mean":
        return float(np.mean(scores))

    raise ValueError(f"Unsupported model-selection mode: {mode}")


def _is_better(
    score: float,
    current_best: float | None,
    direction: str,
) -> bool:
    if direction not in {
        "minimize",
        "maximize",
    }:
        raise ValueError(f"Unsupported optimization direction: {direction}")

    return (
        current_best is None
        or (direction == "minimize" and score < current_best)
        or (direction == "maximize" and score > current_best)
    )


def _score_to_weight(
    score: float,
    direction: str,
) -> float:
    score = float(score)

    if not np.isfinite(score):
        raise ValueError(f"Cannot convert non-finite score to weight: {score}")

    if direction == "minimize":
        if score < 0:
            raise ValueError(
                f"A minimized weighting metric must be non-negative. Received: {score}"
            )

        return 1.0 / max(
            score,
            1e-8,
        )

    if direction == "maximize":
        return max(
            score,
            1e-8,
        )

    raise ValueError(f"Unsupported optimization direction: {direction}")
