from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd

from pftnc.train.evaluation.metrics import compute_metrics, validate_metrics
from pftnc.train.models.manager import ModelManager
from pftnc.train.schemas import EffectiveModelParams
from pftnc.train.targets.manager import TargetTransformManager
from pftnc.train.tracking import MlflowTracker

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Resolve configured parameters or run Optuna studies per fold."""

    def __init__(
        self,
        config: Any,
        models: ModelManager,
        targets: TargetTransformManager,
        tracker: MlflowTracker,
        seed: int,
    ) -> None:
        self.config = config
        self.models = models
        self.targets = targets
        self.tracker = tracker
        self.seed = seed
        self._configure_optuna_logging()

    @property
    def enabled(self) -> bool:
        return bool(self.config and self.config.enabled)

    def resolve(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.DataFrame,
        y_val: pd.DataFrame,
    ) -> EffectiveModelParams:
        if not self.enabled:
            return self.models.configured_params()
        return self._tune(X_train, X_val, y_train, y_val)

    def _tune(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.DataFrame,
        y_val: pd.DataFrame,
    ) -> EffectiveModelParams:
        tuning = self.config
        if tuning is None:
            raise ValueError("tuning config must be present")
        if tuning.search_space is None:
            raise ValueError("tuning.search_space must be set")
        if tuning.objective_metric is None:
            raise ValueError("tuning.objective_metric must be set")
        validate_metrics([tuning.objective_metric])
        if tuning.direction not in {"minimize", "maximize"}:
            raise ValueError("tuning.direction must be either 'minimize' or 'maximize'")

        logger.info(
            "Starting hyperparameter tuning | mode=%s | metric=%s | "
            "direction=%s | n_trials=%d",
            self.models.mode,
            tuning.objective_metric,
            tuning.direction,
            tuning.n_trials,
        )

        if self.models.mode == "multi_target":
            return self._tune_multi_target(
                X_train,
                X_val,
                y_train,
                y_val,
            )
        return self._tune_per_target(
            X_train,
            X_val,
            y_train,
            y_val,
        )

    def _tune_multi_target(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.DataFrame,
        y_val: pd.DataFrame,
    ) -> dict[str, Any]:
        tuning = self.config
        base_params = self.models.configured_params()
        metric_name = tuning.objective_metric
        aggregation = getattr(tuning, "objective_aggregation", "mean")
        target_weights = getattr(tuning, "target_weights", None)

        if not isinstance(base_params, dict):
            raise ValueError("Multi-target tuning requires one base-parameter mapping.")

        def objective(trial: optuna.Trial) -> float:
            sampled = self._sample_params(trial)
            trial_params = self.models.merge_params(base_params, sampled)

            with self.tracker.nested_run(run_name=f"trial_{trial.number}"):
                self.tracker.log_params(trial_params)
                self.tracker.log_param(
                    "objective_aggregation",
                    aggregation,
                )

                model = self.models.create_adapter(trial_params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[
                        (X_train, y_train),
                        (X_val, y_val),
                    ],
                    verbose=False,
                )

                predictions = np.asarray(model.predict(X_val))
                if predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)
                if predictions.ndim != 2 or predictions.shape[1] != len(y_val.columns):
                    raise ValueError(
                        "Prediction output shape does not match the number "
                        "of model outputs during tuning."
                    )

                pred_frame = pd.DataFrame(
                    predictions,
                    columns=y_val.columns,
                    index=y_val.index,
                )
                pred_physical = self.targets.inverse_predictions(
                    X_val,
                    pred_frame,
                )
                truth_physical = self.targets.inverse_predictions(
                    X_val,
                    y_val,
                )

                metric_results = compute_metrics(
                    truth_physical,
                    pred_physical,
                    [metric_name],
                )
                scores = {
                    target: target_metrics[metric_name]
                    for target, target_metrics in metric_results.items()
                }
                for target, score in scores.items():
                    self.tracker.log_metric(
                        f"tuning/{target}/{metric_name}",
                        score,
                    )

                score = aggregate_target_scores(
                    scores,
                    aggregation,
                    target_weights,
                    tuning.direction,
                )
                self.tracker.log_metric(
                    f"tuning/objective/{metric_name}",
                    score,
                )

            self.tracker.log_metric(
                f"tuning/objective/{metric_name}",
                score,
                step=trial.number,
            )
            return score

        study = optuna.create_study(
            direction=tuning.direction,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(objective, n_trials=tuning.n_trials)

        effective = self.models.merge_params(
            base_params,
            study.best_params,
        )
        logger.info(
            "Multi-target tuning completed | best_value=%.4f | "
            "effective_best_params=%s",
            study.best_value,
            effective,
        )
        return effective

    def _tune_per_target(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.DataFrame,
        y_val: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        tuning = self.config
        base_params = self.models.configured_params()
        metric_name = tuning.objective_metric
        outputs = list(y_train.columns)

        if not isinstance(base_params, dict):
            raise ValueError("Per-target tuning requires params_by_target.")

        missing = set(outputs) - set(base_params)
        extra = set(base_params) - set(outputs)
        if missing or extra:
            raise ValueError(
                "Per-target tuning parameters do not match model outputs. "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        effective: dict[str, dict[str, Any]] = {}
        for target_index, target in enumerate(outputs):
            target_base_params = base_params[target]
            if not isinstance(target_base_params, dict):
                raise ValueError(
                    f"Configured parameters for '{target}' must be a mapping."
                )

            safe_target = self.models.safe_name(target)
            logger.info(
                "Starting independent tuning for model output: %s",
                target,
            )

            def objective(
                trial: optuna.Trial,
                *,
                output_column: str = target,
                output_base_params: dict[str, Any] = target_base_params,
                output_safe_name: str = safe_target,
            ) -> float:
                sampled = self._sample_params(trial)
                trial_params = self.models.merge_params(
                    output_base_params,
                    sampled,
                )

                with self.tracker.nested_run(
                    run_name=f"trial_{trial.number}",
                    description=(
                        f"Optuna trial {trial.number} for model output "
                        f"'{output_column}'."
                    ),
                    tags={
                        "run_type": "optuna_trial",
                        "model_output": output_column,
                        "trial_number": trial.number,
                    },
                ):
                    self.tracker.log_params(trial_params)
                    model = self.models.create_adapter(trial_params)
                    model.fit(
                        X_train,
                        y_train[output_column],
                        eval_set=[
                            (X_train, y_train[output_column]),
                            (X_val, y_val[output_column]),
                        ],
                        verbose=False,
                    )
                    target_predictions = np.asarray(model.predict(X_val)).reshape(-1)
                    if len(target_predictions) != len(y_val):
                        raise ValueError(
                            f"Prediction length for '{output_column}' does "
                            "not match validation data."
                        )

                    predicted_outputs = y_val.copy()
                    predicted_outputs[output_column] = target_predictions

                    pred_physical = self.targets.inverse_predictions(
                        X_val,
                        predicted_outputs,
                    )
                    truth_physical = self.targets.inverse_predictions(
                        X_val,
                        y_val,
                    )
                    physical_target = self.targets.physical_target_for_output(
                        output_column
                    )
                    score = compute_metrics(
                        truth_physical[[physical_target]],
                        pred_physical[[physical_target]],
                        [metric_name],
                    )[physical_target][metric_name]
                    metric_key = f"tuning/{output_safe_name}/score/{metric_name}"

                    self.tracker.log_metric(metric_key, score)
                    self.tracker.log_metric(
                        f"tuning/{output_safe_name}/objective/{metric_name}",
                        score,
                    )

                self.tracker.log_metric(
                    f"tuning/{output_safe_name}/objective/{metric_name}",
                    score,
                    step=trial.number,
                )
                return score

            with self.tracker.nested_run(
                run_name=f"tuning_{safe_target}",
                description=(
                    f"Independent hyperparameter tuning for model output '{target}'."
                ),
                tags={
                    "run_type": "target_tuning",
                    "model_output": target,
                },
            ):
                self.tracker.log_params(
                    {
                        "n_trials": tuning.n_trials,
                        "objective_metric": metric_name,
                        "direction": tuning.direction,
                    }
                )

                study = optuna.create_study(
                    direction=tuning.direction,
                    sampler=optuna.samplers.TPESampler(seed=self.seed + target_index),
                )
                study.optimize(objective, n_trials=tuning.n_trials)

                target_best = self.models.merge_params(
                    target_base_params,
                    study.best_params,
                )
                effective[target] = target_best

                self.tracker.log_metric(
                    f"best_{metric_name}",
                    float(study.best_value),
                )
                self.tracker.log_param(
                    "best_trial",
                    study.best_trial.number,
                )
                self.tracker.log_params(
                    {f"best_{name}": value for name, value in target_best.items()}
                )

                logger.info(
                    "Independent tuning completed | output=%s | "
                    "best_value=%.4f | effective_best_params=%s",
                    target,
                    study.best_value,
                    target_best,
                )

        return effective

    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        sampled: dict[str, Any] = {}
        for name, bounds in self.config.search_space.items():
            if (
                isinstance(bounds, list)
                and len(bounds) == 2
                and all(isinstance(value, int) for value in bounds)
            ):
                sampled[name] = trial.suggest_int(
                    name,
                    bounds[0],
                    bounds[1],
                )
            elif (
                isinstance(bounds, list)
                and len(bounds) == 2
                and all(isinstance(value, (int, float)) for value in bounds)
                and any(isinstance(value, float) for value in bounds)
            ):
                sampled[name] = trial.suggest_float(
                    name,
                    bounds[0],
                    bounds[1],
                )
            else:
                sampled[name] = trial.suggest_categorical(name, bounds)
        return sampled

    @staticmethod
    def _configure_optuna_logging() -> None:
        optuna.logging.disable_default_handler()
        optuna_logger = optuna.logging.get_logger("optuna")
        optuna_logger.setLevel(logging.INFO)


def aggregate_target_scores(
    per_target_scores: dict[str, float],
    aggregation: str,
    target_weights: dict[str, float] | None,
    direction: str,
) -> float:
    scores = list(per_target_scores.values())
    targets = list(per_target_scores)

    if not scores:
        raise ValueError("per_target_scores is empty; cannot aggregate.")
    if aggregation == "mean":
        return sum(scores) / len(scores)
    if aggregation == "worst":
        return max(scores) if direction == "minimize" else min(scores)
    if aggregation == "weighted":
        if not target_weights:
            raise ValueError(
                "objective_aggregation is 'weighted' but target_weights is not set"
            )
        missing = set(targets) - set(target_weights)
        if missing:
            raise ValueError(
                f"target_weights is missing entries for: {missing}. "
                f"All targets must have a weight: {targets}"
            )
        total_weight = sum(target_weights[target] for target in targets)
        if total_weight <= 0:
            raise ValueError("target_weights must sum to a positive number.")
        return (
            sum(
                per_target_scores[target] * target_weights[target] for target in targets
            )
            / total_weight
        )
    raise ValueError(
        f"Unknown objective_aggregation: '{aggregation}'. "
        "Valid options: 'mean', 'weighted', 'worst'."
    )
