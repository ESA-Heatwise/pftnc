from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pftnc.config import ModelMode
from pftnc.train.models.base import ModelAdapter
from pftnc.train.schemas import (
    EffectiveModelParams,
    ModelBundleReference,
    ModelReference,
)
from pftnc.train.tracking import MlflowTracker

logger = logging.getLogger(__name__)


class ModelManager:
    """Fit, persist, load, and predict with configured model bundles."""

    def __init__(
        self,
        adapter_cls: type[ModelAdapter],
        model_config: Any,
        seed: int,
        tracker: MlflowTracker,
        model_outputs: list[str],
    ) -> None:
        self.adapter_cls = adapter_cls
        self.config = model_config
        self.seed = seed
        self.tracker = tracker
        self.model_outputs = model_outputs
        self._validate_configuration()

    @property
    def mode(self) -> ModelMode:
        return self.config.mode

    def create_adapter(self, params: dict[str, Any]) -> ModelAdapter:
        return self.adapter_cls(params, seed=self.seed)

    def configured_params(self) -> EffectiveModelParams:
        if self.mode == "multi_target":
            params = self.config.params
            if params is None:
                raise ValueError(
                    "training.model.params must be set for multi_target mode."
                )
            return dict(params)

        params_by_target = self.config.params_by_target
        if params_by_target is None:
            raise ValueError(
                "training.model.params_by_target must be set for per_target mode."
            )
        return {
            target: dict(target_params)
            for target, target_params in params_by_target.items()
        }

    def fit(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.DataFrame,
        y_val: pd.DataFrame,
        params: EffectiveModelParams,
        output_dir: Path,
    ) -> ModelBundleReference:
        if self.mode == "multi_target":
            return self._fit_multi_target(
                X_train,
                X_val,
                y_train,
                y_val,
                params,
                output_dir,
            )
        return self._fit_per_target(
            X_train,
            X_val,
            y_train,
            y_val,
            params,
            output_dir,
        )

    def predict(
        self,
        bundle: ModelBundleReference,
        X: pd.DataFrame,
        output_columns: pd.Index,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        outputs = list(output_columns)

        if bundle.is_multi_target:
            reference = bundle.models["__multi_target__"]
            adapter = self.adapter_cls.load_from_mlflow(reference.uri)
            predictions = np.asarray(adapter.predict(X))

            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            if predictions.ndim != 2 or predictions.shape[1] != len(outputs):
                raise ValueError(
                    "Prediction output shape does not match number of model outputs."
                )

            return (
                pd.DataFrame(
                    predictions,
                    columns=outputs,
                    index=X.index,
                ),
                {"__multi_target__": adapter.model},
            )

        missing = set(outputs) - set(bundle.models)
        extra = set(bundle.models) - set(outputs)
        if missing or extra:
            raise ValueError(
                "Per-target model references do not match model outputs. "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        prediction_columns: dict[str, np.ndarray] = {}
        loaded_models: dict[str, Any] = {}

        for target in outputs:
            adapter = self.adapter_cls.load_from_mlflow(bundle.models[target].uri)
            target_predictions = np.asarray(adapter.predict(X)).reshape(-1)
            if len(target_predictions) != len(X):
                raise ValueError(f"Prediction length for '{target}' does not match X.")
            prediction_columns[target] = target_predictions
            loaded_models[target] = adapter.model

        return (
            pd.DataFrame(prediction_columns, index=X.index),
            loaded_models,
        )

    def selected_model_uri(
        self,
        bundle: ModelBundleReference,
        target: str,
        requires_complete_bundle: bool,
    ) -> str | dict[str, str]:
        if bundle.is_multi_target:
            return bundle.model_uri
        if not requires_complete_bundle and target in bundle.models:
            return bundle.models[target].uri
        return bundle.model_uri

    def log_effective_params(self, params: EffectiveModelParams) -> None:
        if self.mode == "multi_target":
            self.tracker.log_params(params)
            return

        flattened: dict[str, Any] = {}
        for target, target_params in params.items():
            safe_target = self.safe_name(target)
            for name, value in target_params.items():
                flattened[f"model.{safe_target}.{name}"] = value
        self.tracker.log_params(flattened)

    @staticmethod
    def merge_params(
        base_params: dict[str, Any],
        sampled_params: dict[str, Any],
    ) -> dict[str, Any]:
        return {**base_params, **sampled_params}

    @staticmethod
    def safe_name(value: str) -> str:
        safe_value = re.sub(r"[^\w.-]+", "_", value).strip("_.")
        return safe_value or "model_output"

    def _fit_multi_target(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.DataFrame,
        y_val: pd.DataFrame,
        params: EffectiveModelParams,
        output_dir: Path,
    ) -> ModelBundleReference:
        model = self.create_adapter(params)
        logger.info("Fitting one multi-target model")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=10,
        )

        save_path_raw = Path(self.config.save_path)
        model_path = (output_dir / save_path_raw.name).resolve()
        logger.info("Saving model to: %s", model_path)
        model.save(model_path)

        model_info = model.log_to_mlflow("model")
        summary = model.get_training_summary()
        self._log_training_summary(summary)

        reference = ModelReference(
            output_name=None,
            uri=f"models:/{model_info.model_id}",
            local_path=model_path,
            model_info=str(model_info),
            training_summary=summary,
        )
        return ModelBundleReference(
            mode="multi_target",
            models={"__multi_target__": reference},
            feature_columns=tuple(X_train.columns),
        )

    def _fit_per_target(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.DataFrame,
        y_val: pd.DataFrame,
        params: EffectiveModelParams,
        output_dir: Path,
    ) -> ModelBundleReference:
        expected_outputs = list(y_train.columns)
        missing = set(expected_outputs) - set(params)
        extra = set(params) - set(expected_outputs)
        if missing or extra:
            raise ValueError(
                "Per-target effective parameters do not match y columns. "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        references: dict[str, ModelReference] = {}
        for target in expected_outputs:
            safe_target = self.safe_name(target)
            logger.info("Fitting independent model for output: %s", target)

            with self.tracker.nested_run(
                run_name=f"model_{safe_target}",
                description=(
                    f"Final model for output '{target}' within the current fold."
                ),
            ):
                model = self.create_adapter(params[target])
                model.fit(
                    X_train,
                    y_train[target],
                    eval_set=[
                        (X_train, y_train[target]),
                        (X_val, y_val[target]),
                    ],
                    verbose=10,
                )

                model_path = self._target_model_path(output_dir, target)
                logger.info(
                    "Saving model for output %s to: %s",
                    target,
                    model_path,
                )
                model.save(model_path)

                model_info = model.log_to_mlflow(f"model_{safe_target}")
                summary = model.get_training_summary()
                self._log_training_summary(summary, target=target)

                references[target] = ModelReference(
                    output_name=target,
                    uri=f"models:/{model_info.model_id}",
                    local_path=model_path,
                    model_info=str(model_info),
                    training_summary=summary,
                )

        return ModelBundleReference(
            mode="per_target",
            models=references,
            feature_columns=tuple(X_train.columns),
        )

    def _target_model_path(self, output_dir: Path, target: str) -> Path:
        save_path_raw = Path(self.config.save_path)
        suffix = save_path_raw.suffix or ".json"
        stem = save_path_raw.stem if save_path_raw.suffix else save_path_raw.name
        return (output_dir / f"{stem}_{self.safe_name(target)}{suffix}").resolve()

    def _log_training_summary(
        self,
        summary: dict[str, Any],
        target: str | None = None,
    ) -> None:
        prefix = "training"
        if target is not None:
            prefix = f"{prefix}/{self.safe_name(target)}"

        for name, value in summary.items():
            if value is not None and isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                self.tracker.log_metric(f"{prefix}/{name}", float(value))

    def _validate_configuration(self) -> None:
        if self.mode not in {"multi_target", "per_target"}:
            raise ValueError(
                "training.model.mode must be either 'multi_target' or 'per_target'."
            )

        params = getattr(self.config, "params", None)
        params_by_target = getattr(self.config, "params_by_target", None)

        if self.mode == "multi_target":
            if params is None:
                raise ValueError(
                    "training.model.params must be set when "
                    "training.model.mode='multi_target'."
                )
            if params_by_target is not None:
                raise ValueError(
                    "training.model.params_by_target must not be set when "
                    "training.model.mode='multi_target'."
                )
            return

        if params is not None:
            raise ValueError(
                "training.model.params must not be set when "
                "training.model.mode='per_target'. Use "
                "training.model.params_by_target instead."
            )
        if not params_by_target:
            raise ValueError(
                "training.model.params_by_target must be set when "
                "training.model.mode='per_target'."
            )

        expected = set(self.model_outputs)
        configured = set(params_by_target)
        missing = expected - configured
        extra = configured - expected
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing={sorted(missing)}")
            if extra:
                details.append(f"extra={sorted(extra)}")
            raise ValueError(
                "training.model.params_by_target keys must exactly match "
                "the model-output columns. "
                + ", ".join(details)
                + f". Expected: {sorted(expected)}"
            )
