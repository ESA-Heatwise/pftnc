from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class ModelIOContract(BaseModel):
    """Columns exchanged with one fitted model."""

    model_config = ConfigDict(frozen=True)

    input_columns: list[str]

    output_columns: list[str]


class ModelIOSchema(BaseModel):
    """Columns exchanged between a prepared dataset and model bundle."""

    model_config = ConfigDict(frozen=True)

    models: dict[str, ModelIOContract]
    physical_target_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def build_model_io_schema(
    *,
    input_columns: list[str],
    model_output_columns: list[str],
    physical_target_columns: list[str],
    model_mode: str,
) -> ModelIOSchema:
    """Build a bundle schema for multi-target or per-target model mode."""
    if model_mode == "multi_target":
        models = {
            "__multi_target__": ModelIOContract(
                input_columns=input_columns,
                output_columns=model_output_columns,
            )
        }
    elif model_mode == "per_target":
        models = {
            output: ModelIOContract(
                input_columns=input_columns,
                output_columns=[output],
            )
            for output in model_output_columns
        }
    else:
        raise ValueError(f"Unsupported model mode: {model_mode!r}")

    return ModelIOSchema(
        models=models,
        physical_target_columns=physical_target_columns,
    )


def write_model_io_schema(path: Path, schema: ModelIOSchema) -> Path:
    """Write the run-level model input/output schema to YAML."""
    expected = schema.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        yaml.safe_dump(expected, stream, sort_keys=False)
    return path


def validate_model_io_schema(path: Path, schema: ModelIOSchema) -> None:
    """Validate a fold's model I/O schema against the run-level YAML artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Model I/O schema does not exist: {path}")

    with path.open() as stream:
        existing = yaml.safe_load(stream)
    expected = schema.to_dict()
    if existing != expected:
        raise ValueError(
            "Model input/output columns differ between folds: "
            f"existing={existing}, current={expected}"
        )


def log_model_io(schema: ModelIOSchema, *, fold_name: str) -> None:
    """Log each model's exact input and output columns before fitting."""
    for model_name, contract in schema.models.items():
        logger.info(
            "Model columns before fit | fold=%s | model=%s | "
            "n_inputs=%d | inputs=%s | outputs=%s",
            fold_name,
            model_name,
            len(contract.input_columns),
            contract.input_columns,
            contract.output_columns,
        )


def build_model_selection_metadata(
    *,
    best_model: Any,
    fold_results: dict[str, Any],
    metric: str,
    direction: str,
    mode: str,
    target: str | None,
    physical_targets: list[str],
) -> dict[str, Any]:
    """Build the model-selection record."""
    if direction not in {"minimize", "maximize"}:
        raise ValueError(f"Unsupported selection direction: {direction}")
    if mode not in {"mean", "median", "worst"}:
        raise ValueError(f"Unsupported selection mode: {mode}")

    target_scores = {
        physical_target: {
            fold: float(result.validation_metrics[physical_target][metric])
            for fold, result in fold_results.items()
        }
        for physical_target in physical_targets
    }

    fold_scores: dict[str, float] = {}
    for fold, result in fold_results.items():
        scores = [
            float(result.validation_metrics[physical_target][metric])
            for physical_target in physical_targets
        ]
        if target is not None:
            fold_scores[fold] = float(result.validation_metrics[target][metric])
        elif mode == "mean":
            fold_scores[fold] = float(np.mean(scores))
        elif mode == "median":
            fold_scores[fold] = float(np.median(scores))
        else:
            fold_scores[fold] = max(scores) if direction == "minimize" else min(scores)

    def weight(score: float) -> float:
        if not np.isfinite(score):
            raise ValueError("Cannot weight a non-finite validation score")
        if direction == "minimize":
            if score < 0:
                raise ValueError("A minimized metric must be non-negative")
            return 1.0 / max(score, 1e-8)
        return max(score, 1e-8)

    global_scores = {
        fold: float(
            np.mean(
                [
                    target_scores[physical_target][fold]
                    for physical_target in physical_targets
                ]
            )
        )
        for fold in fold_results
    }

    return {
        "metric": metric,
        "direction": direction,
        "mode": mode,
        "target": target,
        "fold_scores": fold_scores,
        "target_scores": target_scores,
        "best_overall": {
            "fold": best_model.overall.fold,
            "score": best_model.overall.score,
        },
        "best_per_target": {
            physical_target: {
                "fold": best_model.per_target[physical_target].fold,
                "score": best_model.per_target[physical_target].score,
            }
            for physical_target in physical_targets
        },
        "ensemble_weights": {
            "global": {fold: weight(score) for fold, score in global_scores.items()},
            "per_target": {
                physical_target: {fold: weight(score) for fold, score in scores.items()}
                for physical_target, scores in target_scores.items()
            },
        },
    }
