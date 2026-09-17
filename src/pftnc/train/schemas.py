from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from pftnc.provenance import SCHEMA_VERSION, portable_path, workspace_root

MetricResults: TypeAlias = dict[str, dict[str, float]]
EffectiveModelParams: TypeAlias = dict[str, Any] | dict[str, dict[str, Any]]
ModelUri: TypeAlias = str | dict[str, str]
ModelArtifactPath: TypeAlias = str | dict[str, str]
ModelInfoValue: TypeAlias = str | dict[str, str]
TrainingSummary: TypeAlias = dict[str, Any] | dict[str, dict[str, Any]]


class ResultModel(BaseModel):
    """Base class for JSON-serializable training results."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )


class ModelReference(ResultModel):
    """Reference to one trained model and its persisted artifacts."""

    output_name: str | None = None
    uri: str
    local_path: Path
    model_info: str
    training_summary: dict[str, Any] = Field(default_factory=dict)


class ModelBundleReference(ResultModel):
    """One multi-target model or a set of independent output models."""

    mode: Literal["multi_target", "per_target"]
    models: dict[str, ModelReference]
    feature_columns: tuple[str, ...]

    @property
    def is_multi_target(self) -> bool:
        return self.mode == "multi_target"

    @property
    def model_uri(self) -> ModelUri:
        if self.is_multi_target:
            return self.models["__multi_target__"].uri
        return {name: model.uri for name, model in self.models.items()}

    def model_path_relative_to(self, run_dir: Path) -> ModelArtifactPath:
        """Return model locations relative to their movable training run."""
        if self.is_multi_target:
            return portable_path(
                self.models["__multi_target__"].local_path,
                root=run_dir,
            )
        return {
            name: portable_path(model.local_path, root=run_dir)
            for name, model in self.models.items()
        }

    @property
    def model_info(self) -> ModelInfoValue:
        if self.is_multi_target:
            return self.models["__multi_target__"].model_info
        return {name: model.model_info for name, model in self.models.items()}

    @property
    def training_summary(self) -> TrainingSummary:
        if self.is_multi_target:
            return self.models["__multi_target__"].training_summary
        return {name: model.training_summary for name, model in self.models.items()}


class FoldResult(ResultModel):
    """Complete output produced by one fold training run."""

    fold_name: str
    dataset_version: str
    dataset_path: Path
    fold_dataset_path: Path
    model_outputs: list[str]
    model_bundle: ModelBundleReference
    params: EffectiveModelParams
    train_metrics: MetricResults
    validation_metrics: MetricResults
    test_metrics: MetricResults
    fit_kwargs: dict[str, Any] = Field(default_factory=dict)
    target_transforms: dict[str, Any] | None = None

    def to_summary_dict(self, *, run_dir: Path) -> dict[str, Any]:
        """Return the portable summary for one fold."""

        workspace = workspace_root(run_dir)
        dataset_path = portable_path(self.dataset_path, root=workspace)
        fold_dataset_path = portable_path(self.fold_dataset_path, root=workspace)
        model_path = self.model_bundle.model_path_relative_to(run_dir)

        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "fold_name": self.fold_name,
            "dataset_version": self.dataset_version,
            "dataset_path": dataset_path,
            "fold_dataset_path": fold_dataset_path,
            "model_mode": self.model_bundle.mode,
            "model_outputs": self.model_outputs,
            "feature_columns": list(self.model_bundle.feature_columns),
            "model_uri": self.model_bundle.model_uri,
            "model_path": model_path,
            "model_info": self.model_bundle.model_info,
            "params": self.params,
            "train_metrics": self.train_metrics,
            "training_summary": self.model_bundle.training_summary,
            "val_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "fit_kwargs": self.fit_kwargs,
        }
        if self.target_transforms is not None:
            summary["target_transforms"] = self.target_transforms
        return summary


class SelectedModel(ResultModel):
    """One selected fold model or model bundle."""

    model_uri: ModelUri
    fold: str
    score: float
    metric: str | None = None
    target: str | None = None
    mode: str | None = None


class BestModelResult(ResultModel):
    """Overall and per-target validation-based selections."""

    overall: SelectedModel
    per_target: dict[str, SelectedModel]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.model_dump(mode="json"),
            "per_target": {
                target: {
                    "model_uri": selection.model_uri,
                    "fold": selection.fold,
                    "score": selection.score,
                }
                for target, selection in self.per_target.items()
            },
        }


class TrainingRunResult(ResultModel):
    """Top-level result of one complete training invocation."""

    trainer_run_id: str
    description: str | None = None
    model_mode: Literal["multi_target", "per_target"]
    dataset_version: str
    dataset_path: Path
    folds: dict[str, FoldResult]
    best_model: BestModelResult
    final_test: dict[str, Any]

    def to_summary_dict(self, *, run_dir: Path) -> dict[str, Any]:
        workspace = workspace_root(run_dir)
        return {
            "schema_version": SCHEMA_VERSION,
            "trainer_run_id": self.trainer_run_id,
            "description": self.description,
            "model_mode": self.model_mode,
            "dataset_version": self.dataset_version,
            "dataset_path": portable_path(self.dataset_path, root=workspace),
            "folds": {
                name: result.to_summary_dict(run_dir=run_dir)
                for name, result in self.folds.items()
            },
            "best_model": self.best_model.to_summary_dict(),
            "final_test": self.final_test,
        }
