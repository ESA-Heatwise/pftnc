from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from pftnc.provenance import workspace_root
from pftnc.utils.utils import resolve_path

ModelMode = Literal["multi_target", "per_target"]


class ProjectConfig(BaseModel):
    seed: int


class DatasetSchema(BaseModel):
    date_column: str
    group_column: str
    sensors: dict[str, list[str]]
    targets: list[str]


class DatasetSplitConfig(BaseModel):
    final_test_year: int


class RollingFeatureConfig(BaseModel):
    windows: list[int]
    statistics: list[str]


class DatasetMetadata(BaseModel):
    sites: list[str]
    dataset_schema: DatasetSchema
    splits: DatasetSplitConfig | None = None


class DatasetInputConfig(BaseModel):
    # direct dataset mode
    path: Path | None = None

    # registry-managed mode
    base_path: Path | None = None
    registry_path: Path | None = None

    # optional:
    # if omitted -> use latest
    version: str | None = None

    @model_validator(mode="after")
    def validate_reference_mode(self):

        direct_mode = self.path is not None

        registry_mode = self.base_path is not None and self.registry_path is not None

        if direct_mode == registry_mode:
            raise ValueError(
                "DatasetInputConfig must use exactly one mode:\n"
                "- direct path mode (path)\n"
                "- registry mode (base_path + registry_path)"
            )

        return self


class DatasetOutputConfig(BaseModel):
    base_path: Path
    registry_path: Path
    description: str


class DatasetSpec(BaseModel):
    storage: DatasetOutputConfig
    metadata: DatasetMetadata


class DatasetArtifact(BaseModel):
    base_path: Path
    registry_path: Path
    version: str

    def to_input_config(self):

        return DatasetInputConfig(
            base_path=self.base_path,
            registry_path=self.registry_path,
            version=self.version,
        )


class FeatureOperationsConfig(BaseModel):
    rolling: RollingFeatureConfig | None = None


class FeatureEngineeringConfig(BaseModel):
    # i/o config
    input_dataset: DatasetInputConfig
    output_dataset: DatasetSpec

    # feature engineering
    features: FeatureOperationsConfig


class FeatureSelectionConfig(BaseModel):
    """Model-input selection applied when a feature dataset is trained."""

    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    type: Literal["xgboost"]
    mode: ModelMode = "multi_target"
    save_path: Path

    # Valid only for mode="multi_target".
    params: dict[str, Any] | None = None

    # Valid only for mode="per_target".
    params_by_target: (
        dict[
            str,
            dict[str, Any],
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def validate_params_for_mode(self) -> "ModelConfig":
        if self.mode == "multi_target":
            if self.params_by_target is not None:
                raise ValueError(
                    "model.params_by_target is only valid when model.mode='per_target'."
                )

            # No params means the underlying model library uses its defaults.
            if self.params is None:
                self.params = {}

            return self

        if self.params is not None:
            raise ValueError(
                "model.params is only valid when "
                "model.mode='multi_target'. Use "
                "model.params_by_target for per_target mode."
            )

        if not self.params_by_target:
            raise ValueError(
                "model.params_by_target must be provided when model.mode='per_target'."
            )

        return self


class BestModelSelectionConfig(BaseModel):
    metric: str
    direction: Literal["minimize", "maximize"]
    target: str | None = None
    mode: Literal["mean", "median", "worst"] = "mean"


class TargetTransformConfig(BaseModel):
    """Configuration for a transformation applied to one target."""

    type: Literal["absolute", "log_delta"] = "absolute"
    reference_feature: str | None = None

    @model_validator(mode="after")
    def validate_transform(self) -> "TargetTransformConfig":
        if self.type == "log_delta" and not self.reference_feature:
            raise ValueError("log_delta requires reference_feature.")

        if self.type == "absolute" and self.reference_feature is not None:
            raise ValueError("absolute does not use reference_feature.")

        return self


class LogFractionsConfig(BaseModel):
    """Configuration for the joint log-fractions transformation."""

    denominator: str
    epsilon: float = Field(default=1e-8, gt=0)


class TargetTransformationsConfig(BaseModel):
    """Target transformations applied during training."""

    per_target: dict[str, TargetTransformConfig] = Field(default_factory=dict)
    log_fractions: LogFractionsConfig | None = None

    @model_validator(mode="after")
    def validate_compatibility(
        self,
    ) -> "TargetTransformationsConfig":
        if self.log_fractions is None:
            return self

        non_absolute = [
            target
            for target, config in self.per_target.items()
            if config.type != "absolute"
        ]

        if non_absolute:
            raise ValueError(
                "log_fractions cannot be combined with non-absolute "
                "per-target transformations. Conflicting targets: "
                f"{sorted(non_absolute)}"
            )

        return self


class TrainingConfig(BaseModel):
    experiment_name: str
    description: str
    output_dir: Path
    input_dataset: DatasetInputConfig

    feature_selection: FeatureSelectionConfig = Field(
        default_factory=FeatureSelectionConfig
    )

    model: ModelConfig

    metrics: list[str]

    best_model_selection: BestModelSelectionConfig

    target_transformations: TargetTransformationsConfig = Field(
        default_factory=TargetTransformationsConfig
    )

    @model_validator(mode="after")
    def validate_model_and_target_transformations(self) -> "TrainingConfig":
        if (
            self.model.mode == "per_target"
            and self.target_transformations.log_fractions is not None
        ):
            raise ValueError(
                "Joint target transformations cannot be used with "
                "model.mode='per_target'. Use model.mode='multi_target'."
            )
        return self


class TuningConfig(BaseModel):
    enabled: bool = False
    n_trials: int = 20
    direction: Literal["minimize", "maximize"] = "minimize"
    objective_metric: str | None = None
    search_space: dict[str, Any] | None = None
    objective_aggregation: Literal["mean", "weighted", "worst"] = "mean"
    target_weights: dict[str, float] | None = None

    @model_validator(mode="after")
    def validate_when_enabled(self):
        if not self.enabled:
            return self

        if self.objective_metric is None:
            raise ValueError("objective_metric is required when tuning.enabled=True")

        if self.search_space is None:
            raise ValueError("search_space is required when tuning.enabled=True")

        if self.objective_aggregation == "weighted" and self.target_weights is None:
            raise ValueError(
                "target_weights is required when objective_aggregation='weighted'"
            )

        return self


class AppConfig(BaseModel):
    project: ProjectConfig
    feature_engineering: FeatureEngineeringConfig
    training: TrainingConfig
    tuning: TuningConfig | None = None


def load_config(path: str | Path) -> AppConfig:

    path = Path(path).resolve()

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    raw.pop("schema_version", None)
    workspace = workspace_root(path)

    fe_input = raw["feature_engineering"]["input_dataset"]

    if fe_input.get("path") is not None:
        fe_input["path"] = resolve_path(
            fe_input["path"],
            workspace,
        )

    if fe_input.get("base_path") is not None:
        fe_input["base_path"] = resolve_path(
            fe_input["base_path"],
            workspace,
        )

    if fe_input.get("registry_path") is not None:
        fe_input["registry_path"] = resolve_path(
            fe_input["registry_path"],
            workspace,
        )

    fe_output = raw["feature_engineering"]["output_dataset"]["storage"]

    fe_output["base_path"] = resolve_path(
        fe_output["base_path"],
        workspace,
    )

    fe_output["registry_path"] = resolve_path(
        fe_output["registry_path"],
        workspace,
    )

    training = raw["training"]

    training["output_dir"] = resolve_path(
        training["output_dir"],
        workspace,
    )

    training_input = training["input_dataset"]

    if training_input.get("path") is not None:
        training_input["path"] = resolve_path(
            training_input["path"],
            workspace,
        )

    if training_input.get("base_path") is not None:
        training_input["base_path"] = resolve_path(
            training_input["base_path"],
            workspace,
        )

    if training_input.get("registry_path") is not None:
        training_input["registry_path"] = resolve_path(
            training_input["registry_path"],
            workspace,
        )

    training["model"]["save_path"] = resolve_path(
        training["model"]["save_path"],
        workspace,
    )

    return AppConfig(**raw)
