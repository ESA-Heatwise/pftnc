from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

from pftnc.provenance import workspace_root


class InferenceModelConfig(BaseModel):
    bundle_path: Path
    selection: Literal[
        "best_overall",
        "best_per_target",
        "ensemble_simple",
        "ensemble_weighted",
        "ensemble_weighted_per_target",
    ] = "best_overall"


class InferenceInputConfig(BaseModel):
    path: Path


class InferenceConfig(BaseModel):
    model: InferenceModelConfig
    input_dataset: InferenceInputConfig
    save_model_inputs: bool = True
    save_outputs: bool = True
    save_index: bool = False
    mode: Literal["latest", "from_date"] = "latest"
    start_date: date | None = None
    output_path: Path | None = None

    @model_validator(mode="after")
    def validate_configuration(self) -> "InferenceConfig":
        if self.mode == "from_date" and not self.start_date:
            raise ValueError("start_date is required when mode='from_date'")
        if self.mode == "latest" and self.start_date is not None:
            raise ValueError("start_date is only valid when mode='from_date'")
        if self.save_outputs and self.output_path is None:
            raise ValueError("output_path is required when save_outputs=true")
        if not self.save_outputs and self.output_path is not None:
            raise ValueError("output_path must be omitted when save_outputs=false")
        return self


def load_inference_config(path: str | Path) -> InferenceConfig:
    """Load inference configuration and resolve paths relative to its file."""
    path = Path(path).resolve()
    with path.open("r") as stream:
        raw = yaml.safe_load(stream) or {}

    raw.pop("schema_version", None)
    workspace = workspace_root(path)
    model = raw["model"]
    input_dataset = raw["input_dataset"]
    for section, key in ((model, "bundle_path"), (input_dataset, "path")):
        value = Path(section[key])
        section[key] = str(
            value if value.is_absolute() else (workspace / value).resolve()
        )

    output_value = raw.get("output_path")
    if output_value not in (None, ""):
        output = Path(str(output_value))
        raw["output_path"] = str(
            output if output.is_absolute() else (workspace / output).resolve()
        )
    else:
        raw["output_path"] = None
    return InferenceConfig(**raw)
