from pathlib import Path

import yaml
from pydantic import BaseModel

from pftnc.provenance import workspace_root
from pftnc.utils.utils import resolve_path


class SensorSimConfig(BaseModel):
    revisit_days: int
    noise_std: float | None = None
    columns: list[str]


class SimulationConfig(BaseModel):
    enabled: bool = True
    base_path: Path
    registry_path: Path
    description: str
    site_name: str
    sensors: dict[str, SensorSimConfig]
    cloud_mask: dict[str, float]
    date_column: str
    input_path: str


class ProjectConfig(BaseModel):
    seed: int = 42


class DatasetSimConfig(BaseModel):
    project: ProjectConfig
    simulation: SimulationConfig | None = None


def load_dataset_gen_config(path: str | Path) -> DatasetSimConfig:
    path = Path(path).resolve()

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    raw.pop("schema_version", None)
    workspace = workspace_root(path)

    sim = raw.get("simulation")

    if sim is not None:
        sim["base_path"] = resolve_path(sim["base_path"], workspace)
        sim["registry_path"] = resolve_path(sim["registry_path"], workspace)
        sim["input_path"] = resolve_path(sim["input_path"], workspace)

    return DatasetSimConfig(**raw)
