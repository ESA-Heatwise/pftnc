from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow
import yaml

from pftnc.provenance import portable_config_snapshot

logger = logging.getLogger(__name__)


class MlflowTracker:
    """Small facade around the MLflow operations used by training."""

    from pftnc.config import AppConfig
    from pftnc.train.models.base import ModelAdapter

    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def configure(self, model_adapter_cls: type[ModelAdapter]) -> None:
        mlflow.set_experiment(self.experiment_name)
        model_adapter_cls.configure_mlflow_tracking()

    @contextmanager
    def training_run(
        self,
        run_name: str,
        description: str | None,
    ) -> Iterator[None]:
        with mlflow.start_run(
            run_name=run_name,
            description=description,
        ):
            yield

    @contextmanager
    def nested_run(
        self,
        run_name: str,
        description: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        with mlflow.start_run(
            run_name=run_name,
            nested=True,
            description=description,
        ):
            if tags:
                mlflow.set_tags(tags)
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        if params:
            mlflow.log_params(params)

    def log_param(self, name: str, value: Any) -> None:
        mlflow.log_param(name, value)

    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
    ) -> None:
        if step is None:
            mlflow.log_metric(name, float(value))
        else:
            mlflow.log_metric(name, float(value), step=step)

    def log_artifact(
        self,
        path: Path,
        artifact_path: str | None = None,
    ) -> None:
        mlflow.log_artifact(
            str(path),
            artifact_path=artifact_path,
        )

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        mlflow.log_dict(payload, artifact_file)

    def log_config(
        self,
        config: AppConfig,
        run_dir: Path,
    ) -> Path:
        effective_config = config.model_dump(mode="json")
        effective_config["training"]["model"]["save_path"] = Path(
            config.training.model.save_path
        ).name
        effective_config = portable_config_snapshot(
            effective_config,
            anchor=run_dir,
        )

        self.log_dict(effective_config, "config.json")

        config_path = run_dir / "config.yml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as stream:
            yaml.safe_dump(
                effective_config,
                stream,
                sort_keys=False,
            )

        self.log_artifact(config_path)
        return config_path


class ArtifactStore:
    """Write local JSON/CSV artifacts before they are logged to MLflow."""

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as stream:
            json.dump(payload, stream, indent=2, default=str)
        return path
