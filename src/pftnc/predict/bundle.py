from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from pftnc.provenance import (
    SCHEMA_VERSION,
    load_artifact_lineage,
    repository_root,
    resolve_workspace_path,
)
from pftnc.utils.utils import read_yaml_mapping

MANIFEST_NAME = "manifest.yml"


def _run_relative_path(value: str) -> tuple[str, ...]:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Expected a safe run-relative path, got {value!r}")
    return path.parts


def _resolve_model_path(value: str, *, training_run_path: Path) -> Path:
    path = training_run_path.joinpath(*_run_relative_path(value)).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact {value!r} was not found in training run {training_run_path}"
        )
    return path


def _require_schema_version(payload: dict[str, Any], *, path: Path) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} is not a schema-version {SCHEMA_VERSION} PFTNC artifact. "
            "Re-run training with the current PFTNC package before packaging."
        )


def package_model_bundle(training_run_path: Path, output_path: Path) -> Path:
    """Copy runtime artifacts from a portable training run into a model bundle."""
    training_run_path = training_run_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    required = {
        "summary": training_run_path / "run_summary.json",
        "config": training_run_path / "config.yml",
        "schema": training_run_path / "model_io_schema.yml",
        "selection": training_run_path / "model_selection.yml",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Required {name} artifact not found: {path}")

    summary = json.loads(required["summary"].read_text())
    _require_schema_version(summary, path=required["summary"])
    config = read_yaml_mapping(required["config"])
    _require_schema_version(config, path=required["config"])
    schema = read_yaml_mapping(required["schema"])
    selection = read_yaml_mapping(required["selection"])
    dataset_path = resolve_workspace_path(
        summary["dataset_path"], anchor=training_run_path
    )
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Training dataset is not available at {dataset_path}. Ensure the data "
            "and experiments repositories are sibling Git repositories."
        )
    dataset_config = read_yaml_mapping(dataset_path / "config.yml")
    dataset_metadata = read_yaml_mapping(dataset_path / "metadata.yml")
    dataset_lineage = load_artifact_lineage(dataset_path)

    if output_path.exists():
        if not output_path.is_dir() or any(output_path.iterdir()):
            raise FileExistsError(
                f"Bundle output must be a new empty directory: {output_path}"
            )
    output_path.mkdir(parents=True, exist_ok=True)

    folds: dict[str, Any] = {}
    training_metrics: dict[str, Any] = {}
    mlflow_uris: dict[str, Any] = {}
    for fold_name, fold in summary["folds"].items():
        raw_paths = fold["model_path"]
        paths = (
            {"__multi_target__": raw_paths} if isinstance(raw_paths, str) else raw_paths
        )
        packaged: dict[str, str] = {}
        for model_name, raw_path in paths.items():
            source = _resolve_model_path(raw_path, training_run_path=training_run_path)
            destination = output_path / "models" / fold_name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            packaged[model_name] = destination.relative_to(output_path).as_posix()
        folds[fold_name] = {"models": packaged}
        training_metrics[fold_name] = {
            "train": fold.get("train_metrics", {}),
            "validation": fold.get("val_metrics", {}),
            "test": fold.get("test_metrics", {}),
        }
        mlflow_uris[fold_name] = fold.get("model_uri")

    feature_engineering = dataset_config.get("features")
    if feature_engineering is None:
        feature_engineering = dataset_config.get("feature_engineering", {}).get(
            "features", {}
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_type": config["training"]["model"]["type"],
        "model_mode": summary["model_mode"],
        "dataset_version": summary["dataset_version"],
        "model_io": schema,
        "dataset_metadata": dataset_metadata,
        "feature_engineering": feature_engineering,
        "feature_selection": config["training"].get("feature_selection", {}),
        "target_transformations": config["training"].get("target_transformations", {}),
        "selection": selection,
        "folds": folds,
    }
    with (output_path / MANIFEST_NAME).open("w") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)

    with (output_path / "training_metrics.yml").open("w") as stream:
        yaml.safe_dump(
            {"folds": training_metrics, "final_test": summary.get("final_test", {})},
            stream,
            sort_keys=False,
        )
    with (output_path / "provenance.yml").open("w") as stream:
        yaml.safe_dump(
            {
                "schema_version": SCHEMA_VERSION,
                "created_from": repository_root(training_run_path).name,
                "source_experiment": summary.get("trainer_run_id"),
                "dataset_version": summary.get("dataset_version"),
                "dataset_lineage": dataset_lineage,
                "dataset_provenance_available": dataset_lineage is not None,
                "mlflow_model_uris": mlflow_uris,
            },
            stream,
            sort_keys=False,
        )
    return output_path


def load_manifest(bundle_path: Path) -> dict[str, Any]:
    bundle_path = bundle_path.expanduser().resolve()
    manifest_path = bundle_path / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model bundle manifest not found: {manifest_path}")
    manifest = read_yaml_mapping(manifest_path)
    if not manifest.get("folds"):
        raise ValueError("Model bundle contains no fold models")
    return manifest
