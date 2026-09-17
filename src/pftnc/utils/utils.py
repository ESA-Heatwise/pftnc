import importlib.metadata
import json
import os
import platform
import uuid
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from pprint import pprint
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import yaml


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML file that must contain a mapping."""
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def resolve_dataset_file(path: Path) -> Path:
    """Resolve a dataset file or a directory containing one CSV/Parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if path.is_file():
        if path.suffix not in {".csv", ".parquet"}:
            raise ValueError(f"Unsupported dataset type: {path}")
        return path
    files = list(path.glob("*.parquet")) + list(path.glob("*.csv"))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one CSV or Parquet file in {path}; found {len(files)}"
        )
    return files[0]


def get_or_create_experiment(experiment_name: str):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Taken from mlflow.org

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """
    import mlflow

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


def show_datasets_info(run_id: str):
    """
    Display readable information about all datasets used in an MLflow run
    """
    import mlflow

    run = mlflow.get_run(run_id)
    datasets = run.inputs.dataset_inputs

    print(f"Number of datasets used: {len(datasets)}\n")
    if "data_source" in run.data.params:
        print(f"Data Source: {run.data.params['data_source']}")
    for dataset_input in datasets:
        dataset = dataset_input.dataset
        tags = dataset_input.tags

        info = {
            "Role": tags[0].value,
            "Digest": dataset.digest,
            "Name": dataset.name,
            "Profile": json.loads(dataset.profile),
            "Schema": json.loads(dataset.schema),
            "Source": json.loads(dataset.source),
        }

        print(f"Dataset (Role: {info['Role']}):")
        pprint(info, indent=2, width=100)
        print("\n" + "=" * 80 + "\n")


def read_registry(regsitry_path: Path) -> dict[str, Any]:
    """
    Load dataset registry YAML file.

    Args:
        registry_path: Path to registry YAML file.

    Returns:
        Registry dictionary with dataset metadata.
    """
    if regsitry_path.exists():
        data = yaml.safe_load(regsitry_path.read_text())

        if data is None:
            data = {"datasets": {}}

        return data

    else:
        return {"datasets": {}}


def get_registry_key_and_paths(
    registry_path: Path, base_path: Path
) -> tuple[str, Path]:
    """
    Helper to extract a uniform registry key and an absolute disk dataset path.
    """
    abs_base_path = base_path.resolve()

    project_root = None
    root_indicators = {"pixi.toml", "pyproject.toml", "setup.py", ".git", ".pixi"}

    for parent in registry_path.resolve().parents:
        if any((parent / indicator).exists() for indicator in root_indicators):
            project_root = parent
            break

    if project_root is None:
        project_root = registry_path.resolve().parent

    base_key = os.path.relpath(abs_base_path, project_root).replace("\\", "/")

    if base_key == ".":
        base_key = abs_base_path.name

    return base_key, abs_base_path


def validate_registry(
    registry_path: Path,
    base_path: Path,
) -> None:
    """
    Validate that registry versions match dataset folders on disk.

    Args:
        registry_path: Dataset registry path.
        base_path: Dataset base directory.

    Raises:
        RuntimeError: If disk and registry versions differ.
    """
    registry = read_registry(registry_path)
    base_key, abs_base_path = get_registry_key_and_paths(registry_path, base_path)

    disk_versions = (
        {p.name for p in abs_base_path.iterdir() if p.is_dir()}
        if abs_base_path.exists()
        else set()
    )

    yaml_versions = set(registry.get("datasets", {}).get(base_key, {}).keys())

    if disk_versions != yaml_versions:
        raise RuntimeError(
            "Dataset registry mismatch\n"
            f"Dataset: {base_key}\n"
            f"Disk versions: {sorted(disk_versions)}\n"
            f"Registry versions: {sorted(yaml_versions)}"
        )


def generate_dataset_version(
    cfg: Any,
    site: str,
) -> str:
    """
    Generate a unique dataset version identifier.

    Args:
        cfg: Dataset configuration object.
        site: Site or dataset identifier.

    Returns:
        Unique dataset version string.
    """
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ%z")

    # stable hash of config - if multiple users created the dataset using same configurations
    cfg_dict = cfg.model_dump(mode="json")
    cfg_json = json.dumps(cfg_dict, sort_keys=True)
    cfg_hash = sha1(cfg_json.encode()).hexdigest()[:8]

    # randomness prevents collisions
    rand = uuid.uuid4().hex[:6]

    return f"{site}_{timestamp}_{cfg_hash}_{rand}"


def register_dataset_version(
    registry_path: Path,
    base_path: Path,
    description: str,
    version: str,
) -> None:
    """
    Register a dataset version in the YAML registry.

    Args:
        registry_path: Path to registry YAML file.
        base_path: Base dataset directory.
        description: Human-readable dataset description.
        version: Dataset version identifier.
    """
    registry = read_registry(registry_path)
    base_key, _ = get_registry_key_and_paths(registry_path, base_path)

    if "datasets" not in registry:
        registry["datasets"] = {}

    if base_key not in registry["datasets"]:
        registry["datasets"][base_key] = {}

    registry["datasets"][base_key][version] = {
        "path": "/".join([base_key, version]),
        "description": description,
        "created": datetime.now(UTC).isoformat(),
    }

    registry_path.parent.mkdir(parents=True, exist_ok=True)

    with registry_path.open("w") as f:
        yaml.safe_dump(registry, f, sort_keys=False)


def save_feature_schema(
    df: "pd.DataFrame",
    path: Path,
) -> None:
    """
    Save dataframe feature schema metadata to YAML.

    Args:
        df: Feature dataframe.
        path: Output directory path.
    """
    schema = {"columns": list(df.columns), "n_features": len(df.columns)}

    with open(path / "feature_schema.yml", "w") as f:
        yaml.safe_dump(schema, f)


def check_target_quality(
    df: pd.DataFrame,
    target_cols: list[str],
) -> None:
    """
    Raise if any target column contains NaN or negative values.

    Call this on the prepared dataframe before build_dataset.
    Negative concentrations are physically invalid and cause NaN in the
    log-fractions transform; NaN targets have no valid label.
    """
    issues = {}
    nan_counts = df[target_cols].isna().sum()
    neg_counts = (df[target_cols] < 0).sum()
    if nan_counts.any():
        issues["nan"] = nan_counts[nan_counts > 0].to_dict()
    if neg_counts.any():
        issues["negative"] = neg_counts[neg_counts > 0].to_dict()
    if issues:
        raise ValueError(
            "Target quality check failed:\n"
            + "\n".join(f"  {k}: {v}" for k, v in issues.items())
            + "\nFix the data before calling build_dataset."
        )
    print(f"Target quality OK — {len(df)} rows, no NaN or negative values in targets.")


def save_environment_versions(
    output_path: str | Path = "versions.txt",
) -> None:
    """Write the Python runtime and installed-distribution versions to a file."""
    try:
        pftnc_version = importlib.metadata.version("pftnc")
    except importlib.metadata.PackageNotFoundError:
        pftnc_version = "unknown"

    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"] or distribution.name
        if name:
            packages[name] = distribution.version
    packages.pop("pftnc", None)
    packages.pop("PFTNC", None)
    lines = [
        f"pftnc=={pftnc_version}",
        f"python=={platform.python_version()}",
        f"platform=={platform.platform()}",
        "",
        "installed_packages:",
        *(
            f"{name}=={version}"
            for name, version in sorted(
                packages.items(), key=lambda item: item[0].lower()
            )
        ),
        "",
    ]
    Path(output_path).write_text("\n".join(lines))


def is_url(value: str) -> bool:
    """
    Return True if value is an HTTP/HTTPS URL.
    """
    parsed = urlparse(str(value))

    return parsed.scheme in {"http", "https"}


def resolve_path(
    path_value: str | Path,
    config_dir: Path,
) -> str:
    """
    Resolve relative filesystem paths against the config directory.

    URLs are returned unchanged.

    Args:
        path_value: Relative path, absolute path, or URL.
        config_dir: Directory containing the config file.

    Returns:
        Resolved absolute path or original URL.
    """
    if is_url(str(path_value)):
        return str(path_value)

    path = Path(path_value)

    if not path.is_absolute():
        path = config_dir / path

    return str(path.resolve())
