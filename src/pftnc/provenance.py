"""Portable workspace paths and dataset lineage."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from pftnc.utils.utils import (
    get_registry_key_and_paths,
    read_registry,
    read_yaml_mapping,
    resolve_dataset_file,
)

SCHEMA_VERSION = 1
PROVENANCE_NAME = "provenance.yml"


def repository_root(path: Path) -> Path:
    """Return the Git repository containing *path* (or its nearest parent)."""
    candidate = path if path.is_dir() else path.parent
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            return parent
    return candidate


def workspace_root(path: Path) -> Path:
    """Return the common parent of the sibling repositories for *path*."""
    return repository_root(path).parent


def _safe_workspace_parts(value: str | Path) -> tuple[str, ...]:
    """Validate a persisted workspace-relative path."""
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(
            f"Expected a non-empty workspace-relative path without '..', got {value!r}"
        )
    return path.parts


def portable_path(path: str | Path, *, root: Path) -> str:
    """Serialize an absolute path relative to the workspace *root*."""
    value = Path(path).resolve()
    try:
        return value.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Path {value} is outside workspace root {root}; all inputs and "
            "artifacts must live in sibling repositories under one workspace."
        ) from exc


def portable_config(value: Any, *, root: Path) -> Any:
    """Convert absolute path values in a config snapshot to workspace paths."""
    if isinstance(value, dict):
        return {key: portable_config(item, root=root) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_config(item, root=root) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return portable_path(value, root=root)
    return value


def resolve_workspace_path(value: str | Path, *, anchor: Path) -> Path:
    """Resolve one persisted workspace-relative path from a repository file."""
    return (workspace_root(anchor) / Path(*_safe_workspace_parts(value))).resolve()


def portable_config_snapshot(config: dict[str, Any], *, anchor: Path) -> dict[str, Any]:
    """Create a versioned, workspace-relative config snapshot."""
    snapshot = portable_config(config, root=workspace_root(anchor))
    return {
        "schema_version": SCHEMA_VERSION,
        **snapshot,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(base_path: Path, registry_path: Path, version: str) -> Path:
    registry = read_registry(registry_path)
    key, _ = get_registry_key_and_paths(registry_path, base_path)
    entry = registry.get("datasets", {}).get(key, {}).get(version)
    if entry is None:
        raise ValueError(f"Dataset version {version!r} is not registered for {key!r}")
    configured = Path(entry.get("path", ""))
    repository = repository_root(registry_path)
    artifact = configured if configured.is_absolute() else repository / configured
    if not artifact.exists():
        raise FileNotFoundError(
            f"Registered dataset artifact does not exist: {artifact}"
        )
    return artifact.resolve()


def _raw_lineage(path: Path, *, workspace: Path) -> dict[str, Any]:
    path = resolve_dataset_file(path)
    return {
        "stage": "raw_input",
        "artifact": portable_path(path, root=workspace),
        "sha256": file_sha256(path),
    }


def load_artifact_lineage(artifact_path: Path) -> dict[str, Any] | None:
    """Load explicit lineage, returning ``None`` when it is unavailable.

    PFTNC never infers lineage from an older dataset's config or writes a new
    provenance file into that dataset. A new downstream artifact records the
    unavailable boundary instead.
    """
    artifact_path = artifact_path.resolve()
    provenance_path = artifact_path / PROVENANCE_NAME
    if provenance_path.exists():
        payload = read_yaml_mapping(provenance_path)
        lineage = payload.get("lineage")
        if not isinstance(lineage, dict) or not lineage.get("stage"):
            raise ValueError(f"Invalid provenance payload: {provenance_path}")
        return lineage
    return None


def _unknown_dataset_lineage(artifact_path: Path, *, workspace: Path) -> dict[str, Any]:
    """Represent an input dataset whose historical lineage was not recorded."""
    return {
        "stage": "unknown_input_dataset",
        "artifact": portable_path(artifact_path, root=workspace),
        "version": artifact_path.name,
        "provenance_available": False,
    }


def _lineage_for_input(
    input_config: dict[str, Any],
    *,
    workspace: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    if input_config.get("path") is not None:
        raw_path = input_config["path"]
        path = resolve_workspace_path(raw_path, anchor=artifact_path)
        return _raw_lineage(path, workspace=workspace)

    base_path = input_config.get("base_path")
    registry_path = input_config.get("registry_path")
    version = input_config.get("version")
    if base_path is None or registry_path is None or version is None:
        raise ValueError(
            "Dataset provenance requires path or base_path, registry_path, version"
        )
    base = resolve_workspace_path(base_path, anchor=artifact_path)
    registry = resolve_workspace_path(registry_path, anchor=artifact_path)
    input_artifact = _artifact_path(base, registry, str(version))
    return load_artifact_lineage(input_artifact) or _unknown_dataset_lineage(
        input_artifact, workspace=workspace
    )


def build_provenance(
    *,
    stage: str,
    artifact_path: Path,
    version: str,
    input_config: dict[str, Any],
) -> dict[str, Any]:
    """Build lineage for a new artifact, reusing upstream provenance when present."""
    workspace = workspace_root(artifact_path)
    lineage = {
        "stage": stage,
        "artifact": portable_path(artifact_path, root=workspace),
        "version": version,
        "inputs": [
            _lineage_for_input(
                input_config,
                workspace=workspace,
                artifact_path=artifact_path,
            )
        ],
    }
    return {"schema_version": SCHEMA_VERSION, "lineage": lineage}


def write_provenance(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
