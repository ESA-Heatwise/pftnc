from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pftnc.config import DatasetInputConfig
from pftnc.provenance import repository_root
from pftnc.utils.utils import (
    get_registry_key_and_paths,
    read_registry,
    resolve_dataset_file,
    validate_registry,
)

logger = logging.getLogger(__name__)


def read_input_dataset(storage: DatasetInputConfig) -> pd.DataFrame:
    """Read a preprocessing or inference input dataset from path or registry."""

    # Direct path mode
    if storage.path is not None:
        dataset_file = resolve_dataset_file(Path(storage.path))
    # Registry mode
    else:
        if storage.registry_path is None or storage.base_path is None:
            raise ValueError("Registry mode requires registry_path and base_path")
        if storage.version is None:
            raise ValueError("Registry mode requires a dataset version")

        registry_path = Path(storage.registry_path)
        base_path = Path(storage.base_path)
        validate_registry(registry_path, base_path)
        base_key, _ = get_registry_key_and_paths(registry_path, base_path)
        versions = read_registry(registry_path).get("datasets", {}).get(base_key)
        if versions is None or storage.version not in versions:
            available = [] if versions is None else list(versions)
            raise ValueError(
                f"Dataset version {storage.version!r} not found; "
                f"available versions: {available}"
            )
        registered_path = Path(versions[storage.version]["path"])
        if not registered_path.is_absolute():
            registered_path = repository_root(registry_path) / registered_path
        dataset_file = resolve_dataset_file(registered_path)

    frame = (
        pd.read_parquet(dataset_file)
        if dataset_file.suffix == ".parquet"
        else pd.read_csv(dataset_file)
    )
    logger.info(
        "Loaded dataset %s | rows=%d columns=%d",
        dataset_file,
        len(frame),
        len(frame.columns),
    )
    return frame
