from __future__ import annotations

import logging
import shutil
from pathlib import Path

import yaml

from pftnc.config import DatasetArtifact, FeatureEngineeringConfig
from pftnc.provenance import (
    build_provenance,
    portable_config_snapshot,
    write_provenance,
)
from pftnc.utils.utils import (
    generate_dataset_version,
    register_dataset_version,
    save_environment_versions,
    save_feature_schema,
    validate_registry,
)

from .dataset import prepare_feature_engineered_dataset
from .features import engineer_features
from .folds import write_year_folds
from .io import read_input_dataset

logger = logging.getLogger(__name__)


def preprocess_dataset(feature_cfg: FeatureEngineeringConfig) -> DatasetArtifact:
    """Create and register one final, model-ready dataset artifact."""
    output = feature_cfg.output_dataset
    storage = output.storage
    metadata = output.metadata
    schema = metadata.dataset_schema

    if storage.base_path is None or storage.registry_path is None:
        raise ValueError("Output base_path and registry_path must be configured")
    if metadata.splits is None:
        raise ValueError("Dataset splits config is required")

    validate_registry(storage.registry_path, storage.base_path)
    site = "_".join(metadata.sites)
    version = generate_dataset_version(feature_cfg, site)
    dataset_path = Path(storage.base_path) / version

    try:
        raw = read_input_dataset(feature_cfg.input_dataset)
        engineered = engineer_features(
            raw,
            feature_config=feature_cfg.features,
            dataset_schema=schema,
        )
        prepared = prepare_feature_engineered_dataset(
            engineered,
            dataset_schema=schema,
            gap_only=True,
        )

        write_year_folds(
            prepared,
            output_path=dataset_path,
            dataset_schema=schema,
            final_test_year=metadata.splits.final_test_year,
        )

        # The schema now describes the exact model matrix written to every fold.
        save_feature_schema(prepared.X, dataset_path)
        config_snapshot = portable_config_snapshot(
            feature_cfg.model_dump(mode="json"),
            anchor=storage.registry_path,
        )
        (dataset_path / "config.yml").write_text(
            yaml.safe_dump(config_snapshot, sort_keys=False)
        )
        (dataset_path / "metadata.yml").write_text(
            yaml.safe_dump(metadata.model_dump(mode="json"), sort_keys=False)
        )
        write_provenance(
            dataset_path / "provenance.yml",
            build_provenance(
                stage="feature_engineering",
                artifact_path=dataset_path,
                version=version,
                input_config=config_snapshot["input_dataset"],
            ),
        )
        save_environment_versions(dataset_path / "versions.txt")
    except Exception:
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        raise

    register_dataset_version(
        storage.registry_path,
        storage.base_path,
        storage.description,
        version,
    )
    logger.info("Created model-ready dataset %s at %s", version, dataset_path)
    return DatasetArtifact(
        base_path=storage.base_path,
        registry_path=storage.registry_path,
        version=version,
    )
