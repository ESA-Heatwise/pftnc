import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pftnc.dataset_simulation_config import DatasetSimConfig
from pftnc.provenance import (
    build_provenance,
    portable_config_snapshot,
    write_provenance,
)
from pftnc.simulation.simulations import apply_simulations, build_simulations
from pftnc.utils.utils import (
    generate_dataset_version,
    register_dataset_version,
    save_environment_versions,
    save_feature_schema,
    validate_registry,
)

logger = logging.getLogger(__name__)


def simulate_dataset(cfg: DatasetSimConfig, raw_df: pd.DataFrame | None = None) -> Path:
    if cfg.simulation is None:
        raise ValueError("Simulation config is required")

    logger.info("Simulating dataset")

    if raw_df is None:
        input_path = cfg.simulation.input_path

        logger.info(f"Loading input data from {input_path}")

        raw_df = pd.read_csv(input_path, index_col=0)

    raw_df["time"] = pd.to_datetime(raw_df["time"])
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    rng = np.random.default_rng(cfg.project.seed)

    simulations = build_simulations(cfg, rng)

    logger.info("Applying simulations")
    simulated_df = apply_simulations(raw_df, simulations)

    validate_registry(cfg.simulation.registry_path, cfg.simulation.base_path)
    version = generate_dataset_version(cfg, cfg.simulation.site_name)

    base = cfg.simulation.base_path
    path = base / version
    path.mkdir(parents=True)

    register_dataset_version(
        cfg.simulation.registry_path, base, cfg.simulation.description, version
    )

    dataset_path = cfg.simulation.base_path / version

    simulated_df.to_parquet(dataset_path / "simulated.parquet")

    save_feature_schema(simulated_df, dataset_path)

    config_path = dataset_path / "config.yml"
    config_snapshot = portable_config_snapshot(
        cfg.model_dump(mode="json"),
        anchor=cfg.simulation.registry_path,
    )

    with open(config_path, "w") as f:
        yaml.safe_dump(
            config_snapshot,
            f,
            sort_keys=False,
        )

    write_provenance(
        dataset_path / "provenance.yml",
        build_provenance(
            stage="simulation",
            artifact_path=dataset_path,
            version=version,
            input_config={"path": config_snapshot["simulation"]["input_path"]},
        ),
    )

    save_environment_versions(dataset_path / "versions.txt")

    logger.info("Simulated data available at")

    return dataset_path
