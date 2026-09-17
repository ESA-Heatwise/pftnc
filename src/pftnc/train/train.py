from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pftnc.config import AppConfig
from pftnc.train.training import Trainer, set_global_seed

logger = logging.getLogger(__name__)


def run_training(
    config: AppConfig,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Execute one configured training run."""

    logger.info(
        "[RUN] Starting training run with config file: %s",
        config_path,
    )

    seed = config.project.seed
    logger.info("[SEED] Setting global seed: %s", seed)
    set_global_seed(seed)

    logger.debug("[CONFIG] Training config: %s", config.training)
    trainer = Trainer(config=config)

    logger.info("[PIPELINE] Executing training pipeline")
    result = trainer.train()

    logger.info("[TRAINING] Training completed successfully")
    logger.info(
        "[TRAINING] Run ID: %s",
        result.get("trainer_run_id"),
    )
    return result
