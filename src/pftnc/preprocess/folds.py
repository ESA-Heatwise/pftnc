from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pftnc.utils.utils import check_target_quality

from ..config import DatasetSchema
from .dataset import PreparedTrainingDataset


def _json_default(value: Any) -> int | float:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_split(
    directory: Path,
    name: str,
    *,
    X: pd.DataFrame,
    y: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    X.to_parquet(directory / f"{name}_X.parquet")
    y.to_parquet(directory / f"{name}_y.parquet")
    metadata.to_parquet(directory / f"{name}_metadata.parquet")


def write_year_folds(
    prepared: PreparedTrainingDataset,
    *,
    output_path: Path,
    dataset_schema: DatasetSchema,
    final_test_year: int,
) -> dict[str, dict[str, Any]]:
    """Generate year-based train/validation/test dataset folds."""
    output_path.mkdir(parents=True, exist_ok=False)

    check_target_quality(prepared.y, list(prepared.y.columns))
    years = prepared.metadata["year"]
    available_years = sorted(int(year) for year in years.dropna().unique())
    if final_test_year not in available_years:
        raise ValueError(
            f"final_test_year={final_test_year} not present; "
            f"available years: {available_years}"
        )

    training_years = [year for year in available_years if year != final_test_year]
    if len(training_years) < 3:
        raise ValueError("At least three non-final-test years are required")

    folds: dict[str, dict[str, Any]] = {}
    for index, validation_year in enumerate(training_years):
        test_year = training_years[(index + 1) % len(training_years)]
        train_years = [
            year for year in training_years if year not in {validation_year, test_year}
        ]
        fold_name = f"fold_{validation_year}"
        fold_path = output_path / fold_name
        fold_path.mkdir()

        split_masks = {
            "train": years.isin(train_years),
            "val": years.eq(validation_year),
            "test": years.eq(test_year),
        }
        for split_name, mask in split_masks.items():
            _write_split(
                fold_path,
                split_name,
                X=prepared.X.loc[mask],
                y=prepared.y.loc[mask],
                metadata=prepared.metadata.loc[mask],
            )

        folds[fold_name] = {
            "train_years": train_years,
            "val_year": validation_year,
            "test_year": test_year,
        }

    final_path = output_path / "final_test"
    final_path.mkdir()
    final_mask = years.eq(final_test_year)
    _write_split(
        final_path,
        "test",
        X=prepared.X.loc[final_mask],
        y=prepared.y.loc[final_mask],
        metadata=prepared.metadata.loc[final_mask],
    )

    (output_path / "folds.json").write_text(
        json.dumps(folds, indent=2, default=_json_default)
    )
    return folds
