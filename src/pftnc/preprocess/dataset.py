from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pftnc.utils.feature_columns import clean_feature_name, clean_feature_names

from .features import CHIME_COLUMNS

logger = logging.getLogger(__name__)

SITE_METADATA_COLUMNS = ["site_lat", "site_lon"]
PREDICTION_METADATA_COLUMNS = [
    "time",
    "date",
    "site",
    *SITE_METADATA_COLUMNS,
    *CHIME_COLUMNS,
]


@dataclass(frozen=True)
class PreparedTrainingDataset:
    X: pd.DataFrame
    y: pd.DataFrame
    metadata: pd.DataFrame
    feature_columns: list[str]


def extract_prediction_metadata(
    df: pd.DataFrame,
    *,
    date_column: str,
    group_column: str,
) -> pd.DataFrame:
    requested = list(
        dict.fromkeys([date_column, group_column, *PREDICTION_METADATA_COLUMNS])
    )
    available = [column for column in requested if column in df.columns]
    metadata = df[available].copy()
    if date_column not in metadata:
        raise ValueError(f"Missing date column: {date_column}")

    date = pd.to_datetime(metadata[date_column], errors="coerce")
    if date.isna().any():
        raise ValueError(f"Invalid values found in date column {date_column!r}")

    metadata["date"] = date.dt.normalize()
    metadata["year"] = date.dt.year.astype("Int64")
    metadata["month"] = date.dt.month.astype("Int64")
    metadata["month_name"] = date.dt.strftime("%b")
    metadata["iso_week"] = date.dt.isocalendar().week
    return metadata


def get_chime_gap_mask(df: pd.DataFrame) -> pd.Series:
    missing_columns = [column for column in CHIME_COLUMNS if column not in df]
    if missing_columns:
        raise ValueError(f"Missing required CHIME columns: {missing_columns}")

    missing = df[CHIME_COLUMNS].isna()
    partially_missing = missing.any(axis=1) & ~missing.all(axis=1)
    if partially_missing.any():
        logger.warning(
            "%d rows have partially missing current CHIME values and are excluded",
            int(partially_missing.sum()),
        )
    return missing.all(axis=1)


def prepare_feature_engineered_dataset(
    engineered_df: pd.DataFrame,
    *,
    dataset_schema: Any,
    gap_only: bool = True,
) -> PreparedTrainingDataset:
    """Prepare a dataset containing all candidate features for later training."""
    targets = [clean_feature_name(column) for column in dataset_schema.targets]
    missing_targets = [column for column in targets if column not in engineered_df]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    frame = engineered_df.copy()
    metadata = extract_prediction_metadata(
        frame,
        date_column=dataset_schema.date_column,
        group_column=dataset_schema.group_column,
    )

    if gap_only:
        mask = get_chime_gap_mask(frame)
        frame = frame.loc[mask].copy()
        metadata = metadata.loc[mask].copy()

    clean = clean_feature_names(frame)
    excluded = {
        *targets,
        clean_feature_name(dataset_schema.date_column),
        clean_feature_name(dataset_schema.group_column),
        "time",
        "date",
        "site",
        *SITE_METADATA_COLUMNS,
        "year",
        "month",
        "month_name",
        "iso_week",
    }
    feature_columns = [column for column in clean.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No engineered feature columns were found")

    y = clean.loc[:, targets].copy()
    nan_counts = y.isna().sum()
    if nan_counts.any():
        raise ValueError(
            "Training targets contain missing values: "
            f"{nan_counts[nan_counts > 0].to_dict()}"
        )

    X = clean.loc[:, feature_columns].copy()
    metadata.index = X.index
    return PreparedTrainingDataset(
        X=X,
        y=y,
        metadata=metadata,
        feature_columns=feature_columns,
    )
