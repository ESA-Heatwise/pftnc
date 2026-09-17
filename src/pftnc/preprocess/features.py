from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pftnc.utils.feature_columns import rolling_feature_name

logger = logging.getLogger(__name__)

CHIME_COLUMNS = [
    "chime_diatoms_ug_per_l",
    "chime_cyanobacteria_ug_per_l",
    "chime_others_ug_per_l",
]


def compute_rolling_slope(values: pd.Series) -> float:
    """Return the OLS slope over the non-missing values in a rolling window."""
    valid = ~values.isna()
    y = values[valid].to_numpy(dtype=float)
    if len(y) < 2:
        return np.nan

    t = np.flatnonzero(valid).astype(float)
    t -= t.mean()
    denominator = np.dot(t, t)
    if denominator == 0:
        return np.nan
    return float(np.dot(t, y) / denominator)


def fill_missing_dates(
    df: pd.DataFrame,
    *,
    date_column: str,
    group_column: str,
    freq: str = "D",
) -> pd.DataFrame:
    """Reindex each group independently to a continuous date range."""
    frame = df.copy()
    frame[date_column] = pd.to_datetime(frame[date_column])

    parts: list[pd.DataFrame] = []
    for group_value, group in frame.groupby(group_column, dropna=False, sort=False):
        if group[date_column].duplicated().any():
            duplicates = group.loc[group[date_column].duplicated(), date_column]
            raise ValueError(
                f"Duplicate dates for {group_column}={group_value!r}: "
                f"{duplicates.astype(str).tolist()}"
            )

        full_range = pd.date_range(
            start=group[date_column].min(),
            end=group[date_column].max(),
            freq=freq,
        )
        expanded = (
            group.set_index(date_column)
            .reindex(full_range)
            .rename_axis(date_column)
            .reset_index()
        )
        expanded[group_column] = group_value
        parts.append(expanded)

    return pd.concat(parts, ignore_index=True)


def engineer_features(
    df: pd.DataFrame,
    *,
    feature_config: Any,
    dataset_schema: Any,
) -> pd.DataFrame:
    """
    Generate reusable time-series features from raw observations.

    This function does not filter rows using targets, create folds, or write files,
    so the exact same implementation can be used by offline preprocessing and
    inference.
    """
    synthetic_col = "_is_synthetic"

    frame = df.reset_index(drop=False).copy()
    date_col = dataset_schema.date_column
    group_col = dataset_schema.group_column

    frame[synthetic_col] = False

    frame = fill_missing_dates(
        frame,
        date_column=date_col,
        group_column=group_col,
    )
    frame[synthetic_col] = frame[synthetic_col].isna()

    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values([group_col, date_col]).reset_index(drop=True)

    sensor_columns = [
        column for columns in dataset_schema.sensors.values() for column in columns
    ]
    missing_sensors = [column for column in sensor_columns if column not in frame]
    if missing_sensors:
        raise ValueError(f"Missing sensor columns: {missing_sensors}")

    rolling = feature_config.rolling
    if rolling is None:
        raise ValueError("Rolling feature config is required")

    windows = list(rolling.windows)
    statistics = list(rolling.statistics)
    supported = {"mean", "median", "max", "slope", "std", "count"}
    unknown = sorted(set(statistics) - supported)
    if unknown:
        raise ValueError(f"Unsupported rolling statistics: {unknown}")

    new_features: dict[str, pd.Series] = {}
    grouped = frame.groupby(group_col, sort=False, dropna=False)
    total = len(sensor_columns) * len(windows) * len(statistics)

    with tqdm(total=total, desc="Generating rolling features") as progress:
        for column in sensor_columns:
            for window in windows:
                rolled = grouped[column].rolling(window=window, min_periods=1)
                for statistic in statistics:
                    name = rolling_feature_name(column, statistic, window)
                    if statistic == "slope":
                        result = rolled.apply(compute_rolling_slope, raw=False)
                    else:
                        result = getattr(rolled, statistic)()
                    new_features[name] = result.reset_index(level=0, drop=True)
                    progress.update(1)

    dates = frame[date_col]
    for column in sensor_columns:
        last_valid_date = (
            dates.where(frame[column].notna())
            .groupby(frame[group_col], dropna=False)
            .transform("ffill")
        )
        new_features[f"{column}_days_since_obs"] = (dates - last_valid_date).dt.days

    missing_chime = [column for column in CHIME_COLUMNS if column not in frame]
    if missing_chime:
        raise ValueError(
            "Cannot create last-observed CHIME features. "
            f"Missing columns: {missing_chime}"
        )

    for column in CHIME_COLUMNS:
        new_features[f"{column}_last_observed"] = grouped[column].transform(
            lambda values: values.shift(1).ffill()
        )

    day_of_year = frame[date_col].dt.dayofyear
    new_features["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    new_features["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    result = pd.concat(
        [frame, pd.DataFrame(new_features, index=frame.index)],
        axis=1,
    )

    synthetic_rows = int(result[synthetic_col].sum())

    # Synthetic rows were required for feature calculation, but they are not
    # real observations and must not become model samples.
    result = (
        result.loc[~result[synthetic_col]]
        .drop(columns=[synthetic_col])
        .reset_index(drop=True)
    )

    logger.info(
        "Removed %d synthetic rows added by fill_missing_dates",
        synthetic_rows,
    )

    logger.info("Generated %d engineered features", len(new_features))
    return result
