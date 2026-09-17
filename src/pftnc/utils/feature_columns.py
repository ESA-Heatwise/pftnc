from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import pandas as pd


def clean_feature_name(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name).strip("_")


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = pd.Index(
        [clean_feature_name(str(column)) for column in cleaned.columns]
    )

    duplicated = cleaned.columns[cleaned.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            "Feature-name normalization created duplicate columns: "
            f"{sorted(set(duplicated))}"
        )
    return cleaned


def rolling_feature_name(sensor: str, statistic: str, window: int) -> str:
    return clean_feature_name(f"{sensor}_{statistic}_{window}")


def rolling_feature_names(
    sensors: Mapping[str, Iterable[str]],
    windows: Iterable[int],
    statistics: Iterable[str],
) -> list[str]:
    return [
        rolling_feature_name(sensor, statistic, window)
        for sensor_columns in sensors.values()
        for sensor in sensor_columns
        for window in windows
        for statistic in statistics
    ]
