from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pftnc.preprocess.features import (
    CHIME_COLUMNS,
    compute_rolling_slope,
    engineer_features,
    fill_missing_dates,
)


def test_compute_rolling_slope_uses_original_positions_when_values_are_missing():
    values = pd.Series([1.0, np.nan, 5.0])

    assert compute_rolling_slope(values) == pytest.approx(2.0)
    assert np.isnan(compute_rolling_slope(pd.Series([1.0])))


def test_fill_missing_dates_fills_each_group_independently(raw_preprocess_frame):
    result = fill_missing_dates(
        raw_preprocess_frame[["site", "date", "sensor1"]],
        date_column="date",
        group_column="site",
    )

    group_a = result[result["site"] == "A"]
    group_b = result[result["site"] == "B"]

    assert group_a["date"].tolist() == list(pd.date_range("2024-01-01", "2024-01-03"))
    assert group_b["date"].tolist() == list(pd.date_range("2024-01-01", "2024-01-02"))
    assert pd.isna(group_a.loc[group_a["date"] == "2024-01-02", "sensor1"]).all()


def test_fill_missing_dates_rejects_duplicate_group_dates(raw_preprocess_frame):
    duplicate = pd.concat([raw_preprocess_frame, raw_preprocess_frame.iloc[[0]]])

    with pytest.raises(ValueError, match="Duplicate dates"):
        fill_missing_dates(duplicate, date_column="date", group_column="site")


def test_engineer_features_creates_rolling_calendar_and_last_observed_features(
    raw_preprocess_frame,
    preprocess_feature_config,
    preprocess_schema,
):
    result = engineer_features(
        raw_preprocess_frame,
        feature_config=preprocess_feature_config,
        dataset_schema=preprocess_schema,
    )

    expected_columns = {
        "sensor1_mean_2",
        "sensor1_max_2",
        "sensor1_count_2",
        "sensor1_slope_2",
        "sensor1_days_since_obs",
        "chime_diatoms_ug_per_l_last_observed",
        "doy_sin",
        "doy_cos",
    }
    assert expected_columns.issubset(result.columns)
    assert len(result) == len(raw_preprocess_frame)
    assert "_is_synthetic" not in result.columns


def test_engineer_features_rejects_unknown_statistic(
    raw_preprocess_frame,
    preprocess_schema,
):
    config = SimpleNamespace(
        rolling=SimpleNamespace(windows=[2], statistics=["unknown"]),
    )

    with pytest.raises(ValueError, match="Unsupported rolling statistics"):
        engineer_features(
            raw_preprocess_frame,
            feature_config=config,
            dataset_schema=preprocess_schema,
        )


@pytest.mark.parametrize(
    "missing_column",
    ["sensor1", *CHIME_COLUMNS],
)
def test_engineer_features_rejects_required_missing_columns(
    raw_preprocess_frame,
    preprocess_feature_config,
    preprocess_schema,
    missing_column,
):
    values = raw_preprocess_frame.drop(columns=[missing_column])

    with pytest.raises(ValueError):
        engineer_features(
            values,
            feature_config=preprocess_feature_config,
            dataset_schema=preprocess_schema,
        )
