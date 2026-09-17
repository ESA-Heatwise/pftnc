import pandas as pd
import pytest

from pftnc.preprocess.dataset import (
    clean_feature_names,
    extract_prediction_metadata,
    prepare_feature_engineered_dataset,
)


def test_clean_feature_names_normalizes_columns_and_rejects_collisions():
    clean = clean_feature_names(pd.DataFrame({"sensor-1": [1], "other value": [2]}))
    assert list(clean.columns) == ["sensor_1", "other_value"]

    with pytest.raises(ValueError, match="duplicate columns"):
        clean_feature_names(pd.DataFrame({"a-b": [1], "a_b": [2]}))


def test_extract_prediction_metadata_adds_calendar_columns(model_input_frame):
    metadata = extract_prediction_metadata(
        model_input_frame,
        date_column="date",
        group_column="site",
    )

    assert metadata["year"].tolist() == [2024, 2024, 2024]
    assert metadata["month_name"].tolist() == ["Jan"] * 3
    assert "site" in metadata


def test_extract_prediction_metadata_rejects_missing_or_invalid_dates():
    with pytest.raises(ValueError, match="Missing date column"):
        extract_prediction_metadata(
            pd.DataFrame({"site": ["A"]}),
            date_column="date",
            group_column="site",
        )

    with pytest.raises(ValueError, match="Invalid values"):
        extract_prediction_metadata(
            pd.DataFrame({"site": ["A"], "date": ["not-a-date"]}),
            date_column="date",
            group_column="site",
        )


def test_prepare_feature_engineered_dataset_retains_all_candidate_features(
    model_input_frame,
    preprocess_schema,
):
    prepared = prepare_feature_engineered_dataset(
        model_input_frame,
        dataset_schema=preprocess_schema,
        gap_only=False,
    )

    assert prepared.feature_columns == [
        "sensor1",
        "sensor1_mean_2",
        "extra_feature",
        "chime_diatoms_ug_per_l",
        "chime_cyanobacteria_ug_per_l",
        "chime_others_ug_per_l",
    ]
    assert "target" not in prepared.X
    assert "date" not in prepared.X


def test_site_coordinates_are_metadata_not_candidate_features(
    model_input_frame,
    preprocess_schema,
):
    frame = model_input_frame.assign(
        site_lat=[51.0, 51.0, 51.0],
        site_lon=[7.0, 7.0, 7.0],
    )

    prepared = prepare_feature_engineered_dataset(
        frame,
        dataset_schema=preprocess_schema,
        gap_only=False,
    )

    assert "site_lat" not in prepared.X
    assert "site_lon" not in prepared.X
    assert prepared.metadata["site_lat"].tolist() == [51.0] * 3
    assert prepared.metadata["site_lon"].tolist() == [7.0] * 3
