from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from pftnc.config import FeatureOperationsConfig, FeatureSelectionConfig
from pftnc.train.data import (
    CHIME_RAW_COLUMNS,
    clean_feature_names,
    extract_prediction_metadata,
    get_chime_gap_mask,
    get_model_feature_columns,
    load_feature_operations_config,
    prepare_split,
)


def test_load_feature_operations_config_supports_legacy_artifact_shape(
    tmp_path: Path,
):
    (tmp_path / "config.yml").write_text(
        yaml.safe_dump(
            {
                "feature_engineering": {
                    "features": {"rolling": {"windows": [3], "statistics": ["mean"]}}
                }
            }
        )
    )

    config = load_feature_operations_config(tmp_path)

    assert config.rolling.windows == [3]


def test_extract_prediction_metadata_derives_calendar_columns():
    raw = pd.DataFrame(
        {"date": ["2024-01-03", "2024-02-04"], "feature": [1.0, 2.0]},
        index=[10, 20],
    )

    metadata = extract_prediction_metadata(raw)

    assert metadata["date"].tolist() == list(
        pd.to_datetime(["2024-01-03", "2024-02-04"])
    )
    assert metadata["year"].tolist() == [2024, 2024]
    assert metadata["month_name"].tolist() == ["Jan", "Feb"]
    assert metadata["iso_week"].tolist() == [1, 5]


def test_clean_feature_names_rejects_normalization_collisions():
    with pytest.raises(ValueError, match="duplicate columns"):
        clean_feature_names(pd.DataFrame({"sensor-a": [1], "sensor_a": [2]}))


def test_chime_gap_mask_requires_complete_missing_vector():
    values = pd.DataFrame(
        {
            CHIME_RAW_COLUMNS[0]: [None, None, 1.0],
            CHIME_RAW_COLUMNS[1]: [None, 2.0, None],
            CHIME_RAW_COLUMNS[2]: [None, None, None],
        }
    )

    assert get_chime_gap_mask(values).tolist() == [True, False, False]


def test_prepare_split_filters_gap_rows_and_keeps_model_features(
    target_names,
    dataset_metadata,
):
    raw_X = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "sensor_a": [1.0, 2.0],
            **{column: [None, 1.0] for column in CHIME_RAW_COLUMNS},
        },
        index=[10, 20],
    )
    raw_y = pd.DataFrame(
        {target: [1.0, 2.0] for target in target_names},
        index=[10, 20],
    )
    prepared, columns = prepare_split(
        raw_X,
        raw_y,
        "train",
        dataset_metadata,
        feature_config=FeatureOperationsConfig(rolling=None),
        feature_selection=FeatureSelectionConfig(),
    )

    assert columns == ["sensor_a"]
    assert prepared.X.index.tolist() == [10]
    assert prepared.y.index.tolist() == [10]
    assert prepared.metadata.loc[10, "year"] == 2024


def test_model_feature_selection_is_training_configured(dataset_metadata):
    metadata = SimpleNamespace(
        dataset_schema=SimpleNamespace(
            targets=dataset_metadata.dataset_schema.targets,
            sensors={
                "sensor": ["sensor_a"],
                "chime": CHIME_RAW_COLUMNS,
            },
        )
    )
    frame = pd.DataFrame(
        columns=[
            "sensor_a",
            "sensor_a_mean_3",
            *CHIME_RAW_COLUMNS,
            "chime_diatoms_ug_per_l_mean_3",
            "chime_cyanobacteria_ug_per_l_mean_3",
            "chime_others_ug_per_l_mean_3",
            "chime_diatoms_ug_per_l_last_observed",
        ]
    )
    feature_config = FeatureOperationsConfig(
        rolling={"windows": [3], "statistics": ["mean"]}
    )

    without_chime = get_model_feature_columns(
        frame,
        metadata,
        feature_config=feature_config,
        feature_selection=FeatureSelectionConfig(
            include=["chime_diatoms_ug_per_l_last_observed"]
        ),
    )
    assert without_chime == [
        "chime_diatoms_ug_per_l_last_observed",
        "sensor_a",
        "sensor_a_mean_3",
        "chime_diatoms_ug_per_l_mean_3",
        "chime_cyanobacteria_ug_per_l_mean_3",
        "chime_others_ug_per_l_mean_3",
    ]

    with_chime = get_model_feature_columns(
        frame,
        metadata,
        feature_config=feature_config,
        feature_selection=FeatureSelectionConfig(include=CHIME_RAW_COLUMNS),
    )
    assert with_chime == [
        *CHIME_RAW_COLUMNS,
        "sensor_a",
        "sensor_a_mean_3",
        "chime_diatoms_ug_per_l_mean_3",
        "chime_cyanobacteria_ug_per_l_mean_3",
        "chime_others_ug_per_l_mean_3",
    ]


def test_prepare_split_uses_persisted_metadata_and_preserves_row_alignment(
    target_names,
    dataset_metadata,
):
    index = pd.Index([10, 20])
    raw_X = pd.DataFrame({"sensor_a": [1.0, 2.0]}, index=index)
    raw_y = pd.DataFrame(
        {target: [1.0, 2.0] for target in target_names},
        index=index,
    )
    metadata = pd.DataFrame(
        {"time": pd.to_datetime(["2024-01-01", "2024-01-02"]), "year": [2024, 2024]},
        index=index,
    )

    prepared, columns = prepare_split(
        raw_X,
        raw_y,
        "train",
        dataset_metadata,
        raw_metadata=metadata,
        feature_config=FeatureOperationsConfig(rolling=None),
        feature_selection=FeatureSelectionConfig(),
    )

    assert columns == ["sensor_a"]
    assert prepared.X.index.equals(prepared.metadata.index)
    assert prepared.y.index.equals(prepared.metadata.index)


@pytest.mark.parametrize(
    "mutation",
    ["row_count", "index", "target"],
)
def test_prepare_split_rejects_incompatible_inputs(
    target_names,
    dataset_metadata,
    mutation,
):
    raw_X = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "sensor_a": [1.0],
            **{column: [None] for column in CHIME_RAW_COLUMNS},
        },
        index=[10],
    )
    raw_y = pd.DataFrame({target: [1.0] for target in target_names}, index=[10])
    if mutation == "row_count":
        raw_y = pd.concat([raw_y, raw_y])
    elif mutation == "index":
        raw_y.index = [11]
    else:
        raw_y = raw_y.drop(columns=[target_names[0]])

    with pytest.raises(ValueError):
        prepare_split(
            raw_X,
            raw_y,
            "train",
            dataset_metadata,
            feature_config=FeatureOperationsConfig(rolling=None),
            feature_selection=FeatureSelectionConfig(),
        )
