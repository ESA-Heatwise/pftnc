import pandas as pd
import pytest

from pftnc.train.evaluation.exports import PredictionExporter


def test_prediction_exporter_writes_metadata_features_truth_and_predictions(
    prepared_split,
    spy_tracker,
    tmp_path,
):
    exporter = PredictionExporter(spy_tracker)
    predictions = prepared_split.y + 1.0

    path = exporter.export(
        tmp_path,
        "validation",
        prepared_split.X,
        prepared_split.metadata,
        prepared_split.y,
        predictions,
    )
    exported = pd.read_csv(path)

    assert path.name == "validation_full_predictions.csv"
    assert "date" in exported
    assert "feature" in exported
    assert "diatoms_ug_per_l_true" in exported
    assert "diatoms_ug_per_l_pred" in exported
    assert spy_tracker.artifacts[0][1] == "validation/predictions"


def test_prediction_exporter_rejects_missing_metadata_rows(
    prepared_split,
    spy_tracker,
    tmp_path,
):
    metadata = prepared_split.metadata.iloc[:1]

    with pytest.raises(ValueError, match="missing rows"):
        PredictionExporter(spy_tracker).export(
            tmp_path,
            "validation",
            prepared_split.X,
            metadata,
            prepared_split.y,
            prepared_split.y,
        )
