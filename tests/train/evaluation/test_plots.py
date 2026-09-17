import pandas as pd
import pytest

from pftnc.train.evaluation.plots import EvaluationPlotter


def plot_metadata(index):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * len(index)),
            "month": [1] * len(index),
            "iso_week": [1] * len(index),
        },
        index=index,
    )


def test_plotter_writes_prediction_and_metric_plots(
    prepared_split,
    spy_tracker,
    tmp_path,
):
    plotter = EvaluationPlotter(spy_tracker, ["mse"])
    predictions = prepared_split.y + 1.0
    metadata = plot_metadata(prepared_split.y.index)

    plotter.prediction_plots(
        tmp_path,
        "test",
        metadata,
        prepared_split.y,
        predictions,
    )
    plotter.metric_comparison_plots(
        {target: {"mse": 1.0} for target in prepared_split.y.columns},
        "test",
        tmp_path,
    )

    assert len(list((tmp_path / "plots").glob("*.png"))) == 9
    assert len(list((tmp_path / "metric_comparisons").glob("*.png"))) == 1


def test_plotter_rejects_missing_calendar_metadata(
    prepared_split,
    spy_tracker,
    tmp_path,
):
    with pytest.raises(ValueError, match="missing columns"):
        EvaluationPlotter(spy_tracker, ["mse"]).prediction_plots(
            tmp_path,
            "test",
            pd.DataFrame(index=prepared_split.y.index),
            prepared_split.y,
            prepared_split.y,
        )
