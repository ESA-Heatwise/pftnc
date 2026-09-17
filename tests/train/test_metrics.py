import pandas as pd
import pytest

from pftnc.train.evaluation.metrics import compute_metrics, validate_metrics


def test_compute_metrics_returns_one_score_per_target():
    truth = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0]})
    predictions = pd.DataFrame({"a": [2.0, 2.0], "b": [1.0, 5.0]})

    result = compute_metrics(truth, predictions, ["mse", "mae"])

    assert result == {
        "a": {"mse": 0.5, "mae": 0.5},
        "b": {"mse": 1.0, "mae": 1.0},
    }


def test_compute_metrics_rejects_misaligned_rows():
    truth = pd.DataFrame(
        {"target": [1.0, 2.0]},
        index=pd.Index([10, 20]),
    )
    predictions = pd.DataFrame(
        {"target": [1.0, 2.0]},
        index=pd.Index([10, 30]),
    )

    with pytest.raises(ValueError, match="matching indexes"):
        compute_metrics(truth, predictions, ["mse"])


def test_validate_metrics_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Unsupported metrics"):
        validate_metrics(["not_a_metric"])
