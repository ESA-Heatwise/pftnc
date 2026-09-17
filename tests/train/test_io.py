from types import SimpleNamespace

from pftnc.train.io import build_model_selection_metadata


def test_model_selection_metadata_records_all_ensemble_weights():
    fold_results = {
        "fold_a": SimpleNamespace(
            validation_metrics={
                "target_a": {"rmse": 2.0},
                "target_b": {"rmse": 4.0},
            }
        ),
        "fold_b": SimpleNamespace(
            validation_metrics={
                "target_a": {"rmse": 4.0},
                "target_b": {"rmse": 2.0},
            }
        ),
    }
    best_model = SimpleNamespace(
        overall=SimpleNamespace(fold="fold_a", score=3.0),
        per_target={
            "target_a": SimpleNamespace(fold="fold_a", score=2.0),
            "target_b": SimpleNamespace(fold="fold_b", score=2.0),
        },
    )

    selection = build_model_selection_metadata(
        best_model=best_model,
        fold_results=fold_results,
        metric="rmse",
        direction="minimize",
        mode="mean",
        target=None,
        physical_targets=["target_a", "target_b"],
    )

    assert selection["metric"] == "rmse"
    assert selection["best_overall"]["fold"] == "fold_a"
    assert selection["best_per_target"]["target_b"]["fold"] == "fold_b"
    assert selection["ensemble_weights"]["global"] == {
        "fold_a": 1 / 3,
        "fold_b": 1 / 3,
    }
    assert selection["ensemble_weights"]["per_target"] == {
        "target_a": {"fold_a": 0.5, "fold_b": 0.25},
        "target_b": {"fold_a": 0.25, "fold_b": 0.5},
    }
