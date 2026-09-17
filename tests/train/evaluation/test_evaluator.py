from types import SimpleNamespace

import pandas as pd
import pytest


def test_evaluate_predictions_computes_metrics_and_reports(
    evaluator,
    prepared_split,
    spy_tracker,
    spy_exporter,
    spy_plotter,
    tmp_path,
):
    predictions = prepared_split.y.copy()
    predictions.iloc[0, 0] += 1.0

    result = evaluator.evaluate_predictions(
        name="fold_2015",
        predictions=predictions,
        split_data=prepared_split,
        run_dir=tmp_path,
        step=2015,
    )

    assert result["diatoms_ug_per_l"]["mse"] == pytest.approx(1 / 3)
    assert result["cyanobacterial_ug_per_l"]["mse"] == 0.0
    assert spy_exporter.calls[0][1] == "final_test/fold_2015"
    assert spy_plotter.prediction_calls[0][1] == "final_test/fold_2015"
    assert spy_tracker.metrics
    assert all(metric[2] == 2015 for metric in spy_tracker.metrics)
    assert spy_tracker.artifacts[0][1] == "fold_2015"


def test_evaluate_predictions_rejects_wrong_target_columns(
    evaluator,
    prepared_split,
    tmp_path,
):
    predictions = pd.DataFrame(
        {"wrong_target": [1.0, 2.0, 3.0]},
        index=prepared_split.X.index,
    )

    with pytest.raises(ValueError, match="do not match physical targets"):
        evaluator.evaluate_predictions(
            name="bad_predictions",
            predictions=predictions,
            split_data=prepared_split,
            run_dir=tmp_path,
        )


def test_evaluate_model_predicts_inverts_and_logs_shap(
    evaluator,
    prepared_split,
    model_space_targets,
    target_transform_manager,
    spy_tracker,
    tmp_path,
    monkeypatch,
):
    loaded_model = object()
    monkeypatch.setattr(
        evaluator.models,
        "predict",
        lambda bundle, X, columns: (
            model_space_targets,
            {"__multi_target__": loaded_model},
        ),
        raising=False,
    )
    shap_calls = []
    monkeypatch.setattr(
        evaluator.shap_logger,
        "log",
        lambda *args: shap_calls.append(args),
    )
    bundle = SimpleNamespace(is_multi_target=True)

    result = evaluator.evaluate_model(
        bundle,
        prepared_split,
        tmp_path,
        split="validation",
        step=7,
    )

    assert set(result) == set(target_transform_manager.physical_targets)
    assert shap_calls[0][1] is loaded_model
    assert shap_calls[0][4] == "validation/7"
    assert all(metric[2] == 7 for metric in spy_tracker.metrics)
