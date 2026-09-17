from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pftnc.train.final_test import (
    ModelSelector,
    _aggregate_selection_scores,
    _get_metric_score,
    _is_better,
    _score_to_weight,
)


def test_final_test_uses_consistent_feature_columns(final_test_runner):
    first = SimpleNamespace(
        model_bundle=SimpleNamespace(feature_columns=("a", "b")),
    )
    second = SimpleNamespace(
        model_bundle=SimpleNamespace(feature_columns=("a", "b")),
    )

    assert final_test_runner._resolve_feature_columns(
        {"fold_1": first, "fold_2": second}
    ) == ["a", "b"]


def test_final_test_rejects_inconsistent_feature_columns(final_test_runner):
    results = {
        "fold_1": SimpleNamespace(
            model_bundle=SimpleNamespace(feature_columns=("a",)),
        ),
        "fold_2": SimpleNamespace(
            model_bundle=SimpleNamespace(feature_columns=("b",)),
        ),
    }

    with pytest.raises(ValueError, match="different feature columns"):
        final_test_runner._resolve_feature_columns(results)


def test_final_test_simple_ensemble_averages_fold_predictions(final_test_runner):
    first = pd.DataFrame({"target": [1.0, 3.0]})
    second = pd.DataFrame({"target": [3.0, 5.0]})

    result = final_test_runner._simple_ensemble([first, second])

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({"target": [2.0, 4.0]}),
    )


def test_final_test_weighted_ensemble_uses_total_weight(final_test_runner):
    first = pd.DataFrame({"target": [2.0, 4.0]})
    second = pd.DataFrame({"target": [4.0, 8.0]})

    result = final_test_runner._weighted_ensemble(
        weighted_predictions=[first * 1.0, second * 3.0],
        weights=[1.0, 3.0],
    )

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({"target": [3.5, 7.0]}),
    )


def test_final_test_per_target_ensemble_can_use_different_weights(
    final_test_runner,
):
    index = pd.Index([10, 20])
    result = final_test_runner._per_target_weighted_ensemble(
        weighted_predictions={
            "a": [
                pd.Series([1.0, 2.0], index=index),
                pd.Series([3.0, 4.0], index=index),
            ],
            "b": [
                pd.Series([30.0, 60.0], index=index),
                pd.Series([30.0, 40.0], index=index),
            ],
        },
        weights={"a": [1.0, 1.0], "b": [3.0, 1.0]},
        targets=["a", "b"],
        index=index,
    )

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({"a": [2.0, 3.0], "b": [15.0, 25.0]}, index=index),
    )


@pytest.mark.parametrize(
    ("fold", "expected"),
    [("fold_2015", 2015), ("fold_without_year", None), ("2015", None)],
)
def test_final_test_fold_step_is_safe(final_test_runner, fold, expected):
    assert final_test_runner._fold_step(fold) == expected


def test_final_test_validates_prediction_index_and_values(final_test_runner):
    truth = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([1, 2]))

    final_test_runner._validate_predictions("fold_1", truth.copy(), truth)

    with pytest.raises(ValueError, match="index mismatch"):
        final_test_runner._validate_predictions(
            "fold_1",
            truth.set_axis([2, 3]),
            truth,
        )

    invalid = truth.copy()
    invalid.loc[1, "a"] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        final_test_runner._validate_predictions("fold_1", invalid, truth)


def test_metric_score_requires_finite_existing_value():
    metrics = {"a": {"mse": 2.0}}

    assert _get_metric_score(metrics, "a", "mse") == 2.0

    with pytest.raises(ValueError, match="missing"):
        _get_metric_score(metrics, "b", "mse")
    with pytest.raises(ValueError, match="finite"):
        _get_metric_score({"a": {"mse": np.inf}}, "a", "mse")


def test_selection_helpers_respect_direction_and_mode():
    assert _aggregate_selection_scores([1.0, 3.0], "mean", "minimize") == 2.0
    assert _aggregate_selection_scores([1.0, 3.0], "median", "minimize") == 2.0
    assert _aggregate_selection_scores([1.0, 3.0], "worst", "minimize") == 3.0
    assert _aggregate_selection_scores([1.0, 3.0], "worst", "maximize") == 1.0
    assert _is_better(1.0, 2.0, "minimize")
    assert _is_better(2.0, 1.0, "maximize")


@pytest.mark.parametrize(
    ("score", "direction", "expected"),
    [(2.0, "minimize", 0.5), (2.0, "maximize", 2.0)],
)
def test_score_to_weight(score, direction, expected):
    assert _score_to_weight(score, direction) == expected


def test_model_selector_selects_lowest_overall_fold(target_transform_manager):
    config = SimpleNamespace(
        metric="mse",
        direction="minimize",
        target=None,
        mode="mean",
    )
    models = SimpleNamespace(
        selected_model_uri=lambda bundle, target, requires_complete_bundle: (
            bundle.model_uri
        ),
    )
    selector = ModelSelector(config, ["mse"], target_transform_manager, models)

    def result(uri, score):
        bundle = SimpleNamespace(is_multi_target=True, model_uri=uri)
        return SimpleNamespace(
            model_bundle=bundle,
            validation_metrics={
                target: {"mse": score}
                for target in target_transform_manager.physical_targets
            },
        )

    selected = selector.select(
        {"fold_1": result("uri-1", 2.0), "fold_2": result("uri-2", 1.0)}
    )

    assert selected.overall.fold == "fold_2"
    assert selected.overall.model_uri == "uri-2"
    assert set(selected.per_target) == set(target_transform_manager.physical_targets)


def test_final_test_run_scores_folds_and_three_ensembles(
    final_test_runner,
    prepared_split,
    model_space_targets,
    target_transform_manager,
    spy_tracker,
    tmp_path,
):
    final_test_runner.repository.load_final_test = lambda model_columns: prepared_split

    evaluator_calls = []

    def evaluate_predictions(**kwargs):
        evaluator_calls.append(kwargs)
        return {
            target: {"mse": 0.0} for target in target_transform_manager.physical_targets
        }

    final_test_runner.models = SimpleNamespace(
        predict=lambda bundle, X, output_columns: (model_space_targets.copy(), {}),
    )
    evaluator = SimpleNamespace(evaluate_predictions=evaluate_predictions)
    final_test_runner.evaluator = evaluator
    final_test_runner.tracker = spy_tracker

    def fold_result():
        return SimpleNamespace(
            model_bundle=SimpleNamespace(
                feature_columns=tuple(prepared_split.X.columns)
            ),
            validation_metrics={
                target: {"mse": 1.0}
                for target in target_transform_manager.physical_targets
            },
        )

    results = final_test_runner.run(
        {"fold_2015": fold_result(), "fold_2016": fold_result()},
        tmp_path,
    )

    assert set(results) == {
        "fold_2015",
        "fold_2016",
        "ensemble_simple",
        "ensemble_weighted",
        "ensemble_weighted_per_target",
    }
    assert len(evaluator_calls) == 5
    assert evaluator_calls[0]["step"] == 2015
    assert (tmp_path / "final_test_per_fold.csv").is_file()
