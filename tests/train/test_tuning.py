import pytest

from pftnc.train.tuning import aggregate_target_scores


class FakeTrial:
    def suggest_int(self, name, low, high):
        return (name, low, high)

    def suggest_float(self, name, low, high):
        return (name, low, high)

    def suggest_categorical(self, name, choices):
        return (name, choices)


def test_tuner_samples_integer_float_and_categorical_parameters(tuner):
    result = tuner._sample_params(FakeTrial())

    assert result == {
        "integer_param": ("integer_param", 1, 3),
        "float_param": ("float_param", 0.1, 0.9),
        "choice_param": ("choice_param", ["a", "b"]),
    }


def test_tuner_disabled_returns_configured_parameters(tuner):
    tuner.config.enabled = False

    assert tuner.resolve(None, None, None, None) == {"depth": 2}


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("objective_metric", "not_supported", "Unsupported metrics"),
        ("direction", "sideways", "direction"),
        ("search_space", None, "search_space"),
    ],
)
def test_tuner_rejects_invalid_configuration(tuner, attribute, value, message):
    setattr(tuner.config, attribute, value)

    with pytest.raises(ValueError, match=message):
        tuner._tune(None, None, None, None)


def test_multi_target_tuning_returns_best_sampled_parameters(
    tuner,
    features,
    model_space_targets,
):
    class TrialModel:
        def fit(self, *args, **kwargs):
            return None

        def predict(self, X):
            return model_space_targets.to_numpy()

    tuner.config.search_space = {"depth": [2, 2]}
    tuner.config.n_trials = 1
    tuner.models.configured_params = lambda: {"base": 1}
    tuner.models.merge_params = lambda base, sampled: {**base, **sampled}
    tuner.models.create_adapter = lambda params: TrialModel()

    result = tuner._tune_multi_target(
        features,
        features,
        model_space_targets,
        model_space_targets,
    )

    assert result == {"base": 1, "depth": 2}


def test_aggregate_target_scores_mean():
    assert (
        aggregate_target_scores(
            {"a": 1.0, "b": 3.0},
            aggregation="mean",
            target_weights=None,
            direction="minimize",
        )
        == 2.0
    )


def test_aggregate_target_scores_weighted():
    assert aggregate_target_scores(
        {"a": 1.0, "b": 3.0},
        aggregation="weighted",
        target_weights={"a": 1.0, "b": 3.0},
        direction="minimize",
    ) == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("minimize", 3.0), ("maximize", 1.0)],
)
def test_aggregate_target_scores_worst_uses_optimization_direction(
    direction,
    expected,
):
    assert (
        aggregate_target_scores(
            {"a": 1.0, "b": 3.0},
            aggregation="worst",
            target_weights=None,
            direction=direction,
        )
        == expected
    )


def test_aggregate_target_scores_rejects_incomplete_weights():
    with pytest.raises(ValueError, match="missing entries"):
        aggregate_target_scores(
            {"a": 1.0, "b": 3.0},
            aggregation="weighted",
            target_weights={"a": 1.0},
            direction="minimize",
        )


def test_aggregate_target_scores_rejects_unknown_aggregation():
    with pytest.raises(ValueError, match="Unknown objective_aggregation"):
        aggregate_target_scores(
            {"a": 1.0},
            aggregation="unknown",
            target_weights=None,
            direction="minimize",
        )
