from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pftnc.train.models.manager import ModelManager


def test_model_manager_fits_and_saves_multi_target_bundle(
    model_manager,
    fake_adapter,
    features,
    model_space_targets,
    tmp_path,
):
    bundle = model_manager.fit(
        features,
        features,
        model_space_targets,
        model_space_targets,
        params={"depth": 3},
        output_dir=tmp_path,
    )

    assert bundle.is_multi_target
    assert bundle.feature_columns == tuple(features.columns)
    assert bundle.models["__multi_target__"].uri == "models:/model"
    assert (
        bundle.models["__multi_target__"].local_path
        == (tmp_path / "model.json").resolve()
    )
    assert fake_adapter.created[0].fit_calls[0][2]["verbose"] == 10


def test_model_manager_predicts_multi_target_bundle(
    model_manager,
    fake_adapter,
    features,
    model_space_targets,
):
    fake_adapter.predictions_by_uri["models:/model"] = np.ones(
        (len(features), len(model_space_targets.columns))
    )
    bundle = SimpleNamespace(
        is_multi_target=True,
        models={"__multi_target__": SimpleNamespace(uri="models:/model")},
    )

    predictions, loaded = model_manager.predict(
        bundle,
        features,
        pd.Index(model_space_targets.columns),
    )

    assert predictions.shape == model_space_targets.shape
    assert predictions.index.equals(features.index)
    assert "__multi_target__" in loaded


def test_model_manager_rejects_wrong_prediction_shape(
    model_manager,
    fake_adapter,
    features,
    model_space_targets,
):
    fake_adapter.predictions_by_uri["models:/model"] = np.ones((len(features), 1))
    bundle = SimpleNamespace(
        is_multi_target=True,
        models={"__multi_target__": SimpleNamespace(uri="models:/model")},
    )

    with pytest.raises(ValueError, match="shape does not match"):
        model_manager.predict(
            bundle,
            features,
            pd.Index(model_space_targets.columns),
        )


@pytest.mark.parametrize(
    "bad_config",
    [
        SimpleNamespace(
            mode="unknown", params={}, params_by_target=None, save_path=Path("m.json")
        ),
        SimpleNamespace(
            mode="multi_target",
            params=None,
            params_by_target=None,
            save_path=Path("m.json"),
        ),
        SimpleNamespace(
            mode="multi_target",
            params={},
            params_by_target={},
            save_path=Path("m.json"),
        ),
    ],
)
def test_model_manager_rejects_invalid_multi_target_configuration(
    fake_adapter,
    target_transform_manager,
    spy_tracker,
    bad_config,
):
    with pytest.raises(ValueError):
        ModelManager(
            fake_adapter,
            bad_config,
            seed=1,
            tracker=spy_tracker,
            model_outputs=target_transform_manager.model_outputs,
        )


def test_model_manager_safe_name_and_parameter_merge():
    assert ModelManager.safe_name("target / value") == "target_value"
    assert ModelManager.safe_name("...") == "model_output"
    assert ModelManager.merge_params({"a": 1}, {"b": 2, "a": 3}) == {
        "a": 3,
        "b": 2,
    }
