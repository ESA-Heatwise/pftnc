import pandas as pd
import pytest


def test_log_fraction_manager_round_trip(
    target_transform_manager,
    features,
    physical_targets,
    target_names,
):
    model_values = target_transform_manager.transform_targets(
        features,
        physical_targets,
    )

    reconstructed = target_transform_manager.inverse_predictions(
        features,
        model_values,
    )

    pd.testing.assert_frame_equal(
        reconstructed,
        physical_targets,
        check_dtype=False,
        rtol=1e-10,
        atol=1e-10,
    )
    assert target_transform_manager.physical_targets == target_names
    assert target_transform_manager.model_outputs == [
        "total_ug_per_l",
        "log_fractions_diatoms_ug_per_l",
        "log_fractions_cyanobacterial_ug_per_l",
    ]


def test_log_fraction_manager_requires_all_model_outputs(
    target_transform_manager,
    features,
    model_space_targets,
):
    incomplete = model_space_targets.drop(columns=["total_ug_per_l"])

    with pytest.raises(ValueError, match="Missing log-fraction model outputs"):
        target_transform_manager.inverse_predictions(features, incomplete)


def test_physical_truth_returns_only_configured_targets(
    target_transform_manager,
    physical_targets,
):
    values = physical_targets.assign(unused=1.0)

    result = target_transform_manager.physical_truth(values)

    assert result.equals(physical_targets)
