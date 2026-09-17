import pandas as pd
import pytest


def test_single_target_manager_transforms_and_inverts_each_target(
    single_target_transform_manager,
):
    X = pd.DataFrame({"reference": [1.0, 2.0]}, index=[1, 2])
    physical = pd.DataFrame(
        {"target_a": [3.0, 4.0], "target_b": [2.0, 6.0]},
        index=X.index,
    )

    model_values = single_target_transform_manager.transform_targets(X, physical)
    reconstructed = single_target_transform_manager.inverse_predictions(X, model_values)

    assert single_target_transform_manager.model_outputs == [
        "target_a",
        "target_b_log_delta",
    ]
    pd.testing.assert_frame_equal(
        reconstructed,
        physical,
        check_dtype=False,
        rtol=1e-10,
        atol=1e-10,
    )
    assert (
        single_target_transform_manager.physical_target_for_output("target_b_log_delta")
        == "target_b"
    )


def test_log_delta_manager_rejects_missing_reference_feature(
    single_target_transform_manager,
):
    X = pd.DataFrame(index=[1])
    values = pd.DataFrame({"target_a": [1.0], "target_b": [2.0]}, index=[1])

    with pytest.raises(ValueError, match="Reference feature"):
        single_target_transform_manager.transform_targets(X, values)


def test_single_target_manager_rejects_unknown_output(
    single_target_transform_manager,
):
    with pytest.raises(ValueError, match="does not map"):
        single_target_transform_manager.physical_target_for_output("missing")
