from pathlib import Path

import pandas as pd
import pytest

from pftnc.inference_config import InferenceConfig, load_inference_config
from pftnc.predict.inference import _select_rows


def test_latest_mode_selects_last_row_per_site():
    X = pd.DataFrame({"feature": [1, 2, 3, 4]}, index=[0, 1, 2, 3])
    metadata = pd.DataFrame(
        {
            "site": ["a", "a", "b", "b"],
            "date": pd.to_datetime(
                ["2024-01-03", "2024-01-02", "2024-01-01", "2024-01-04"]
            ),
        },
        index=X.index,
    )
    config = InferenceConfig(
        model={"bundle_path": "bundle"},
        input_dataset={"path": "input.csv"},
        output_path="predictions.csv",
    )

    selected_X, selected_metadata = _select_rows(X, metadata, config, "site")

    assert selected_X["feature"].tolist() == [1, 4]
    assert selected_metadata["site"].tolist() == ["a", "b"]


def test_date_mode_is_inclusive():
    X = pd.DataFrame({"feature": [1, 2, 3]}, index=[0, 1, 2])
    metadata = pd.DataFrame(
        {
            "site": ["a", "a", "a"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        },
        index=X.index,
    )
    config = InferenceConfig(
        model={"bundle_path": "bundle"},
        input_dataset={"path": "input.csv"},
        mode="from_date",
        start_date="2024-01-02",
        output_path="predictions.csv",
    )

    selected_X, _ = _select_rows(X, metadata, config, "site")

    assert selected_X["feature"].tolist() == [2, 3]


def test_inference_config_requires_start_date(tmp_path: Path):
    path = tmp_path / "inference.yml"
    path.write_text(
        "model:\n  bundle_path: bundle\n"
        "input_dataset:\n  path: input.csv\n"
        "mode: from_date\noutput_path: predictions.csv\n"
    )

    with pytest.raises(ValueError, match="start_date"):
        load_inference_config(path)


def test_inference_config_accepts_unquoted_yaml_date(tmp_path: Path):
    path = tmp_path / "inference.yml"
    path.write_text(
        "model:\n  bundle_path: bundle\n"
        "input_dataset:\n  path: input.csv\n"
        "mode: from_date\nstart_date: 2023-12-01\n"
        "output_path: predictions.csv\n"
    )

    config = load_inference_config(path)

    assert config.start_date.isoformat() == "2023-12-01"


def test_inference_config_resolves_paths_from_workspace_root(tmp_path: Path):
    workspace = tmp_path / "workspace"
    experiments = workspace / "my-experiments"
    (experiments / ".git").mkdir(parents=True)
    config_path = experiments / "config" / "inference.yml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "model:\n  bundle_path: my-experiments/model_bundle\n"
        "input_dataset:\n  path: my-data/data/input.csv\n"
        "output_path: my-experiments/outputs/predictions.csv\n"
    )

    config = load_inference_config(config_path)

    assert config.model.bundle_path == workspace / "my-experiments" / "model_bundle"
    assert config.input_dataset.path == workspace / "my-data" / "data" / "input.csv"
    assert config.save_outputs is True
    assert config.save_index is False


def test_inference_config_accepts_output_controls(tmp_path: Path):
    path = tmp_path / "inference.yml"
    path.write_text(
        "model:\n  bundle_path: bundle\n"
        "input_dataset:\n  path: input.csv\n"
        "save_outputs: false\nsave_index: true\n"
    )

    config = load_inference_config(path)

    assert config.save_outputs is False
    assert config.save_index is True


def test_inference_config_requires_output_path_when_outputs_are_enabled():
    with pytest.raises(ValueError, match="output_path is required"):
        InferenceConfig(
            model={"bundle_path": "bundle"},
            input_dataset={"path": "input.csv"},
            save_outputs=True,
        )


def test_inference_config_rejects_output_path_when_outputs_are_disabled():
    with pytest.raises(ValueError, match="output_path must be omitted"):
        InferenceConfig(
            model={"bundle_path": "bundle"},
            input_dataset={"path": "input.csv"},
            save_outputs=False,
            output_path="predictions.csv",
        )


def test_inference_config_allows_disabled_outputs_without_output_path():
    config = InferenceConfig(
        model={"bundle_path": "bundle"},
        input_dataset={"path": "input.csv"},
        save_outputs=False,
    )

    assert config.output_path is None
