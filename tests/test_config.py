import pytest
import yaml
from pydantic import ValidationError

from pftnc.config import AppConfig, load_config


def test_load_valid_config(config_path):
    config = load_config(config_path)

    assert isinstance(config, AppConfig)
    assert config.project.seed == 42
    assert config.training.experiment_name == "test_experiment"


def test_invalid_config_raises_validation_error(
    tmp_path,
    app_config,
):
    app_config["project"]["seed"] = "not_an_int"

    path = tmp_path / "invalid_config.yaml"

    with open(path, "w") as f:
        yaml.safe_dump(app_config, f)

    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_resolves_paths_from_workspace_root(tmp_path, app_config):
    workspace = tmp_path / "workspace"
    data_repo = workspace / "my-data"
    experiments_repo = workspace / "my-experiments"
    (data_repo / ".git").mkdir(parents=True)
    config_path = experiments_repo / "config" / "config.yml"
    (experiments_repo / ".git").mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    app_config["feature_engineering"]["input_dataset"] = {
        "base_path": "my-data/data/simulated/site",
        "registry_path": "my-data/data/simulated/site/registry.yml",
        "version": "v1",
    }
    app_config["training"]["output_dir"] = "my-experiments/outputs"
    app_config["training"]["input_dataset"] = {
        "base_path": "my-data/data/feature_engineered/site",
        "registry_path": "my-data/data/feature_engineered/site/registry.yml",
        "version": "v1",
    }
    config_path.write_text(yaml.safe_dump(app_config))

    config = load_config(config_path)

    assert config.training.output_dir == workspace / "my-experiments" / "outputs"
    assert config.training.input_dataset.base_path == (
        workspace / "my-data" / "data" / "feature_engineered" / "site"
    )
