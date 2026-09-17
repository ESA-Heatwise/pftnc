from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from pftnc.cli import app

runner = CliRunner()


def test_create_config_copies_packaged_templates(tmp_path: Path) -> None:
    output = tmp_path / "config"

    result = runner.invoke(app, ["create-config", "--output-path", str(output)])

    assert result.exit_code == 0
    assert sorted(path.name for path in output.iterdir()) == [
        "config.yml",
        "dataset_simulation_config.yml",
        "inference.yml",
    ]


def test_create_config_copies_single_yaml_file(tmp_path: Path) -> None:
    source = tmp_path / "custom-name.yaml"
    output = tmp_path / "output"
    source.write_text("key: value\n")

    result = runner.invoke(
        app,
        [
            "create-config",
            "--source-path",
            str(source),
            "--output-path",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert (output / source.name).read_text() == "key: value\n"


def test_create_config_copies_supported_configs_from_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "config.yml").write_text("config: true\n")
    (source / "metadata.yml").write_text("metadata: true\n")
    (source / "metrics.yaml").write_text("metrics: true\n")

    result = runner.invoke(
        app,
        [
            "create-config",
            "--source-path",
            str(source),
            "--output-path",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert [path.name for path in output.iterdir()] == ["config.yml"]


def test_create_config_rejects_empty_source_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    result = runner.invoke(app, ["create-config", "--source-path", str(source)])

    assert result.exit_code != 0
    assert "No YAML config files found" in result.stderr


def test_simulate_command_loads_config_and_runs_pipeline(tmp_path, monkeypatch) -> None:
    config = object()
    calls = []
    output = tmp_path / "simulated"

    monkeypatch.setattr(
        "pftnc.dataset_simulation_config.load_dataset_gen_config",
        lambda path: calls.append(("load", path)) or config,
    )
    monkeypatch.setattr(
        "pftnc.simulation.simulate.simulate_dataset",
        lambda value: calls.append(("simulate", value)) or output,
    )

    result = runner.invoke(app, ["simulate", str(tmp_path / "simulation.yml")])

    assert result.exit_code == 0
    assert calls == [("load", tmp_path / "simulation.yml"), ("simulate", config)]
    assert str(output) in result.stdout


def test_preprocess_command_loads_config_and_runs_pipeline(
    tmp_path, monkeypatch
) -> None:
    feature_config = object()
    artifact = SimpleNamespace(base_path=tmp_path, version="v1")
    calls = []
    config_path = tmp_path / "config.yml"

    monkeypatch.setattr(
        "pftnc.config.load_config",
        lambda path: (
            calls.append(("load", path))
            or SimpleNamespace(feature_engineering=feature_config)
        ),
    )
    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.preprocess_dataset",
        lambda value: calls.append(("preprocess", value)) or artifact,
    )

    result = runner.invoke(app, ["preprocess", str(config_path)])

    assert result.exit_code == 0
    assert calls == [("load", config_path), ("preprocess", feature_config)]
    assert "v1" in result.stdout


def test_train_command_rejects_missing_config(tmp_path) -> None:
    result = runner.invoke(app, ["train", str(tmp_path / "missing.yml")])

    assert result.exit_code != 0
    assert "Config file not found" in result.stderr


def test_train_command_loads_config_and_runs_pipeline(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("schema_version: 1\n")
    config = object()
    calls = []

    monkeypatch.setattr(
        "pftnc.config.load_config",
        lambda path: calls.append(("load", path)) or config,
    )
    monkeypatch.setattr(
        "pftnc.train.train.run_training",
        lambda value, path: calls.append(("train", value, path)),
    )

    result = runner.invoke(app, ["train", str(config_path)])

    assert result.exit_code == 0
    assert calls == [("load", config_path), ("train", config, config_path)]


def test_package_model_command_creates_bundle(tmp_path, monkeypatch) -> None:
    training_run = tmp_path / "run"
    output = tmp_path / "bundle"
    bundle = output / "manifest.yml"

    monkeypatch.setattr(
        "pftnc.predict.bundle.package_model_bundle",
        lambda run_path, output_path: bundle,
    )

    result = runner.invoke(
        app,
        [
            "package-model",
            "--training-run-path",
            str(training_run),
            "--output-path",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert str(bundle) in result.stdout


def test_infer_command_runs_inference_without_saving(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "inference.yml"
    config = SimpleNamespace(
        output_path=tmp_path / "predictions.csv",
        save_outputs=False,
    )
    calls = []

    monkeypatch.setattr(
        "pftnc.inference_config.load_inference_config",
        lambda path: calls.append(("load", path)) or config,
    )
    monkeypatch.setattr(
        "pftnc.predict.inference.run_inference",
        lambda value: calls.append(("infer", value)) or [1, 2, 3],
    )

    result = runner.invoke(app, ["infer", str(config_path)])

    assert result.exit_code == 0
    assert calls == [("load", config_path), ("infer", config)]
    assert "Rows: 3" in result.stdout
    assert "save_outputs=false" in result.stdout
