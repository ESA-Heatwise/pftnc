import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

from pftnc.utils import (
    check_target_quality,
    generate_dataset_version,
    get_or_create_experiment,
    is_url,
    read_registry,
    register_dataset_version,
    resolve_path,
    save_environment_versions,
    save_feature_schema,
    show_datasets_info,
    validate_registry,
)
from pftnc.utils.utils import get_registry_key_and_paths


def test_get_or_create_experiment_returns_existing_id(monkeypatch):
    mock_experiment = SimpleNamespace(experiment_id="123")

    get_by_name = Mock(return_value=mock_experiment)
    create_experiment = Mock()

    monkeypatch.setattr(
        "mlflow.get_experiment_by_name",
        get_by_name,
    )

    monkeypatch.setattr(
        "mlflow.create_experiment",
        create_experiment,
    )

    result = get_or_create_experiment("my-exp")

    assert result == "123"

    get_by_name.assert_called_once_with("my-exp")
    create_experiment.assert_not_called()


def test_get_or_create_experiment_creates_new_experiment(monkeypatch):
    get_by_name = Mock(return_value=None)
    create_experiment = Mock(return_value="456")

    monkeypatch.setattr(
        "mlflow.get_experiment_by_name",
        get_by_name,
    )

    monkeypatch.setattr(
        "mlflow.create_experiment",
        create_experiment,
    )

    result = get_or_create_experiment("new-exp")

    assert result == "456"

    create_experiment.assert_called_once_with("new-exp")


def test_show_datasets_info_prints_expected_output(monkeypatch, capsys):
    dataset = SimpleNamespace(
        digest="abc123",
        name="dataset-name",
        profile=json.dumps({"rows": 100}),
        schema=json.dumps({"columns": ["a", "b"]}),
        source=json.dumps({"path": "/tmp/data"}),
    )

    tag = SimpleNamespace(value="training")

    dataset_input = SimpleNamespace(
        dataset=dataset,
        tags=[tag],
    )

    run = SimpleNamespace(
        inputs=SimpleNamespace(dataset_inputs=[dataset_input]),
        data=SimpleNamespace(params={"data_source": "s3"}),
    )

    get_run = Mock(return_value=run)

    monkeypatch.setattr(
        "mlflow.get_run",
        get_run,
    )

    show_datasets_info("run-123")

    captured = capsys.readouterr()

    assert "Number of datasets used: 1" in captured.out
    assert "Data Source: s3" in captured.out
    assert "dataset-name" in captured.out
    assert "training" in captured.out
    assert "abc123" in captured.out


def test_show_datasets_info_handles_missing_data_source(monkeypatch, capsys):
    dataset = SimpleNamespace(
        digest="digest",
        name="dataset",
        profile=json.dumps({}),
        schema=json.dumps({}),
        source=json.dumps({}),
    )

    dataset_input = SimpleNamespace(
        dataset=dataset,
        tags=[SimpleNamespace(value="validation")],
    )

    run = SimpleNamespace(
        inputs=SimpleNamespace(dataset_inputs=[dataset_input]),
        data=SimpleNamespace(params={}),
    )

    monkeypatch.setattr(
        "mlflow.get_run",
        Mock(return_value=run),
    )

    show_datasets_info("run-456")

    captured = capsys.readouterr()

    assert "Data Source:" not in captured.out
    assert "validation" in captured.out


def test_show_datasets_info_handles_multiple_datasets(monkeypatch, capsys):
    dataset_inputs = []

    for idx in range(2):
        dataset = SimpleNamespace(
            digest=f"digest-{idx}",
            name=f"dataset-{idx}",
            profile=json.dumps({}),
            schema=json.dumps({}),
            source=json.dumps({}),
        )

        dataset_inputs.append(
            SimpleNamespace(
                dataset=dataset,
                tags=[SimpleNamespace(value=f"role-{idx}")],
            )
        )

    run = SimpleNamespace(
        inputs=SimpleNamespace(dataset_inputs=dataset_inputs),
        data=SimpleNamespace(params={}),
    )

    monkeypatch.setattr(
        "mlflow.get_run",
        Mock(return_value=run),
    )

    show_datasets_info("run-789")

    captured = capsys.readouterr()

    assert "Number of datasets used: 2" in captured.out
    assert "dataset-0" in captured.out
    assert "dataset-1" in captured.out


def test_read_registry_returns_empty_when_file_missing(tmp_path):
    registry_path = tmp_path / "registry.yaml"

    result = read_registry(registry_path)

    assert result == {"datasets": {}}


def test_read_registry_returns_empty_when_yaml_is_none(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("null")

    result = read_registry(registry_path)

    assert result == {"datasets": {}}


def test_read_registry_reads_valid_yaml(tmp_path):
    registry_path = tmp_path / "registry.yaml"

    expected = {"datasets": {"dataset_a": {"v1": {}}}}

    registry_path.write_text(yaml.safe_dump(expected))

    result = read_registry(registry_path)

    assert result == expected


def test_validate_registry_passes_when_versions_match(tmp_path):
    base_path = tmp_path / "dataset"
    base_path.mkdir()

    (base_path / "v1").mkdir()
    (base_path / "v2").mkdir()

    registry_path = tmp_path / "registry.yaml"

    base_key, _ = get_registry_key_and_paths(
        registry_path,
        base_path,
    )

    registry = {
        "datasets": {
            base_key: {
                "v1": {},
                "v2": {},
            }
        }
    }

    registry_path.write_text(yaml.safe_dump(registry))

    validate_registry(registry_path, base_path)


def test_validate_registry_raises_on_mismatch(tmp_path):
    base_path = tmp_path / "dataset"
    base_path.mkdir()

    (base_path / "v1").mkdir()

    registry_path = tmp_path / "registry.yaml"

    base_key, _ = get_registry_key_and_paths(
        registry_path,
        base_path,
    )

    registry = {
        "datasets": {
            base_key: {
                "v1": {},
                "v2": {},
            }
        }
    }

    registry_path.write_text(yaml.safe_dump(registry))

    with pytest.raises(RuntimeError, match="Dataset registry mismatch"):
        validate_registry(registry_path, base_path)


def test_validate_registry_handles_missing_base_path(tmp_path):
    base_path = tmp_path / "missing_dataset"
    registry_path = tmp_path / "registry.yaml"
    base_key, _ = get_registry_key_and_paths(
        registry_path,
        base_path,
    )
    registry = {
        "datasets": {
            base_key: {
                "v1": {},
            }
        }
    }
    registry_path.write_text(yaml.safe_dump(registry))
    with pytest.raises(RuntimeError):
        validate_registry(registry_path, base_path)


def test_register_dataset_version_creates_registry(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    base_path = tmp_path / "datasets"

    register_dataset_version(
        registry_path=registry_path,
        base_path=base_path,
        description="test dataset",
        version="v1",
    )

    assert registry_path.exists()

    data = yaml.safe_load(registry_path.read_text())

    base_key, _ = get_registry_key_and_paths(
        registry_path,
        base_path,
    )

    assert "datasets" in data
    assert base_key in data["datasets"]
    assert "v1" in data["datasets"][base_key]


def test_register_dataset_version_writes_expected_fields(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    base_path = tmp_path / "datasets"

    register_dataset_version(
        registry_path=registry_path,
        base_path=base_path,
        description="my dataset",
        version="v123",
    )

    data = yaml.safe_load(registry_path.read_text())

    base_key, _ = get_registry_key_and_paths(
        registry_path,
        base_path,
    )

    entry = data["datasets"][base_key]["v123"]

    assert entry["description"] == "my dataset"
    assert entry["path"] == f"{base_key}/v123"
    assert "created" in entry


def test_register_dataset_version_appends_existing_registry(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    base_path = tmp_path / "datasets"
    base_key, _ = get_registry_key_and_paths(
        registry_path,
        base_path,
    )

    initial = {
        "datasets": {
            base_key: {
                "v1": {
                    "path": "x",
                    "description": "old",
                    "created": "y",
                }
            }
        }
    }

    registry_path.write_text(yaml.safe_dump(initial))

    register_dataset_version(
        registry_path=registry_path,
        base_path=base_path,
        description="new dataset",
        version="v2",
    )

    data = yaml.safe_load(registry_path.read_text())
    print(data)

    assert "v1" in data["datasets"][base_key]
    assert "v2" in data["datasets"][base_key]


def test_register_dataset_version_creates_datasets_key(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry.yaml"
    base_path = tmp_path / "datasets"

    monkeypatch.setattr("pftnc.utils.utils.read_registry", lambda _: {})

    register_dataset_version(
        registry_path=registry_path,
        base_path=base_path,
        description="desc",
        version="v1",
    )

    data = yaml.safe_load(registry_path.read_text())

    assert "datasets" in data


def test_generate_dataset_version_has_expected_format():
    class MockCfg:
        def model_dump(self, mode="json"):
            return {
                "a": 1,
                "b": 2,
            }

    version = generate_dataset_version(MockCfg(), "siteA")

    pattern = r"siteA_\d{8}T\d{6}Z?_?\w*_\w{8}_\w{6}"

    assert re.match(pattern, version)


def test_generate_dataset_version_is_unique():
    class MockCfg:
        def model_dump(self, mode="json"):
            return {
                "a": 1,
                "b": 2,
            }

    version1 = generate_dataset_version(MockCfg(), "siteA")
    version2 = generate_dataset_version(MockCfg(), "siteA")

    assert version1 != version2


def test_generate_dataset_version_changes_when_config_changes():
    class CfgA:
        def model_dump(self, mode="json"):
            return {"x": 1}

    class CfgB:
        def model_dump(self, mode="json"):
            return {"x": 2}

    version1 = generate_dataset_version(CfgA(), "siteA")
    version2 = generate_dataset_version(CfgB(), "siteA")

    hash1 = version1.split("_")[2]
    hash2 = version2.split("_")[2]

    assert hash1 != hash2


def test_save_feature_schema_writes_yaml(tmp_path):
    df = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        }
    )

    save_feature_schema(df, tmp_path)

    schema_path = tmp_path / "feature_schema.yml"

    assert schema_path.exists()


def test_save_feature_schema_contains_expected_content(tmp_path):
    df = pd.DataFrame(
        {
            "feature1": [1],
            "feature2": [2],
            "feature3": [3],
        }
    )

    save_feature_schema(df, tmp_path)

    schema = yaml.safe_load((tmp_path / "feature_schema.yml").read_text())

    assert schema["columns"] == [
        "feature1",
        "feature2",
        "feature3",
    ]

    assert schema["n_features"] == 3


def test_save_feature_schema_handles_empty_dataframe(tmp_path):
    df = pd.DataFrame()

    save_feature_schema(df, tmp_path)

    schema = yaml.safe_load((tmp_path / "feature_schema.yml").read_text())

    assert schema["columns"] == []
    assert schema["n_features"] == 0


def test_check_target_quality_passes_for_valid_targets(capsys):
    df = pd.DataFrame({"target": [1.0, 2.0, 3.0]})

    check_target_quality(df, ["target"])

    captured = capsys.readouterr()

    assert "Target quality OK" in captured.out


def test_check_target_quality_raises_on_nan():
    df = pd.DataFrame({"target": [1.0, None, 3.0]})

    with pytest.raises(ValueError, match="nan"):
        check_target_quality(df, ["target"])


def test_check_target_quality_raises_on_negative():
    df = pd.DataFrame({"target": [1.0, -2.0, 3.0]})

    with pytest.raises(ValueError, match="negative"):
        check_target_quality(df, ["target"])


def test_check_target_quality_reports_multiple_issues():
    df = pd.DataFrame(
        {
            "target1": [1.0, None],
            "target2": [-1.0, 2.0],
        }
    )

    with pytest.raises(ValueError) as exc:
        check_target_quality(df, ["target1", "target2"])

    message = str(exc.value)

    assert "nan" in message
    assert "negative" in message


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 1.0, 2.0],
        [100.0, 0.0, 50.0],
        [1e-9, 5.0, 10.0],
    ],
)
def test_check_target_quality_accepts_valid_values(values):
    df = pd.DataFrame({"target": values})

    check_target_quality(df, ["target"])


def test_check_target_quality_reports_correct_column_names():
    df = pd.DataFrame(
        {
            "a_": [1.0, None],
            "b_": [1.0, 2.0],
            "c_": [-1.0, 3.0],
        }
    )

    with pytest.raises(ValueError) as exc:
        check_target_quality(df, ["a_", "b_", "c_"])

    message = str(exc.value)

    assert "a_" in message
    assert "c_" in message
    assert "b_" not in message


def test_save_environment_versions_writes_output(
    tmp_path,
    monkeypatch,
):
    output_path = tmp_path / "versions.txt"

    monkeypatch.setattr(
        "pftnc.utils.utils.importlib.metadata.version",
        lambda name: "1.2.3",
    )
    monkeypatch.setattr(
        "pftnc.utils.utils.importlib.metadata.distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": "numpy"}, version="2.2.0", name="numpy"),
            SimpleNamespace(
                metadata={"Name": "pandas"}, version="2.3.1", name="pandas"
            ),
        ],
    )

    save_environment_versions(output_path)

    assert output_path.exists()

    content = output_path.read_text()

    assert "pftnc==1.2.3" in content
    assert "numpy==2.2.0" in content
    assert "pandas==2.3.1" in content


@pytest.mark.parametrize(
    "value",
    [
        "http://example.com/data.parquet",
        "https://example.com/data.parquet",
        "https://storage.googleapis.com/file.csv",
    ],
)
def test_is_url_returns_true_for_valid_urls(value):
    assert is_url(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "/tmp/data.csv",
        "../data/file.csv",
        "relative/path.csv",
        "C:/data/file.csv",
    ],
)
def test_is_url_returns_false_for_filesystem_paths(value):
    assert is_url(value) is False


def test_resolve_path_returns_absolute_path_unchanged(
    tmp_path,
):
    absolute_path = tmp_path / "data.csv"

    result = resolve_path(
        absolute_path,
        tmp_path,
    )

    assert Path(result) == absolute_path.resolve()


def test_resolve_path_resolves_relative_paths(
    tmp_path,
):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    result = resolve_path(
        "../data/file.csv",
        config_dir,
    )

    expected = (config_dir / "../data/file.csv").resolve()

    assert Path(result) == expected


def test_resolve_path_keeps_urls_unchanged():
    url = "https://example.com/data.parquet"

    result = resolve_path(
        url,
        Path("/tmp"),
    )

    assert result == url


def test_resolve_path_handles_string_paths(
    tmp_path,
):
    result = resolve_path(
        "data/file.csv",
        tmp_path,
    )

    expected = (tmp_path / "data/file.csv").resolve()

    assert Path(result) == expected


def test_resolve_path_handles_path_objects(
    tmp_path,
):
    relative = Path("data/file.csv")

    result = resolve_path(
        relative,
        tmp_path,
    )

    expected = (tmp_path / relative).resolve()

    assert Path(result) == expected
