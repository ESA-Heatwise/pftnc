import pytest

from pftnc.preprocess.preprocess import preprocess_dataset


def test_preprocess_dataset_runs_pipeline_and_registers_artifact(
    preprocess_config,
    prepared_training_dataset,
    monkeypatch,
):
    calls = []
    dataset_path = preprocess_config.output_dataset.storage.base_path / "v1"

    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.validate_registry",
        lambda *args: calls.append("validate"),
    )
    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.generate_dataset_version",
        lambda *args: "v1",
    )
    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.read_input_dataset",
        lambda storage: calls.append("read") or object(),
    )
    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.engineer_features",
        lambda *args, **kwargs: calls.append("engineer") or object(),
    )
    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.prepare_feature_engineered_dataset",
        lambda *args, **kwargs: calls.append("prepare") or prepared_training_dataset,
    )

    def write_folds(prepared, output_path, **kwargs):
        calls.append("folds")
        output_path.mkdir(parents=True)
        return {"fold_2021": {}}

    monkeypatch.setattr("pftnc.preprocess.preprocess.write_year_folds", write_folds)
    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.save_feature_schema",
        lambda *args: calls.append("schema"),
    )
    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.save_environment_versions",
        lambda *args: calls.append("versions"),
    )
    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.register_dataset_version",
        lambda *args: calls.append("register"),
    )

    result = preprocess_dataset(preprocess_config)

    assert result.version == "v1"
    assert result.base_path == preprocess_config.output_dataset.storage.base_path
    assert calls == [
        "validate",
        "read",
        "engineer",
        "prepare",
        "folds",
        "schema",
        "versions",
        "register",
    ]
    assert (dataset_path / "config.yml").is_file()
    assert (dataset_path / "metadata.yml").is_file()


def test_preprocess_dataset_removes_partial_artifact_after_failure(
    preprocess_config,
    prepared_training_dataset,
    monkeypatch,
):
    dataset_path = preprocess_config.output_dataset.storage.base_path / "v1"
    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.validate_registry", lambda *args: None
    )
    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.generate_dataset_version", lambda *args: "v1"
    )
    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.read_input_dataset", lambda storage: object()
    )
    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.engineer_features",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.prepare_feature_engineered_dataset",
        lambda *args, **kwargs: prepared_training_dataset,
    )

    def fail_after_creating_output(prepared, output_path, **kwargs):
        output_path.mkdir(parents=True)
        raise RuntimeError("fold generation failed")

    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.write_year_folds",
        fail_after_creating_output,
    )

    with pytest.raises(RuntimeError, match="fold generation failed"):
        preprocess_dataset(preprocess_config)

    assert not dataset_path.exists()


def test_preprocess_dataset_requires_split_configuration(
    preprocess_config,
    monkeypatch,
):
    preprocess_config.output_dataset.metadata.splits = None
    monkeypatch.setattr(
        "pftnc.preprocess.preprocess.validate_registry", lambda *args: None
    )

    with pytest.raises(ValueError, match="splits config"):
        preprocess_dataset(preprocess_config)
