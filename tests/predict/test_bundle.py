import json
from pathlib import Path

import pytest
import yaml

from pftnc.predict.bundle import load_manifest, package_model_bundle


def write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def write_bundle_source(run: Path, dataset: Path) -> None:
    (run / "fold_2020").mkdir(parents=True)
    dataset.mkdir(parents=True)
    (run / "fold_2020" / "model.json").write_text("model")
    write_yaml(
        dataset / "config.yml",
        {
            "schema_version": 1,
            "features": {"rolling": None},
        },
    )
    write_yaml(
        dataset / "metadata.yml",
        {
            "sites": ["a"],
            "dataset_schema": {
                "date_column": "date",
                "group_column": "site",
                "sensors": {"s": ["sensor"]},
                "targets": ["target"],
            },
        },
    )
    write_yaml(
        dataset / "provenance.yml",
        {
            "schema_version": 1,
            "lineage": {
                "stage": "feature_engineering",
                "artifact": "pftnc-data/data/feature_engineered/site/v1",
            },
        },
    )
    write_yaml(
        run / "config.yml",
        {
            "schema_version": 1,
            "training": {
                "model": {"type": "xgboost"},
                "feature_selection": {},
                "target_transformations": {},
            },
        },
    )
    write_yaml(
        run / "model_io_schema.yml",
        {
            "models": {
                "__multi_target__": {
                    "input_columns": ["sensor"],
                    "output_columns": ["target"],
                }
            },
            "physical_target_columns": ["target"],
        },
    )
    write_yaml(
        run / "model_selection.yml",
        {
            "metric": "rmse",
            "direction": "minimize",
            "mode": "mean",
            "best_overall": {"fold": "fold_2020", "score": 1.0},
            "best_per_target": {"target": {"fold": "fold_2020", "score": 1.0}},
            "ensemble_weights": {
                "global": {"fold_2020": 1.0},
                "per_target": {"target": {"fold_2020": 1.0}},
            },
        },
    )
    (run / "run_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trainer_run_id": "run-2020",
                "model_mode": "multi_target",
                "dataset_version": "v1",
                "dataset_path": "pftnc-data/data/feature_engineered/site/v1",
                "folds": {
                    "fold_2020": {
                        "model_path": "fold_2020/model.json",
                        "model_uri": "models:/model-2020",
                        "val_metrics": {"target": {"rmse": 1.0}},
                    }
                },
                "final_test": {},
            }
        )
    )


def test_package_model_bundle_resolves_sibling_repositories(tmp_path: Path):
    workspace = tmp_path / "workspace"
    experiments = workspace / "pftnc-experiments"
    data_repo = workspace / "pftnc-data"
    (experiments / ".git").mkdir(parents=True)
    (data_repo / ".git").mkdir(parents=True)
    run = experiments / "outputs" / "run-2020"
    dataset = data_repo / "data" / "feature_engineered" / "site" / "v1"
    write_bundle_source(run, dataset)

    output = package_model_bundle(run, tmp_path / "bundle")

    assert (output / "models/fold_2020/model.json").read_text() == "model"
    manifest = load_manifest(output)
    assert manifest["schema_version"] == 1
    assert manifest["feature_engineering"] == {"rolling": None}
    metrics = yaml.safe_load((output / "training_metrics.yml").read_text())
    assert metrics["folds"]["fold_2020"]["validation"] == {"target": {"rmse": 1.0}}


def test_package_model_bundle_reports_missing_dataset_provenance(tmp_path: Path):
    workspace = tmp_path / "workspace"
    experiments = workspace / "pftnc-experiments"
    data_repo = workspace / "pftnc-data"
    (experiments / ".git").mkdir(parents=True)
    (data_repo / ".git").mkdir(parents=True)
    run = experiments / "outputs" / "run-2020"
    dataset = data_repo / "data" / "feature_engineered" / "site" / "v1"
    write_bundle_source(run, dataset)
    (dataset / "provenance.yml").unlink()

    output = package_model_bundle(run, tmp_path / "bundle")

    provenance = yaml.safe_load((output / "provenance.yml").read_text())
    assert provenance["dataset_lineage"] is None
    assert provenance["dataset_provenance_available"] is False


def test_package_model_bundle_rejects_legacy_experiment(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_summary.json").write_text("{}")
    write_yaml(run / "config.yml", {"schema_version": 1})
    write_yaml(run / "model_io_schema.yml", {})
    write_yaml(run / "model_selection.yml", {})

    with pytest.raises(ValueError, match="schema-version 1"):
        package_model_bundle(run, tmp_path / "bundle")
