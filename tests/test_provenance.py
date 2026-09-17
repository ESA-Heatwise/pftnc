import yaml

from pftnc.provenance import build_provenance, load_artifact_lineage, portable_config


def test_portable_config_uses_workspace_relative_paths(tmp_path):
    workspace = tmp_path / "workspace"
    path = workspace / "pftnc-data" / "data" / "input.csv"

    result = portable_config({"path": str(path), "version": "v1"}, root=workspace)

    assert result == {"path": "pftnc-data/data/input.csv", "version": "v1"}


def test_build_provenance_reuses_existing_upstream_lineage(tmp_path):
    workspace = tmp_path / "workspace"
    data_repo = workspace / "pftnc-data"
    (data_repo / ".git").mkdir(parents=True)
    simulation_base = data_repo / "data" / "simulated" / "site"
    simulation_artifact = simulation_base / "sim-v1"
    simulation_artifact.mkdir(parents=True)
    simulation_registry = simulation_base / "simulated.yaml"
    simulation_registry.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "data/simulated/site": {
                        "sim-v1": {"path": str(simulation_artifact)}
                    }
                }
            }
        )
    )
    upstream_lineage = {
        "stage": "custom_simulation",
        "artifact": "pftnc-data/data/simulated/site/sim-v1",
        "inputs": [{"stage": "raw_input", "artifact": "pftnc-data/raw.csv"}],
    }
    (simulation_artifact / "provenance.yml").write_text(
        yaml.safe_dump({"schema_version": 1, "lineage": upstream_lineage})
    )

    feature_artifact = data_repo / "data" / "feature_engineered" / "site" / "fe-v1"
    feature_artifact.mkdir(parents=True)
    payload = build_provenance(
        stage="feature_engineering",
        artifact_path=feature_artifact,
        version="fe-v1",
        input_config={
            "base_path": "pftnc-data/data/simulated/site",
            "registry_path": "pftnc-data/data/simulated/site/simulated.yaml",
            "version": "sim-v1",
        },
    )

    assert payload["lineage"]["stage"] == "feature_engineering"
    assert payload["lineage"]["inputs"] == [upstream_lineage]


def test_load_artifact_lineage_returns_none_when_provenance_is_missing(tmp_path):
    workspace = tmp_path / "workspace"
    data_repo = workspace / "pftnc-data"
    (data_repo / ".git").mkdir(parents=True)
    artifact = data_repo / "data" / "simulated" / "site" / "sim-v1"
    artifact.mkdir(parents=True)

    assert load_artifact_lineage(artifact) is None


def test_build_provenance_marks_a_missing_upstream_lineage(tmp_path):
    workspace = tmp_path / "workspace"
    data_repo = workspace / "pftnc-data"
    (data_repo / ".git").mkdir(parents=True)
    input_base = data_repo / "data" / "simulated" / "site"
    input_artifact = input_base / "sim-v1"
    input_artifact.mkdir(parents=True)
    registry = input_base / "simulated.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "data/simulated/site": {"sim-v1": {"path": str(input_artifact)}}
                }
            }
        )
    )
    output = data_repo / "data" / "feature_engineered" / "site" / "fe-v1"
    output.mkdir(parents=True)

    payload = build_provenance(
        stage="feature_engineering",
        artifact_path=output,
        version="fe-v1",
        input_config={
            "base_path": "pftnc-data/data/simulated/site",
            "registry_path": "pftnc-data/data/simulated/site/simulated.yaml",
            "version": "sim-v1",
        },
    )

    assert payload["lineage"]["inputs"] == [
        {
            "stage": "unknown_input_dataset",
            "artifact": "pftnc-data/data/simulated/site/sim-v1",
            "version": "sim-v1",
            "provenance_available": False,
        }
    ]
