from types import SimpleNamespace


def make_simulation_cfg(tmp_path):
    (tmp_path / ".git").mkdir()
    input_path = tmp_path / "input.csv"
    input_path.write_text("time,date,sensor1\n2024-01-01,2024-01-01,1\n")

    sensor = SimpleNamespace(
        columns=["sensor1"],
        revisit_days=2,
        noise_std=0.1,
    )

    simulation = SimpleNamespace(
        input_path=input_path,
        base_path=tmp_path / "datasets",
        registry_path=tmp_path / "registry.yaml",
        description="test simulation",
        site_name="elbe",
        date_column="date",
        cloud_mask={
            "winter": 1.0,
            "summer": 0.0,
            "default": 0.0,
        },
        sensors={
            "s2": sensor,
        },
    )

    project = SimpleNamespace(
        seed=42,
    )

    cfg = SimpleNamespace(
        project=project,
        simulation=simulation,
        model_dump=lambda mode="json": {
            "simulation": {"input_path": str(simulation.input_path)},
        },
    )

    return cfg
