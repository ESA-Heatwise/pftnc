import pandas as pd
import pytest

from pftnc.simulation.simulate import simulate_dataset
from tests.simulation.config import make_simulation_cfg


def test_simulate_dataset_runs_pipeline(
    tmp_path,
    monkeypatch,
):
    cfg = make_simulation_cfg(tmp_path)

    raw_df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2024-01-01",
                ]
            ),
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                ]
            ),
            "sensor1": [1.0],
        }
    )

    monkeypatch.setattr(
        "pftnc.simulation.simulate.validate_registry",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        "pftnc.simulation.simulate.register_dataset_version",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        "pftnc.simulation.simulate.save_feature_schema",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        "pftnc.simulation.simulate.save_environment_versions",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        "pftnc.simulation.simulate.generate_dataset_version",
        lambda *args, **kwargs: "v1",
    )

    result = simulate_dataset(
        cfg,
        raw_df=raw_df,
    )

    assert result.exists()

    assert (result / "simulated.parquet").exists()

    assert (result / "config.yml").exists()


def test_simulate_dataset_raises_when_simulation_missing():
    class Obj:
        pass

    cfg = Obj()

    cfg.simulation = None

    with pytest.raises(ValueError, match="Simulation config"):
        simulate_dataset(cfg)
