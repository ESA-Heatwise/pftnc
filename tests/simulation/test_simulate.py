import numpy as np
import pandas as pd
import pytest

from pftnc.simulation.simulations import (
    CloudMask,
    GaussianNoise,
    RevisitFrequency,
    Simulation,
    apply_simulations,
    build_simulations,
)
from tests.simulation.config import make_simulation_cfg


def test_simulation_base_class_raises():
    sim = Simulation()

    with pytest.raises(NotImplementedError):
        sim.apply(pd.DataFrame())


def test_revisit_frequency_masks_skipped_dates():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )

    sim = RevisitFrequency(
        columns=["x"],
        freq_days=2,
        date_col="date",
    )

    result = sim.apply(df)

    assert pd.isna(result.loc[1, "x"])
    assert not pd.isna(result.loc[0, "x"])


def test_cloud_mask_masks_winter_values():
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "x": [1.0, 2.0],
        }
    )

    sim = CloudMask(
        columns=["x"],
        date_col="date",
        probs={
            "winter": 1.0,
            "summer": 0.0,
            "default": 0.0,
        },
        rng=rng,
    )

    result = sim.apply(df)

    assert result["x"].isna().all()


def test_gaussian_noise_changes_values():
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "x": [1.0, 1.0, 1.0],
        }
    )

    sim = GaussianNoise(
        columns=["x"],
        std=0.5,
        rng=rng,
    )

    result = sim.apply(df)

    assert not result["x"].equals(df["x"])


def test_apply_simulations_runs_all_transforms():
    class AddOne(Simulation):
        def apply(self, df):
            df = df.copy()
            df["x"] += 1
            return df

    df = pd.DataFrame(
        {
            "x": [1],
        }
    )

    result = apply_simulations(
        df,
        [AddOne(), AddOne()],
    )

    assert result["x"].iloc[0] == 3


def test_build_simulations_returns_expected_transforms(tmp_path):
    cfg = make_simulation_cfg(tmp_path)

    rng = np.random.default_rng(42)

    transforms = build_simulations(cfg, rng)

    assert len(transforms) == 3

    assert isinstance(transforms[0], RevisitFrequency)
    assert isinstance(transforms[1], GaussianNoise)
    assert isinstance(transforms[2], CloudMask)


def test_build_simulations_raises_when_simulation_missing():
    class Obj:
        pass

    cfg = Obj()

    cfg.simulation = None

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError, match="Simulation config"):
        build_simulations(cfg, rng)
