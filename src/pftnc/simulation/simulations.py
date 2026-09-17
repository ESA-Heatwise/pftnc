import numpy as np
import pandas as pd

from pftnc.dataset_simulation_config import DatasetSimConfig


class Simulation:
    """
    Base class for dataframe simulation transforms.
    """

    def apply(self, df: pd.DataFrame):
        """
        Apply the simulation transform to a dataframe.

        Args:
            df: Input dataframe.

        Returns:
            Transformed dataframe.
        """
        raise NotImplementedError


class RevisitFrequency(Simulation):
    """
    Simulate lower revisit frequency by masking observations on skipped dates.
    """

    def __init__(
        self,
        columns: list[str],
        freq_days: int,
        date_col: str,
    ):
        self.columns = columns
        self.freq_days = freq_days
        self.date_col = date_col

    def apply(self, df: pd.DataFrame):
        df = df.copy()
        unique_dates = sorted(df[self.date_col].unique())
        keep_dates = set(unique_dates[:: self.freq_days])
        mask = ~df[self.date_col].isin(keep_dates)
        df.loc[mask, self.columns] = np.nan
        return df


class CloudMask(Simulation):
    """
    Simulate cloud coverage by randomly masking observations by season.
    """

    def __init__(
        self,
        columns: list[str],
        date_col: str,
        probs: dict[str, float],
        rng: np.random.Generator,
    ):
        self.columns = columns
        self.date_col = date_col
        self.probs = probs
        self.rng = rng

    def apply(self, df: pd.DataFrame):
        df = df.copy()
        month = df[self.date_col].dt.month
        prob = np.where(
            month.isin([12, 1, 2]),
            self.probs["winter"],
            np.where(
                month.isin([6, 7, 8]), self.probs["summer"], self.probs["default"]
            ),
        )
        mask = self.rng.random(len(df)) < prob
        df.loc[mask, self.columns] = np.nan
        return df


class GaussianNoise(Simulation):
    """
    Add Gaussian noise to selected dataframe columns.
    """

    def __init__(
        self,
        columns: list[str],
        std: float,
        rng: np.random.Generator,
    ):
        self.columns = columns
        self.std = std or 0.01
        self.rng = rng

    def apply(self, df: pd.DataFrame):
        df = df.copy()
        noise = self.rng.normal(0, self.std, size=(len(df), len(self.columns)))
        df[self.columns] = df[self.columns] + noise

        return df


def apply_simulations(df: pd.DataFrame, simulations: list[Simulation]) -> pd.DataFrame:
    """
    Sequentially apply simulation transforms to a dataframe.

    Args:
        df: Input dataframe.
        simulations: List of simulation transforms.

    Returns:
        Simulated dataframe.
    """
    for s in simulations:
        df = s.apply(df)

    return df


def build_simulations(
    cfg: DatasetSimConfig,
    rng: np.random.Generator,
) -> list[Simulation]:
    """
    Build simulation pipeline from configuration.

    Args:
        cfg: Simulation configuration object.
        rng: NumPy random number generator.

    Returns:
        List of configured simulation transforms.
    """

    sim_cfg = cfg.simulation

    if sim_cfg is None:
        raise ValueError("Simulation config is required")

    transforms: list[Simulation] = []

    all_sensor_cols: list[str] = []

    for sensor_name, sensor_cfg in sim_cfg.sensors.items():
        columns = sensor_cfg.columns

        transforms.append(
            RevisitFrequency(
                columns=columns,
                freq_days=sensor_cfg.revisit_days,
                date_col=sim_cfg.date_column,
            )
        )

        if sensor_cfg.noise_std is not None:
            transforms.append(
                GaussianNoise(columns=columns, std=sensor_cfg.noise_std, rng=rng)
            )

        all_sensor_cols.extend(columns)

    transforms.append(
        CloudMask(
            columns=all_sensor_cols,
            date_col=sim_cfg.date_column,
            probs=sim_cfg.cloud_mask,
            rng=rng,
        )
    )

    return transforms
