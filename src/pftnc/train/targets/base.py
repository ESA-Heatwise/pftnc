from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class SingleTargetTransformation(ABC):
    """Transform one physical target to and from model-output space."""

    type: str

    @abstractmethod
    def transform(
        self,
        X: pd.DataFrame,
        values: pd.Series,
        target: str,
    ) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def inverse(
        self,
        X: pd.DataFrame,
        values: pd.Series,
        target: str,
    ) -> pd.Series:
        raise NotImplementedError

    def output_names(
        self,
        targets: list[str],
    ) -> list[str]:
        """Return model-output names for physical targets."""
        if self.type == "absolute":
            return list(targets)

        return [f"{target}_{self.type}" for target in targets]


class JointTargetTransformation(ABC):
    """Base class for transformations involving multiple targets."""

    type: str

    @abstractmethod
    def transform(
        self,
        X: pd.DataFrame,
        values: pd.DataFrame,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def inverse(
        self,
        X: pd.DataFrame,
        values: pd.DataFrame,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def output_names(self, targets: list[str]) -> list[str]:
        """Return the columns produced in model-output space."""
        raise NotImplementedError
