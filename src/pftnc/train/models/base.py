from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mlflow.models.model import ModelInfo


class ModelAdapter(ABC):
    """Framework-neutral interface implemented by model backends."""

    model: Any

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 1,
    ) -> None:
        del params, seed

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_training_summary(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> ModelAdapter:
        raise NotImplementedError

    @abstractmethod
    def log_to_mlflow(self, name: str = "model") -> ModelInfo:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_from_mlflow(cls, uri: str) -> ModelAdapter:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def configure_mlflow_tracking(cls) -> None:
        raise NotImplementedError
