from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.models.model import ModelInfo

from pftnc.train.models.base import ModelAdapter


class XGBoostAdapter(ModelAdapter):
    """XGBoost implementation of the model adapter interface."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 1,
    ) -> None:
        from xgboost import XGBRegressor

        effective_params = dict(params or {})
        effective_params["random_state"] = seed
        self.model = XGBRegressor(**effective_params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        **kwargs: Any,
    ) -> None:
        eval_set = kwargs.get("eval_set")
        verbose = kwargs.get("verbose", False)

        if eval_set:
            self.model.fit(
                X,
                y,
                eval_set=eval_set,
                verbose=verbose,
            )
        else:
            self.model.fit(X, y, verbose=verbose)

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        del kwargs
        return np.asarray(self.model.predict(X))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: Path) -> XGBoostAdapter:
        from xgboost import XGBRegressor

        self.model = XGBRegressor()
        self.model.load_model(str(path))
        return self

    def log_to_mlflow(self, name: str = "model") -> ModelInfo:
        return mlflow.xgboost.log_model(
            xgb_model=self.model,
            name=name,
            model_format="json",
        )

    @classmethod
    def load_from_mlflow(cls, uri: str) -> XGBoostAdapter:
        adapter = cls()
        adapter.model = mlflow.xgboost.load_model(uri)
        return adapter

    @classmethod
    def configure_mlflow_tracking(cls) -> None:
        mlflow.xgboost.autolog(
            log_models=False,
            log_datasets=False,
            silent=False,
        )

    def get_training_summary(self) -> dict[str, Any]:
        try:
            best_iteration = self.model.best_iteration
        except (AttributeError, ValueError):
            best_iteration = None

        try:
            best_score = self.model.best_score
        except (AttributeError, ValueError):
            best_score = None

        return {
            "best_iteration": best_iteration,
            "best_score": best_score,
        }
