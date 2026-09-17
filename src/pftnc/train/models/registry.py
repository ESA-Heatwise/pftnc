from __future__ import annotations

from pftnc.train.models.base import ModelAdapter
from pftnc.train.models.xgboost import XGBoostAdapter

MODEL_ADAPTER_REGISTRY: dict[str, type[ModelAdapter]] = {
    "xgboost": XGBoostAdapter,
}


def get_model_adapter(model_type: str) -> type[ModelAdapter]:
    try:
        return MODEL_ADAPTER_REGISTRY[model_type]
    except KeyError as error:
        supported = sorted(MODEL_ADAPTER_REGISTRY)
        raise ValueError(
            f"Unsupported model type '{model_type}'. Supported model types: {supported}"
        ) from error
