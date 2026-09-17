from __future__ import annotations

import numpy as np
import pandas as pd

from pftnc.train.targets.base import SingleTargetTransformation


class LogDeltaTransformationSingle(SingleTargetTransformation):
    """Learn the log-space delta from a non-negative reference feature."""

    type = "log_delta"

    def __init__(self, reference_feature: str) -> None:
        self.reference_feature = reference_feature

    def transform(
        self,
        X: pd.DataFrame,
        values: pd.Series,
        target: str,
    ) -> pd.Series:
        actual = self._validated_values(values, target)
        reference = self._reference(X, target, actual.index)
        return np.log1p(actual) - np.log1p(reference)

    def inverse(
        self,
        X: pd.DataFrame,
        values: pd.Series,
        target: str,
    ) -> pd.Series:
        model_values = pd.to_numeric(values, errors="coerce")
        invalid = model_values.isna() | ~np.isfinite(model_values)
        if invalid.any():
            raise ValueError(
                f"Model output '{target}' contains {int(invalid.sum())} "
                "missing or non-finite values."
            )

        reference = self._reference(X, target, model_values.index)
        return np.expm1(np.log1p(reference) + model_values.astype(float)).clip(lower=0)

    def _reference(
        self,
        X: pd.DataFrame,
        target: str,
        expected_index: pd.Index,
    ) -> pd.Series:
        if self.reference_feature not in X.columns:
            raise ValueError(
                f"Reference feature '{self.reference_feature}' "
                f"for target '{target}' is missing from X."
            )

        reference = pd.to_numeric(
            X[self.reference_feature],
            errors="coerce",
        ).reindex(expected_index)

        invalid = reference.isna() | ~np.isfinite(reference)
        if invalid.any():
            raise ValueError(
                f"Reference feature '{self.reference_feature}' "
                f"contains {int(invalid.sum())} missing or non-finite "
                f"values for target '{target}'."
            )
        if (reference < 0).any():
            raise ValueError(
                f"Reference feature '{self.reference_feature}' "
                f"contains negative values for target '{target}'."
            )
        return reference.astype(float)

    @staticmethod
    def _validated_values(values: pd.Series, target: str) -> pd.Series:
        actual = pd.to_numeric(values, errors="coerce")
        invalid = actual.isna() | ~np.isfinite(actual)
        if invalid.any():
            raise ValueError(
                f"Target '{target}' contains {int(invalid.sum())} "
                "missing or non-finite values."
            )
        if (actual < 0).any():
            raise ValueError(f"Target '{target}' contains negative values.")
        return actual.astype(float)
