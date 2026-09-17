from __future__ import annotations

import pandas as pd

from pftnc.train.targets.base import SingleTargetTransformation


class AbsoluteTransformationSingle(SingleTargetTransformation):
    """Identity transformation for targets learned in physical units."""

    type = "absolute"

    def transform(
        self,
        X: pd.DataFrame,
        values: pd.Series,
        target: str,
    ) -> pd.Series:
        del X, target
        return values.copy()

    def inverse(
        self,
        X: pd.DataFrame,
        values: pd.Series,
        target: str,
    ) -> pd.Series:
        del X, target
        return values.copy()
