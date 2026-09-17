from __future__ import annotations

import numpy as np
import pandas as pd

from pftnc.train.targets.base import JointTargetTransformation


class LogFractionsTransformation(JointTargetTransformation):
    """Transform concentrations into total and log-fraction outputs."""

    type = "log_fractions"

    def __init__(
        self,
        *,
        original_targets: list[str],
        denominator: str,
        epsilon: float,
    ) -> None:
        if not original_targets:
            raise ValueError("At least one original target is required.")

        if len(set(original_targets)) != len(original_targets):
            raise ValueError("Original target names must be unique.")

        if denominator not in original_targets:
            raise ValueError(
                f"Denominator '{denominator}' is not present in "
                f"original targets: {original_targets}."
            )

        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError(
                "Log-fractions epsilon must be finite and greater than zero."
            )

        self.original_targets = list(original_targets)
        self.denominator = denominator
        self.epsilon = float(epsilon)

    @property
    def numerators(self) -> list[str]:
        return [
            target for target in self.original_targets if target != self.denominator
        ]

    def output_names(
        self,
        targets: list[str],
    ) -> list[str]:
        if targets != self.original_targets:
            raise ValueError(
                "Log fractions require the complete physical target "
                f"bundle in this order: {self.original_targets}."
            )

        return [
            "total_ug_per_l",
            *[f"log_fractions_{target}" for target in self.numerators],
        ]

    def transform(
        self,
        X: pd.DataFrame,
        values: pd.DataFrame,
    ) -> pd.DataFrame:
        del X

        physical = self._validated_physical_values(values)
        total = physical.sum(axis=1)

        transformed = pd.DataFrame(
            index=physical.index,
        )
        transformed["total_ug_per_l"] = total

        for target in self.numerators:
            output_name = f"log_fractions_{target}"

            transformed[output_name] = np.log(
                (physical[target] + self.epsilon) / (total + self.epsilon)
            )

        transformed = transformed[self.output_names(self.original_targets)]

        self._validate_finite(
            transformed,
            context="transformed log-fraction labels",
        )

        return transformed

    def inverse(
        self,
        X: pd.DataFrame,
        values: pd.DataFrame,
    ) -> pd.DataFrame:
        del X

        model_values = self._validated_model_values(values)
        total = model_values["total_ug_per_l"].clip(lower=0)

        physical = pd.DataFrame(
            index=model_values.index,
        )

        for target in self.numerators:
            output_name = f"log_fractions_{target}"

            physical[target] = (
                (total + self.epsilon) * np.exp(model_values[output_name])
                - self.epsilon
            ).clip(lower=0)

        physical[self.denominator] = (
            total - physical[self.numerators].sum(axis=1)
        ).clip(lower=0)

        physical = physical[self.original_targets]

        self._validate_finite(
            physical,
            context="inverse-transformed physical targets",
        )

        return physical

    def _validated_physical_values(
        self,
        values: pd.DataFrame,
    ) -> pd.DataFrame:
        missing = [
            target for target in self.original_targets if target not in values.columns
        ]
        if missing:
            raise ValueError(f"Missing physical target columns: {missing}")

        physical = values[self.original_targets].apply(
            pd.to_numeric,
            errors="coerce",
        )

        self._validate_finite(
            physical,
            context="physical target labels",
        )

        negative_counts = (physical < 0).sum()
        if negative_counts.any():
            raise ValueError(
                "Physical target labels contain negative values: "
                f"{negative_counts[negative_counts > 0].to_dict()}"
            )

        return physical.astype(float)

    def _validated_model_values(
        self,
        values: pd.DataFrame,
    ) -> pd.DataFrame:
        output_names = self.output_names(self.original_targets)

        missing = [column for column in output_names if column not in values.columns]
        if missing:
            raise ValueError(f"Missing log-fraction model outputs: {missing}")

        model_values = values[output_names].apply(
            pd.to_numeric,
            errors="coerce",
        )

        self._validate_finite(
            model_values,
            context="log-fraction model outputs",
        )

        return model_values.astype(float)

    @staticmethod
    def _validate_finite(
        values: pd.DataFrame,
        *,
        context: str,
    ) -> None:
        nan_counts = values.isna().sum()

        inf_counts = pd.Series(
            np.isinf(values.to_numpy(dtype=float)).sum(axis=0),
            index=values.columns,
        )

        if nan_counts.any() or inf_counts.any():
            raise ValueError(
                f"Label validation failed ({context}):\n"
                f"  NaN: "
                f"{nan_counts[nan_counts > 0].to_dict()}\n"
                f"  Inf: "
                f"{inf_counts[inf_counts > 0].to_dict()}"
            )
