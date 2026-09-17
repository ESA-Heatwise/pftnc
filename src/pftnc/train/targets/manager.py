from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from pftnc.config import (
    DatasetMetadata,
    LogFractionsConfig,
    TargetTransformationsConfig,
    TargetTransformConfig,
)
from pftnc.train.targets.absolute import AbsoluteTransformationSingle
from pftnc.train.targets.base import (
    JointTargetTransformation,
    SingleTargetTransformation,
)
from pftnc.train.targets.log_delta import LogDeltaTransformationSingle
from pftnc.train.targets.log_fractions import LogFractionsTransformation


class TargetTransformManager:
    """Build and delegate configured target transformations."""

    def __init__(
        self,
        target_transform_config: TargetTransformationsConfig,
        dataset_metadata: DatasetMetadata,
    ) -> None:
        self.targets = list(dataset_metadata.dataset_schema.targets)

        self.joint: JointTargetTransformation | None = None
        self.single: dict[str, SingleTargetTransformation] = {}

        if target_transform_config.log_fractions is not None:
            self.joint = self._build_joint(target_transform_config.log_fractions)
        else:
            self.single = self._build_single(
                target_transform_config.per_target,
            )

    @property
    def physical_targets(self) -> list[str]:
        return list(self.targets)

    @property
    def model_outputs(self) -> list[str]:
        if self.joint is not None:
            return self.joint.output_names(self.targets)

        return [
            self.single[target].output_names([target])[0] for target in self.targets
        ]

    @property
    def requires_complete_bundle_for_target_selection(
        self,
    ) -> bool:
        return self.joint is not None

    def transform_targets(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform physical labels into model-output space."""
        if self.joint is not None:
            return self.joint.transform(
                X,
                y[self.targets],
            )

        transformed = pd.DataFrame(
            index=y.index,
        )

        for target in self.targets:
            transformation = self.single[target]
            output_name = transformation.output_names([target])[0]

            transformed[output_name] = transformation.transform(
                X,
                y[target],
                target,
            )

        return transformed[self.model_outputs]

    def inverse_predictions(
        self,
        X: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert model outputs back into physical concentrations."""
        if self.joint is not None:
            return self.joint.inverse(
                X,
                predictions,
            )[self.targets]

        physical = pd.DataFrame(
            index=predictions.index,
        )

        for target in self.targets:
            transformation = self.single[target]
            output_name = transformation.output_names([target])[0]

            physical[target] = transformation.inverse(
                X,
                predictions[output_name],
                target,
            )

        return physical[self.targets]

    def physical_truth(
        self,
        y: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return labels already stored in physical target space."""
        return y[self.targets].copy()

    def physical_target_for_output(self, output_name: str) -> str:
        """Return the physical target represented by one model output."""
        for target in self.targets:
            transformation = self.single[target]
            if transformation.output_names([target]) == [output_name]:
                return target

        raise ValueError(
            f"Model output '{output_name}' does not map to a physical target."
        )

    def _build_joint(
        self,
        config: LogFractionsConfig,
    ) -> JointTargetTransformation:
        builder = JOINT_TRANSFORM_BUILDERS["log_fractions"]
        return builder(config, self.targets)

    def _build_single(
        self,
        configured: dict[
            str,
            TargetTransformConfig,
        ],
    ) -> dict[str, SingleTargetTransformation]:
        transformations: dict[str, SingleTargetTransformation] = {}

        for target in self.targets:
            config = configured.get(target)

            transform_config = config or TargetTransformConfig()
            builder = SINGLE_TRANSFORM_BUILDERS.get(transform_config.type)
            if builder is None:
                raise ValueError(
                    f"Unsupported single-target transformation: {transform_config.type}"
                )
            transformations[target] = builder(transform_config)

        return transformations


SingleTransformBuilder = Callable[
    [TargetTransformConfig],
    SingleTargetTransformation,
]
JointTransformBuilder = Callable[
    [LogFractionsConfig, list[str]],
    JointTargetTransformation,
]


def build_absolute(
    config: TargetTransformConfig,
) -> SingleTargetTransformation:
    del config
    return AbsoluteTransformationSingle()


def build_log_delta(
    config: TargetTransformConfig,
) -> SingleTargetTransformation:
    if config.reference_feature is None:
        raise ValueError("log_delta requires reference_feature.")

    return LogDeltaTransformationSingle(
        config.reference_feature,
    )


def build_log_fractions(
    config: LogFractionsConfig,
    targets: list[str],
) -> JointTargetTransformation:
    if not config.denominator:
        raise ValueError("log_fractions requires denominator.")

    return LogFractionsTransformation(
        original_targets=targets,
        denominator=config.denominator,
        epsilon=config.epsilon,
    )


SINGLE_TRANSFORM_BUILDERS: dict[str, SingleTransformBuilder] = {
    "absolute": build_absolute,
    "log_delta": build_log_delta,
}


JOINT_TRANSFORM_BUILDERS: dict[str, JointTransformBuilder] = {
    "log_fractions": build_log_fractions,
}
