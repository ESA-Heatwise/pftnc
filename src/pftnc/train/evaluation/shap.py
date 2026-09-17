from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pftnc.train.tracking import MlflowTracker

plt.ioff()
logger = logging.getLogger(__name__)


class ShapLogger:
    """Generate and log SHAP summary, bar, and waterfall plots."""

    def __init__(self, tracker: MlflowTracker, seed: int) -> None:
        self.tracker = tracker
        self.seed = seed

    def log(
        self,
        run_dir: Path,
        model: Any,
        X: pd.DataFrame,
        targets: pd.Index,
        split: str,
    ) -> None:
        import shap

        shap_dir = run_dir / "shap" / Path(split)
        X_sample = X.sample(
            min(200, len(X)),
            random_state=self.seed,
        )
        if X_sample.empty:
            logger.warning(
                "Skipping SHAP because split '%s' has no rows.",
                split,
            )
            return

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_per_target = [np.asarray(values) for values in shap_values]
        else:
            values = np.asarray(shap_values)
            if values.ndim == 2:
                shap_per_target = [values]
            elif values.ndim == 3:
                shap_per_target = [
                    values[:, :, index] for index in range(values.shape[2])
                ]
            else:
                raise ValueError(f"Unsupported SHAP output shape: {values.shape}")

        if len(shap_per_target) != len(targets):
            raise ValueError(
                "SHAP output count does not match target labels. "
                f"outputs={len(shap_per_target)}, targets={len(targets)}"
            )

        expected_values = np.asarray(explainer.expected_value)

        for index, target in enumerate(targets):
            values = shap_per_target[index]
            with warnings.catch_warnings():
                self._summary_plot(
                    shap,
                    values,
                    X_sample,
                    shap_dir,
                    str(target),
                    split,
                )
                self._bar_plot(
                    shap,
                    values,
                    X_sample,
                    shap_dir,
                    str(target),
                    split,
                )
                self._waterfall_plot(
                    shap,
                    values,
                    expected_values,
                    index,
                    X_sample,
                    shap_dir,
                    str(target),
                    split,
                )

    def _summary_plot(
        self,
        shap: Any,
        values: np.ndarray,
        X_sample: pd.DataFrame,
        shap_dir: Path,
        target: str,
        split: str,
    ) -> None:
        shap.summary_plot(values, X_sample, show=False)
        path = shap_dir / f"{target}_summary.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        self.tracker.log_artifact(path, artifact_path=f"{split}/shap")

    def _bar_plot(
        self,
        shap: Any,
        values: np.ndarray,
        X_sample: pd.DataFrame,
        shap_dir: Path,
        target: str,
        split: str,
    ) -> None:
        shap.plots.bar(
            shap.Explanation(
                values=values,
                data=X_sample.values,
                feature_names=X_sample.columns,
            ),
            show=False,
        )
        path = shap_dir / f"{target}_bar.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        self.tracker.log_artifact(path, artifact_path=f"{split}/shap")

    def _waterfall_plot(
        self,
        shap: Any,
        values: np.ndarray,
        expected_values: np.ndarray,
        target_index: int,
        X_sample: pd.DataFrame,
        shap_dir: Path,
        target: str,
        split: str,
    ) -> None:
        if expected_values.ndim == 0:
            base_value = float(expected_values)
        else:
            base_value = expected_values.reshape(-1)[target_index]

        explanation = shap.Explanation(
            values=values[0],
            base_values=base_value,
            data=X_sample.iloc[0],
            feature_names=X_sample.columns,
        )
        shap.plots.waterfall(explanation, show=False)
        path = shap_dir / f"{target}_waterfall_0.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        self.tracker.log_artifact(path, artifact_path=f"{split}/shap")
