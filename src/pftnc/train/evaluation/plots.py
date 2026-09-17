from __future__ import annotations

import calendar
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pftnc.train.evaluation.metrics import validate_metrics, validate_prediction_frames
from pftnc.train.schemas import MetricResults
from pftnc.train.tracking import MlflowTracker

plt.ioff()
logger = logging.getLogger(__name__)


class EvaluationPlotter:
    """Generate prediction and metric-comparison plots."""

    def __init__(
        self,
        tracker: MlflowTracker,
        metric_names: list[str],
    ) -> None:
        validate_metrics(metric_names)
        self.tracker = tracker
        self.metric_names = tuple(metric_names)

    def prediction_plots(
        self,
        run_dir: Path,
        split: str,
        metadata: pd.DataFrame,
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
    ) -> None:
        validate_prediction_frames(y_true, y_pred)
        metadata = self._align_metadata(metadata, y_true.index)

        required = {"date", "month", "iso_week"}
        missing = required - set(metadata.columns)
        if missing:
            raise ValueError(
                f"Prediction metadata is missing columns: {sorted(missing)}"
            )

        plot_dir = run_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        for target in y_true.columns:
            plot_data = pd.DataFrame(
                {
                    "actual": y_true[target],
                    "predicted": y_pred[target],
                    "date": metadata["date"],
                    "month": metadata["month"],
                    "iso_week": metadata["iso_week"],
                },
                index=y_true.index,
            ).dropna(
                subset=[
                    "actual",
                    "predicted",
                    "date",
                    "month",
                    "iso_week",
                ]
            )

            if plot_data.empty:
                logger.warning(
                    "No valid rows available for plot: split=%s, target=%s",
                    split,
                    target,
                )
                continue

            lower = min(
                float(plot_data["actual"].min()),
                float(plot_data["predicted"].min()),
            )
            upper = max(
                float(plot_data["actual"].max()),
                float(plot_data["predicted"].max()),
            )

            if lower == upper:
                lower -= 1.0
                upper += 1.0

            self._save_scatter(
                plot_data=plot_data,
                lower=lower,
                upper=upper,
                target=str(target),
                split=split,
                plot_dir=plot_dir,
                suffix="overall",
                color_by=None,
            )
            self._save_scatter(
                plot_data=plot_data,
                lower=lower,
                upper=upper,
                target=str(target),
                split=split,
                plot_dir=plot_dir,
                suffix="by_month",
                color_by="month",
            )
            self._save_scatter(
                plot_data=plot_data,
                lower=lower,
                upper=upper,
                target=str(target),
                split=split,
                plot_dir=plot_dir,
                suffix="by_iso_week",
                color_by="iso_week",
            )

    def metric_comparison_plots(
        self,
        metrics: MetricResults,
        split: str,
        run_dir: Path,
    ) -> None:
        if not metrics:
            return

        targets = list(metrics)
        display_names = [
            target.removesuffix("_ug_per_l").replace("_", " ").title()
            for target in targets
        ]

        plot_dir = run_dir / "metric_comparisons"
        plot_dir.mkdir(parents=True, exist_ok=True)

        for metric_name in self.metric_names:
            missing_targets = [
                target for target in targets if metric_name not in metrics[target]
            ]
            if missing_targets:
                raise ValueError(
                    f"Metric '{metric_name}' is missing for targets: {missing_targets}"
                )

            values = [float(metrics[target][metric_name]) for target in targets]

            fig, axis = plt.subplots(figsize=(8, 4))
            path = plot_dir / f"{split}_{metric_name}_by_target.png"
            path.parent.mkdir(parents=True, exist_ok=True)

            try:
                axis.bar(display_names, values)
                axis.set_title(
                    f"{split.replace('_', ' ').title()} {metric_name.upper()} by target"
                )
                axis.set_xlabel("Target")
                axis.set_ylabel(metric_name.upper())

                if metric_name in {"r2", "explained_variance"}:
                    axis.axhline(0, linestyle="--", linewidth=1)

                axis.tick_params(axis="x", rotation=20)
                fig.tight_layout()
                fig.savefig(path, dpi=150, bbox_inches="tight")
            finally:
                plt.close(fig)

            if not path.is_file():
                raise RuntimeError(f"Metric comparison plot was not created: {path}")

            self.tracker.log_artifact(
                path,
                artifact_path=(f"evaluation/{split}/metric_comparisons"),
            )

    def _save_scatter(
        self,
        plot_data: pd.DataFrame,
        lower: float,
        upper: float,
        target: str,
        split: str,
        plot_dir: Path,
        suffix: str,
        color_by: str | None,
    ) -> None:
        fig, axis = plt.subplots(figsize=(7, 6))
        path = plot_dir / f"{split}_{target}_pred_vs_true_{suffix}.png"
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if color_by is None:
                axis.scatter(
                    plot_data["actual"],
                    plot_data["predicted"],
                    alpha=0.6,
                )
                title_suffix = ""

            elif color_by == "month":
                scatter = axis.scatter(
                    plot_data["actual"],
                    plot_data["predicted"],
                    c=plot_data["month"].astype(int),
                    cmap=plt.get_cmap("tab20", 12),
                    alpha=0.7,
                )
                colorbar = fig.colorbar(
                    scatter,
                    ax=axis,
                    ticks=range(1, 13),
                    boundaries=np.arange(0.5, 13.5, 1),
                )
                colorbar.ax.set_yticklabels(list(calendar.month_abbr)[1:])
                colorbar.set_label("Month")
                title_suffix = " colored by month"

            elif color_by == "iso_week":
                scatter = axis.scatter(
                    plot_data["actual"],
                    plot_data["predicted"],
                    c=plot_data["iso_week"].astype(int),
                    cmap="viridis",
                    vmin=1,
                    vmax=53,
                    alpha=0.7,
                )
                colorbar = fig.colorbar(
                    scatter,
                    ax=axis,
                    ticks=[1, 10, 20, 30, 40, 50, 53],
                )
                colorbar.set_label("ISO week of year")
                title_suffix = " colored by ISO week"

            else:
                raise ValueError(f"Unsupported color_by value: {color_by}")

            axis.plot(
                [lower, upper],
                [lower, upper],
                "r--",
                linewidth=1,
                label="Perfect prediction",
            )
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{split} — {target}\nActual vs predicted{title_suffix}")
            axis.set_xlabel("Actual")
            axis.set_ylabel("Predicted")
            axis.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)

        if not path.is_file():
            raise RuntimeError(f"Prediction plot was not created: {path}")

        self.tracker.log_artifact(
            path,
            artifact_path=f"{split}/plots/actual_vs_predicted",
        )

    @staticmethod
    def _align_metadata(
        metadata: pd.DataFrame,
        expected_index: pd.Index,
    ) -> pd.DataFrame:
        if not metadata.index.is_unique:
            raise ValueError("Prediction metadata must have a unique index.")

        missing_rows = expected_index.difference(metadata.index)
        if not missing_rows.empty:
            raise ValueError(
                "Prediction metadata is missing rows required for plots. "
                f"Missing count: {len(missing_rows)}"
            )

        return metadata.loc[expected_index].copy()
