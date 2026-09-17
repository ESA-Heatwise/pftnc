from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pftnc.config import AppConfig
from pftnc.provenance import portable_path, workspace_root
from pftnc.train.data import CHIME_RAW_COLUMNS  # noqa: F401
from pftnc.train.data import DatasetRepository, FoldDataset
from pftnc.train.evaluation.evaluator import Evaluator
from pftnc.train.evaluation.exports import PredictionExporter
from pftnc.train.evaluation.metrics import validate_metrics
from pftnc.train.evaluation.plots import EvaluationPlotter
from pftnc.train.evaluation.shap import ShapLogger
from pftnc.train.final_test import FinalTestRunner, ModelSelector
from pftnc.train.io import (
    build_model_io_schema,
    build_model_selection_metadata,
    log_model_io,
    validate_model_io_schema,
    write_model_io_schema,
)
from pftnc.train.models.base import ModelAdapter
from pftnc.train.models.manager import ModelManager
from pftnc.train.models.registry import get_model_adapter
from pftnc.train.models.xgboost import XGBoostAdapter  # noqa: F401
from pftnc.train.schemas import FoldResult, TrainingRunResult
from pftnc.train.targets.manager import TargetTransformManager
from pftnc.train.tracking import ArtifactStore, MlflowTracker
from pftnc.train.tuning import HyperparameterTuner
from pftnc.utils.utils import save_environment_versions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingComponents:
    """Concrete components used by the training orchestration."""

    repository: DatasetRepository
    targets: TargetTransformManager
    models: ModelManager
    tuner: HyperparameterTuner
    evaluator: Evaluator
    selector: ModelSelector
    final_test: FinalTestRunner
    tracker: MlflowTracker
    artifacts: ArtifactStore


def build_training_components(
    config: AppConfig,
    model_cls: type[ModelAdapter] | None = None,
) -> TrainingComponents:
    """Construct the default training component graph."""

    validate_metrics(config.training.metrics)

    resolved_model_cls = model_cls or get_model_adapter(config.training.model.type)

    repository = DatasetRepository(
        config.training.input_dataset,
        feature_selection=config.training.feature_selection,
    )
    targets = TargetTransformManager(
        target_transform_config=config.training.target_transformations,
        dataset_metadata=repository.metadata,
    )
    tracker = MlflowTracker(config.training.experiment_name)
    artifacts = ArtifactStore()
    models = ModelManager(
        adapter_cls=resolved_model_cls,
        model_config=config.training.model,
        seed=config.project.seed,
        tracker=tracker,
        model_outputs=targets.model_outputs,
    )

    exporter = PredictionExporter(tracker)
    plotter = EvaluationPlotter(
        tracker=tracker,
        metric_names=list(config.training.metrics),
    )
    shap_logger = ShapLogger(
        tracker=tracker,
        seed=config.project.seed,
    )
    evaluator = Evaluator(
        metric_names=list(config.training.metrics),
        models=models,
        targets=targets,
        tracker=tracker,
        artifacts=artifacts,
        exporter=exporter,
        plotter=plotter,
        shap_logger=shap_logger,
    )
    tuner = HyperparameterTuner(
        config=config.tuning,
        models=models,
        targets=targets,
        tracker=tracker,
        seed=config.project.seed,
    )
    selector = ModelSelector(
        selection_config=config.training.best_model_selection,
        metric_names=list(config.training.metrics),
        targets=targets,
        models=models,
    )
    final_test = FinalTestRunner(
        repository=repository,
        models=models,
        targets=targets,
        evaluator=evaluator,
        tracker=tracker,
        selection_config=config.training.best_model_selection,
        metric_names=list(config.training.metrics),
    )

    return TrainingComponents(
        repository=repository,
        targets=targets,
        models=models,
        tuner=tuner,
        evaluator=evaluator,
        selector=selector,
        final_test=final_test,
        tracker=tracker,
        artifacts=artifacts,
    )


class Trainer:
    """Coordinate fold training, selection, and final-test evaluation."""

    def __init__(
        self,
        model_cls: type[ModelAdapter] | None = None,
        config: AppConfig | None = None,
        components: TrainingComponents | None = None,
    ) -> None:
        if config is None:
            raise ValueError("config must be provided")

        self.model_cls = model_cls or get_model_adapter(config.training.model.type)
        self.config = config
        self.components = components or build_training_components(
            config,
            self.model_cls,
        )

        output_dir = config.training.output_dir
        if output_dir is None:
            raise ValueError("training.output_dir must be set")
        self.output_root = Path(output_dir).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

        repository = self.components.repository
        # Compatibility attributes retained from the former monolithic Trainer.
        self.dataset_path = repository.dataset_path
        self.site_name = repository.site_name
        self.dataset_version = repository.dataset_version
        self.dataset_meta = repository.metadata
        self.fold_dirs = repository.fold_dirs

        logger.info("Site name: %s", repository.site_name)
        logger.info("Dataset version: %s", repository.dataset_version)
        logger.info("Dataset path: %s", repository.dataset_path)
        logger.info("Output root: %s", self.output_root)
        logger.info("Model mode: %s", self.components.models.mode)
        logger.info("Model outputs: %s", self.components.targets.model_outputs)
        logger.info(
            "Folds found (%d): %s",
            len(repository.fold_dirs),
            [path.name for path in repository.fold_dirs],
        )
        logger.info(
            "Physical targets: %s",
            self.components.targets.physical_targets,
        )
        logger.info(
            "Model outputs: %s",
            self.components.targets.model_outputs,
        )
        logger.info(
            "Joint target transformation enabled: %s",
            self.components.targets.requires_complete_bundle_for_target_selection,
        )

    def train(self) -> dict[str, Any]:
        """Execute the complete configured training workflow."""

        self.components.tracker.configure(self.model_cls)

        run_id = self._generate_run_id()
        run_dir = (self.output_root / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Experiment name: %s",
            self.config.training.experiment_name,
        )
        logger.info("Run ID: %s", run_id)
        logger.info("Run directory: %s", run_dir)
        logger.info(
            "Number of folds to process: %d",
            len(self.components.repository.fold_dirs),
        )

        with self.components.tracker.training_run(
            run_name=run_id,
            description=self.config.training.description,
        ):
            versions_path = run_dir / "versions.txt"
            save_environment_versions(versions_path)
            self.components.tracker.log_artifact(versions_path)
            self.components.tracker.log_config(self.config, run_dir)
            self._log_run_metadata()

            first_fold = self.components.repository.load_fold(
                self.components.repository.fold_dirs[0]
            )
            run_schema = build_model_io_schema(
                input_columns=list(first_fold.train.X.columns),
                model_output_columns=list(self.components.targets.model_outputs),
                physical_target_columns=list(self.components.targets.physical_targets),
                model_mode=self.components.models.mode,
            )
            schema_path = run_dir / "model_io_schema.yml"
            write_model_io_schema(schema_path, run_schema)
            self.components.tracker.log_artifact(schema_path)
            logger.info("Model input/output schema saved to: %s", schema_path)

            fold_results: dict[str, FoldResult] = {}
            for fold_path in self.components.repository.fold_dirs:
                fold_name = fold_path.name
                logger.info("[FOLD START] %s", fold_name)
                fold_results[fold_name] = self._train_fold(
                    fold_name,
                    fold_path,
                    run_dir,
                )
                logger.info("[FOLD END] %s", fold_name)

            logger.info("Selecting best model from validation results")
            best_model = self.components.selector.select(fold_results)

            selection_metadata = build_model_selection_metadata(
                best_model=best_model,
                fold_results=fold_results,
                metric=self.config.training.best_model_selection.metric,
                direction=self.config.training.best_model_selection.direction,
                mode=self.config.training.best_model_selection.mode,
                target=self.config.training.best_model_selection.target,
                physical_targets=list(self.components.targets.physical_targets),
            )
            selection_path = run_dir / "model_selection.yml"
            with selection_path.open("w") as stream:
                yaml.safe_dump(selection_metadata, stream, sort_keys=False)
            self.components.tracker.log_artifact(selection_path)

            logger.info("Running final test evaluation")
            final_test_results = self.components.final_test.run(
                fold_results,
                run_dir,
            )

            result = TrainingRunResult(
                trainer_run_id=run_id,
                description=self.config.training.description,
                model_mode=self.components.models.mode,
                dataset_version=(self.components.repository.dataset_version),
                dataset_path=self.components.repository.dataset_path,
                folds=fold_results,
                best_model=best_model,
                final_test=final_test_results,
            )
            summary = result.to_summary_dict(run_dir=run_dir)
            summary_path = self.components.artifacts.write_json(
                run_dir / "run_summary.json",
                summary,
            )
            self.components.tracker.log_artifact(summary_path)

            logger.info("Training run completed successfully")
            logger.info("Run summary saved to: %s", summary_path)

        return summary

    def _train_fold(
        self,
        fold_name: str,
        fold_path: Path,
        run_dir: Path,
    ) -> FoldResult:
        with self.components.tracker.nested_run(run_name=fold_name):
            dataset = self.components.repository.load_fold(fold_path)
            self._log_fold_shapes(fold_name, dataset)

            y_train_model = self.components.targets.transform_targets(
                dataset.train.X,
                dataset.train.y,
            )
            y_val_model = self.components.targets.transform_targets(
                dataset.validation.X,
                dataset.validation.y,
            )

            model_io = build_model_io_schema(
                input_columns=list(dataset.train.X.columns),
                model_output_columns=list(y_train_model.columns),
                physical_target_columns=list(self.components.targets.physical_targets),
                model_mode=self.components.models.mode,
            )
            log_model_io(model_io, fold_name=fold_name)
            schema_path = run_dir / "model_io_schema.yml"
            validate_model_io_schema(schema_path, model_io)

            if self.components.tuner.enabled:
                logger.info(
                    "Hyperparameter tuning enabled for fold: %s",
                    fold_name,
                )
            else:
                logger.info(
                    "Hyperparameter tuning disabled for fold: %s",
                    fold_name,
                )

            effective_params = self.components.tuner.resolve(
                dataset.train.X,
                dataset.validation.X,
                y_train_model,
                y_val_model,
            )
            logger.info(
                "Effective parameters for fold %s: %s",
                fold_name,
                effective_params,
            )
            self.components.models.log_effective_params(effective_params)

            fold_dir = (run_dir / fold_name).resolve()
            fold_dir.mkdir(parents=True, exist_ok=True)

            model_bundle = self.components.models.fit(
                X_train=dataset.train.X,
                X_val=dataset.validation.X,
                y_train=y_train_model,
                y_val=y_val_model,
                params=effective_params,
                output_dir=fold_dir,
            )

            train_metrics = self.components.evaluator.evaluate_model(
                model_bundle,
                dataset.train,
                fold_dir,
                split="train",
            )
            validation_metrics = self.components.evaluator.evaluate_model(
                model_bundle,
                dataset.validation,
                fold_dir,
                split="validation",
            )
            test_metrics = self.components.evaluator.evaluate_model(
                model_bundle,
                dataset.test,
                fold_dir,
                split="test",
            )

            fold_result = FoldResult(
                fold_name=fold_name,
                dataset_version=(self.components.repository.dataset_version),
                dataset_path=self.components.repository.dataset_path,
                fold_dataset_path=fold_path,
                model_outputs=list(y_train_model.columns),
                model_bundle=model_bundle,
                params=effective_params,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                fit_kwargs={
                    "eval_set": ["train", "validation"],
                    "verbose": 10,
                },
                target_transforms=(
                    self.config.training.target_transformations.model_dump()
                ),
            )

            summary_path = self.components.artifacts.write_json(
                fold_dir / "fold_summary.json",
                fold_result.to_summary_dict(run_dir=run_dir),
            )
            self.components.tracker.log_artifact(summary_path)
            return fold_result

    def _log_run_metadata(self) -> None:
        repository = self.components.repository
        workspace = workspace_root(repository.dataset_path)
        self.components.tracker.log_params(
            {
                "dataset_version": repository.dataset_version,
                "dataset_path": portable_path(
                    repository.dataset_path,
                    root=workspace,
                ),
                "model_type": self.config.training.model.type,
                "model_mode": self.components.models.mode,
                "model_outputs": ", ".join(self.components.targets.model_outputs),
                "n_folds": len(repository.fold_dirs),
                "tuning_enabled": self.components.tuner.enabled,
                "targets": ", ".join(self.components.targets.physical_targets),
                "log_fractions_enabled": (
                    self.config.training.target_transformations.log_fractions
                    is not None
                ),
            }
        )

    @staticmethod
    def _log_fold_shapes(
        fold_name: str,
        dataset: FoldDataset,
    ) -> None:
        logger.info(
            "Fold %s | train=%d | val=%d | test=%d",
            fold_name,
            len(dataset.train.X),
            len(dataset.validation.X),
            len(dataset.test.X),
        )
        logger.debug(
            "Fold %s shapes | X_train=%s | X_val=%s | X_test=%s | "
            "y_train=%s | y_val=%s | y_test=%s",
            fold_name,
            dataset.train.X.shape,
            dataset.validation.X.shape,
            dataset.test.X.shape,
            dataset.train.y.shape,
            dataset.validation.y.shape,
            dataset.test.y.shape,
        )

    def _generate_run_id(self) -> str:
        return (
            f"{self.components.repository.site_name}_mlrun_"
            f"{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
        )


def set_global_seed(seed: int) -> None:
    """Set Python and NumPy process-level random seeds."""

    random.seed(seed)
    np.random.seed(seed)
