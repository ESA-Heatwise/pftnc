from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pftnc.config import (
    DatasetMetadata,
    DatasetSchema,
    FeatureOperationsConfig,
    TargetTransformationsConfig,
)
from pftnc.inference_config import InferenceConfig
from pftnc.preprocess.dataset import extract_prediction_metadata
from pftnc.preprocess.features import engineer_features
from pftnc.preprocess.io import resolve_dataset_file
from pftnc.provenance import portable_config_snapshot
from pftnc.train.models.registry import get_model_adapter
from pftnc.train.targets.manager import TargetTransformManager
from pftnc.utils.feature_columns import clean_feature_names

from .bundle import load_manifest

logger = logging.getLogger(__name__)


def _read_input(path: Path) -> pd.DataFrame:
    dataset_file = resolve_dataset_file(path)
    if dataset_file.suffix == ".parquet":
        return pd.read_parquet(dataset_file)
    return pd.read_csv(dataset_file)


def _target_manager(manifest: dict[str, Any]) -> TargetTransformManager:
    metadata = DatasetMetadata(
        sites=[],
        dataset_schema=DatasetSchema(
            date_column=manifest["dataset_metadata"]["dataset_schema"]["date_column"],
            group_column=manifest["dataset_metadata"]["dataset_schema"]["group_column"],
            sensors=manifest["dataset_metadata"]["dataset_schema"]["sensors"],
            targets=list(manifest["model_io"]["physical_target_columns"]),
        ),
    )
    transformations = TargetTransformationsConfig.model_validate(
        manifest.get("target_transformations") or {}
    )
    return TargetTransformManager(
        target_transform_config=transformations,
        dataset_metadata=metadata,
    )


def _prepare_input(
    raw: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = manifest["dataset_metadata"]["dataset_schema"]
    date_column = schema["date_column"]
    group_column = schema["group_column"]
    required = {date_column, group_column}
    required.update(
        column for columns in schema["sensors"].values() for column in columns
    )
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Inference input is missing required raw columns: {missing}")

    feature_config = FeatureOperationsConfig.model_validate(
        manifest["feature_engineering"]
    )
    dataset_schema = DatasetSchema.model_validate(schema)
    engineered = engineer_features(
        raw,
        feature_config=feature_config,
        dataset_schema=dataset_schema,
    )
    clean = clean_feature_names(engineered)
    metadata = extract_prediction_metadata(
        engineered,
        date_column=date_column,
        group_column=group_column,
    )
    columns = list(manifest["model_io"]["models"].values())[0]["input_columns"]
    missing_model = [column for column in columns if column not in clean.columns]
    if missing_model:
        raise ValueError(
            f"Inference data is missing model input columns: {missing_model}"
        )
    return clean.loc[:, columns].copy(), metadata


def _select_rows(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    config: InferenceConfig,
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(metadata["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Inference input contains invalid dates")
    if config.mode == "from_date":
        start = pd.Timestamp(config.start_date).normalize()
        mask = dates >= start
        if not mask.any():
            raise ValueError(f"No inference rows found on or after {start.date()}")
        return X.loc[mask], metadata.loc[mask]

    ordered = metadata.assign(_inference_date=dates).sort_values(
        [group_column, "_inference_date"], kind="stable"
    )
    selected = (
        ordered.groupby(group_column, sort=False)
        .tail(1)
        .drop(columns="_inference_date")
    )
    return X.loc[selected.index], metadata.loc[selected.index]


def _load_fold_predictions(
    bundle_path: Path,
    manifest: dict[str, Any],
    fold_name: str,
    X: pd.DataFrame,
    targets: TargetTransformManager,
) -> pd.DataFrame:
    adapter_cls = get_model_adapter(manifest["model_type"])
    fold = manifest["folds"][fold_name]
    model_contracts = manifest["model_io"]["models"]
    model_outputs = list(manifest["model_io"]["models"].values())[0]["output_columns"]
    raw_predictions: dict[str, np.ndarray] = {}
    if manifest["model_mode"] == "multi_target":
        model_path = bundle_path / fold["models"]["__multi_target__"]
        adapter = adapter_cls().load(model_path)
        values = np.asarray(adapter.predict(X))
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape != (len(X), len(model_outputs)):
            raise ValueError(
                f"Model output shape does not match bundle schema: {values.shape}"
            )
        raw = pd.DataFrame(values, columns=model_outputs, index=X.index)
    else:
        for output in model_outputs:
            if output not in fold["models"]:
                raise ValueError(f"Fold {fold_name} has no model for output {output!r}")
            adapter = adapter_cls().load(bundle_path / fold["models"][output])
            values = np.asarray(adapter.predict(X)).reshape(-1)
            raw_predictions[output] = values
        raw = pd.DataFrame(raw_predictions, index=X.index)[model_outputs]
    del model_contracts
    return targets.inverse_predictions(X, raw)


def run_inference(config: InferenceConfig) -> pd.DataFrame:
    bundle_path = config.model.bundle_path.expanduser().resolve()
    manifest = load_manifest(bundle_path)
    raw = _read_input(config.input_dataset.path)
    X, metadata = _prepare_input(raw, manifest)
    group_column = manifest["dataset_metadata"]["dataset_schema"]["group_column"]
    X, metadata = _select_rows(X, metadata, config, group_column)
    targets = _target_manager(manifest)
    fold_predictions = {
        fold: _load_fold_predictions(bundle_path, manifest, fold, X, targets)
        for fold in manifest["folds"]
    }
    selection = config.model.selection
    target_names = list(targets.physical_targets)
    if selection == "best_overall":
        fold = manifest["selection"]["best_overall"]["fold"]
        predictions = fold_predictions[fold]
    elif selection == "best_per_target":
        predictions = pd.DataFrame(index=X.index)
        for target in target_names:
            fold = manifest["selection"]["best_per_target"][target]["fold"]
            predictions[target] = fold_predictions[fold][target]
    elif selection == "ensemble_simple":
        predictions = sum(fold_predictions.values()) / len(fold_predictions)
    else:
        if selection == "ensemble_weighted":
            weights = manifest["selection"]["ensemble_weights"]["global"]
            predictions = sum(
                (fold_predictions[fold] * weight for fold, weight in weights.items()),
                start=pd.DataFrame(0.0, index=X.index, columns=target_names),
            ) / sum(weights.values())
        else:
            predictions = pd.DataFrame(index=X.index)
            for target in target_names:
                weights = manifest["selection"]["ensemble_weights"]["per_target"][
                    target
                ]
                predictions[target] = sum(
                    (
                        fold_predictions[fold][target] * weight
                        for fold, weight in weights.items()
                    ),
                    start=pd.Series(0.0, index=X.index),
                ) / sum(weights.values())

    result = metadata.copy()
    if config.save_model_inputs:
        # Preserve the exact model inputs alongside the metadata and
        # predictions.
        model_inputs = X.drop(columns=metadata.columns.intersection(X.columns))
        result = pd.concat([result, model_inputs], axis=1)
    for target in target_names:
        result[f"{target}_pred"] = predictions[target]
    if config.save_outputs:
        assert config.output_path is not None
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(config.output_path, index=config.save_index)

        inference_config_path = config.output_path.with_name("inference_config.yml")
        config_snapshot = portable_config_snapshot(
            config.model_dump(mode="json"),
            anchor=inference_config_path,
        )
        with inference_config_path.open("w") as stream:
            yaml.safe_dump(
                config_snapshot,
                stream,
                sort_keys=False,
            )

    return result
