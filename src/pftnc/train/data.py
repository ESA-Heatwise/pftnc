from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pftnc.config import (
    DatasetInputConfig,
    DatasetMetadata,
    FeatureOperationsConfig,
    FeatureSelectionConfig,
)
from pftnc.utils.feature_columns import (
    clean_feature_name,
    clean_feature_names,
    rolling_feature_names,
)
from pftnc.utils.utils import get_registry_key_and_paths

logger = logging.getLogger(__name__)

CHIME_RAW_COLUMNS = [
    "chime_diatoms_ug_per_l",
    "chime_cyanobacteria_ug_per_l",
    "chime_others_ug_per_l",
]
SITE_METADATA_COLUMNS = ["site_lat", "site_lon"]

PREDICTION_METADATA_COLUMNS = [
    "time",
    "date",
    "site",
    *SITE_METADATA_COLUMNS,
    *CHIME_RAW_COLUMNS,
]


@dataclass(frozen=True)
class PreparedSplit:
    X: pd.DataFrame
    y: pd.DataFrame
    metadata: pd.DataFrame


@dataclass(frozen=True)
class FoldDataset:
    train: PreparedSplit
    validation: PreparedSplit
    test: PreparedSplit


class DatasetRepository:
    """Discover, validate, and load prepared training datasets."""

    def __init__(
        self,
        storage_config: DatasetInputConfig,
        feature_selection: FeatureSelectionConfig | None = None,
    ) -> None:
        if storage_config.base_path is None:
            raise ValueError("training.input_dataset.base_path must be set")
        if storage_config.registry_path is None:
            raise ValueError("training.input_dataset.registry_path must be set")
        if storage_config.version is None:
            raise ValueError("training.input_dataset.version must be set")

        self.storage_config = storage_config
        self.dataset_path = validate_dataset_version(
            storage_config.base_path,
            storage_config.registry_path,
            storage_config.version,
        )
        self.site_name = self.dataset_path.parent.name
        self.dataset_version = self.dataset_path.name
        self.metadata = load_dataset_metadata(self.dataset_path)
        self.feature_config = load_feature_operations_config(self.dataset_path)
        self.feature_selection = feature_selection or FeatureSelectionConfig()
        self.fold_dirs = list_fold_dirs(self.dataset_path)

    def load_fold(self, fold_path: Path) -> FoldDataset:
        raw_train_X = read_dataframe(fold_path / "train_X")
        raw_train_y = read_dataframe(fold_path / "train_y")
        raw_train_metadata = read_dataframe(fold_path / "train_metadata")
        raw_val_X = read_dataframe(fold_path / "val_X")
        raw_val_y = read_dataframe(fold_path / "val_y")
        raw_val_metadata = read_dataframe(fold_path / "val_metadata")
        raw_test_X = read_dataframe(fold_path / "test_X")
        raw_test_y = read_dataframe(fold_path / "test_y")
        raw_test_metadata = read_dataframe(fold_path / "test_metadata")

        return self._prepare_dataset(
            raw_train_X,
            raw_train_y,
            raw_train_metadata,
            raw_val_X,
            raw_val_y,
            raw_val_metadata,
            raw_test_X,
            raw_test_y,
            raw_test_metadata,
        )

    def load_final_test(self, model_columns: list[str]) -> PreparedSplit:
        final_path = self.dataset_path / "final_test"
        raw_X = read_dataframe(final_path / "test_X")
        raw_y = read_dataframe(final_path / "test_y")
        raw_metadata = read_dataframe(final_path / "test_metadata")
        return self._prepare_split(
            raw_X,
            raw_y,
            split_name="final_test",
            model_columns=model_columns,
            raw_metadata=raw_metadata,
        )

    def _prepare_dataset(
        self,
        train_X: pd.DataFrame,
        train_y: pd.DataFrame,
        train_metadata: pd.DataFrame,
        val_X: pd.DataFrame,
        val_y: pd.DataFrame,
        val_metadata: pd.DataFrame,
        test_X: pd.DataFrame,
        test_y: pd.DataFrame,
        test_metadata: pd.DataFrame,
    ) -> FoldDataset:
        train = self._prepare_split(
            train_X, train_y, "train", raw_metadata=train_metadata
        )
        model_columns = list(train.X.columns)
        validation = self._prepare_split(
            val_X,
            val_y,
            "validation",
            model_columns=model_columns,
            raw_metadata=val_metadata,
        )
        test = self._prepare_split(
            test_X,
            test_y,
            "test",
            model_columns=model_columns,
            raw_metadata=test_metadata,
        )
        return FoldDataset(
            train=train,
            validation=validation,
            test=test,
        )

    def _prepare_split(
        self,
        raw_X: pd.DataFrame,
        raw_y: pd.DataFrame,
        split_name: str,
        model_columns: list[str] | None = None,
        raw_metadata: pd.DataFrame | None = None,
    ) -> PreparedSplit:
        prepared, _ = prepare_split(
            raw_X=raw_X,
            raw_y=raw_y,
            raw_metadata=raw_metadata,
            split_name=split_name,
            dataset_metadata=self.metadata,
            feature_config=self.feature_config,
            feature_selection=self.feature_selection,
            model_columns=model_columns,
        )
        return prepared


def read_registry(registry_path: Path) -> dict[str, Any]:
    if registry_path.exists():
        data = yaml.safe_load(registry_path.read_text())
        return data if data is not None else {"datasets": {}}
    return {"datasets": {}}


def validate_dataset_version(
    base_path: Path,
    registry_path: Path,
    version: str,
) -> Path:
    """
    Confirm that the configured dataset version exists on disk and in the registry.
    """
    base_key, absolute_base_path = get_registry_key_and_paths(
        registry_path,
        base_path,
    )
    dataset_path = Path(absolute_base_path) / version

    if not dataset_path.exists():
        raise RuntimeError(
            f"Dataset version {version} does not exist at {dataset_path}"
        )

    registry = read_registry(registry_path)
    dataset_registry = registry.get("datasets", {}).get(base_key, {})
    if version not in dataset_registry:
        raise RuntimeError(
            f"Dataset version {version} not registered for dataset {base_key}"
        )
    return dataset_path


def list_fold_dirs(dataset_path: Path) -> list[Path]:
    """Return sorted fold directories inside dataset_path."""

    fold_dirs = sorted(
        path
        for path in dataset_path.iterdir()
        if path.is_dir() and path.name.startswith("fold_")
    )
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found in {dataset_path}")
    return fold_dirs


def load_dataset_metadata(dataset_path: Path) -> DatasetMetadata:
    metadata_path = dataset_path / "metadata.yml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    with metadata_path.open() as stream:
        raw = yaml.safe_load(stream)
    return DatasetMetadata(**raw)


def load_feature_operations_config(dataset_path: Path) -> FeatureOperationsConfig:
    """Load the feature-generation config stored with a dataset artifact."""
    config_path = dataset_path / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Feature configuration not found: {config_path}")
    with config_path.open() as stream:
        raw = yaml.safe_load(stream)
    features = raw.get("features")
    if features is None:
        features = raw.get("feature_engineering", {}).get("features")
    return FeatureOperationsConfig(**(features or {}))


def read_dataframe(path_without_suffix: Path) -> pd.DataFrame:
    """Load a split from Parquet or CSV, preferring Parquet when both exist."""

    parquet_path = path_without_suffix.with_suffix(".parquet")
    csv_path = path_without_suffix.with_suffix(".csv")

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        unnamed = [
            column for column in frame.columns if str(column).startswith("Unnamed:")
        ]
        if unnamed:
            frame = frame.drop(columns=unnamed)
        return frame

    raise FileNotFoundError(
        f"Dataset split not found. Expected one of: {parquet_path}, {csv_path}"
    )


def extract_prediction_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract non-model metadata used for prediction exports and plots.

    These columns are kept separate from X so they are never passed
    into the model.
    """

    available_columns = [
        column for column in PREDICTION_METADATA_COLUMNS if column in df.columns
    ]
    metadata = df[available_columns].copy()

    if "date" in metadata.columns:
        date = pd.to_datetime(metadata["date"], errors="coerce")
    elif "time" in metadata.columns:
        date = pd.to_datetime(metadata["time"], errors="coerce")
    else:
        raise ValueError(
            "Neither 'date' nor 'time' is available for prediction metadata."
        )

    metadata["date"] = date.dt.normalize()
    metadata["year"] = date.dt.year.astype("Int64")
    metadata["month"] = date.dt.month.astype("Int64")
    metadata["month_name"] = date.dt.strftime("%b")
    metadata["iso_week"] = date.dt.isocalendar().week
    return metadata


def get_model_feature_columns(
    df: pd.DataFrame,
    dataset_metadata: DatasetMetadata,
    feature_config: FeatureOperationsConfig,
    feature_selection: FeatureSelectionConfig,
) -> list[str]:
    schema = dataset_metadata.dataset_schema
    selection = feature_selection

    sensor_columns = [
        clean_feature_name(column)
        for sensor_columns in schema.sensors.values()
        for column in sensor_columns
    ]
    sensor_columns = [
        column for column in sensor_columns if column not in CHIME_RAW_COLUMNS
    ]

    rolling_columns: list[str] = []
    if feature_config.rolling is not None:
        rolling_columns = rolling_feature_names(
            schema.sensors,
            feature_config.rolling.windows,
            feature_config.rolling.statistics,
        )

    included_columns = [clean_feature_name(column) for column in selection.include]
    excluded_columns = {clean_feature_name(column) for column in selection.exclude}

    feature_columns = list(
        dict.fromkeys(included_columns + sensor_columns + rolling_columns)
    )
    feature_columns = [
        column for column in feature_columns if column not in excluded_columns
    ]
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Configured model features are missing: {missing}")
    return feature_columns


def get_chime_gap_mask(df: pd.DataFrame) -> pd.Series:
    """
    Return rows where all three current CHIME values are missing.

    These are the rows for which the gap-filling model is needed.
    """

    missing_columns = [
        column for column in CHIME_RAW_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required CHIME columns: {missing_columns}")

    missing = df[CHIME_RAW_COLUMNS].isna()
    partially_missing = missing.any(axis=1) & ~missing.all(axis=1)
    if partially_missing.any():
        logger.warning(
            "%d rows have partially missing CHIME targets. These rows "
            "will not be used because training requires a complete target "
            "vector for each row.",
            int(partially_missing.sum()),
        )
    return missing.all(axis=1)


def prepare_split(
    raw_X: pd.DataFrame,
    raw_y: pd.DataFrame,
    split_name: str,
    dataset_metadata: DatasetMetadata,
    model_columns: list[str] | None = None,
    raw_metadata: pd.DataFrame | None = None,
    feature_config: FeatureOperationsConfig | None = None,
    feature_selection: FeatureSelectionConfig | None = None,
) -> tuple[PreparedSplit, list[str]]:
    """
    Filter, align, clean, and select model inputs for one dataset split.

    When ``model_columns`` is omitted, columns are resolved from the dataset
    metadata. Supplying ``model_columns`` enforces the training split's feature
    set and order for train, validation, test, and final-test data.
    """

    if len(raw_X) != len(raw_y):
        raise ValueError(
            f"{split_name} X/y row counts differ: {len(raw_X)} != {len(raw_y)}"
        )
    if not raw_X.index.is_unique:
        raise ValueError(f"{split_name} X index must be unique.")
    if not raw_y.index.is_unique:
        raise ValueError(f"{split_name} y index must be unique.")
    if not raw_X.index.equals(raw_y.index):
        raise ValueError(f"{split_name} X and y indexes do not match.")

    physical_targets = list(dataset_metadata.dataset_schema.targets)
    missing_targets = [
        target for target in physical_targets if target not in raw_y.columns
    ]
    if missing_targets:
        raise ValueError(
            f"{split_name} split is missing target columns: {missing_targets}"
        )

    if raw_metadata is None:
        metadata = extract_prediction_metadata(raw_X)
        gap_mask = get_chime_gap_mask(raw_X)
    else:
        if not raw_metadata.index.is_unique:
            raise ValueError(f"{split_name} metadata index must be unique.")
        if not raw_metadata.index.equals(raw_X.index):
            raise ValueError(f"{split_name} X and metadata indexes do not match.")
        metadata = raw_metadata.copy()
        gap_mask = pd.Series(True, index=raw_X.index)
    total_rows = len(raw_X)

    filtered_X = raw_X.loc[gap_mask].copy()
    filtered_y = raw_y.loc[
        gap_mask,
        physical_targets,
    ].copy()
    filtered_metadata = metadata.loc[gap_mask].copy()

    logger.info(
        "CHIME gap rows | %s=%d/%d",
        split_name,
        len(filtered_X),
        total_rows,
    )

    clean_X = clean_feature_names(filtered_X)

    if model_columns is None:
        if feature_config is None or feature_selection is None:
            raise ValueError(
                "feature_config and feature_selection are required when "
                "model_columns is not provided"
            )
        selected_columns = get_model_feature_columns(
            clean_X,
            dataset_metadata,
            feature_config=feature_config,
            feature_selection=feature_selection,
        )
    else:
        selected_columns = list(model_columns)
        missing = [
            column for column in selected_columns if column not in clean_X.columns
        ]
        if missing:
            raise ValueError(f"{split_name} split is missing model features: {missing}")

    return (
        PreparedSplit(
            X=clean_X[selected_columns].copy(),
            y=filtered_y,
            metadata=filtered_metadata,
        ),
        selected_columns,
    )


def load_dataset_splits(
    dataset_metadata: DatasetMetadata,
    dataset_path: Path,
    feature_config: FeatureOperationsConfig,
    feature_selection: FeatureSelectionConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load train/val/test splits for a given fold,
    returning feature matrices and target vectors, for train, validation,
    and test splits.
    """

    train, model_columns = prepare_split(
        raw_X=read_dataframe(dataset_path / "train_X"),
        raw_y=read_dataframe(dataset_path / "train_y"),
        raw_metadata=read_dataframe(dataset_path / "train_metadata"),
        split_name="train",
        dataset_metadata=dataset_metadata,
        feature_config=feature_config,
        feature_selection=feature_selection,
    )
    validation, _ = prepare_split(
        raw_X=read_dataframe(dataset_path / "val_X"),
        raw_y=read_dataframe(dataset_path / "val_y"),
        raw_metadata=read_dataframe(dataset_path / "val_metadata"),
        split_name="validation",
        dataset_metadata=dataset_metadata,
        feature_config=feature_config,
        feature_selection=feature_selection,
        model_columns=model_columns,
    )
    test, _ = prepare_split(
        raw_X=read_dataframe(dataset_path / "test_X"),
        raw_y=read_dataframe(dataset_path / "test_y"),
        raw_metadata=read_dataframe(dataset_path / "test_metadata"),
        split_name="test",
        dataset_metadata=dataset_metadata,
        feature_config=feature_config,
        feature_selection=feature_selection,
        model_columns=model_columns,
    )

    return (
        train.X,
        validation.X,
        test.X,
        train.y,
        validation.y,
        test.y,
        train.metadata,
        validation.metadata,
        test.metadata,
    )
