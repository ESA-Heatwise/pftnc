from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from pftnc.config import (
    DatasetMetadata,
    DatasetSchema,
    LogFractionsConfig,
    TargetTransformationsConfig,
    TargetTransformConfig,
)
from pftnc.preprocess.folds import PreparedTrainingDataset
from pftnc.train.data import PreparedSplit
from pftnc.train.evaluation.evaluator import Evaluator
from pftnc.train.final_test import FinalTestRunner
from pftnc.train.models.manager import ModelManager
from pftnc.train.targets.manager import TargetTransformManager
from pftnc.train.tuning import HyperparameterTuner

BASE_CONFIG = {
    "project": {
        "seed": 42,
    },
    "feature_engineering": {
        "input_dataset": {
            "base_path": "data/",
            "registry_path": "data/registry.yaml",
            "version": "test_version",
        },
        "output_dataset": {
            "storage": {
                "base_path": "data/output",
                "registry_path": "data/output.yaml",
                "description": "test dataset",
            },
            "metadata": {
                "sites": ["test_site"],
                "dataset_schema": {
                    "date_column": "date",
                    "group_column": "id",
                    "sensors": {
                        "s1": ["f1", "f2"],
                    },
                    "targets": ["target"],
                },
                "splits": {
                    "final_test_year": 2022,
                },
            },
        },
        "features": {
            "rolling": {
                "windows": [3],
                "statistics": ["mean"],
            }
        },
    },
    "training": {
        "experiment_name": "test_experiment",
        "description": "test description",
        "output_dir": "outputs",
        "input_dataset": {
            "base_path": "data/output",
            "registry_path": "data/output.yaml",
            "version": "test_version",
        },
        "model": {
            "type": "xgboost",
            "save_path": "model.json",
            "params": {
                "n_estimators": 10,
                "max_depth": 4,
                "learning_rate": 0.1,
            },
        },
        "metrics": ["rmse"],
        "best_model_selection": {
            "metric": "rmse",
            "direction": "minimize",
        },
    },
    "tuning": {
        "enabled": False,
    },
}


def deep_update(base: dict, updates: dict):
    """Recursively update nested dictionaries."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def make_config(**overrides):
    """Create a fresh config dictionary for tests."""
    config = deepcopy(BASE_CONFIG)

    deep_update(config, overrides)

    return config


@pytest.fixture
def app_config():
    """Standard minimal valid config."""
    return make_config()


@pytest.fixture
def app_with_tuning_config():
    """Config with tuning enabled."""
    return make_config(
        tuning={
            "enabled": True,
            "objective_metric": "rmse",
            "search_space": {
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1],
            },
        }
    )


@pytest.fixture
def config_path(tmp_path, app_config):
    """Write config to temporary YAML file and return path."""
    path = tmp_path / "config.yaml"

    with open(path, "w") as f:
        yaml.safe_dump(app_config, f)

    return path


@pytest.fixture
def make_config_file(tmp_path):
    """Factory fixture for creating arbitrary config files."""

    def _make_config_file(**overrides):
        cfg = make_config(**overrides)

        path = tmp_path / "config.yaml"

        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)

        return path

    return _make_config_file


TARGETS = [
    "diatoms_ug_per_l",
    "cyanobacterial_ug_per_l",
    "others_ug_per_l",
]


@pytest.fixture
def target_names() -> list[str]:
    return TARGETS.copy()


@pytest.fixture
def physical_targets(target_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            target_names[0]: [10.0, 0.0, 30.0],
            target_names[1]: [2.0, 5.0, 0.0],
            target_names[2]: [8.0, 5.0, 10.0],
        },
        index=pd.Index([10, 20, 30], name="sample"),
    )


@pytest.fixture
def features(physical_targets: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reference": [1.0, 2.0, 4.0],
            "feature": [0.1, 0.2, 0.3],
        },
        index=physical_targets.index,
    )


@pytest.fixture
def dataset_metadata(target_names: list[str]):
    return SimpleNamespace(
        dataset_schema=SimpleNamespace(
            targets=target_names,
            sensors={"sensor": ["sensor_a"]},
        )
    )


@pytest.fixture
def model_space_targets(
    target_transform_manager: TargetTransformManager,
    features: pd.DataFrame,
    physical_targets: pd.DataFrame,
) -> pd.DataFrame:
    return target_transform_manager.transform_targets(
        features,
        physical_targets,
    )


@pytest.fixture
def target_transform_manager(target_names: list[str]) -> TargetTransformManager:
    metadata = DatasetMetadata(
        sites=[],
        dataset_schema=DatasetSchema(
            date_column="date",
            group_column="site",
            sensors={},
            targets=target_names,
        ),
    )
    target_transform_config = TargetTransformationsConfig(
        log_fractions=LogFractionsConfig(
            denominator=target_names[-1],
            epsilon=1e-8,
        ),
    )
    return TargetTransformManager(target_transform_config, metadata)


@pytest.fixture
def single_target_transform_manager() -> TargetTransformManager:
    metadata = DatasetMetadata(
        sites=[],
        dataset_schema=DatasetSchema(
            date_column="date",
            group_column="site",
            sensors={},
            targets=["target_a", "target_b"],
        ),
    )
    config = TargetTransformationsConfig(
        per_target={
            "target_a": TargetTransformConfig(type="absolute"),
            "target_b": TargetTransformConfig(
                type="log_delta",
                reference_feature="reference",
            ),
        },
    )
    return TargetTransformManager(config, metadata)


@pytest.fixture
def prepared_split(
    features: pd.DataFrame,
    physical_targets: pd.DataFrame,
) -> PreparedSplit:
    metadata = pd.DataFrame(
        {"date": ["2020-01-01", "2020-01-02", "2020-01-03"]},
        index=features.index,
    )
    return PreparedSplit(
        X=features,
        y=physical_targets,
        metadata=metadata,
    )


class SpyTracker:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, int | None]] = []
        self.artifacts: list[tuple[str, str | None]] = []
        self.params = []

    @contextmanager
    def nested_run(self, run_name, **kwargs):
        yield

    def log_params(self, params):
        self.params.append(params)

    def log_param(self, name, value):
        self.params.append({name: value})

    def log_metric(self, name, value, step=None):
        self.metrics.append((name, float(value), step))

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((str(path), artifact_path))


class SpyExporter:
    def __init__(self) -> None:
        self.calls = []

    def export(self, run_dir, split, X, metadata, y_true, y_pred):
        self.calls.append((run_dir, split, X, metadata, y_true, y_pred))
        return run_dir / f"{split}.csv"


class SpyPlotter:
    def __init__(self) -> None:
        self.prediction_calls = []
        self.metric_calls = []

    def prediction_plots(self, run_dir, split, metadata, y_true, y_pred):
        self.prediction_calls.append((run_dir, split, metadata, y_true, y_pred))

    def metric_comparison_plots(self, metrics, split, run_dir):
        self.metric_calls.append((metrics, split, run_dir))


class NoOpShapLogger:
    def log(self, *args, **kwargs):
        raise AssertionError("SHAP logging is not expected in this test")


@pytest.fixture
def spy_tracker() -> SpyTracker:
    return SpyTracker()


@pytest.fixture
def spy_exporter() -> SpyExporter:
    return SpyExporter()


@pytest.fixture
def spy_plotter() -> SpyPlotter:
    return SpyPlotter()


@pytest.fixture
def evaluator(
    target_transform_manager: TargetTransformManager,
    spy_tracker: SpyTracker,
    spy_exporter: SpyExporter,
    spy_plotter: SpyPlotter,
) -> Evaluator:
    return Evaluator(
        metric_names=["mse", "mae"],
        models=SimpleNamespace(),
        targets=target_transform_manager,
        tracker=spy_tracker,
        artifacts=SimpleNamespace(
            write_json=lambda path, payload: path,
        ),
        exporter=spy_exporter,
        plotter=spy_plotter,
        shap_logger=NoOpShapLogger(),
    )


class FakeAdapter:
    predictions_by_uri = {}
    created = []

    def __init__(self, params=None, seed=1):
        self.params = params or {}
        self.seed = seed
        self.model = SimpleNamespace(name="fake-model")
        self.fit_calls = []
        self.saved_paths = []
        FakeAdapter.created.append(self)

    def fit(self, X, y, **kwargs):
        self.fit_calls.append((X, y, kwargs))

    def predict(self, X, **kwargs):
        return self.prediction

    def save(self, path):
        self.saved_paths.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake model")

    def log_to_mlflow(self, name="model"):
        return SimpleNamespace(model_id=name)

    @classmethod
    def load_from_mlflow(cls, uri):
        adapter = cls()
        adapter.prediction = cls.predictions_by_uri[uri]
        return adapter

    @classmethod
    def configure_mlflow_tracking(cls):
        return None

    def get_training_summary(self):
        return {"best_iteration": 3, "best_score": 0.25}


@pytest.fixture
def fake_adapter():
    FakeAdapter.predictions_by_uri = {}
    FakeAdapter.created = []
    return FakeAdapter


@pytest.fixture
def model_manager(fake_adapter, target_transform_manager, spy_tracker):
    config = SimpleNamespace(
        mode="multi_target",
        params={"depth": 2},
        params_by_target=None,
        save_path=Path("model.json"),
    )
    return ModelManager(
        adapter_cls=fake_adapter,
        model_config=config,
        seed=7,
        tracker=spy_tracker,
        model_outputs=target_transform_manager.model_outputs,
    )


@pytest.fixture
def final_test_runner(target_transform_manager, spy_tracker):
    return FinalTestRunner(
        repository=SimpleNamespace(),
        models=SimpleNamespace(),
        targets=target_transform_manager,
        evaluator=SimpleNamespace(),
        tracker=spy_tracker,
        selection_config=SimpleNamespace(metric="mse", direction="minimize"),
        metric_names=["mse", "mae"],
    )


@pytest.fixture
def tuner(model_manager, target_transform_manager, spy_tracker):
    config = SimpleNamespace(
        enabled=True,
        objective_metric="mse",
        direction="minimize",
        objective_aggregation="mean",
        target_weights=None,
        n_trials=1,
        search_space={
            "integer_param": [1, 3],
            "float_param": [0.1, 0.9],
            "choice_param": ["a", "b"],
        },
    )
    return HyperparameterTuner(
        config=config,
        models=model_manager,
        targets=target_transform_manager,
        tracker=spy_tracker,
        seed=3,
    )


@pytest.fixture
def preprocess_schema():
    return SimpleNamespace(
        date_column="date",
        group_column="site",
        sensors={"sensor_group": ["sensor1"]},
        targets=["target"],
    )


@pytest.fixture
def rolling_feature_config():
    return SimpleNamespace(
        windows=[2],
        statistics=["mean", "max", "count", "slope"],
    )


@pytest.fixture
def preprocess_feature_config(rolling_feature_config):
    return SimpleNamespace(rolling=rolling_feature_config)


@pytest.fixture
def raw_preprocess_frame():
    return pd.DataFrame(
        {
            "site": ["A", "A", "B", "B"],
            "date": [
                "2024-01-01",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
            ],
            "sensor1": [1.0, 3.0, 10.0, 12.0],
            "extra_feature": [5.0, 6.0, 7.0, 8.0],
            "target": [10.0, 30.0, 100.0, 120.0],
            "chime_diatoms_ug_per_l": [1.0, 2.0, 3.0, 4.0],
            "chime_cyanobacteria_ug_per_l": [1.0, 2.0, 3.0, 4.0],
            "chime_others_ug_per_l": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def model_input_frame():
    return pd.DataFrame(
        {
            "site": ["A", "A", "A"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "sensor1": [10.0, 11.0, 12.0],
            "sensor1_mean_2": [1.0, 2.0, 3.0],
            "extra_feature": [5.0, 6.0, 7.0],
            "target": [10.0, 20.0, 30.0],
            "chime_diatoms_ug_per_l": [None, 1.0, None],
            "chime_cyanobacteria_ug_per_l": [None, 1.0, None],
            "chime_others_ug_per_l": [None, 1.0, None],
        }
    )


@pytest.fixture
def prepared_training_dataset():
    index = pd.Index(range(4))
    return PreparedTrainingDataset(
        X=pd.DataFrame({"sensor1_mean_2": [1.0, 2.0, 3.0, 4.0]}, index=index),
        y=pd.DataFrame({"target": [10.0, 20.0, 30.0, 40.0]}, index=index),
        metadata=pd.DataFrame(
            {
                "year": [2021, 2022, 2023, 2024],
                "date": pd.to_datetime(
                    ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"]
                ),
            },
            index=index,
        ),
        feature_columns=["sensor1_mean_2"],
    )


@pytest.fixture
def preprocess_config(tmp_path, preprocess_schema, rolling_feature_config):
    (tmp_path / ".git").mkdir()
    input_path = tmp_path / "input.csv"
    input_path.write_text("date,id,target\n2024-01-01,a,1\n")
    metadata = SimpleNamespace(
        sites=["site_a"],
        dataset_schema=preprocess_schema,
        splits=SimpleNamespace(final_test_year=2024),
        model_dump=lambda mode="json": {"targets": ["target"]},
    )
    storage = SimpleNamespace(
        base_path=tmp_path / "datasets",
        registry_path=tmp_path / "registry.yml",
        description="test dataset",
    )
    return SimpleNamespace(
        input_dataset=SimpleNamespace(path=input_path),
        output_dataset=SimpleNamespace(storage=storage, metadata=metadata),
        features=SimpleNamespace(rolling=rolling_feature_config),
        model_dump=lambda mode="json": {
            "input_dataset": {"path": str(input_path)},
            "features": {},
        },
    )
