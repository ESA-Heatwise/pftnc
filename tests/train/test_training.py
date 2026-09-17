from types import SimpleNamespace

from pftnc.train.data import FoldDataset
from pftnc.train.io import build_model_io_schema, write_model_io_schema
from pftnc.train.schemas import FoldResult, ModelBundleReference, ModelReference
from pftnc.train.training import Trainer, set_global_seed


def test_seed_helper_makes_numpy_reproducible():
    import numpy as np

    set_global_seed(12)
    first = np.random.random(3)
    set_global_seed(12)

    assert np.array_equal(first, np.random.random(3))


def test_fold_summary_serializes_portable_dataset_and_model_paths(tmp_path):
    workspace = tmp_path / "workspace"
    data_repo = workspace / "pftnc-data"
    experiments = workspace / "pftnc-experiments"
    (data_repo / ".git").mkdir(parents=True)
    (experiments / ".git").mkdir(parents=True)
    dataset = data_repo / "data" / "feature_engineered" / "site" / "v1"
    fold_dataset = dataset / "fold_2020"
    run = experiments / "outputs" / "run-2020"
    model = run / "fold_2020" / "model.json"
    fold_dataset.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    model.write_text("model")
    bundle = ModelBundleReference(
        mode="multi_target",
        models={
            "__multi_target__": ModelReference(
                uri="models:/fake",
                local_path=model,
                model_info="fake",
            )
        },
        feature_columns=("sensor",),
    )
    result = FoldResult(
        fold_name="fold_2020",
        dataset_version="v1",
        dataset_path=dataset,
        fold_dataset_path=fold_dataset,
        model_outputs=["target"],
        model_bundle=bundle,
        params={},
        train_metrics={},
        validation_metrics={},
        test_metrics={},
    )

    summary = result.to_summary_dict(run_dir=run)

    assert summary["schema_version"] == 1
    assert summary["dataset_path"] == "pftnc-data/data/feature_engineered/site/v1"
    assert summary["fold_dataset_path"] == (
        "pftnc-data/data/feature_engineered/site/v1/fold_2020"
    )
    assert summary["model_path"] == "fold_2020/model.json"


def test_train_fold_transforms_fits_and_evaluates_all_splits(
    target_transform_manager,
    prepared_split,
    spy_tracker,
    tmp_path,
):
    bundle = ModelBundleReference(
        mode="multi_target",
        models={
            "__multi_target__": ModelReference(
                uri="models:/fake",
                local_path=tmp_path / "model.json",
                model_info="fake",
            )
        },
        feature_columns=tuple(prepared_split.X.columns),
    )
    dataset = FoldDataset(
        train=prepared_split,
        validation=prepared_split,
        test=prepared_split,
    )
    evaluated_splits = []

    components = SimpleNamespace(
        repository=SimpleNamespace(
            dataset_version="v1",
            dataset_path=tmp_path / "dataset",
            load_fold=lambda path: dataset,
        ),
        targets=target_transform_manager,
        tuner=SimpleNamespace(
            enabled=False,
            resolve=lambda *args: {"depth": 2},
        ),
        models=SimpleNamespace(
            mode="multi_target",
            log_effective_params=lambda params: None,
            fit=lambda **kwargs: bundle,
        ),
        evaluator=SimpleNamespace(
            evaluate_model=lambda bundle, data_split, run_dir, split: (
                evaluated_splits.append(split)
                or {
                    target: {"mse": 0.0}
                    for target in target_transform_manager.physical_targets
                }
            ),
        ),
        tracker=spy_tracker,
        artifacts=SimpleNamespace(write_json=lambda path, payload: path),
    )
    config = SimpleNamespace(
        training=SimpleNamespace(
            target_transformations=SimpleNamespace(
                model_dump=lambda: {"type": "log_fractions"},
            ),
        ),
    )
    trainer = Trainer.__new__(Trainer)
    trainer.components = components
    trainer.config = config

    write_model_io_schema(
        tmp_path / "model_io_schema.yml",
        build_model_io_schema(
            input_columns=list(prepared_split.X.columns),
            model_output_columns=list(target_transform_manager.model_outputs),
            physical_target_columns=list(target_transform_manager.physical_targets),
            model_mode="multi_target",
        ),
    )

    result = trainer._train_fold("fold_2020", tmp_path / "fold_2020", tmp_path)

    assert result.fold_name == "fold_2020"
    assert evaluated_splits == ["train", "validation", "test"]
    assert result.params == {"depth": 2}

    schema = (tmp_path / "model_io_schema.yml").read_text()
    assert "input_columns:" in schema
    assert "output_columns:" in schema
    assert "physical_target_columns:" in schema
