import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from pftnc.preprocess.folds import write_year_folds


def test_write_year_folds_creates_rotating_training_folds(
    prepared_training_dataset,
    tmp_path,
):
    schema = SimpleNamespace()

    folds = write_year_folds(
        prepared_training_dataset,
        output_path=tmp_path / "dataset",
        dataset_schema=schema,
        final_test_year=2024,
    )

    assert set(folds) == {"fold_2021", "fold_2022", "fold_2023"}
    assert folds["fold_2021"] == {
        "train_years": [2023],
        "val_year": 2021,
        "test_year": 2022,
    }
    assert (tmp_path / "dataset" / "final_test" / "test_X.parquet").is_file()
    assert (tmp_path / "dataset" / "fold_2021" / "val_metadata.parquet").is_file()
    assert json.loads((tmp_path / "dataset" / "folds.json").read_text()) == folds

    validation = pd.read_parquet(tmp_path / "dataset" / "fold_2021" / "val_y.parquet")
    assert validation["target"].tolist() == [10.0]

    metadata = pd.read_parquet(
        tmp_path / "dataset" / "fold_2021" / "val_metadata.parquet"
    )
    assert metadata.index.tolist() == validation.index.tolist()


def test_write_year_folds_rejects_missing_final_test_year(
    prepared_training_dataset,
    tmp_path,
):
    with pytest.raises(ValueError, match="final_test_year"):
        write_year_folds(
            prepared_training_dataset,
            output_path=tmp_path / "missing_final_year",
            dataset_schema=SimpleNamespace(),
            final_test_year=2030,
        )


def test_write_year_folds_requires_three_training_years(
    prepared_training_dataset,
    tmp_path,
):
    short = replace(
        prepared_training_dataset,
        X=prepared_training_dataset.X.iloc[:3],
        y=prepared_training_dataset.y.iloc[:3],
        metadata=prepared_training_dataset.metadata.iloc[:3],
    )

    with pytest.raises(ValueError, match="three non-final-test years"):
        write_year_folds(
            short,
            output_path=tmp_path / "too_few_years",
            dataset_schema=SimpleNamespace(),
            final_test_year=2023,
        )
