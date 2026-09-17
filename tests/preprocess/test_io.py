from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pftnc.preprocess.io import read_input_dataset, resolve_dataset_file


def test_resolve_dataset_file_accepts_file_or_single_file_directory(tmp_path):
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"value": [1, 2]}).to_csv(csv_path, index=False)

    assert resolve_dataset_file(csv_path) == csv_path
    assert resolve_dataset_file(tmp_path) == csv_path


def test_resolve_dataset_file_rejects_missing_unsupported_and_ambiguous_paths(
    tmp_path,
):
    with pytest.raises(FileNotFoundError):
        resolve_dataset_file(tmp_path / "missing.csv")

    unsupported = tmp_path / "input.txt"
    unsupported.write_text("data")
    with pytest.raises(ValueError, match="Unsupported dataset type"):
        resolve_dataset_file(unsupported)

    (tmp_path / "one.csv").write_text("value\n1\n")
    (tmp_path / "two.csv").write_text("value\n2\n")
    with pytest.raises(ValueError, match="exactly one"):
        resolve_dataset_file(tmp_path)


def test_read_input_dataset_reads_direct_csv(tmp_path):
    path = tmp_path / "input.csv"
    expected = pd.DataFrame({"value": [1, 2]})
    expected.to_csv(path, index=False)

    result = read_input_dataset(SimpleNamespace(path=path))

    pd.testing.assert_frame_equal(result, expected)


def test_read_input_dataset_requires_complete_registry_configuration():
    with pytest.raises(ValueError, match="Registry mode"):
        read_input_dataset(
            SimpleNamespace(
                path=None,
                registry_path=None,
                base_path=Path("datasets"),
                version="v1",
            )
        )
