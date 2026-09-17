from typer.testing import CliRunner

import pftnc
from pftnc.cli import app


def test_package_exposes_version() -> None:
    assert pftnc.__version__


def test_cli_exposes_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == pftnc.__version__
