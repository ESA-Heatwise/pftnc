import logging
import shutil
from importlib import resources
from pathlib import Path

import typer

from pftnc import __version__
from pftnc.logging import setup_logger

setup_logger()

logger = logging.getLogger(__name__)

app = typer.Typer(help="PFTNC (Phytoplankton Functional Types - NowCasting) CLI")


def version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the installed PFTNC version and exit.",
    ),
) -> None:
    """Configure global CLI options."""


PACKAGE_CONFIG_DIR = Path(str(resources.files("pftnc") / "config"))
PACKAGED_CONFIG_FILES = (
    "config.yml",
    "dataset_simulation_config.yml",
    "inference.yml",
)


@app.command(
    help="""
    Generate simulated datasets from raw observations using the
    configured sensor simulations, revisit schedules, cloud masking,
    and noise models.
    """
)
def simulate(config_path: Path):
    """
    Generate simulated datasets from raw observations.
    """
    from pftnc.dataset_simulation_config import load_dataset_gen_config
    from pftnc.simulation.simulate import simulate_dataset

    typer.echo("Loading simulation configuration...")
    config = load_dataset_gen_config(config_path)

    typer.echo("Running dataset simulation pipeline...")
    output_path = simulate_dataset(config)

    typer.secho(
        f"Simulation completed successfully.\nDataset saved at:\n{output_path}",
        fg=typer.colors.GREEN,
        bold=True,
    )


@app.command(
    help="""
    Run preprocessing and feature engineering pipeline on a dataset.
    """
)
def preprocess(config_path: Path):
    """
    Run feature engineering pipeline.
    """
    from pftnc.preprocess.preprocess import preprocess_dataset

    from .config import load_config

    typer.echo("Loading preprocessing configuration...")
    config = load_config(config_path)

    typer.echo("Running preprocessing pipeline...")
    artifact = preprocess_dataset(config.feature_engineering)

    typer.secho(
        f"Preprocessing completed successfully.\nDataset saved at:\n"
        f"{str(artifact.base_path / artifact.version)}",
        fg=typer.colors.GREEN,
        bold=True,
    )


@app.command(
    help="""
    Create example configuration files in a target directory.

    By default, packaged configs are copied into ./config.

    Optionally provide:
    --source-path to copy one YAML file or supported config files from another
    directory.
    --output-path to control destination location.
    """
)
def create_config(
    output_path: Path = typer.Option(
        Path("./config"),
        help="Directory where config files will be copied.",
    ),
    source_path: Path | None = typer.Option(
        None,
        help="Optional source config directory to copy from.",
    ),
):
    """
    Copy template configuration files.
    """
    if source_path is None:
        source_dir = PACKAGE_CONFIG_DIR
        source_files = [source_dir / filename for filename in PACKAGED_CONFIG_FILES]
        logger.info(
            "Using packaged configuration templates from %s",
            source_dir,
        )

    else:
        source = source_path.resolve()

        if not source.exists():
            logger.error(
                "Source config path does not exist: %s",
                source,
            )

            raise typer.BadParameter(f"Source config path not found: {source}")

        if source.is_file():
            if source.suffix.lower() not in {".yml", ".yaml"}:
                raise typer.BadParameter(
                    f"Source config file must be YAML (.yml or .yaml): {source}"
                )
            source_files = [source]
        elif source.is_dir():
            source_files = [
                source / filename
                for filename in PACKAGED_CONFIG_FILES
                if (source / filename).is_file()
            ]
        else:
            raise typer.BadParameter(
                f"Source config path must be a file or directory: {source}"
            )

        logger.info(
            "Using user-provided config source: %s",
            source,
        )

    missing_files = [path for path in source_files if not path.is_file()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise typer.BadParameter(f"Config file(s) not found: {missing}")

    if not source_files:
        source_label = source_path.resolve() if source_path is not None else source_dir
        raise typer.BadParameter(f"No YAML config files found in {source_label}")

    output_dir = output_path.resolve()
    logger.info(
        "Creating config directory at %s",
        output_dir,
    )

    if output_dir.exists() and not output_dir.is_dir():
        raise typer.BadParameter(f"Output path is not a directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for source_file in source_files:
        destination = output_dir / source_file.name
        if source_file == destination:
            raise typer.BadParameter(
                f"Source and destination are the same file: {source_file}"
            )
        shutil.copy2(source_file, destination)
        logger.info(
            "Copied %s -> %s",
            source_file,
            destination,
        )

    typer.secho(
        f"Successfully copied {len(source_files)} config file(s) to:\n{output_dir}",
        fg=typer.colors.GREEN,
        bold=True,
    )


@app.command(
    help="""
    Run training pipeline using the provided config file.

    This executes the training workflow including:
    - dataset loading
    - fold-based training
    - tuning (if enabled)
    - evaluation and logging
    """
)
def train(config_path: Path):
    """
    Run model training pipeline.
    """

    logger.info(
        "[TRAINING PIPELINE] Starting training pipeline... loading libraries and config..."
    )

    if not config_path.exists():
        config_not_found_error = f"Config file not found: {config_path}"
        logger.error("[CONFIG] %s", config_not_found_error)
        raise typer.BadParameter(config_not_found_error)

    logger.info("[CONFIG] Loading training configuration")
    from pftnc.config import load_config

    config = load_config(config_path)

    logger.info("[PIPELINE] Running training pipeline")
    from pftnc.train.train import run_training

    run_training(config, config_path)
    logger.info("[PIPELINE] Training completed successfully")


@app.command("package-model")
def package_model(
    training_run_path: Path = typer.Option(
        ...,
        "--training-run-path",
        "--training_run_path",
        help="Completed training run directory containing run_summary.json.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output-path",
        "--output_path",
        help="New directory where the inference bundle will be created.",
    ),
):
    """Create a portable inference bundle from a completed training run."""
    from pftnc.predict.bundle import package_model_bundle

    bundle = package_model_bundle(training_run_path, output_path)
    typer.secho(
        f"Model bundle created successfully at:\n{bundle}",
        fg=typer.colors.GREEN,
        bold=True,
    )


@app.command("infer")
def infer(config_path: Path):
    """Run config-driven inference using a packaged model bundle."""
    from pftnc.inference_config import load_inference_config
    from pftnc.predict.inference import run_inference

    config = load_inference_config(config_path)
    result = run_inference(config)
    output_message = (
        f"Predictions saved at:\n{config.output_path}"
        if config.save_outputs
        else "Outputs were not saved (save_outputs=false)."
    )
    typer.secho(
        f"Inference completed successfully.\nRows: {len(result)}\n{output_message}",
        fg=typer.colors.GREEN,
        bold=True,
    )


if __name__ == "__main__":
    app()
