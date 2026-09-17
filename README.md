# PFTNC

[![Unittest pftnc](https://github.com/ESA-Heatwise/pftnc/actions/workflows/unittest.yml/badge.svg)](https://github.com/ESA-Heatwise/pftnc/actions/workflows/unittest.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff) 
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)


![](content/pftnc-ascii.png)

`PFTNC` **(Phytoplankton Functional Types - NowCasting)** is a reproducible machine 
learning and dataset engineering framework for aquatic remote sensing workflows.

The package is designed to simulate and feature-engineer datasets that mimic 
future ESA Earth observation missions such as **CHIME** and **LSTM**, 
enabling robust experimentation before these missions become fully operational.

The package supports:

- sensor-aware dataset simulation
- reproducible feature engineering
- dataset versioning and lineage tracking
- ML-ready dataset folds generation
- experiment tracking with MLflow
- model training
- model packaging
- inference

PFTNC focuses on robust, reproducible, and configurable workflows for 
building machine learning datasets and training ML models from simulated and 
feature-engineered datasets.


## Getting Started

Once you clone this repo, run:

```bash
pixi i
```

If you are using PyCharm, set your Interpreter following [this](https://pixi.prefix.dev/v0.63.2/integration/editor/jetbrains/#alternate-approach-using-environmentstxt)


### Jupyter Notebooks
Jupyter is already installed in this environment, just run:
```bash
jupyter lab
```

and go the notebooks folder and get started with your exploration.

Move your data to the `data/` folder.

### Mlflow Experiment Tracking

This environment also comes with Mlflow installed.

To run the server, do this:
```bash
mlflow server
```

Then open:

```text
http://127.0.0.1:5000
```

To learn about how to use mlflow, you can read it on MLflow [docs](https://mlflow.org/docs/latest/ml/#traditional-ml-and-deep-learning-with-mlflow)


## CLI Usage

### Create Example Configurations

Create starter configuration files in the current directory:

```bash
pftnc create-config
```

This creates:

```text
./config/
```

folder with example YAML configuration files.

You can also specify a custom output location:

```bash
pftnc create-config --output-path ./my-configs
```

Or copy from another config directory:

```bash
pftnc create-config \
  --source-path ./existing-configs \
  --output-path ./new-configs
```

You can also copy one YAML file with any filename:

```bash
pftnc create-config \
  --source-path ./existing-configs/my-experiment.yaml \
  --output-path ./new-configs
```

When `--source-path` points to a directory, only the supported configuration
filenames are copied (`config.yml`, `dataset_simulation_config.yml`, and
`inference.yml`). To copy an arbitrarily named YAML file, pass the file itself
as `--source-path`.

---

### Dataset Simulation

Generate simulated datasets using configurable:
- revisit frequencies
- cloud masking
- Gaussian sensor noise

```bash
pftnc simulate <path-to-your-config>
```

The simulation pipeline:
- loads raw observations
- applies sensor simulations
- versions the generated dataset
- stores metadata and configuration snapshots
- exports parquet datasets

---

### Feature Engineering / Preprocessing

Run preprocessing and feature engineering pipelines:

```bash
pftnc preprocess <path-to-your-config>
```

Feature engineering currently supports:
- rolling statistics
- temporal features
- transforms target values if specified
- dataset quality validation
- train/validation/test fold generation

Generated datasets are versioned automatically and stored in a registry-managed 
structure.

### Model packaging and inference

After training, create a portable runtime bundle containing the fold models and
the metadata required to reproduce preprocessing:

```bash
pftnc package-model \
  --training-run-path outputs/<training-run> \
  --output-path model_bundle
```

Copy the packaged inference template with `pftnc create-config`, then update its
bundle, input, mode, and output paths. Runtime scoring is config-driven:

```bash
pftnc infer config/inference.yml
```

`latest` scores the final row for each site. `from_date` scores every row from
the inclusive `start_date` through the end of the input. Available model
selection strategies are `best_overall`, `best_per_target`, `ensemble_simple`,
`ensemble_weighted`, and `ensemble_weighted_per_target`.

---


## Testing

As we use pixi, you can run tests using pytest by running this command:

```bash
pixi run tests
```

## Formatting and linting


```bash
pixi run format
```

and 
```bash
pixi run lint
```

## Development

Run formatting:

```bash
pixi run format
```

Run linting (will also run mypy):

```bash
pixi run lint
```

Run tests:

```bash
pixi run test
```
