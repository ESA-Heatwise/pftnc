## Version 0.1.1

- Made `pftnc create-config` accept a single YAML file with any filename.
- Directory sources now copy only the supported configuration filenames.
- Updated documentation and expanded CLI test coverage.

## Version 0.1.0

- Exposed the package version through `pftnc.__version__` and the
  `pftnc --version` CLI option.
- Initial public release.



## Version 0.0.9

- Made new datasets, experiment runs, and model bundles portable across
  machines with sibling data and experiments repositories: dataset references
  are workspace-relative, model references are training-run-relative, and
  bundle contents are self-contained. Model packaging accepts schema-version-1
  runs produced by the current package.
- Simplified provenance: new artifacts reuse an input dataset's
  `provenance.yml` when present; otherwise they record an explicit
  `unknown_input_dataset` boundary without guessing, recreating, or modifying
  older datasets. Older datasets therefore remain trainable.
- Training runs now write `versions.txt` with the PFTNC, Python, platform, and
  installed package versions. This is environment-agnostic and no longer
  depends on Pixi.
- Inference now supports `save_outputs` (default `true`) to control writing the
  prediction CSV and resolved inference config together, plus `save_index`
  (default `false`) to control CSV index output.
- Updated configuration templates and `USAGE.md` based on the changes mentioned
  above.
- Fixed wheel packaging so built distributions include the complete `pftnc` 
  package and CLI, not only metadata.
- Removed unnecessary files genrerated from cookiecutter.

## Version 0.0.8

- Added portable inference workflow for trained models:
  - added `pftnc package-model` to create a self-contained bundle from a
    completed training run;
  - added `pftnc infer <inference.yml>` for config-driven predictions from raw
    CSV or Parquet input;
  - added inference configuration support for `latest` and `from_date` row
    selection modes;
- Added a dedicated Pydantic inference configuration module and support for
  saving the resolved `inference_config.yml` beside inference predictions;
- Inference outputs now optionally include the exact model input features via
  the `save_model_inputs` configuration option.
- Added model input/output schema tracking through `model_io_schema.yml` so
  training and inference use the same feature and target contracts.
- Expanded preprocessing artifacts with `feature_schema.yml`, fold metadata,
  dataset metadata, provenance, and Pixi package-version records, in addition
  to the train/validation/test Parquet files.
- Updated `USAGE.md` with the inference commands, configuration, model bundle
  layout, output format, and preprocessing/training artifact descriptions.
- Remove unnecessary files created from cookiecutter.


## Version 0.0.7

- Added support for loading input datasets from both Parquet and CSV formats 
  (earlier it was just Parquet).
- Added gap-only training/evaluation rows where current CHIME values are missing.
- Added last_observed CHIME features in preprocessing after masked rolling features.
- Use `Agg` backend in Matplotlib for CLI plotting stability.
- Added configurable per-target `log_delta` target transformations for training.
- Added inverse transformation of `log_delta` predictions back to physical 
  concentrations before metrics, plots, CSV exports, and ensembles.
- The CLI logs are now written to `logs/pftnc_<timestamp>.log` files.
- Added configurable model modes for either one multi-target model or one 
  independent model per target.
- Added `per-target` model parameter configuration.
- Added independent Optuna hyperparameter tuning studies for each target in 
 `per-target` mode.
- Added target-specific MLflow child runs for training and tuning to avoid mixed
  autologged metrics.
- Added support for evaluating and ensembling predictions from both `multi-target`
  and `per-target` models.
- Added `description` to the main MLflow training runs
- Refactor `train` and `preprocess` submodule into further modular submodules
  for readability, extensibility and better testability.
- Refactored and added more tests for the `train` and `preprocess` submodules.
- Made model feature selection configurable: rolling features and non-CHIME raw
  sensors are selected automatically, while experiment-specific columns are
  controlled by training `include` and `exclude` lists.
- Moved model feature selection from preprocessing to training so multiple
  experiments can reuse one feature-engineered dataset with different inputs.
- Expanded the generated `config.yml` template with feature-selection, model,
  target-transformation, metric, and tuning documentation.


## Version 0.0.6

### Features

- Converted training workflow from NOTEBOOK 3 into a modular Python pipeline under:
  - `src/pftnc/train/train.py`
  - `src/pftnc/train/training.py`
- Added CLI integration for training:
  - `pftnc train <config.yml>`
- Implemented fold-based training pipeline with:
  - k-fold training and evaluation
  - optional Optuna hyperparameter tuning
  - MLflow logging for models, metrics, artifacts
- Standardized experiment outputs under:
  - `<output_dir>/<run_id>/<fold_x>/`
- Implemented run ID naming convention:
  - `<site>_mlrun_<timestamp>`
- Added structured logging across training pipeline with:
  - module and function-level context
  - semantic log markers (`[CONFIG]`, `[PIPELINE]`, `[FOLD START]`, etc.)
- Added final test evaluation stage:
  - evaluates all fold models on held-out dataset
  - compares ensemble strategies:
    - simple mean
    - validation-weighted
    - per-target weighted
- Added per-fold artifact logging:
  - predictions CSV
  - plots
- Use explicit template filenames when copying config files instead of globbing.

### Testing

- Added pytest-based unit tests for training pipeline
- Tests use mocking for:
  - model training
  - MLflow interactions
  - dataset loading
- Moved shared fixtures into `tests/conftest.py`

### Improvements

- Added structured logging in training module
- Improved type annotations (including `ModelInfo` for MLflow logging)
- Added assertions for config validation in training workflow
- Improved path handling using `expanduser().resolve()`


## Version 0.0.5

- Refactored project structure by moving pipeline code from notebooks into the
  - `src/pftnc/simulation` package
  - `src/pftnc/preprocess` package
- Added pytest-based test suite covering preprocessing, simulation and utilities.
- Added Pixi environment/package version tracking for reproducibility
- Added path resolution supporting relative paths, absolute paths, and URLs for `AppConfig`
- Added centralized package-wide logging using Python `logging`
- Added CLI subcommands for 
  - dataset simulation (`simulation`)
  - preprocessing workflows (`preprocess`)
  - creating configs (`create-config`)
- Fixed typing and validation issues identified by mypy
- Updated README.md
- Updated GitHub workflow `unittest.yml`

### Bug fixes

- Fixed a bug where the data pipeline would crash with a `FileNotFoundError` if 
  commands were run from any directory other than the exact project root.
- Changed `register_dataset_version` and  `read_input_dataset` so it no longer 
  writes incorrect relative path strings into the registry files when generated 
  from different folders.


## Version 0.0.4

- Updated dataset versioning from `v1, v2...` to `
  {sites}_{timestamp}_{config_hash}_{random_number}`
- Added a `config.yml` file to each simulated dataset directory to ensure
  reproducibility and preserve the exact generation configuration.
- Refactored dataset simulation config out of `AppConfig` into `DatasetSimConfig`
- For `AppConfig`, the following changes were made:
  - Added 
    - `DatasetArtifact` for runtime dataset outputs
    - `training.input_dataset`
  - Removed 
    - `training.dataset_version`
  - Renamed 
    - `DatasetInputStorageConfig` → `DatasetInputConfig`
    - `DatasetOutputStorageConfig` → `DatasetOutputConfig`
    - `DatasetRef` → `DatasetSpec`
  - Decoupled `training` from `feature_engineering.output_dataset`
  - Trainer now resolves datasets from `training.input_dataset`
- Running the feature engineering (FE) step now returns `DatasetArtifact` 
  which can be fed into the training dynamically 
  using `.to_input_config()` method.
- Improved pipeline orchestration and standalone stage execution i.e. 
  model training now can be run in a pipeline with FE 
  step or standalone from a previous output of FE step.
- Implemented initial version of rolling window data handling by filling in 
  NaNs for missing dates, calculating stats and remove the filled NaNs.

## Version 0.0.3

- Renamed ALR (Additive Log-Ratio) transform to **log-fractions** throughout:
  config key `preprocessing.alr` → `preprocessing.log_fractions`, Pydantic class
  `ALRConfig` → `LogFractionsConfig`, metadata file `alr_meta.json` →
  `log_fractions_meta.json`, and all output columns `alr_*` → `log_fractions_*`;
  the transform computes `log((t + ε) / (total + ε))` per non-denominator target,
  which is a log fractional share of total — not standard ALR
- Added `check_target_quality` QC step in nb02: raises with a full diagnostic
  report if any target column contains NaN or negative values; must pass before
  `build_dataset` is called; `generate_year_folds` also raises rather than
  silently dropping rows, so data issues are always surfaced explicitly
- Added log-fractions target transform: composition targets
  (`diatoms`, `cyanobacteria`, `others`) are transformed to
  `(total_ug_per_l, log_fractions_diatoms_ug_per_l, log_fractions_cyanobacterial_ug_per_l)`
  before saving fold parquets; controlled via `preprocessing.log_fractions` in
  config (`enabled`, `epsilon`, `denominator`)
- All metrics and ensemble outputs are back-transformed to physical units (µg/L)
  before logging, so MLflow values are always interpretable regardless of
  log-fractions setting
- Added `std` and `count` as configurable rolling statistics in `run_preprocessor`
- Added `{sensor}_days_since_obs` feature per sensor: days elapsed since the last
  non-NaN observation, giving the model a direct signal for how stale each input is
- Added cyclic day-of-year features (`doy_sin`, `doy_cos`) to capture seasonal
  bloom dynamics without requiring very long rolling windows
- Extended default rolling windows from `[3, 5, 10, 15, 30]` to
  `[3, 5, 10, 15, 30, 60]` days
- Added `LogFractionsConfig` Pydantic model to `config.py`; `PreprocessingConfig`
  gains an optional `log_fractions` field
- `load_dataset_splits` now reads all columns from y parquets directly, making it
  transparent to whether log-fractions targets or raw targets are stored
- `Trainer` auto-detects `log_fractions_meta.json` in the dataset version directory
  and applies back-transform in `_evaluate_model`, `_tune`, and `_final_test`
- Optuna objective is now optimised in physical space when log-fractions is enabled
- Added per-target best model selection: `_select_best_model` now returns both an
  overall winner and an independent best-fold per target
- Added per-target weighted ensemble (`ensemble_weighted_per_target`) in final test:
  each target column is weighted independently based on per-target val scores
- Added configurable Optuna objective aggregation via `objective_aggregation` in
  config: `mean`, `worst` (most conservative), or `weighted` (custom
  per-target weights via `target_weights` in config)
- MLflow metric naming unified to `{split}/{target}/{metric}` throughout
  (e.g. `val/diatoms_ug_per_l/rmse`) which renders as a folder tree in the UI
- Optuna objective metric key renamed from `tuning/all_targets/{metric}` to
  `tuning/objective/{metric}` to accurately reflect what it is
- Prediction scatter plots now include a perfect-prediction reference line
- Final test printout shows all three ensembles side by side for easy comparison
- Weighted ensemble weighting now respects `best_model_selection.direction`
  for both error metrics and r2
- Updated the `direction` parameter in `TuningConfig`  to use a `Literal` type to strictly enforce 
  `minimize` or `maximize` values. 
- Added validation checks to ensure `per_target_scores` is not empty, `target_weights` 
  are properly configured, prediction shape matches the number of targets
  and valid folds exist before model selection.
- Updated `_evaluate_ensemble` to now also logs csv and plots to mlflow.
- Removed the redundant `_save_ensemble_csv` function.


### Bug fixes
- Fixed negative target values (e.g. `others = CHL − diatoms − cyano` going
  negative due to measurement noise) causing NaN in the log-fractions transform;
  `check_target_quality` now raises before dataset generation so the issue is
  visible and must be resolved explicitly
- Fixed `compute_slope` to preserve original time-axis positions when NaNs are
  present in the rolling window (previously used sequential indices, ignoring
  gaps); degenerate cases now return `np.nan` instead of `0`
- `build_dataset` hardcoded the dataset path to `"v1"` instead of using the
  version string returned by `generate_year_folds`
- `log_metrics` closure in `Trainer.train()` captured the wrong `result`
  due to late binding, so all folds were logging the last fold's metrics
- Final test weighted ensemble hardcoded `"rmse"` as the weight metric,
  crashing if `rmse` was not in the configured metrics list
- `_select_best_model` accepted `metric` and `direction` as arguments
  but immediately overwrote them from config, silently discarding the inputs
- Optuna objective score used `metric_fn(y_val.values, preds)` on 2D
  arrays, relying on sklearn's silent `multioutput='uniform_average'` default
  with no way to override it but now computed per-target explicitly


## Version 0.0.2 

- Added Data generation, preprocessing, and model training notebooks
- Implemented MLflow logging throughout the training pipeline (metrics, artifacts, 
  SHAP plots, prediction CSVs)
- Added initial user guide for the MLOps pipeline
- Improved `config.yml` structure

## Version 0.0.1

- Scaffolded the project structure using the [Gaiaflow cookiecutter template](https://github.com/bcdev/gaiaflow-cookiecutter)
- Implemented the initial configuration system using `config.yml` and Pydantic `BaseModel`
- Added the first version of the CLI interface `pft`
