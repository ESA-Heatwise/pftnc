# PFTNC Usage Guide

## Before running project commands

This guide uses Pixi, but PFTNC may also be installed into another supported
Python environment. Activate the environment in every terminal before running
any `pftnc`, `mlflow`, Python, or Jupyter command:

```bash
pixi shell
```

All commands below assume that the `pixi shell` is active. If you open a new
terminal, change repository, or start a second terminal for MLflow, run
`pixi shell` again in that terminal first.

## 1. Workspace and repository setup

Create two sibling Git repositories: one for your data and one for your
experiments. PFTNC is installed as a package; cloning the PFTNC source is only
needed when developing PFTNC itself.

```text
workspace/
├── my-data/
│   └── .git/
└── my-experiments/
    └── .git/
```

For example:

```bash
mkdir -p workspace/my-data workspace/my-experiments
git -C workspace/my-data init
git -C workspace/my-experiments init
```

> NOTE: You can skip this step if you already have your data and experiments
> repositories available in the manner desribed above.

All PFTNC config paths are relative to `workspace/`. For example:

```yaml
base_path: my-data/data/simulated/site-a
output_dir: my-experiments/outputs
```

Do not use absolute paths or `../` paths in new configs. The repository names
are part of the portable artifact references, so retain them when moving a
workspace to another machine.

This project's own development workspace uses `pftnc-data` and
`pftnc-experiments`; those are examples, not required names.

### Using a local PFTNC checkout

Only do this when:

* developing `pftnc` locally, or
* testing functionality that exists locally but has not been released yet.

If developing unreleased PFTNC functionality, place the PFTNC checkout beside
your two repositories and configure a local editable dependency:

```toml
pftnc = { path = "../pftnc", editable = true }
```

Then refresh the environment:

```bash
pixi i
```

> For normal usage with a released `pftnc` version, **do not uncomment this 
> dependency**, instead update the tag to the newest version or whichever 
> version is required.

---

# 2. Simulate the raw dataset

Run this from your data repository, for example `my-data`.

If the `config/` directory does not exist:

```bash
pftnc create-config
```

Update:

```text
config/data_simulation_config.yml
```

The important fields are:

```yaml
project:
  seed: 42

simulation:
  enabled: true

  # Raw source dataset
  input_path: my-data/raw/dataset.csv

  # Where simulated datasets will be stored
  base_path: my-data/data/simulated/your_site

  # Registry tracking generated dataset versions
  registry_path: my-data/data/simulated/your_site/simulated.yaml

  description: "Simulated dataset"
  site_name: your_site
  date_column: time

  sensors:
    chime:
      revisit_days: 11
      noise_std: null
      columns:
        - chime_diatoms_ug_per_l
        - chime_cyanobacteria_ug_per_l
        - chime_others_ug_per_l

    lstm:
      revisit_days: 2
      noise_std: null
      columns:
        - lstm_lswt_c

    s2:
      revisit_days: 5
      noise_std: 0.03
      columns:
        - s2_chl_ug_per_l
        - s2_tur_mg_per_l
        - s2_cdom_1_per_m

    s3:
      revisit_days: 1
      noise_std: 0.04
      columns:
        - s3_chl_ug_per_l
        - s3_tur_mg_per_l
        - s3_cdom_1_per_m

  cloud_mask:
    winter: 0.75
    summer: 0.25
    default: 0.5
```

Run:

```bash
pftnc simulate config/data_simulation_config.yml
```

The command creates the simulated dataset and registers a dataset version in the configured registry.

---

# 3. Run feature engineering

Open:

```text
config/config.yml
```

Point `feature_engineering.input_dataset` to the simulated dataset created above.

## Registry-managed input

Recommended when consuming a simulated dataset:

```yaml
feature_engineering:
  input_dataset:
    base_path: my-data/data/simulated/your_site
    registry_path: my-data/data/simulated/your_site/simulated.yaml
    version: your_simulated_dataset_version
```

You can also use a dataset directly:

```yaml
feature_engineering:
  input_dataset:
    path: my-data/raw/dataset.csv
```

> Use **either** `path` **or** `base_path + registry_path`, not both.

Configure the output dataset and schema:

```yaml
  output_dataset:
    storage:
      base_path: my-data/data/feature_engineered/your_site
      registry_path: my-data/data/feature_engineered/your_site/feature_engineered.yaml
      description: Feature engineered dataset

    metadata:
      sites:
        - your_site

      dataset_schema:
        date_column: time
        group_column: site

        sensors:
          chime:
            - chime_diatoms_ug_per_l
            - chime_cyanobacteria_ug_per_l
            - chime_others_ug_per_l

          lstm:
            - lstm_lswt_c

          s2:
            - s2_chl_ug_per_l
            - s2_tur_mg_per_l
            - s2_cdom_1_per_m

          s3:
            - s3_chl_ug_per_l
            - s3_tur_mg_per_l
            - s3_cdom_1_per_m

        targets:
          - diatoms_ug_per_l
          - cyanobacterial_ug_per_l
          - others_ug_per_l

      splits:
        final_test_year: 2023

  features:
    rolling:
      windows: [3]
      statistics:
        - mean
        # - median
        # - max
        # - slope
        # - std
        # - count
```

Feature generation follows this rule:

- Rolling features are generated for every configured sensor, window, and
  statistic.
- Original sensors, `days_since_obs`, `last_observed`, and calendar features
  are retained as candidate columns.
- No model-specific feature selection happens during preprocessing.

The resulting candidate-column list is saved as `feature_schema.yml` in the
feature-engineered dataset artifact. Each training experiment selects its own
model columns from this shared dataset.

Then run:

```bash
# Run `pixi shell` first in this terminal.
pftnc preprocess config/config.yml
```

`pftnc preprocess` creates and registers the reusable feature-engineered
dataset.

After this step, note the **feature-engineered dataset registry and version**. You will use those values in the experiments repository.

## Preprocessing outputs

Each registered feature-engineered dataset version contains the model-ready
folds and the metadata needed to reproduce them:

```text
<feature_engineered_base_path>/<version>/
├── fold_<year>/
│   ├── train_X.parquet
│   ├── train_y.parquet
│   ├── train_metadata.parquet
│   ├── val_X.parquet
│   ├── val_y.parquet
│   ├── val_metadata.parquet
│   ├── test_X.parquet
│   ├── test_y.parquet
│   └── test_metadata.parquet
├── final_test/
│   ├── test_X.parquet
│   ├── test_y.parquet
│   └── test_metadata.parquet
├── folds.json
├── feature_schema.yml
├── config.yml
├── metadata.yml
├── provenance.yml
└── versions.txt
```

The `X` files contain candidate model features, the `y` files contain target
columns, and the metadata files contain dates, site/group information, and
other non-model fields. `feature_schema.yml` describes the generated feature
set; training applies experiment-specific feature selection afterward.

New datasets include an explicit `provenance.yml` lineage from `pftnc 
v0.0.9`. An older dataset without that file remains usable for training,
but its historical lineage is
reported as unavailable; PFTNC does not guess from its config or modify it.

---

# 4. Train the model

Move to:

```bash
cd ../my-experiments
```

Create the experiment configuration files. For fresh config templates, copy the
packaged examples into `config/`:

```bash
pftnc create-config --output-path config/
```

> NOTE: Default output path is `config/`, so you can simply do this:


```bash
pftnc create-config
```

To copy configuration files from your data repository instead, use its config
directory as the source. Only the supported configuration filenames are copied:
`config.yml`, `dataset_simulation_config.yml`, and `inference.yml`.

```bash
pftnc create-config \
  --source-path ../my-data/config \
  --output-path config/
```

`--source-path` may also point to one YAML file with any filename. After copying
the files, update the relevant configuration with the feature-engineered
dataset registry and version produced in the previous step.

Start the MLflow tracking server in a separate terminal. In that terminal,
enter the experiments repository and activate its Pixi environment first:

```bash
cd ../my-experiments
pixi shell
mlflow server
```

Keep this terminal running. MLflow will be available at:

```text
http://127.0.0.1:5000
```

In the terminal where you run training, activate the environment separately:

```bash
pixi shell
```

Update the `training` section in `config/config.yml` so that `input_dataset` points to the feature-engineered dataset.

## Feature selection

Feature selection is applied during training and is shared by both model modes.
The same feature-engineered dataset can therefore be reused by experiments with
different model inputs.

```yaml
training:
  feature_selection:
    # Automatically selected:
    #   - original non-CHIME sensor columns, such as lstm_*, s2_* and s3_*;
    #   - all rolling columns generated by features.rolling, including rolling
    #     CHIME features.
    # Raw CHIME columns are not automatic features.
    include: []
    exclude: []
```

Use `include` to add optional columns such as:

```yaml
training:
  feature_selection:
    include:
      - chime_diatoms_ug_per_l_last_observed
      - doy_sin
      - doy_cos
```

Raw CHIME columns can also be added explicitly. Use `exclude` to remove
automatically selected or included columns. If a column appears in both lists,
`exclude` takes precedence.

## Model modes

`pftnc` supports two training modes:

* `multi_target`: one model configuration is used to predict all targets jointly.
* `per_target`: each target gets its own model configuration.

### Option A — Multi-target

Use:

```yaml
training:
  experiment_name: pft_nc_test1
  description: describe your experiment
  output_dir: my-experiments/outputs

  input_dataset:
    base_path: my-data/data/feature_engineered/your_site
    registry_path: my-data/data/feature_engineered/your_site/feature_engineered.yaml
    version: your_feature_engineered_dataset_version

  model:
    type: xgboost
    mode: multi_target
    save_path: model.json

    params:
      objective: reg:squarederror
      eval_metric: rmse
      n_estimators: 500
      max_depth: 6
      learning_rate: 0.05
      subsample: 0.8
      colsample_bytree: 0.8
      reg_lambda: 1
      reg_alpha: 0
      early_stopping_rounds: 50
      tree_method: hist

  metrics:
    - mse
    - rmse
    - r2

  best_model_selection:
    metric: rmse
    direction: minimize
```

For `multi_target`, configure the model under:

```yaml
model:
  params:
```

Do **not** use `params_by_target`.

---

### Option B — Per-target

Use `per_target` when you want different XGBoost parameters for each target:

```yaml
training:
  model:
    type: xgboost
    mode: per_target
    save_path: model.json

    params_by_target:
      diatoms_ug_per_l:
        objective: reg:squarederror
        eval_metric: rmse
        n_estimators: 20
        max_depth: 2
        learning_rate: 0.1
        early_stopping_rounds: 5
        tree_method: hist

      cyanobacterial_ug_per_l:
        objective: reg:squarederror
        eval_metric: rmse
        n_estimators: 20
        max_depth: 2
        learning_rate: 0.1
        early_stopping_rounds: 5
        tree_method: hist

      others_ug_per_l:
        objective: reg:squarederror
        eval_metric: rmse
        n_estimators: 20
        max_depth: 2
        learning_rate: 0.1
        early_stopping_rounds: 5
        tree_method: hist
```

For `per_target`, configure models under:

```yaml
model:
  params_by_target:
```

Do **not** also provide `model.params`.

---

## Target transformations

Target transformations are configured under:

```yaml
training:
  target_transformations:
```

### Per-target transformations

Each target can use either:

* `absolute` — train directly on the target value.
* `log_delta` — train relative to a reference feature.

Example:

```yaml
training:
  target_transformations:
    per_target:
      diatoms_ug_per_l:
        type: log_delta
        reference_feature: chime_diatoms_ug_per_l_last_observed

      cyanobacterial_ug_per_l:
        type: log_delta
        reference_feature: chime_cyanobacteria_ug_per_l_last_observed

      others_ug_per_l:
        type: log_delta
        reference_feature: chime_others_ug_per_l_last_observed
```

When using `log_delta`, `reference_feature` is required.

For an absolute target:

```yaml
target_transformations:
  per_target:
    diatoms_ug_per_l:
      type: absolute
```

`absolute` does not use `reference_feature`.

### Joint transformations

So far, the pipeline supports `log_fractions` joint tranformation. A joint
transformation means that all the targets are required to create
new targets (i.e. they are dependent on each other).

For multi-target training, you can use the joint `log_fractions` transformation:

```yaml
training:
  target_transformations:
    log_fractions:
      denominator: others_ug_per_l
      epsilon: 0.01
```

Important compatibility rules:

* `log_fractions` can only be used with `model.mode: multi_target`.
* `log_fractions` cannot be combined with non-`absolute` per-target transformations such as `log_delta`.
* `per_target` model mode cannot use the joint `log_fractions` transformation.

A useful mental model is:

```text
multi_target
├── model.params
├── absolute targets
├── log_delta targets
└── joint log_fractions

per_target
├── model.params_by_target
├── absolute targets
└── log_delta targets
```



## Tuning

Tuning can be enabled independently:

```yaml
tuning:
  enabled: true
  n_trials: 20
  direction: minimize
  objective_metric: rmse
  objective_aggregation: mean

  search_space:
    n_estimators: [50, 100, 500]
    learning_rate: [0.01, 0.05, 0.1]
    max_depth: [3, 4, 5, 6, 7, 8]
    subsample: [0.5, 0.8, 1.0]
    colsample_bytree: [0.3, 0.6, 0.8, 1.0]
    reg_lambda: [0, 1, 5, 50]
    reg_alpha: [0, 0.1, 1, 10]
```

When `tuning.enabled: true`, `objective_metric` and `search_space` are required.

For weighted aggregation:

```yaml
objective_aggregation: weighted

target_weights:
  diatoms_ug_per_l: 0.5
  cyanobacterial_ug_per_l: 0.3
  others_ug_per_l: 0.2
```

Then run:

```bash
# Run `pixi shell` first in this terminal.
pftnc train config/config.yml
```

## Training outputs

Training creates a timestamped run directory below `training.output_dir`:

```text
<output_dir>/<run_id>/
├── fold_<year>/
│   ├── <model-file>.json
│   ├── fold_summary.json
├── predictions/
├── plots/
├── metric_comparisons/
├── shap/
├── final_test/
├── final_test_per_fold.csv
├── model_io_schema.yml
├── model_selection.yml
├── run_summary.json
├── config.yml
└── versions.txt
```

The run directory contains fold models, metrics, prediction CSVs, evaluation
plots, SHAP artifacts, final-test ensemble results, the exact model input/output
schema, model-selection metadata, a complete run summary, and `versions.txt`.
The version file records the PFTNC, Python, platform, and installed package
versions used for the run. `model_io_schema.yml` records the columns required
by inference.

## Package a trained model for inference

After training, create a portable inference bundle from the completed run. The
run must contain `run_summary.json`, `config.yml`, `model_io_schema.yml`, and
`model_selection.yml`. Packaging supports schema-version 1 created by the
current PFTNC package; old experiment runs are intentionally not supported:

```bash
pftnc package-model \
  --training-run-path outputs/<run_id> \
  --output-path model_bundle
```

The output directory must be new or empty. The bundle contains the copied fold
models, `manifest.yml`, `training_metrics.yml`, and `provenance.yml`. The
manifest includes the feature-engineering settings, selected model columns,
target transformations, fold-selection information, and dataset metadata, so
inference can recreate the same feature engineering from raw input data.

## Verify provenance and portable artifacts

Use a disposable location such as `my-data/data/deleteme/` and
`my-experiments/outputs/portable-tests/`. All paths in the test configs must
remain workspace-relative; do not use absolute paths or `../`.

Test these cases before relying on a new workspace layout:

1. **Create a new dataset.** Run `pftnc preprocess` with an output registry
   below `my-data/data/deleteme/`. Its generated dataset directory must contain
   `config.yml`, `provenance.yml`, and `versions.txt`. If its input has
   provenance, the new `provenance.yml` must contain that lineage. If the input
   has no provenance, its lineage must instead contain an
   `unknown_input_dataset` entry with `provenance_available: false`.

2. **Train and package using a dataset with provenance.** Configure training
   to use the newly generated dataset, or another registered dataset containing
   `provenance.yml`. For a quick test, disable tuning and use a small model
   budget. Run:

   ```bash
   cd ../my-experiments
   pftnc train config/with-provenance.yml
   pftnc package-model \
     --training-run-path outputs/portable-tests/<run-id> \
     --output-path portable-test-bundle-with-provenance
   ```

   Confirm that the run contains `versions.txt`; that `run_summary.json` has
   workspace-relative `dataset_path` values and run-relative `model_path`
   values; and that the bundle's `provenance.yml` has
   `dataset_provenance_available: true`.

3. **Train and package using an older dataset without provenance.** Point a
   second training config at a registered dataset directory that has no
   `provenance.yml`. Training and packaging must still succeed. The resulting
   bundle's `provenance.yml` must state:

   ```yaml
   dataset_lineage: null
   dataset_provenance_available: false
   ```

   PFTNC does not recreate or alter the older dataset in this case.

4. **Check the stored references.** Inspect the new dataset config/provenance
   and the new run's `config.yml` and `run_summary.json`. Dataset references
   must look like `my-data/data/...`; model paths in a run must look like
   `fold_2024/model.json`. Neither may contain a machine-specific absolute
   path.

5. **Test on another machine.** Clone the data and experiments repositories as
   siblings under a new workspace, install the same PFTNC version, and run
   `pftnc package-model` against the copied schema-version-1 run. It must find
   the dataset from the workspace-relative path and the model from its
   run-relative path. Old experiment runs are intentionally not supported for
   this test.

## Run inference

Copy the packaged `inference.yml` template and edit its paths:

```bash
pftnc create-config --output-path config/
```

Configure the bundle, a raw CSV or Parquet input, row-selection mode, model
selection strategy, and prediction output:

```yaml
model:
  bundle_path: my-experiments/model_bundle
  selection: best_overall

input_dataset:
  path: my-data/data/input.csv

save_model_inputs: true
save_outputs: true       # save predictions.csv and inference_config.yml
save_index: false        # include the dataframe index in predictions.csv

mode: latest                 # one latest row per site
# mode: from_date            # all rows from start_date onward
# start_date: 2024-01-01

output_path: my-experiments/outputs/predictions.csv
```

Run:

```bash
pftnc infer config/inference.yml
```

`save_outputs` defaults to `true`. Set it to `false` when inference should
compute and return predictions without writing either the prediction CSV or the
resolved `inference_config.yml`; in that case, omit `output_path`. `save_index`
defaults to `false` and only controls whether the Pandas row index is included
in the CSV. `save_model_inputs` defaults to `true` and controls whether the
model feature columns are included in the prediction CSV. It has no effect when
`save_outputs` is `false`.

Inference writes the predictions and the resolved configuration used to create
them. The configuration is saved as `inference_config.yml` beside the output
file. The prediction output contains the input metadata, the model input
features, and the generated prediction columns.

The raw input must contain the date column, group/site column, and all sensor
columns declared in the dataset schema used for training. Inference performs
the same date filling and rolling feature engineering, selects the exact model
inputs recorded in the bundle, loads every fold model, applies the requested
fold selection or ensemble, reverses configured target transformations, and
writes metadata plus `<target>_pred` columns to the output CSV.

Available `model.selection` values are:

* `best_overall`
* `best_per_target`
* `ensemble_simple`
* `ensemble_weighted`
* `ensemble_weighted_per_target`

Use `latest` for one prediction row per site/group. Use `from_date` with a
required `start_date` to retain every input row on or after that date.

---

# TL;DR

```bash
# 1. Create sibling data and experiments Git repositories, then install pftnc.
# Use a local checkout only when developing unreleased functionality:
# pftnc = { path = "../pftnc", editable = true }

pixi i

# In this terminal, activate the environment before project commands.
pixi shell

# 2. Simulate
pftnc simulate config/data_simulation_config.yml

# 3. Point feature_engineering.input_dataset
#    to the generated simulation registry/version

pftnc preprocess config/config.yml

# 4. In your experiments repository, start MLflow in a separate terminal first:
#    pixi shell
#    mlflow server
#
#    Then, in the training terminal, activate the environment:
#    pixi shell
#
#    Point training.input_dataset to the
#    feature-engineered registry/version
pftnc train config/config.yml

# 5. Package the completed run and run inference
pftnc package-model \
  --training-run-path outputs/<run_id> \
  --output-path model_bundle
pftnc infer config/inference.yml
```

## Common config mistakes

* Do not configure both `input_dataset.path` and `base_path + registry_path`.
* When `tuning.enabled: true`, both `objective_metric` and `search_space` are required.
* When using `objective_aggregation: weighted`, `target_weights` must also be provided.
* The editable local `pftnc` dependency is intended for **local development / unreleased versions only**.
