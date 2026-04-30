# Hydra Optuna MLflow Sweeper

Hydra Optuna MLflow Sweeper is a general-purpose Hydra sweeper plugin for hyperparameter optimization with Optuna.

This project is based on the original Hydra Optuna Sweeper plugin by Toshihiko Yanase:
https://github.com/toshihikoyanase/hydra-optuna-sweeper/tree/main

## What This Package Adds

In addition to Optuna-based sweeping, this package adds:

- MLflow study and trial hierarchy logging, including parent run propagation to trial jobs.
- Restart behavior for persistent studies with restart_mode:
  - resume: continue an existing study in the same storage.
  - fresh: create a new timestamped study name while keeping the same storage backend.
- Support for persistent SQLite Optuna storage (for example, sqlite:///logs/optuna/mlp_search.db).

## Installation

Using pip:

```bash
pip install -e .
```

Using uv:

```bash
uv sync
```

## Quick Usage

Set the sweeper in your Hydra config:

```yaml
defaults:
  - override /hydra/sweeper: mlflow_optuna
  - override /hydra/launcher: joblib
```

The sweeper injects these runtime overrides for each trial:

- +mlflow_parent_run_id
- +optuna_trial_number

Your training code can use these values to attach nested runs to the study parent run.

`mlflow_study_run_name` controls the top-level study run name created by the sweeper.
When set, that explicit value is used instead of the resolved study name.

Parallel trial execution can be controlled through Hydra's joblib launcher by linking
launcher workers:

```yaml
hydra:
  launcher:
    n_jobs: 4
```

You can also force a dedicated file-only logger for each trial subjob:

```yaml
hydra:
  sweeper:
    optuna_config:
      subjob_job_logging: file_only
```

This injects `hydra/job_logging=file_only` into each trial job.

Example logging config file:

Create `config/hydra/job_logging/file_only.yaml` with:

```yaml
version: 1
formatters:
  simple:
    format: '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
handlers:
  file:
    class: logging.FileHandler
    formatter: simple
    # written to the Hydra run output directory alongside other run artifacts
    filename: ${hydra.runtime.output_dir}/${hydra.job.name}.log
root:
  level: INFO
  handlers: [file]
disable_existing_loggers: false
```

## Recommended Config Example

Below is a production-style example adapted from your config:

```yaml
# @package _global_
defaults:
  - override /hydra/sweeper: mlflow_optuna

# Metric returned by train() (unused by CV)
optimized_metric: "val/loss"

# Vary the CV split seed across trials
split_seed: ${hydra:job.num}

log_system_metrics: false
save_checkpoints: false

hydra:
  mode: "MULTIRUN"
  sweeper:
    optuna_config:
      # Persistent study DB. Re-running the same command with resume
      # continues the same study.
      storage: sqlite:///logs/optuna/mlp_search.db
      study_name: mlp_search
      load_if_exists: true

      # resume: keep same study_name
      # fresh: append timestamp suffix to create a new study in same DB
      restart_mode: resume

      # Top-level MLflow run name (defaults to study_name when null)
      mlflow_study_run_name: null
      direction: minimize
      n_trials: 50

      sampler:
        _target_: optuna.samplers.TPESampler
        seed: 42

      params:
        # Architecture
        model.model.hidden_size: choice(12, 16, 20, 24, 28, 32)
        model.model.num_layers: choice(2, 3, 4, 5)
        model.model.activation: choice("relu", "softplus", "silu")
        model.model.dropout: choice(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

        # Optimization
        model.weight_decay: choice(0, 1e-5, 1e-4, 1e-3)

        # Batch size affects throughput and generalization
        datamodule.batch_size: choice(1024, 2048, 4096)

  # Keep sweep directory simple to avoid unresolved interpolation issues
  sweep:
    dir: logs/multirun/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.num}
```

## Minimal Example App

A minimal runnable example is provided in example/.

```bash
python example/quadratic.py -m 'x=interval(-5.0, 5.0)' 'y=interval(0.0, 10.0)'
```

## Train-Side MLflow Run Setup

In trial jobs (for example `train.py`), consume `mlflow_parent_run_id` injected by the sweeper
to attach each training run under the study run:

```python
from omegaconf import DictConfig
import mlflow


def _start_mlflow_run(cfg: DictConfig):
  """Start an MLflow run using config values and enable autologging."""
  logger_cfg = cfg.trainer.logger
  tracking_uri = logger_cfg.tracking_uri
  experiment_name = cfg.experiment_path
  run_name = cfg.get("run_name")
  parent_run_id = cfg.get("mlflow_parent_run_id")

  mlflow.set_tracking_uri(tracking_uri)
  mlflow.set_experiment(experiment_name)

  start_run_kwargs = {"run_name": run_name}
  if parent_run_id:
    start_run_kwargs["parent_run_id"] = parent_run_id
  return mlflow.start_run(**start_run_kwargs)
```

With `restart_mode: resume`, rerunning the same sweep command with the same
`study_name` and storage backend continues the existing Optuna study.

## Contributing

We welcome contributions! To get started:

1. **Set up the development environment:**
   ```bash
   uv sync
   source .venv/bin/activate
   ```

2. **Install pre-commit hooks:**
   ```bash
   uv run pre-commit install
   ```

3. **Make your changes and run linting/tests:**
   ```bash
   uv run pre-commit run --all-files
   uv run pytest
   ```

4. **Submit a pull request** with a clear description of your changes.

Please ensure your code follows the project's style guidelines (enforced by ruff and pre-commit).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
