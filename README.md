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
```

The sweeper injects these runtime overrides for each trial:

- +mlflow_parent_run_id
- +optuna_trial_number

Your training code can use these values to attach nested runs to the study parent run.

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

      # Set n_jobs > 1 only when your hardware can safely parallelize trials
      n_jobs: 1
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
