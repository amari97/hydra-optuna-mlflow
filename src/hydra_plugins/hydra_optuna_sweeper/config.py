# The MIT License (MIT)
# Copyright (c) 2026, Swiss Data Science Center (SDSC)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class OptunaConfig:
    """Sweeper-specific parameters isolated from task config values."""

    storage: Optional[str] = None
    study_name: Optional[str] = None
    load_if_exists: bool = True
    # Append deterministic hash of sweep-relevant config to avoid collisions.
    append_config_hash: bool = True
    # resume: reuse study_name; fresh: create timestamp-suffixed study_name.
    restart_mode: str = "resume"
    direction: str = "minimize"
    n_trials: int = 20
    sampler: Any = None
    params: Dict[str, Any] = field(default_factory=dict)
    # Name for top-level MLflow study run; defaults to study_name if not set.
    mlflow_study_run_name: Optional[str] = None
    # Optional Hydra job logging config injected into each trial job.
    # Example: "file_only" -> hydra/job_logging=file_only
    subjob_job_logging: Optional[str] = None


@dataclass
class MLflowOptunaSweeperConf:
    _target_: str = (
        "hydra_plugins.hydra_optuna_sweeper.mlflow_optuna_sweeper.MLflowOptunaSweeper"
    )
    optuna_config: OptunaConfig = field(default_factory=OptunaConfig)


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="mlflow_optuna",
    node=MLflowOptunaSweeperConf,
    provider="mlflow_optuna_sweeper",
)


ConfigStore.instance().store(
    group="hydra/job_logging",
    name="file_only",
    node={
        "version": 1,
        "formatters": {
            "simple": {"format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"}
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "formatter": "simple",
                "filename": "${hydra.runtime.output_dir}/${hydra.job.name}.log",
            }
        },
        "root": {"level": "INFO", "handlers": ["file"]},
        "disable_existing_loggers": False,
    },
    provider="mlflow_optuna_sweeper",
)
