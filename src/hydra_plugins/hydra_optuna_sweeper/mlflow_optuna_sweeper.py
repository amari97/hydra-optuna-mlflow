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
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import mlflow
import optuna
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import IntervalSweep
from hydra.core.override_parser.types import Override
from hydra.core.override_parser.types import Transformer
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from omegaconf import DictConfig
from omegaconf import OmegaConf
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.storages import JournalStorage
from optuna.storages import RDBStorage

from .config import OptunaConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distribution factory
# ---------------------------------------------------------------------------


def create_optuna_distribution_from_override(override: Override) -> Any:
    """Convert a Hydra override into an Optuna 3.x distribution (or scalar)."""
    value = override.value()
    if not override.is_sweep_override():
        return value

    if override.is_choice_sweep():
        choices = list(override.sweep_iterator(transformer=Transformer.encode))
        return CategoricalDistribution(choices)

    if override.is_range_sweep():
        choices = list(override.sweep_iterator(transformer=Transformer.encode))
        return CategoricalDistribution(choices)

    if override.is_interval_sweep():
        assert isinstance(value, IntervalSweep)
        is_log = "log" in value.tags
        is_int = "int" in value.tags
        if is_int:
            return IntDistribution(int(value.start), int(value.end), log=is_log)
        return FloatDistribution(float(value.start), float(value.end), log=is_log)

    raise NotImplementedError(f"{override} is not supported by MLflowOptunaSweeper.")


# ---------------------------------------------------------------------------
# Sweeper
# ---------------------------------------------------------------------------


class MLflowOptunaSweeper(Sweeper):
    """Hydra sweeper plugin based on Optuna with nested MLflow run logging.

    MLflow run hierarchy produced
    ------------------------------
    study_run             (top-level, created by sweeper)
    ├── trial_0_meta      (cv.py parent run, nested via parent_run_id)
    │   ├── trial_0_cv=1  (fold run, nested=True inside cv.py)
    │   └── trial_0_cv=2
    └── trial_1_meta
        ├── trial_1_cv=1
        └── trial_1_cv=2

    For train.py the hierarchy is shallower:
    study_run
    ├── trial_0           (training run, nested via parent_run_id)
    └── trial_1

    Registration
    ------------
    The sweeper config is registered as ``mlflow_optuna``; use it with::

        defaults:
        - override /hydra/sweeper: mlflow_optuna
    """

    def __init__(self, optuna_config: OptunaConfig) -> None:
        self.optuna_config = optuna_config

    def setup(
        self,
        *,
        hydra_context: Any,
        task_function: Any,
        config: DictConfig,
    ) -> None:
        self.config = config
        if not HydraConfig.initialized():
            HydraConfig.instance().set_config(config)
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context,
            task_function=task_function,
            config=config,
        )
        # sampler is already constructed by Hydra's _target_ resolution before __init__.
        self.sampler = self.optuna_config.sampler

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_sqlite_storage(self, storage: Optional[str]) -> None:
        """Create parent directory for sqlite:/// paths before Optuna opens DB."""
        if not storage:
            return

        if not storage.startswith("sqlite:///"):
            return

        # SQLAlchemy-style sqlite URL forms:
        # - sqlite:///relative/path.db   -> relative to current working dir
        # - sqlite:////absolute/path.db  -> absolute path
        db_path = storage[len("sqlite:///") :]
        if not db_path:
            return

        resolved = Path(db_path) if db_path.startswith("/") else Path.cwd() / db_path

        parent = resolved.parent
        parent.mkdir(parents=True, exist_ok=True)

    def _resolve_study_name(self) -> Optional[str]:
        """Resolve study name according to restart policy."""
        restart_mode = self.optuna_config.restart_mode
        if restart_mode not in {"resume", "fresh"}:
            raise ValueError(
                "Unsupported restart_mode='"
                f"{restart_mode}'"
                ". Expected one of: 'resume', 'fresh'."
            )

        base_name = self.optuna_config.study_name
        if self.optuna_config.append_config_hash:
            config_hash = self._build_config_hash(self._fixed_overrides)
            stem = base_name or "optuna_study"
            base_name = f"{stem}_{config_hash}"
        if restart_mode == "resume":
            return base_name

        # fresh mode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_name:
            fresh_name = f"{base_name}_fresh_{timestamp}"
        else:
            fresh_name = f"optuna_fresh_{timestamp}"
        log.info("restart_mode=fresh: using fresh study_name='%s'", fresh_name)
        return fresh_name

    def _build_config_hash(self, fixed_overrides: List[str]) -> str:
        """Build a stable hash from sweep-relevant config inputs.

        Excludes Hydra runtime/sweeper control overrides so study identity stays
        stable when only execution controls change (e.g. n_trials override).
        """
        filtered_overrides: List[str] = []
        ignored_prefixes = (
            "hydra.",
            "+hydra.",
        )
        for override in fixed_overrides:
            if override.startswith(ignored_prefixes):
                continue
            filtered_overrides.append(override)

        payload = {
            "overrides": sorted(filtered_overrides),
            "optuna_params": {
                k: str(v) for k, v in sorted(self.optuna_config.params.items())
            },
            "direction": self.optuna_config.direction,
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return digest[:10]

    def _build_search_space(
        self, arguments: List[str]
    ) -> tuple[Dict[str, Any], List[str]]:
        """Return (search_space, fixed_overrides).

        search_space  — {param_name: Optuna distribution}
        fixed_overrides — non-sweep CLI overrides to append verbatim
        """
        parser = OverridesParser.create()

        # Sweeper config params (lower priority)
        search_space: Dict[str, Any] = {}
        if self.optuna_config.params:
            for override in parser.parse_overrides(
                [f"{k}={v}" for k, v in self.optuna_config.params.items()]
            ):
                search_space[override.get_key_element()] = (
                    create_optuna_distribution_from_override(override)
                )

        # CLI arguments (higher priority)
        fixed_overrides: List[str] = []
        if arguments:
            for override in parser.parse_overrides(arguments):
                key = override.get_key_element()
                if override.is_sweep_override():
                    search_space[key] = create_optuna_distribution_from_override(
                        override
                    )
                else:
                    fixed_overrides.append(override.input_line)

        return search_space, fixed_overrides

    def _create_mlflow_study_run(self) -> tuple[Optional[str], Optional[str]]:
        """Create the top-level MLflow study run.

        Returns ``(study_run_id, experiment_id)`` or ``(None, None)`` when
        MLflow is not configured.
        """
        use_mlflow = OmegaConf.select(self.config, "use_mlflow", default=None)
        if use_mlflow is False:
            return None, None

        tracking_uri = OmegaConf.select(
            self.config, "trainer.logger.tracking_uri", default=None
        )
        if not tracking_uri:
            if use_mlflow:
                log.warning(
                    "use_mlflow=True but trainer.logger.tracking_uri is not set. "
                    "Skipping MLflow study run creation."
                )
            return None, None

        experiment_name = OmegaConf.select(
            self.config, "trainer.logger.experiment_name", default=None
        ) or OmegaConf.select(self.config, "experiment_path", default=None)
        study_run_name = (
            self.optuna_config.mlflow_study_run_name
            or self._resolved_study_name
            or OmegaConf.select(self.config, "run_name", default="optuna_study")
        )

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        study_run = client.create_run(
            experiment_id=experiment_id,
            run_name=study_run_name,
            tags={
                "mlflow.note.content": f"Optuna study: {self._resolved_study_name or ''}"
            },
        )
        study_run_id = study_run.info.run_id
        log.info(
            "Created MLflow study run '%s' (run_id=%s)", study_run_name, study_run_id
        )
        return study_run_id, experiment_id

    def _log_best_trial_to_mlflow(
        self,
        study: optuna.Study,
        study_run_id: str,
        experiment_id: str,
    ) -> None:
        """Log best-trial params/metric and best child run id to the study run."""
        try:
            best_trial = study.best_trial
        except ValueError:
            log.warning("No completed trials — skipping best-trial MLflow logging.")
            return

        client = mlflow.tracking.MlflowClient()
        with mlflow.start_run(run_id=study_run_id):
            mlflow.log_params(best_trial.params)
            mlflow.log_metric("best_value", study.best_value)

            # Find the best trial's direct child run via the optuna.trial_number tag
            all_runs = client.search_runs(
                experiment_ids=[experiment_id],
                max_results=self.optuna_config.n_trials * 4,
            )
            best_child = next(
                (
                    r
                    for r in all_runs
                    if r.data.tags.get("mlflow.parentRunId") == study_run_id
                    and r.data.tags.get("optuna.trial_number") == str(best_trial.number)
                ),
                None,
            )
            if best_child is not None:
                mlflow.log_param("best_child_run_id", best_child.info.run_id)
                log.info(
                    "Best trial #%d → MLflow run %s",
                    best_trial.number,
                    best_child.info.run_id,
                )

    def _build_trial_overrides(
        self,
        trial: optuna.trial.FrozenTrial,
        fixed_overrides: List[str],
        study_run_id: Optional[str],
    ) -> tuple[str, ...]:
        """Build one Hydra override tuple for a single Optuna trial."""
        trial_overrides: tuple[str, ...] = tuple(
            f"{name}={val}" for name, val in trial.params.items()
        )
        trial_overrides += tuple(fixed_overrides)
        if self.optuna_config.subjob_job_logging:
            trial_overrides += (
                f"hydra/job_logging={self.optuna_config.subjob_job_logging}",
            )
        if study_run_id is not None:
            trial_overrides += (
                f"+mlflow_parent_run_id={study_run_id}",
                f"+optuna_trial_number={trial.number}",
            )
        return trial_overrides

    def _effective_n_jobs(self) -> int:
        """Resolve parallel worker count from hydra.launcher.n_jobs."""
        launcher_n_jobs = OmegaConf.select(
            self.config, "hydra.launcher.n_jobs", default=1
        )
        try:
            return max(1, int(launcher_n_jobs))
        except (TypeError, ValueError):
            return 1

    def _validate_parallel_storage(self, n_jobs: int) -> None:
        """Require persistent Optuna storage when running trials in parallel."""
        if n_jobs <= 1:
            return

        storage = self.optuna_config.storage

        if isinstance(storage, (RDBStorage, JournalStorage)):
            return

        if isinstance(storage, str):
            # String-based RDB URL configs are converted by Optuna into RDBStorage.
            if storage.startswith(
                (
                    "sqlite://",
                    "mysql://",
                    "postgresql://",
                    "postgresql+",
                    "oracle://",
                    "mariadb://",
                )
            ):
                return

        raise ValueError(
            "Parallel execution requires persistent Optuna storage. "
            "When hydra.launcher.n_jobs > 1, set optuna_config.storage to an "
            "RDB URL (e.g. sqlite:///...) or use JournalStorage."
        )

    def _report_trial_result(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        return_value: Any,
    ) -> None:
        """Tell Optuna a trial result, marking missing values as failed."""
        if return_value is None:
            log.warning("Trial %d returned None — marking as FAIL.", trial.number)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            return

        study.tell(trial, float(return_value))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def sweep(self, arguments: List[str]) -> None:
        search_space, fixed_overrides = self._build_search_space(arguments)
        self._fixed_overrides = fixed_overrides

        if not search_space:
            log.warning(
                "No hyperparameter search space defined. Running %d fixed trial(s).",
                self.optuna_config.n_trials,
            )

        self._resolved_study_name = self._resolve_study_name()
        self._prepare_sqlite_storage(self.optuna_config.storage)

        if (
            self.optuna_config.storage
            and self.optuna_config.load_if_exists
            and not self._resolved_study_name
        ):
            log.warning(
                "Persistent storage is configured but study_name is null; "
                "resume semantics are disabled because Optuna will generate "
                "a new random study name each run."
            )

        n_jobs = self._effective_n_jobs()
        self._validate_parallel_storage(n_jobs)

        study = optuna.create_study(
            study_name=self._resolved_study_name,
            storage=self.optuna_config.storage,
            direction=self.optuna_config.direction,
            sampler=self.sampler,
            load_if_exists=self.optuna_config.load_if_exists,
        )

        study_run_id, experiment_id = self._create_mlflow_study_run()

        n_trials_remaining = self.optuna_config.n_trials
        sweep_status = "FINISHED"
        try:
            while n_trials_remaining > 0:
                batch_size = min(n_trials_remaining, n_jobs)
                trials = [study.ask(search_space) for _ in range(batch_size)]
                overrides_batch = [
                    self._build_trial_overrides(
                        trial=trial,
                        fixed_overrides=fixed_overrides,
                        study_run_id=study_run_id,
                    )
                    for trial in trials
                ]

                job_returns = self.launcher.launch(
                    overrides_batch, initial_job_idx=trials[0].number
                )

                for trial, ret in zip(trials, job_returns):
                    self._report_trial_result(
                        study=study,
                        trial=trial,
                        return_value=ret.return_value,
                    )

                n_trials_remaining -= batch_size

            if study_run_id is not None:
                self._log_best_trial_to_mlflow(study, study_run_id, experiment_id)

        except Exception:
            sweep_status = "FAILED"
            raise
        finally:
            if study_run_id is not None:
                client = mlflow.tracking.MlflowClient()
                run_info = client.get_run(study_run_id)
                if run_info.info.status == "RUNNING":
                    client.set_terminated(study_run_id, sweep_status)

        try:
            best_trial = study.best_trial
            log.info("Best parameters: %s", best_trial.params)
            log.info("Best value: %s", best_trial.value)
        except ValueError:
            log.warning("No completed trials.")
        if self._resolved_study_name:
            log.info("Study name: %s", study.study_name)
        if self.optuna_config.storage:
            log.info("Storage: %s", self.optuna_config.storage)
