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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from hydra_plugins.hydra_optuna_sweeper.config import OptunaConfig
from hydra_plugins.hydra_optuna_sweeper.mlflow_optuna_sweeper import (
    create_optuna_distribution_from_override,
)
from hydra_plugins.hydra_optuna_sweeper.mlflow_optuna_sweeper import (
    MLflowOptunaSweeper,
)
from omegaconf import OmegaConf
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution


class TestCreateOptunaDistribution:
    """Only keep representative conversion paths."""

    def test_scalar_value(self, override_parser):
        """Test non-sweep override returns scalar value."""
        override = override_parser.parse_overrides(["x=5.0"])[0]
        dist = create_optuna_distribution_from_override(override)
        assert dist == 5.0

    def test_choice_sweep(self, override_parser):
        """Choice sweep maps to categorical distribution."""
        override = override_parser.parse_overrides(["optimizer=adam,sgd,rmsprop"])[0]
        dist = create_optuna_distribution_from_override(override)
        assert isinstance(dist, CategoricalDistribution)
        assert len(dist.choices) == 3

    def test_interval_float_sweep(self, override_parser):
        """Test interval sweep with floats."""
        override = override_parser.parse_overrides(["lr=interval(0.001,0.1)"])[0]
        dist = create_optuna_distribution_from_override(override)
        assert isinstance(dist, FloatDistribution)
        assert dist.low == 0.001
        assert dist.high == 0.1
        assert dist.log is False

    def test_range_sweep(self, override_parser):
        """Range sweep also maps to categorical distribution."""
        override = override_parser.parse_overrides(["x=range(0,5)"])[0]
        assert isinstance(
            create_optuna_distribution_from_override(override), CategoricalDistribution
        )


class TestMLflowOptunaSweeper:
    """Keep only behaviorally relevant helper tests."""

    def test_prepare_sqlite_storage_relative_path(self, sweeper, tmp_path):
        """Test SQLite storage preparation with relative path."""
        db_path = str(tmp_path / "studies" / "test.db")
        storage_url = f"sqlite:///{db_path}"

        # Should not raise
        sweeper._prepare_sqlite_storage(storage_url)
        assert Path(db_path).parent.exists()

    def test_resolve_study_name_resume_mode(self):
        """Test study name resolution in resume mode."""
        config = OptunaConfig(
            study_name="my_study",
            restart_mode="resume",
            append_config_hash=False,
        )
        sweeper = MLflowOptunaSweeper(config)
        sweeper._fixed_overrides = []

        name = sweeper._resolve_study_name()
        assert name == "my_study"

    def test_resolve_study_name_fresh_mode(self):
        """Test study name resolution in fresh mode."""
        config = OptunaConfig(
            study_name="my_study",
            restart_mode="fresh",
            append_config_hash=False,
        )
        sweeper = MLflowOptunaSweeper(config)
        sweeper._fixed_overrides = []

        name = sweeper._resolve_study_name()
        assert name.startswith("my_study_fresh_")

    def test_resolve_study_name_invalid_mode(self):
        """Test invalid restart mode raises error."""
        config = OptunaConfig(restart_mode="invalid")
        sweeper = MLflowOptunaSweeper(config)
        sweeper._fixed_overrides = []

        with pytest.raises(ValueError, match="Unsupported restart_mode"):
            sweeper._resolve_study_name()

    def test_build_config_hash_filters_hydra_overrides(self):
        """Test that Hydra overrides are excluded from hash."""
        config = OptunaConfig()
        sweeper = MLflowOptunaSweeper(config)

        overrides_with_hydra = [
            "x=5",
            "hydra.run.dir=/tmp",
            "+hydra.verbose=true",
            "y=10",
        ]

        hash1 = sweeper._build_config_hash(overrides_with_hydra)
        hash2 = sweeper._build_config_hash(["x=5", "y=10"])

        # Same hash because hydra overrides were filtered
        assert hash1 == hash2

    def test_build_search_space_sweep_parameters(self, sweeper):
        """Test search space building from CLI sweep arguments."""
        arguments = ["x=interval(0,10)", "y=choice1,choice2"]
        search_space, fixed = sweeper._build_search_space(arguments)

        assert "x" in search_space
        assert "y" in search_space
        assert isinstance(search_space["x"], FloatDistribution)
        assert isinstance(search_space["y"], CategoricalDistribution)
        assert len(fixed) == 0

    def test_build_search_space_cli_overrides_config(self):
        """Test that CLI arguments override config params."""
        config = OptunaConfig(params={"x": "interval(0,1)"})
        sweeper = MLflowOptunaSweeper(config)

        # CLI should override config params
        search_space, _ = sweeper._build_search_space(["x=interval(5,10)"])

        assert isinstance(search_space["x"], FloatDistribution)
        assert search_space["x"].low == 5.0
        assert search_space["x"].high == 10.0

    def test_create_mlflow_study_run_uses_explicit_name(self, mlflow_enabled_cfg):
        """mlflow_study_run_name should be used as the top-level run name."""
        config = OptunaConfig(mlflow_study_run_name="my-study-run")
        sweeper = MLflowOptunaSweeper(config)
        sweeper._resolved_study_name = "resolved_study"
        sweeper.config = mlflow_enabled_cfg

        mock_client = Mock()
        mock_client.get_experiment_by_name.return_value = SimpleNamespace(
            experiment_id="123"
        )
        mock_client.search_runs.return_value = []
        mock_client.create_run.return_value = SimpleNamespace(
            info=SimpleNamespace(run_id="run-abc")
        )

        with (
            patch(
                "hydra_plugins.hydra_optuna_sweeper.mlflow_optuna_sweeper.mlflow.set_tracking_uri"
            ),
            patch(
                "hydra_plugins.hydra_optuna_sweeper.mlflow_optuna_sweeper.mlflow.set_experiment"
            ),
            patch(
                "hydra_plugins.hydra_optuna_sweeper.mlflow_optuna_sweeper.mlflow.tracking.MlflowClient",
                return_value=mock_client,
            ),
        ):
            run_id, experiment_id = sweeper._create_mlflow_study_run()

        assert run_id == "run-abc"
        assert experiment_id == "123"
        mock_client.create_run.assert_called_once()
        assert mock_client.create_run.call_args.kwargs["run_name"] == "my-study-run"
        assert (
            mock_client.create_run.call_args.kwargs["tags"][
                "hydra_optuna_sweeper.study_name"
            ]
            == "resolved_study"
        )

    def test_create_mlflow_study_run_resume_reuses_existing_run(
        self, mlflow_enabled_cfg
    ):
        """resume mode should reuse an existing MLflow study run instead of creating one."""
        config = OptunaConfig(restart_mode="resume")
        sweeper = MLflowOptunaSweeper(config)
        sweeper._resolved_study_name = "resolved_study"
        sweeper.config = mlflow_enabled_cfg

        mock_client = Mock()
        mock_client.get_experiment_by_name.return_value = SimpleNamespace(
            experiment_id="123"
        )
        mock_client.search_runs.return_value = [
            SimpleNamespace(info=SimpleNamespace(run_id="existing-run"))
        ]

        with (
            patch(
                "hydra_plugins.hydra_optuna_sweeper.mlflow_optuna_sweeper.mlflow.set_tracking_uri"
            ),
            patch(
                "hydra_plugins.hydra_optuna_sweeper.mlflow_optuna_sweeper.mlflow.set_experiment"
            ),
            patch(
                "hydra_plugins.hydra_optuna_sweeper.mlflow_optuna_sweeper.mlflow.tracking.MlflowClient",
                return_value=mock_client,
            ),
        ):
            run_id, experiment_id = sweeper._create_mlflow_study_run()

        assert run_id == "existing-run"
        assert experiment_id == "123"
        mock_client.create_run.assert_not_called()

    def test_build_trial_overrides_adds_parent_run_id_for_train(self):
        """Trial overrides should include parent run metadata used by train.py."""
        sweeper = MLflowOptunaSweeper(OptunaConfig())
        trial = SimpleNamespace(number=7, params={"lr": 0.01})

        overrides = sweeper._build_trial_overrides(
            trial=trial,
            fixed_overrides=["model=resnet"],
            study_run_id="parent-123",
        )

        assert "lr=0.01" in overrides
        assert "model=resnet" in overrides
        assert "+mlflow_parent_run_id=parent-123" in overrides
        assert "+optuna_trial_number=7" in overrides

    def test_build_trial_overrides_injects_subjob_logging_override(self):
        """Optional subjob logging config should be forwarded to each trial."""
        sweeper = MLflowOptunaSweeper(OptunaConfig(subjob_job_logging="file_only"))
        trial = SimpleNamespace(number=2, params={"lr": 0.02})

        overrides = sweeper._build_trial_overrides(
            trial=trial,
            fixed_overrides=[],
            study_run_id=None,
        )

        assert "hydra/job_logging=file_only" in overrides

    def test_effective_n_jobs_from_launcher_interpolation(self):
        """n_jobs is sourced from hydra.launcher.n_jobs."""
        sweeper = MLflowOptunaSweeper(OptunaConfig())
        sweeper.config = OmegaConf.create({"hydra": {"launcher": {"n_jobs": 3}}})

        assert sweeper._effective_n_jobs() == 3

    def test_sweep_resume_reuses_study_name(self):
        """resume mode should call Optuna with stable study name + load_if_exists."""
        config = OptunaConfig(
            study_name="resume_study",
            restart_mode="resume",
            append_config_hash=False,
            load_if_exists=True,
            n_trials=0,
        )
        sweeper = MLflowOptunaSweeper(config)
        sweeper.config = OmegaConf.create({"use_mlflow": False})
        sweeper.sampler = None

        with patch(
            "hydra_plugins.hydra_optuna_sweeper.mlflow_optuna_sweeper.optuna.create_study"
        ) as create_study:
            sweeper.sweep(arguments=[])

        create_study.assert_called_once()
        assert create_study.call_args.kwargs["study_name"] == "resume_study"
        assert create_study.call_args.kwargs["load_if_exists"] is True

    def test_parallel_jobs_require_persistent_storage(self):
        """n_jobs > 1 must use RDB/Journal-backed storage."""
        sweeper = MLflowOptunaSweeper(OptunaConfig(storage=None, n_trials=0))
        sweeper.config = OmegaConf.create({"hydra": {"launcher": {"n_jobs": 2}}})

        with pytest.raises(ValueError, match="Parallel execution requires persistent"):
            sweeper.sweep(arguments=[])

    def test_parallel_jobs_accept_sqlite_storage(self):
        """RDB URL storage should be accepted when n_jobs > 1."""
        config = OptunaConfig(
            storage="sqlite:///tmp_parallel_optuna.db",
            study_name="parallel_study",
            restart_mode="resume",
            append_config_hash=False,
            load_if_exists=True,
            n_trials=0,
        )
        sweeper = MLflowOptunaSweeper(config)
        sweeper.config = OmegaConf.create({"hydra": {"launcher": {"n_jobs": 2}}})
        sweeper.sampler = None

        with patch(
            "hydra_plugins.hydra_optuna_sweeper.mlflow_optuna_sweeper.optuna.create_study"
        ) as create_study:
            sweeper.sweep(arguments=[])

        create_study.assert_called_once()
