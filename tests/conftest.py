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
import tempfile
from pathlib import Path

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra_plugins.hydra_optuna_sweeper.config import OptunaConfig
from hydra_plugins.hydra_optuna_sweeper.mlflow_optuna_sweeper import MLflowOptunaSweeper
from omegaconf import OmegaConf


@pytest.fixture
def override_parser():
    """Reusable Hydra override parser."""
    return OverridesParser.create()


@pytest.fixture
def sweeper():
    """Reusable sweeper with default config."""
    return MLflowOptunaSweeper(OptunaConfig())


@pytest.fixture
def temp_hydra_config_dir():
    """Temporary config directory for compose smoke tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        (config_dir / "config.yaml").write_text("""
defaults:
  - override /hydra/sweeper: mlflow_optuna

x: 1.0
y: 2.0

hydra:
  sweeper:
    optuna_config:
      direction: minimize
      n_trials: 2
""")
        yield config_dir


@pytest.fixture
def mlflow_enabled_cfg():
    """Minimal config with MLflow enabled for unit tests."""
    return OmegaConf.create(
        {
            "use_mlflow": True,
            "experiment_path": "exp/path",
            "trainer": {
                "logger": {
                    "tracking_uri": "file:///tmp/mlruns",
                }
            },
        }
    )
