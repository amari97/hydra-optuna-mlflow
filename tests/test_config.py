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
from hydra_plugins.hydra_optuna_sweeper.config import MLflowOptunaSweeperConf
from hydra_plugins.hydra_optuna_sweeper.config import OptunaConfig


def test_optuna_config_core_defaults():
    """Keep only defaults that drive sweeper behavior."""
    config = OptunaConfig()

    assert config.direction == "minimize"
    assert config.n_trials == 20
    assert config.n_jobs == 1
    assert config.restart_mode == "resume"
    assert config.append_config_hash is True


def test_sweeper_conf_targets_mlflow_optuna_sweeper():
    """Target path is the critical wiring for plugin instantiation."""
    conf = MLflowOptunaSweeperConf()

    assert (
        conf._target_
        == "hydra_plugins.hydra_optuna_sweeper.mlflow_optuna_sweeper.MLflowOptunaSweeper"
    )
    assert isinstance(conf.optuna_config, OptunaConfig)
