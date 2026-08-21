# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from auto_round import autoround
from auto_round.autoround import _split_entry_kwargs


def test_split_entry_kwargs_partitions_owned_fields():
    processor = object()

    grouped = _split_entry_kwargs(
        {"dataset": ["sample"], "scale_dtype": "fp32", "processor": processor, "model_free": False}
    )

    assert grouped["base"]["dataset"] == ["sample"]
    assert grouped["compressor"]["scale_dtype"] == "fp32"
    assert grouped["mllm"]["processor"] is processor
    assert grouped["route"]["model_free"] is False


def test_split_entry_kwargs_ignores_unknown_fields(monkeypatch):
    warnings = []
    monkeypatch.setattr(autoround.logger, "warning_once", lambda message, *args: warnings.append(message % args))

    grouped = _split_entry_kwargs({"unknown_option": 1}, context="test entry")

    assert all(not values for values in grouped.values())
    assert "unknown_option" in warnings[0]


def test_cli_explicit_auto_round_with_zero_iters_selects_rtn():
    from argparse import Namespace

    from auto_round.cli.algorithms import AlgorithmHandler

    args = Namespace(algorithm="auto_round", iters=0)
    configs = AlgorithmHandler.build_configs(args, {})

    assert type(configs[0]).__name__ == "RTNConfig"


def test_cli_awq_rtn_opt_policy():
    from argparse import Namespace

    from auto_round.cli.algorithms import AlgorithmHandler

    cases = [
        (Namespace(algorithm="awq", iters=200, disable_opt_rtn=True), ["AWQConfig", "RTNConfig"], [True, True]),
        (Namespace(algorithm="awq", iters=200, disable_opt_rtn=None), ["AWQConfig", "RTNConfig"], [True, True]),
        (Namespace(algorithm="awq,rtn", iters=200, disable_opt_rtn=False), ["AWQConfig", "RTNConfig"], [False, False]),
    ]

    for args, expected_types, expected_disable_opt_rtn in cases:
        configs = AlgorithmHandler.build_configs(args, {})

        assert [type(config).__name__ for config in configs] == expected_types
        assert [config.disable_opt_rtn for config in configs] == expected_disable_opt_rtn


def test_cli_rtn_inherits_built_awq_config_without_mutating_args(monkeypatch):
    from argparse import Namespace

    from auto_round.cli.algorithms import AWQ, AlgorithmHandler
    from auto_round.algorithms.transforms.awq.config import AWQConfig

    def build_awq_with_opt_rtn(self, args, common_kwargs):
        return AWQConfig(disable_opt_rtn=False)

    monkeypatch.setattr(AWQ, "build", build_awq_with_opt_rtn)

    args = Namespace(algorithm="rtn,awq", iters=200, disable_opt_rtn=None)
    configs = AlgorithmHandler.build_configs(args, {})

    assert [type(config).__name__ for config in configs] == ["RTNConfig", "AWQConfig"]
    assert configs[0].disable_opt_rtn is False
    assert configs[1].disable_opt_rtn is False
    assert args.disable_opt_rtn is None
