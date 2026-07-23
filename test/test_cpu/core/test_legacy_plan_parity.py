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

from auto_round import AutoRound
from auto_round.algorithms.quantization.rtn.config import RTNConfig


def test_legacy_and_algorithm_config_entries_build_equivalent_plans(tiny_opt_model_path):
    common = {
        "model": tiny_opt_model_path,
        "scheme": "W4A16",
        "iters": 0,
        "nsamples": 1,
        "seqlen": 8,
        "dataset": ["local calibration sample"],
        "low_cpu_mem_usage": False,
    }
    legacy = AutoRound(bits=4, group_size=128, **common)
    modern = AutoRound(alg_configs=RTNConfig(bits=4, group_size=128), **common)

    legacy.post_init()
    modern.post_init()

    assert legacy.compression_plan.scheme.value == modern.compression_plan.scheme.value
    assert legacy.compression_plan.layer_config == modern.compression_plan.layer_config
