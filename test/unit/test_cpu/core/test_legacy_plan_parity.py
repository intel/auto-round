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

from auto_round import AutoRound, SignRoundConfig


def test_direct_and_algorithm_config_entries_build_equivalent_plans(tiny_opt_model_path):
    common = {
        "model": tiny_opt_model_path,
        "scheme": "W4A16",
        "iters": 0,
        "nsamples": 1,
        "seqlen": 8,
        "dataset": ["local calibration sample"],
        "low_cpu_mem_usage": False,
    }
    direct = AutoRound(bits=4, group_size=128, **common)
    configured = AutoRound(alg_configs=SignRoundConfig(bits=4, group_size=128, iters=0), **common)

    direct.post_init()
    configured.post_init()

    assert direct.compression_plan.scheme.value == configured.compression_plan.scheme.value
    assert direct.compression_plan.layer_config == configured.compression_plan.layer_config
