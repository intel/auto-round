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

"""
Unit tests for low_cpu_mem_usage option:
When enabled, block weights are offloaded to disk upfront and loaded on demand
during quantization, significantly reducing peak CPU RAM usage.
"""

import torch

from auto_round import AutoRound
from auto_round.utils import offload as offload_module


class TestLowCpuMemUsage:
    """Tests for low_cpu_mem_usage block offloading."""

    def test_option_stored_correctly(self, tiny_opt_model_path):
        """Test that low_cpu_mem_usage option is stored correctly."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            low_cpu_mem_usage=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=32,
        )
        assert autoround.low_cpu_mem_usage is True

        autoround2 = AutoRound(
            tiny_opt_model_path,
            bits=4,
            low_cpu_mem_usage=False,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=32,
        )
        assert autoround2.low_cpu_mem_usage is False

    def test_offloader_disabled_when_low_cpu_mem_false(self, tiny_opt_model_path):
        """Test that the offloader is disabled when low_cpu_mem_usage=False."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            low_cpu_mem_usage=False,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=32,
        )
        assert autoround.low_cpu_mem_usage is False
        # stream_offload_all_blocks should be a no-op when offloader is disabled
        autoround._offloader.stream_offload_all_blocks(autoround.model, [["model.layers.0"]], autoround.device_list)
        assert autoround._offloader._blocks == {}

    def test_stream_offload_blocks_records_blocks(self, tiny_opt_model_path, tmp_path, monkeypatch):
        """Test that stream_offload_all_blocks records offloaded blocks when enabled."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            low_cpu_mem_usage=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=32,
        )

        dummy_block = torch.nn.Linear(4, 4)
        # Monkeypatch get_module used by stream_offload_all_blocks in offload.py

        monkeypatch.setattr(offload_module, "get_module", lambda _model, _name: dummy_block)
        monkeypatch.setattr(torch, "save", lambda *args, **kwargs: None)

        # Force the offloader to think it has a tempdir already
        autoround._offloader._tempdir = str(tmp_path)
        autoround._offloader.stream_offload_all_blocks(autoround.model, [["model.layers.0"]], autoround.device_list)
        assert autoround._offloader.has("model.layers.0")
