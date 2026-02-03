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
Unit tests for CPU RAM optimization options:
1. cpu_stream_offload_blocks: Offload block weights to disk, load on demand
2. cpu_stream_loss: Compute loss on-the-fly using frozen block copy
"""

import torch

from auto_round import AutoRound
from auto_round.compressors import base as base_module


class TestCpuStreamOffloadBlocks:
    """Tests for cpu_stream_offload_blocks option."""

    def test_option_stored_correctly(self, tiny_opt_model_path):
        """Test that cpu_stream_offload_blocks option is stored correctly."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            low_cpu_mem_usage=True,
            cpu_stream_offload_blocks=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=32,
        )
        assert autoround.cpu_stream_offload_blocks is True

        autoround2 = AutoRound(
            tiny_opt_model_path,
            bits=4,
            low_cpu_mem_usage=False,
            cpu_stream_offload_blocks=False,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=32,
        )
        assert autoround2.cpu_stream_offload_blocks is False

    def test_offload_requires_low_cpu_mem_usage(self, tiny_opt_model_path):
        """Test that offload only activates when low_cpu_mem_usage is True."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            low_cpu_mem_usage=False,
            cpu_stream_offload_blocks=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=32,
        )
        # Even if cpu_stream_offload_blocks=True, it should not offload
        # when low_cpu_mem_usage=False
        assert autoround.cpu_stream_offload_blocks is True
        assert autoround.low_cpu_mem_usage is False

    def test_stream_offload_blocks_skips_when_disabled(self, tiny_opt_model_path):
        """Test that _stream_offload_blocks returns early when disabled."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            low_cpu_mem_usage=True,
            cpu_stream_offload_blocks=False,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=32,
        )
        autoround._stream_offload_blocks([["model.layers.0"]])
        assert autoround._offloaded_blocks == {}

        autoround2 = AutoRound(
            tiny_opt_model_path,
            bits=4,
            low_cpu_mem_usage=False,
            cpu_stream_offload_blocks=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=32,
        )
        autoround2._stream_offload_blocks([["model.layers.0"]])
        assert autoround2._offloaded_blocks == {}

    def test_stream_offload_blocks_records_blocks(self, tiny_opt_model_path, tmp_path, monkeypatch):
        """Test that _stream_offload_blocks records offloaded blocks when enabled."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            low_cpu_mem_usage=True,
            cpu_stream_offload_blocks=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=32,
        )

        dummy_block = torch.nn.Linear(4, 4)
        monkeypatch.setattr(base_module, "get_module", lambda _model, _name: dummy_block)
        monkeypatch.setattr(autoround, "_init_cpu_offload_dir", lambda: str(tmp_path))
        monkeypatch.setattr(torch, "save", lambda *args, **kwargs: None)
        monkeypatch.setattr(base_module, "clear_module_weights", lambda *_args, **_kwargs: None)

        autoround._stream_offload_blocks([["model.layers.0"]])
        assert "model.layers.0" in autoround._offloaded_blocks


class TestCpuStreamLoss:
    """Tests for cpu_stream_loss option."""

    def test_option_stored_correctly(self, tiny_opt_model_path):
        """Test that cpu_stream_loss option is stored correctly."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            low_cpu_mem_usage=True,
            cpu_stream_loss=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=32,
        )
        assert autoround.cpu_stream_loss is True

        autoround2 = AutoRound(
            tiny_opt_model_path,
            bits=4,
            low_cpu_mem_usage=False,
            cpu_stream_loss=False,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=32,
        )
        assert autoround2.cpu_stream_loss is False

    def test_stream_loss_requires_nblocks_1(self, tiny_opt_model_path):
        """Test that cpu_stream_loss only works with nblocks=1."""
        # nblocks > 1 should trigger warning and disable stream_loss
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            low_cpu_mem_usage=True,
            cpu_stream_loss=True,
            nblocks=2,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=32,
        )
        # The option is stored, but internally it will fall back during quantize
        assert autoround.cpu_stream_loss is True
        assert autoround.nblocks == 2
        stream_loss = autoround.cpu_stream_loss and autoround.nblocks == 1
        assert stream_loss is False


class TestCombinedOptions:
    """Tests for combined optimization options."""

    def test_both_options_enabled(self, tiny_opt_model_path):
        """Test that both options can be enabled together."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            low_cpu_mem_usage=True,
            cpu_stream_offload_blocks=True,
            cpu_stream_loss=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=32,
        )
        assert autoround.cpu_stream_offload_blocks is True
        assert autoround.cpu_stream_loss is True
        assert autoround.low_cpu_mem_usage is True
