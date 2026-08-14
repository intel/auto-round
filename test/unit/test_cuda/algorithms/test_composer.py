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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

from auto_round.algorithms.composer import AlgorithmComposer, BlockContext, _can_compile_block_forward
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.transforms.awq.config import AWQConfig


class TestCanCompileBlockForward:
    def test_disabled_by_user(self):
        class _Q:
            def can_compile_block_forward(self):
                return True

        class _R:
            def can_compile_block_forward(self):
                return True

        assert _can_compile_block_forward(_Q(), [_R()], user_enabled=False) is False

    def test_quantizer_does_not_support(self):
        class _Q:
            def can_compile_block_forward(self):
                return False

        class _R:
            def can_compile_block_forward(self):
                return True

        assert _can_compile_block_forward(_Q(), [_R()], user_enabled=True) is False

    def test_incompatible_rotation_disables(self):
        class _Q:
            def can_compile_block_forward(self):
                return True

        class _R:
            def can_compile_block_forward(self):
                return False

        assert _can_compile_block_forward(_Q(), [_R()], user_enabled=True) is False

    def test_all_support_compiles(self):
        class _Q:
            def can_compile_block_forward(self):
                return True

        class _R:
            def can_compile_block_forward(self):
                return True

        assert _can_compile_block_forward(_Q(), [_R()], user_enabled=True) is True


class TestAlgorithmComposerConstruction:
    def test_single_rtn_quantizer(self):
        c = AlgorithmComposer([RTNConfig()])
        assert c.preprocessors == []
        assert type(c.block_quantizer).__name__ == "OptimizedRTNQuantizer"
        assert [type(m).__name__ for m in c.members()] == ["OptimizedRTNQuantizer"]

    def test_empty_configs_auto_appends_rtn(self):
        c = AlgorithmComposer([])
        assert type(c.block_quantizer).__name__ == "OptimizedRTNQuantizer"

    def test_awq_preprocessor_then_rtn_quantizer(self):
        c = AlgorithmComposer([AWQConfig(), RTNConfig()])
        assert [type(p).__name__ for p in c.preprocessors] == ["AWQTransform"]
        assert type(c.block_quantizer).__name__ == "OptimizedRTNQuantizer"

    def test_multiple_block_quantizers_raise(self):
        with pytest.raises(ValueError, match="exactly one block-quantization"):
            AlgorithmComposer([RTNConfig(), RTNConfig()])

    def test_duplicate_preprocessors_raise(self):
        with pytest.raises(ValueError, match="Duplicate preprocessor"):
            AlgorithmComposer([AWQConfig(), AWQConfig(), RTNConfig()])

    def test_non_quantization_config_ignored(self):
        class _NotAConfig:
            pass

        c = AlgorithmComposer([_NotAConfig(), RTNConfig()])
        assert type(c.block_quantizer).__name__ == "OptimizedRTNQuantizer"


class TestNeedQuantedInput:
    def test_default_false(self):
        c = AlgorithmComposer([RTNConfig()])
        assert c.need_quanted_input() is False

    def test_preprocessor_enable_quanted_input(self):
        class _Pre:
            enable_quanted_input = True

        c = AlgorithmComposer([RTNConfig()])
        c.preprocessors = [_Pre()]
        assert c.need_quanted_input() is True

    def test_quantizer_enable_quanted_input(self):
        c = AlgorithmComposer([RTNConfig()])
        c.block_quantizer.enable_quanted_input = True
        assert c.need_quanted_input() is True


class TestBlockContext:
    def test_defaults(self):
        block = nn.Linear(4, 4)
        bc = BlockContext(model=block, block_names=["0"], block_name="0", block_index=0)
        assert bc.bs == 1
        assert bc.is_mllm is False
        assert bc.is_diffusion is False
        assert bc.block_cnt == 0
        assert bc.pbar is None


class TestActMaxHooks:
    def _make_block(self):
        class _SmallBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)
                self.linear.bits = 4
                self.linear.act_dynamic = False
                self.linear.act_data_type = "fp"
                self.linear.act_bits = 8
                self.linear.act_group_size = 8

            def forward(self, x):
                return self.linear(x)

        return _SmallBlock()

    def test_registers_hook_and_collects_act_max(self):
        c = AlgorithmComposer([RTNConfig()])
        c.scheme = SimpleNamespace(act_dynamic=False, data_type="int", group_size=-1, act_data_type="fp")
        block = self._make_block()
        handles = c._register_act_max_hooks(block)
        assert len(handles) >= 1
        try:
            with torch.no_grad():
                block(torch.randn(4, 8))
            assert hasattr(block.linear, "act_max")
            assert block.linear.act_max.ndim >= 1
        finally:
            for h in handles:
                h.remove()

    def test_no_hook_for_dynamic_activation(self):
        c = AlgorithmComposer([RTNConfig()])
        block = self._make_block()
        block.linear.act_dynamic = True
        block.linear.act_bits = 16
        handles = c._register_act_max_hooks(block)
        assert handles == []


class TestPrepareFinalize:
    def test_prepare_and_finalize_run_forward_to_members(self):
        calls = []

        class _Pre:
            def prepare_run(self, composer=None):
                calls.append("pre_prepare")

            def finalize_run(self):
                calls.append("pre_finalize")

        c = AlgorithmComposer([RTNConfig()])
        c.preprocessors = [_Pre()]
        c.prepare_run()
        c.finalize_run()
        assert calls == ["pre_prepare", "pre_finalize"]
