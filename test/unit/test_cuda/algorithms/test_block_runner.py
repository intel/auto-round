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
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

from auto_round.algorithms.block_runner import (
    BlockForwardRunner,
    register_block_output,
    register_diffusion_output,
)


class _DummyBlock(nn.Module):
    pass


def _make_runner(**kwargs):
    defaults = dict(
        batch_dim=0,
        batch_size=8,
        device="cpu",
        cache_device="cpu",
        amp=False,
        is_diffusion=False,
        enable_torch_compile=False,
    )
    defaults.update(kwargs)
    return BlockForwardRunner(**defaults)


def _fake_block_forward(block, input_ids, input_others, amp=False, amp_dtype=None, device=None, output_return_id=0):
    return input_ids * 2


def _fake_diffusion_block_forward(
    block, input_ids, input_others, amp=False, amp_dtype=None, device=None, output_return_id=0
):
    # Diffusion blocks return a tuple; hidden_states is the last element here.
    return (torch.zeros_like(input_ids), input_ids * 2)


# ==============================================================================
# Registries
# ==============================================================================


class TestRegistries:
    def test_diffusion_registry_maps_name_to_keys(self):
        register_diffusion_output("MyTestBlock", ["encoder_hidden_states", "hidden_states"])
        assert BlockForwardRunner.DIFFUSION_OUTPUT_CONFIGS["MyTestBlock"] == [
            "encoder_hidden_states",
            "hidden_states",
        ]

    def test_block_output_registry(self):
        register_block_output("MyTestDecBlock", ["hidden_states", "prev_topk"])
        from auto_round.algorithms import block_runner as br

        assert br._BLOCK_OUTPUT_REGISTRY["MyTestDecBlock"] == ["hidden_states", "prev_topk"]

    def test_builtin_diffusion_blocks_registered(self):
        from auto_round.algorithms import block_runner as br

        assert "FluxTransformerBlock" in br._DIFFUSION_OUTPUT_REGISTRY
        assert "WanTransformerBlock" in br._DIFFUSION_OUTPUT_REGISTRY


# ==============================================================================
# BlockForwardRunner core
# ==============================================================================


class TestCountSamples:
    def test_list_input(self):
        r = _make_runner()
        assert r._count_samples([torch.ones(1, 4), torch.ones(1, 4), torch.ones(1, 4)]) == 3

    def test_tensor_input(self):
        r = _make_runner(batch_dim=1)
        assert r._count_samples(torch.ones(2, 4, 8)) == 4

    def test_dict_hidden_states_list(self):
        r = _make_runner()
        assert r._count_samples({"hidden_states": [torch.ones(1, 4), torch.ones(1, 4)]}) == 2


class TestSplitOutputs:
    def test_splits_along_batch_dim(self):
        r = _make_runner(batch_dim=0)
        parts = r.split_outputs(torch.arange(6).reshape(3, 2))
        assert len(parts) == 3
        assert parts[0].shape == (1, 2)


class TestNormalizeOutput:
    def test_tensor_passthrough(self):
        r = _make_runner()
        t = torch.ones(2, 4)
        assert r._normalize_output(t) is t

    def test_tuple_returns_first_tensor(self):
        r = _make_runner()
        assert torch.equal(r._normalize_output((torch.ones(2, 4), torch.zeros(2, 4))), torch.ones(2, 4))

    def test_empty_tuple_raises(self):
        r = _make_runner()
        with pytest.raises(ValueError, match="empty"):
            r._normalize_output(())

    def test_non_tensor_first_raises(self):
        r = _make_runner()
        with pytest.raises(TypeError, match="Block output\\[0\\]"):
            r._normalize_output(("not_a_tensor",))

    def test_diffusion_uses_registered_output_index(self):
        register_diffusion_output("MyTestBlock", ["encoder_hidden_states", "hidden_states"])
        r = _make_runner(is_diffusion=True)
        block = type("MyTestBlock", (nn.Module,), {})()
        out = r._normalize_output((torch.zeros(1, 4), torch.ones(1, 4)), block)
        assert torch.equal(out, torch.ones(1, 4))

    def test_diffusion_hidden_states_index_oob_raises(self):
        r = _make_runner(is_diffusion=True)
        block = type("MyTestBlock2", (nn.Module,), {})()
        register_diffusion_output("MyTestBlock2", ["a", "b", "hidden_states"])
        with pytest.raises(ValueError, match="hidden_states index"):
            r._normalize_output((torch.zeros(1, 4), torch.ones(1, 4)), block)


class TestForwardLLM:
    def test_with_indices_returns_batched_tensor(self):
        inputs = [torch.ones(1, 4) * i for i in range(3)]
        with patch("auto_round.algorithms.block_runner.block_forward", side_effect=_fake_block_forward):
            r = _make_runner(batch_size=8)
            out = r(_DummyBlock(), inputs, {}, indices=torch.tensor([0, 1]))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 4)
        # sample0*2=0, sample1*2=2
        assert torch.equal(out[0], torch.zeros(4))
        assert torch.equal(out[1], torch.full((4,), 2.0))

    def test_without_indices_returns_list(self):
        inputs = [torch.ones(1, 4) * i for i in range(3)]
        with patch("auto_round.algorithms.block_runner.block_forward", side_effect=_fake_block_forward):
            r = _make_runner(batch_size=8)
            out = r(_DummyBlock(), inputs, {})
        assert isinstance(out, list)
        assert len(out) == 3
        assert all(o.shape == (1, 4) for o in out)
        # sample i has value i, *2 -> 2*i; sample 2 -> 4.0
        assert torch.equal(out[2], torch.full((1, 4), 4.0))

    def test_batch_size_one_with_indices(self):
        inputs = [torch.ones(1, 4) * i for i in range(2)]
        with patch("auto_round.algorithms.block_runner.block_forward", side_effect=_fake_block_forward):
            r = _make_runner(batch_size=1)
            out = r(_DummyBlock(), inputs, {}, indices=[0, 1])
        # each per-sample output is unsqueezed along batch_dim before cat
        assert out.shape == (2, 1, 4)

    def test_empty_indices_raises(self):
        with patch("auto_round.algorithms.block_runner.block_forward", side_effect=_fake_block_forward):
            r = _make_runner(batch_size=8)
            with pytest.raises(RuntimeError, match="no outputs"):
                r(_DummyBlock(), [torch.ones(1, 4)], {}, indices=torch.tensor([], dtype=torch.long))


class TestForwardDiffusion:
    def test_dict_input_with_output_dict(self):
        inputs = {"hidden_states": [torch.ones(1, 4) * i for i in range(2)]}
        with patch("auto_round.algorithms.block_runner.block_forward", side_effect=_fake_diffusion_block_forward):
            r = _make_runner(is_diffusion=True, output_config=["encoder_hidden_states", "hidden_states"], batch_size=8)
            out = r(_DummyBlock(), inputs, {})
        assert isinstance(out, list)
        assert len(out) == 2
        assert r.last_output_dict is not None
        assert "hidden_states" in r.last_output_dict


class TestSelectBatch:
    def test_list_input_cat(self):
        r = _make_runner()
        inputs = [torch.ones(1, 4) * i for i in range(3)]
        sel, others = r.select_batch(inputs, {}, torch.tensor([1, 2]))
        assert sel.shape == (2, 4)
        assert torch.equal(sel[0], torch.full((4,), 1.0))
        assert torch.equal(sel[1], torch.full((4,), 2.0))

    def test_shared_cache_key_kept_singleton(self):
        r = _make_runner(shared_cache_keys=("cache",))
        inputs = {"cache": [torch.ones(2, 4)]}
        sel, _ = r.select_batch(inputs, {}, torch.tensor([0]))
        assert sel["cache"].shape == (2, 4)

    def test_others_passthrough_scalars(self):
        r = _make_runner()
        inputs = [torch.ones(1, 4)]
        sel, others = r.select_batch(inputs, {"attention_mask": None, "flag": True}, torch.tensor([0]))
        assert others["attention_mask"] is None
        assert others["flag"] is True


class TestFromOrchestrator:
    def test_creates_from_orchestrator_attrs(self):
        orch = SimpleNamespace(
            batch_dim=0,
            batch_size=4,
            cache_device="cpu",
            amp=False,
            amp_dtype=torch.bfloat16,
            shared_cache_keys=(),
            model_context=SimpleNamespace(is_diffusion=False, output_config=None),
        )
        r = BlockForwardRunner.from_orchestrator(orch, enable_torch_compile=False)
        assert r.batch_size == 4
        assert r.is_diffusion is False
