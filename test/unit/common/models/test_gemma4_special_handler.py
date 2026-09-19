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
"""Unit tests for Gemma4 special model handler changes.

Covers:
1. _get_gemma4_shared_kv_states_global
2. position_ids guard in prepare_special_model_block_inputs
3. _attach_gemma4_rotary_emb (transformers >= 5.6 path)
4. _patch_gemma4_model (transformers < 5.6 path)
5. Gemma4 position embedding replay (block_forward level)
6. _prepare_gemma4_replay_inputs helper
7. prepare_special_model_block_inputs Gemma4 dispatch
8. block_forward Gemma4 integration

Uses real Gemma4TextModel from transformers where available; falls back to
mocks for untestable paths.
"""

import types
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. Tests for _get_gemma4_shared_kv_states_global
# ---------------------------------------------------------------------------


class TestGemma4SharedKvStatesGlobal:
    """Unit tests for _get_gemma4_shared_kv_states_global."""

    def test_returns_shared_ref_when_attached(self):
        """Returns the attached _shared_kv_states_global_ref."""
        from auto_round.special_model_handler import _get_gemma4_shared_kv_states_global

        class FakeBlock(nn.Module):
            pass

        block = FakeBlock()
        shared_dict = {}
        object.__setattr__(block, "_shared_kv_states_global_ref", shared_dict)

        result = _get_gemma4_shared_kv_states_global(block)
        assert result is shared_dict

    def test_returns_empty_dict_when_no_ref(self):
        """Returns a fresh {} when no ref is attached."""
        from auto_round.special_model_handler import _get_gemma4_shared_kv_states_global

        class FakeBlock(nn.Module):
            pass

        block = FakeBlock()
        result = _get_gemma4_shared_kv_states_global(block)
        assert result == {}

    def test_shared_dict_is_mutated_across_layers(self):
        """The same dict reference is returned for multiple layers."""
        from auto_round.special_model_handler import _get_gemma4_shared_kv_states_global

        class FakeBlock(nn.Module):
            pass

        shared_dict = {}
        block0 = FakeBlock()
        block1 = FakeBlock()
        object.__setattr__(block0, "_shared_kv_states_global_ref", shared_dict)
        object.__setattr__(block1, "_shared_kv_states_global_ref", shared_dict)

        result0 = _get_gemma4_shared_kv_states_global(block0)
        result1 = _get_gemma4_shared_kv_states_global(block1)

        assert result0 is result1 is shared_dict
        # Mutating via one layer is visible to the other
        result0["key"] = "value"
        assert result1["key"] == "value"


# ---------------------------------------------------------------------------
# 2. Tests for position_ids guard in prepare_special_model_block_inputs
# ---------------------------------------------------------------------------


class TestPrepareSpecialModelBlockInputsPositionIdsGuard:
    """Unit tests for position_ids normalization in prepare_special_model_block_inputs.

    These tests cover the defensive guard added to handle position_ids arriving
    as a list or None instead of a tensor — which occurs in the MLLM calibration
    pipeline on transformers >= 5.6.
    """

    def _make_fake_block(self):
        class FakeRotaryEmbedding:
            def __call__(self, input_ids, position_ids, lt):
                return torch.ones(*input_ids.shape, 512), torch.ones(*input_ids.shape, 512)

        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_type = "full_attention"
                self.head_dim = 512

        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttention()

        block = FakeBlock()
        block._rotary_emb_ref = [FakeRotaryEmbedding()]
        block._autoround_special_replay = "gemma4"
        block.layer_idx = 0
        return block

    def test_position_ids_as_single_element_list_is_unwrapped(self):
        """Single-element list [tensor] is unwrapped to bare tensor."""
        from auto_round.special_model_handler import prepare_special_model_block_inputs

        block = self._make_fake_block()
        hidden_states = torch.zeros((1, 4, 8), dtype=torch.float32)
        inner_tensor = torch.arange(4).unsqueeze(0)
        input_others = {"position_ids": [inner_tensor], "positional_inputs": []}

        updated_inputs, _ = prepare_special_model_block_inputs(
            block, hidden_states, input_others, positional_inputs=None
        )

        assert updated_inputs["position_ids"] is inner_tensor
        assert not isinstance(updated_inputs["position_ids"], list)

    def test_position_ids_as_empty_list_is_rebuilt_from_seq_len(self):
        """Empty list [] causes position_ids to be rebuilt from sequence length."""
        from auto_round.special_model_handler import prepare_special_model_block_inputs

        block = self._make_fake_block()
        hidden_states = torch.zeros((2, 8, 8), dtype=torch.float32)
        input_others = {"position_ids": [], "positional_inputs": []}

        updated_inputs, _ = prepare_special_model_block_inputs(
            block, hidden_states, input_others, positional_inputs=None
        )

        rebuilt = updated_inputs["position_ids"]
        assert isinstance(rebuilt, torch.Tensor)
        assert rebuilt.shape == (2, 8)
        assert rebuilt[0].tolist() == list(range(8))

    def test_position_ids_as_none_is_rebuilt_from_seq_len(self):
        """position_ids=None causes position_ids to be rebuilt from sequence length."""
        from auto_round.special_model_handler import prepare_special_model_block_inputs

        block = self._make_fake_block()
        hidden_states = torch.zeros((1, 6, 8), dtype=torch.float32)
        input_others = {"position_ids": None, "positional_inputs": []}

        updated_inputs, _ = prepare_special_model_block_inputs(
            block, hidden_states, input_others, positional_inputs=None
        )

        rebuilt = updated_inputs["position_ids"]
        assert isinstance(rebuilt, torch.Tensor)
        assert rebuilt.shape == (1, 6)
        assert rebuilt[0].tolist() == list(range(6))

    def test_position_ids_as_tensor_passes_through_unchanged(self):
        """Bare tensor position_ids is passed through without modification."""
        from auto_round.special_model_handler import prepare_special_model_block_inputs

        block = self._make_fake_block()
        hidden_states = torch.zeros((1, 4, 8), dtype=torch.float32)
        original = torch.arange(4).unsqueeze(0)
        input_others = {"position_ids": original, "positional_inputs": []}

        updated_inputs, _ = prepare_special_model_block_inputs(
            block, hidden_states, input_others, positional_inputs=None
        )

        assert updated_inputs["position_ids"] is original


# ---------------------------------------------------------------------------
# 3. Tests for _attach_gemma4_rotary_emb (transformers >= 5.6 path)
# ---------------------------------------------------------------------------


class TestAttachGemma4RotaryEmb:
    """Unit tests for _attach_gemma4_rotary_emb.

    On transformers >= 5.6 this function attaches _rotary_emb_ref and
    _shared_kv_states_global_ref to every Gemma4 layer.
    """

    @pytest.fixture
    def gemma4_model(self):
        """Minimal real Gemma4TextModel (transformers >= 5.6 path)."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextConfig, Gemma4TextModel

        config = Gemma4TextConfig(
            num_hidden_layers=3,
            num_attention_heads=2,
            intermediate_size=8,
            hidden_size=8,
        )
        model = Gemma4TextModel(config).cpu().eval()
        wrapper = nn.Module()
        wrapper.language_model = model
        return wrapper

    def test_attaches_rotary_emb_ref_to_all_layers(self, gemma4_model):
        from auto_round.special_model_handler import _attach_gemma4_rotary_emb

        _attach_gemma4_rotary_emb(gemma4_model)

        text_model = gemma4_model.language_model
        for i, layer in enumerate(text_model.layers):
            ref = getattr(layer, "_rotary_emb_ref", None)
            assert ref is not None, f"layer {i} missing _rotary_emb_ref"
            assert ref[0] is text_model.rotary_emb

    def test_attaches_shared_kv_states_global_ref_to_all_layers(self, gemma4_model):
        from auto_round.special_model_handler import _attach_gemma4_rotary_emb

        _attach_gemma4_rotary_emb(gemma4_model)

        text_model = gemma4_model.language_model
        shared_refs = [getattr(layer, "_shared_kv_states_global_ref", None) for layer in text_model.layers]
        assert all(r is not None for r in shared_refs), "all layers must have _shared_kv_states_global_ref"
        for i in range(1, len(shared_refs)):
            assert shared_refs[0] is shared_refs[i], "layers must share the same dict"
        shared_refs[0]["_test_marker"] = 99
        for r in shared_refs[1:]:
            assert r.get("_test_marker") == 99

    def test_attaches_special_replay_marker_to_all_layers(self, gemma4_model):
        from auto_round.special_model_handler import _attach_gemma4_rotary_emb

        _attach_gemma4_rotary_emb(gemma4_model)

        text_model = gemma4_model.language_model
        for i, layer in enumerate(text_model.layers):
            marker = getattr(layer, "_autoround_special_replay", None)
            assert marker == "gemma4", f"layer {i} missing gemma4 marker"

    def test_noops_when_gemma4textmodel_not_found(self):
        from auto_round.special_model_handler import _attach_gemma4_rotary_emb

        model = nn.Module()
        result = _attach_gemma4_rotary_emb(model)
        assert result is None


# ---------------------------------------------------------------------------
# 4. Tests for _patch_gemma4_model (transformers < 5.6 path)
# ---------------------------------------------------------------------------


class TestPatchGemma4Model:
    """Unit tests for _patch_gemma4_model.

    On transformers < 5.6 this function replaces each layer's forward with
    a patched version that rebuilds position_embeddings and attention_mask.
    """

    @pytest.fixture
    def gemma4_model(self):
        """Minimal real Gemma4TextModel (transformers >= 5.6, but _patch_gemma4_model
        is called for < 5.6 — we test the patching logic directly)."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextConfig, Gemma4TextModel

        config = Gemma4TextConfig(
            num_hidden_layers=3,
            num_attention_heads=2,
            intermediate_size=8,
            hidden_size=8,
        )
        model = Gemma4TextModel(config).cpu().eval()
        wrapper = nn.Module()
        wrapper.language_model = model
        return wrapper

    def test_patches_all_layers_with_special_replay_marker(self, gemma4_model):
        from auto_round.special_model_handler import _patch_gemma4_model

        result = _patch_gemma4_model(gemma4_model)

        text_model = result.language_model
        for i, layer in enumerate(text_model.layers):
            marker = getattr(layer, "_autoround_special_replay", None)
            assert marker == "gemma4", f"layer {i} missing gemma4 marker"

    def test_patched_forward_accepts_position_ids_none(self):
        """Patched forward must not crash when position_ids is None."""
        from auto_round.special_model_handler import _patch_gemma4_model

        class FakeRotaryEmb(nn.Module):
            def __call__(self, x, position_ids, layer_type):
                return torch.ones(x.shape[0], x.shape[1], 512), torch.ones(x.shape[0], x.shape[1], 512)

        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_type = "full_attention"
                self.head_dim = 512
                self.store_full_length_kv = True

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttention()
                self.layer_idx = 0

            def forward(self, hidden_states, **kwargs):
                return hidden_states

        class FakeTextModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.rotary_emb = FakeRotaryEmb()
                self.layers = nn.ModuleList([FakeLayer()])
                self.config = types.SimpleNamespace()

        model = nn.Module()
        object.__setattr__(model, "language_model", FakeTextModel())

        _patch_gemma4_model(model)

        layer = model.language_model.layers[0]
        hidden_states = torch.zeros((1, 4, 8))
        try:
            layer.forward(hidden_states, position_ids=None)
        except AttributeError:
            pytest.fail("Patched forward crashed on position_ids=None — guard missing or broken")

    def test_noops_when_gemma4textmodel_not_found(self):
        from auto_round.special_model_handler import _patch_gemma4_model

        model = nn.Module()
        result = _patch_gemma4_model(model)
        assert result is model


# ---------------------------------------------------------------------------
# 5. Tests for Gemma4 position embedding replay (block_forward level)
# ---------------------------------------------------------------------------


class TestGemma4PositionEmbeddingReplay:
    """Unit tests for Gemma4 position embedding recomputation in block_forward.

    These test the full block_forward -> prepare_special_model_block_inputs ->
    _prepare_gemma4_replay_inputs -> rotary_emb pipeline with fake blocks.
    """

    def _make_fake_block(self, layer_type="full_attention", head_dim=512):
        """Helper to build a fake Gemma4-like block with _rotary_emb attached."""

        class FakeRotaryEmbedding:
            def __call__(self, input_ids, position_ids, lt):
                hd = 512 if lt == "full_attention" else 256
                shape = (*input_ids.shape, hd)
                return torch.ones(shape), torch.ones(shape)

        class FakeAttention(nn.Module):
            def __init__(self, lt, hd):
                super().__init__()
                self.layer_type = lt
                self.head_dim = hd

        class FakeBlock(nn.Module):
            def __init__(self, lt, hd):
                super().__init__()
                self.self_attn = FakeAttention(lt, hd)

            def forward(self, input_ids, **kwargs):
                return kwargs["position_embeddings"][0]

        block = FakeBlock(layer_type, head_dim)
        block._rotary_emb_ref = [FakeRotaryEmbedding()]
        block._autoround_special_replay = "gemma4"
        return block

    def test_recomputes_when_position_embeddings_missing(self):
        """position_embeddings absent → should be recomputed from position_ids."""
        from auto_round.compressors.utils import block_forward

        block = self._make_fake_block("full_attention", 512)
        input_ids = torch.zeros((1, 4), dtype=torch.float32)
        position_ids = torch.arange(4).unsqueeze(0)
        output = block_forward(
            block,
            input_ids,
            {"position_ids": position_ids, "positional_inputs": []},
            device=torch.device("cpu"),
        )
        assert output.shape[-1] == 512

    def test_recomputes_when_position_embeddings_shape_mismatches(self):
        """position_embeddings present but wrong dim (cached from sliding layer) → recompute."""
        from auto_round.compressors.utils import block_forward

        block = self._make_fake_block("full_attention", 512)
        input_ids = torch.zeros((1, 4), dtype=torch.float32)
        position_ids = torch.arange(4).unsqueeze(0)
        wrong_pe = (torch.ones(1, 4, 256), torch.ones(1, 4, 256))
        output = block_forward(
            block,
            input_ids,
            {"position_ids": position_ids, "position_embeddings": wrong_pe, "positional_inputs": []},
            device=torch.device("cpu"),
        )
        assert output.shape[-1] == 512

    def test_keeps_position_embeddings_when_shape_matches(self):
        """position_embeddings present with correct dim → no recompute."""
        from auto_round.compressors.utils import block_forward

        block = self._make_fake_block("sliding_attention", 256)
        input_ids = torch.zeros((1, 4), dtype=torch.float32)
        position_ids = torch.arange(4).unsqueeze(0)
        correct_pe = (torch.full((1, 4, 256), 42.0), torch.full((1, 4, 256), 42.0))
        output = block_forward(
            block,
            input_ids,
            {"position_ids": position_ids, "position_embeddings": correct_pe, "positional_inputs": []},
            device=torch.device("cpu"),
        )
        assert output.shape[-1] == 256
        assert torch.allclose(output, torch.full((1, 4, 256), 42.0))


# ---------------------------------------------------------------------------
# 6. Tests for _prepare_gemma4_replay_inputs helper
# ---------------------------------------------------------------------------


class TestGemma4ReplayInputHelper:
    """Unit tests for _prepare_gemma4_replay_inputs."""

    def _make_fake_block(self, layer_type="full_attention", head_dim=512, store_full_length_kv=True):
        class FakeRotaryEmbedding:
            def __call__(self, input_states, position_ids, lt):
                hd = 512 if lt == "full_attention" else 256
                shape = (*input_states.shape, hd)
                return torch.ones(shape), torch.ones(shape)

        class FakeAttention(nn.Module):
            def __init__(self, lt, hd, store_kv):
                super().__init__()
                self.layer_type = lt
                self.head_dim = hd
                if store_kv:
                    self.store_full_length_kv = True

        class FakeBlock(nn.Module):
            def __init__(self, lt, hd, store_kv):
                super().__init__()
                self.self_attn = FakeAttention(lt, hd, store_kv)

        block = FakeBlock(layer_type, head_dim, store_full_length_kv)
        block._rotary_emb_ref = [FakeRotaryEmbedding()]
        block._autoround_special_replay = "gemma4"
        return block

    def test_injects_shared_kv_and_rebuilds_missing_position_embeddings(self):
        from auto_round.special_model_handler import _prepare_gemma4_replay_inputs

        block = self._make_fake_block("full_attention", 512, store_full_length_kv=True)
        hidden_states = torch.zeros((1, 4, 8), dtype=torch.float32)
        position_ids = torch.arange(4).unsqueeze(0)
        shared_kv_states = {}

        prepared = _prepare_gemma4_replay_inputs(
            block,
            hidden_states,
            position_ids=position_ids,
            default_shared_kv_states=shared_kv_states,
        )

        assert prepared["shared_kv_states"] is shared_kv_states
        assert prepared["position_embeddings"][0].shape[-1] == 512

    def test_rebuilds_attention_mask_for_sliding_layers(self):
        from auto_round.special_model_handler import _prepare_gemma4_replay_inputs

        block = self._make_fake_block("sliding_attention", 256, store_full_length_kv=False)
        hidden_states = torch.zeros((1, 4, 8), dtype=torch.float32)
        position_ids = torch.arange(4).unsqueeze(0)
        expected_mask = torch.full((1, 1, 4, 4), 7.0)

        with patch(
            "auto_round.special_model_handler._rebuild_gemma4_attention_mask",
            return_value=expected_mask,
        ) as rebuild_mask:
            prepared = _prepare_gemma4_replay_inputs(
                block,
                hidden_states,
                position_ids=position_ids,
                attention_mask=None,
                config=types.SimpleNamespace(),
            )

        rebuild_mask.assert_called_once()
        assert torch.equal(prepared["attention_mask"], expected_mask)

    def test_keeps_other_layer_types_when_position_embeddings_is_dict(self):
        from auto_round.special_model_handler import _prepare_gemma4_replay_inputs

        block = self._make_fake_block("full_attention", 512, store_full_length_kv=False)
        hidden_states = torch.zeros((1, 4, 8), dtype=torch.float32)
        position_ids = torch.arange(4).unsqueeze(0)
        cached_dict = {"sliding_attention": (torch.full((1, 4, 256), 3.0), torch.full((1, 4, 256), 3.0))}

        prepared = _prepare_gemma4_replay_inputs(
            block,
            hidden_states,
            position_ids=position_ids,
            position_embeddings=cached_dict,
        )

        assert prepared["position_embeddings"]["sliding_attention"][0].shape[-1] == 256
        assert prepared["position_embeddings"]["full_attention"][0].shape[-1] == 512


# ---------------------------------------------------------------------------
# 7. Tests for prepare_special_model_block_inputs Gemma4 dispatch
# ---------------------------------------------------------------------------


class TestPrepareSpecialModelBlockInputsDispatch:
    """Unit tests for Gemma4-specific dispatch in prepare_special_model_block_inputs."""

    def test_gemma4_dispatch_rebuilds_mask_and_normalizes_per_layer_input(self):
        from auto_round.special_model_handler import prepare_special_model_block_inputs

        class FakeRotaryEmbedding:
            def __call__(self, input_ids, position_ids, lt):
                hd = 512 if lt == "full_attention" else 256
                shape = (*input_ids.shape, hd)
                return torch.ones(shape), torch.ones(shape)

        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_type = "full_attention"
                self.head_dim = 512

        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttention()
                self._rotary_emb_ref = [FakeRotaryEmbedding()]
                self._autoround_special_replay = "gemma4"
                self._gemma4_config_ref = types.SimpleNamespace()

        block = FakeBlock()
        hidden_states = torch.zeros((1, 4, 8), dtype=torch.float32)
        position_ids = torch.arange(4).unsqueeze(0)
        per_layer_input = torch.ones((1, 2, 8), dtype=torch.float32)
        expected_mask = torch.full((1, 1, 4, 4), 9.0)

        with patch(
            "auto_round.special_model_handler._rebuild_gemma4_attention_mask",
            return_value=expected_mask,
        ) as rebuild_mask:
            updated_inputs, updated_positional = prepare_special_model_block_inputs(
                block,
                hidden_states,
                {"position_ids": position_ids},
                [per_layer_input],
            )

        rebuild_mask.assert_called_once()
        assert torch.equal(updated_inputs["attention_mask"], expected_mask)
        assert updated_positional[0].shape[1] == hidden_states.shape[1]


# ---------------------------------------------------------------------------
# 8. Tests for block_forward Gemma4 integration
# ---------------------------------------------------------------------------


class TestBlockForwardGemma4Dispatch:
    """Unit tests for Gemma4 integration in block_forward."""

    def test_legacy_block_forward_uses_special_model_dispatch(self):
        from auto_round.compressors.utils import block_forward

        class FakeRotaryEmbedding:
            def __call__(self, input_ids, position_ids, lt):
                hd = 512 if lt == "full_attention" else 256
                shape = (*input_ids.shape, hd)
                return torch.ones(shape), torch.ones(shape)

        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_type = "full_attention"
                self.head_dim = 512

        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttention()
                self._rotary_emb_ref = [FakeRotaryEmbedding()]
                self._autoround_special_replay = "gemma4"

            def forward(self, input_ids, **kwargs):
                return kwargs["position_embeddings"][0]

        block = FakeBlock()
        input_ids = torch.zeros((1, 4), dtype=torch.float32)
        position_ids = torch.arange(4).unsqueeze(0)

        output = block_forward(
            block,
            input_ids,
            {"position_ids": position_ids, "positional_inputs": []},
            device=torch.device("cpu"),
        )

        assert output.shape[-1] == 512
