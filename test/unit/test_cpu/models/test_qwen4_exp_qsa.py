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
"""Equivalence tests for the vectorized Qwen4-Exp QSA indexer.

Regression coverage for intel/auto-round#2323: the upstream indexer loops over
every (batch, query) pair, which made the full-attention blocks of
Qwen3.8-Flash-Next take minutes each during tuning.
"""

import unittest

import pytest
import torch

qwen4_exp = pytest.importorskip(
    "transformers.models.qwen4_exp.modeling_qwen4_exp",
    reason="qwen4_exp requires transformers >= 5.16",
)

from auto_round.modeling.qwen4_exp_qsa import qsa_indexer_forward  # noqa: E402


def _config(seq_len: int, ratio: int = 4, budget: int = 64):
    from transformers import Qwen4ExpTextConfig

    cfg = Qwen4ExpTextConfig(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=budget,
        indexer_compress_ratio=ratio,
        hc_count=2,
        hc_lowrank=16,
        ngram_size=3,
        ngram_vocab_size_base=512,
        heads_per_ngram=2,
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ple_layer_ids=[2],
        split_ngram_parts=4,
        make_ngram_vocab_size_divisible_by=128,
        max_position_embeddings=seq_len,
        vocab_size=512,
        partial_rotary_factor=0.25,
        rope_parameters={
            "rope_theta": 10000000,
            "rope_type": "default",
            "partial_rotary_factor": 0.25,
        },
        eos_token_id=1,
    )
    cfg._attn_implementation = "sdpa"
    return cfg


def _build_inputs(cfg, batch, seq_len, device, dtype):
    layer_idx = cfg.layer_types.index("indexed_attention")
    indexer = qwen4_exp.Qwen4ExpTextQSAIndexer(cfg, layer_idx).to(device, dtype)

    hidden = torch.randn(batch, seq_len, cfg.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device)[None, :].expand(batch, -1)
    mask = qwen4_exp.create_causal_mask(
        config=cfg,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
        allow_is_causal_skip=False,
    )
    embeds = qwen4_exp.Qwen4ExpTextRotaryEmbedding(cfg).to(device)(hidden.float(), position_ids)
    return indexer, hidden, mask, embeds


class TestQSAIndexerEquivalence(unittest.TestCase):
    """The vectorized indexer must reproduce the reference mask exactly."""

    def _assert_equal(self, batch, seq_len, ratio=4, budget=64, dtype=torch.bfloat16):
        cfg = _config(seq_len, ratio=ratio, budget=budget)
        device = "cpu"
        indexer, hidden, mask, embeds = _build_inputs(cfg, batch, seq_len, device, dtype)

        with torch.no_grad():
            expected = indexer(hidden, embeds, mask, None)
            actual = qsa_indexer_forward(indexer, hidden, embeds, mask, None)

        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(
            torch.equal(expected, actual),
            f"mask mismatch for batch={batch} seq={seq_len} ratio={ratio} budget={budget}: "
            f"{(expected != actual).sum().item()} differing elements",
        )

    def test_short_sequence(self):
        self._assert_equal(batch=1, seq_len=16)

    def test_single_block(self):
        # seq == ratio leaves exactly one complete block and an empty tail.
        self._assert_equal(batch=2, seq_len=8, ratio=4, budget=64)

    def test_multiple_blocks(self):
        self._assert_equal(batch=2, seq_len=128)

    def test_odd_sequence_length(self):
        self._assert_equal(batch=2, seq_len=37)

    def test_budget_smaller_than_visible_blocks(self):
        # token_budget shrinks below the number of visible blocks: the offending
        # tail must be dropped exactly like the reference does.
        self._assert_equal(batch=1, seq_len=256, ratio=4, budget=8)

    def test_ratio_one(self):
        self._assert_equal(batch=2, seq_len=64, ratio=1, budget=32)

    def test_ratio_eight(self):
        self._assert_equal(batch=2, seq_len=128, ratio=8, budget=64)

    def test_float_mask(self):
        """Float masks must produce the additive variant, not booleans."""
        cfg = _config(64)
        indexer, hidden, mask, embeds = _build_inputs(cfg, 1, 64, "cpu", torch.float32)
        float_mask = torch.where(mask, torch.zeros((), dtype=torch.float32), torch.finfo(torch.float32).min)
        with torch.no_grad():
            expected = indexer(hidden, embeds, float_mask, None)
            actual = qsa_indexer_forward(indexer, hidden, embeds, float_mask, None)
        self.assertTrue(expected.is_floating_point() and actual.is_floating_point())
        self.assertTrue(torch.equal(expected, actual))


class TestQSAIndexerPatch(unittest.TestCase):
    """The installed patch must be idempotent and keep the original around."""

    def test_patch_is_idempotent(self):
        from auto_round.modeling.qwen4_exp_qsa import patch_qwen4_exp_indexer

        cls = qwen4_exp.Qwen4ExpTextQSAIndexer
        self.assertTrue(patch_qwen4_exp_indexer())
        patched = cls.forward
        self.assertTrue(patch_qwen4_exp_indexer())
        # Second call must not re-wrap.
        self.assertIs(patched, cls.forward)
        self.assertTrue(getattr(cls.forward, "_auto_round_vectorized", False))
        self.assertIsNotNone(getattr(cls.forward, "_auto_round_original", None))


if __name__ == "__main__":
    unittest.main()
