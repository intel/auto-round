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

from auto_round.modeling.qwen4_exp_qsa import _upstream_forward, qsa_indexer_forward  # noqa: E402


def _reference(indexer, hidden, embeds, mask):
    """Upstream per-token loop; ``import auto_round`` already patched the class."""
    with torch.no_grad():
        return _upstream_forward()(indexer, hidden, embeds, mask, None)


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


def _block_scores(indexer, hidden, embeds, ratio, batch_idx, query_idx, n_blocks):
    """Upstream's block scores for one query, to tell ties apart from real mistakes."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    dim = indexer.index_head_dim
    seq_len = hidden.shape[1]
    with torch.no_grad():
        qk = indexer.index_qk_proj(hidden)
        q, token_k = torch.split(qk, [indexer.index_n_heads * dim, indexer.index_kv_heads * dim], dim=-1)
        q = indexer.q_layernorm(q.reshape(hidden.shape[0], seq_len, -1, dim))
        q = apply_rotary_pos_emb(q, cos=embeds[0][:, -seq_len:], sin=embeds[1][:, -seq_len:], unsqueeze_dim=2)
        keys = token_k.reshape(hidden.shape[0], seq_len, dim)[batch_idx, : n_blocks * ratio]
        pooled = indexer.k_layernorm(keys.view(n_blocks, ratio, dim).float().mean(dim=1).to(hidden.dtype))
        starts = torch.arange(0, n_blocks * ratio, ratio)
        block_keys = apply_rotary_pos_emb(
            pooled.unsqueeze(1),
            cos=embeds[0][batch_idx].index_select(0, starts),
            sin=embeds[1][batch_idx].index_select(0, starts),
        ).squeeze(1)
        return torch.relu(torch.matmul(q[batch_idx, query_idx].float(), block_keys.float().T)).sum(0)


class TestQSAIndexerEquivalence(unittest.TestCase):
    """The vectorized indexer must select the same keys as the upstream loop."""

    def _compare(self, batch, seq_len, ratio=4, budget=64, dtype=torch.bfloat16):
        cfg = _config(seq_len, ratio=ratio, budget=budget)
        indexer, hidden, mask, embeds = _build_inputs(cfg, batch, seq_len, "cpu", dtype)
        expected = _reference(indexer, hidden, embeds, mask)
        with torch.no_grad():
            actual = qsa_indexer_forward(indexer, hidden, embeds, mask, None)
        self.assertEqual(expected.shape, actual.shape)
        return cfg, indexer, hidden, mask, embeds, expected, actual

    def _assert_equal(self, *args, **kwargs):
        *_, expected, actual = self._compare(*args, **kwargs)
        self.assertTrue(torch.equal(expected, actual), f"{(expected != actual).sum().item()} differing elements")

    def _assert_equivalent(self, batch, seq_len, ratio, budget):
        """Equal up to torch.topk's unspecified tie-breaking between equal block scores."""
        cfg, indexer, hidden, mask, embeds, expected, actual = self._compare(batch, seq_len, ratio, budget)
        for b in range(batch):
            for q in torch.nonzero((expected[b, 0] != actual[b, 0]).any(dim=-1)).flatten().tolist():
                n_blocks = int(mask[b, 0, q].sum()) // ratio
                scores = _block_scores(indexer, hidden, embeds, ratio, b, q, n_blocks)

                def chosen(m):
                    return sorted(
                        {k // ratio for k in m[b, 0, q].nonzero().flatten().tolist() if k // ratio < n_blocks}
                    )

                self.assertEqual(
                    sorted(scores[chosen(expected)].tolist()),
                    sorted(scores[chosen(actual)].tolist()),
                    f"query {q}: selected blocks differ and are not tied",
                )

    def test_short_sequence(self):
        self._assert_equal(batch=1, seq_len=16)

    def test_single_block(self):
        self._assert_equal(batch=2, seq_len=8, ratio=4, budget=64)

    def test_odd_sequence_length(self):
        self._assert_equal(batch=2, seq_len=37)

    def test_budget_smaller_than_visible_blocks(self):
        self._assert_equivalent(batch=1, seq_len=256, ratio=4, budget=8)

    def test_multiple_blocks(self):
        self._assert_equivalent(batch=2, seq_len=128, ratio=4, budget=64)

    def test_ratio_one(self):
        self._assert_equivalent(batch=2, seq_len=64, ratio=1, budget=32)

    def test_ratio_eight(self):
        self._assert_equivalent(batch=2, seq_len=128, ratio=8, budget=64)


class TestQSAIndexerMaskContract(unittest.TestCase):
    """Mask dtype and layout follow upstream, for the installed transformers version."""

    def test_float_mask_follows_upstream_dtype(self):
        cfg = _config(64)
        indexer, hidden, mask, embeds = _build_inputs(cfg, 1, 64, "cpu", torch.float32)
        float_mask = torch.where(mask, torch.zeros((), dtype=torch.float32), torch.finfo(torch.float32).min)
        expected = _reference(indexer, hidden, embeds, float_mask)
        with torch.no_grad():
            actual = qsa_indexer_forward(indexer, hidden, embeds, float_mask, None)
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertTrue(torch.equal(expected, actual))

    def test_bool_mask_stays_bool(self):
        cfg = _config(64)
        indexer, hidden, mask, embeds = _build_inputs(cfg, 1, 64, "cpu", torch.float32)
        with torch.no_grad():
            actual = qsa_indexer_forward(indexer, hidden, embeds, mask, None)
        self.assertEqual(actual.dtype, torch.bool)

    def test_left_padded_mask_matches_upstream(self):
        """Non-prefix rows (left padding) must keep the real visible key positions."""
        cfg = _config(32, ratio=4, budget=16)
        indexer, hidden, _, embeds = _build_inputs(cfg, 2, 32, "cpu", torch.float32)
        attention_mask = torch.ones(2, 32, dtype=torch.long)
        attention_mask[0, :8] = 0
        mask = qwen4_exp.create_causal_mask(
            config=cfg,
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=torch.arange(32)[None, :].expand(2, -1),
            allow_is_causal_skip=False,
        )
        self.assertEqual(int(mask[0, 0, 31].nonzero()[0]), 8)  # row starts after the padding
        expected = _reference(indexer, hidden, embeds, mask)
        with torch.no_grad():
            actual = qsa_indexer_forward(indexer, hidden, embeds, mask, None)
        self.assertTrue(torch.equal(expected, actual))


class TestQSAIndexerPatch(unittest.TestCase):
    """The installed patch must be idempotent and keep the original around."""

    def test_patch_is_idempotent(self):
        from auto_round.modeling.qwen4_exp_qsa import patch_qwen4_exp_indexer

        cls = qwen4_exp.Qwen4ExpTextQSAIndexer
        self.assertTrue(patch_qwen4_exp_indexer())
        patched = cls.forward
        self.assertTrue(patch_qwen4_exp_indexer())
        self.assertIs(patched, cls.forward)
        self.assertTrue(getattr(cls.forward, "_auto_round_vectorized", False))
        self.assertIsNotNone(getattr(cls.forward, "_auto_round_original", None))


if __name__ == "__main__":
    unittest.main()
