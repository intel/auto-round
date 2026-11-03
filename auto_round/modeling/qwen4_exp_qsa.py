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
"""Vectorized QSA indexer for Qwen4-Exp (Qwen3.8-Flash-Next).

``transformers``' ``Qwen4ExpTextQSAIndexer.forward`` decides which compressed key
blocks a query may attend to with a Python double loop::

    for batch_idx in range(batch_size):
        for query_idx in range(seq_length):
            ...

Every inner step issues ``torch.nonzero`` / ``index_select`` / ``mean`` / ``topk``
on per-token slices, so one full-attention layer fires thousands of tiny kernels.
Measured on a toy Qwen4Exp layer at 2x1024 tokens the indexer alone issued 2042
``aten::topk`` and 7752 ``aten::gather`` calls and burned ~1.5 s of CPU time
against ~0.2 s of GPU time -- it is launch-bound, not compute-bound. During
quantization the indexer is re-entered for every full-attention block on every
tuning iteration, which is why those blocks dominate wall clock
(intel/auto-round#2323).

This module rebuilds the same selection with batched tensor ops. The key
observation is that the calibration mask is a causal prefix: for query ``q`` the
visible key set is ``[0, len_q)``, therefore

* the number of complete compression blocks is ``len_q // compress_ratio``;
* those blocks are always the *first* ones, so ``block_token_indices`` is just
  ``arange(n_blocks * compress_ratio)`` and is **independent of the query** -- the
  pooled block keys can be computed once per batch instead of once per token;
* only the trailing incomplete block has to be handled per query.

That removes both loops. The produced mask is numerically equivalent to the
reference implementation; see ``test/unit/test_cpu/test_qwen4_exp_qsa.py``.
"""

from __future__ import annotations

import math
from functools import wraps

import torch

_PAD_SENTINEL = -1
"""Padding value the reference implementation stores in unused budget slots."""

# Cap on the ``(batch, seq, heads, block_chunk)`` score buffer used while merging
# top-k results chunk by chunk.
_SCORE_ELEMENT_BUDGET = 1 << 25  # ~32M float32 elements ≈ 128 MiB


def _visible_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Normalize a 4D attention mask to a boolean "key is visible" mask."""
    if attention_mask.dtype == torch.bool:
        return attention_mask
    return attention_mask == 0


def _visible_lengths(visible: torch.Tensor) -> torch.Tensor:
    """Length of the visible *prefix* for every ``(batch, query)`` pair.

    ``visible`` has shape ``(batch, 1, seq_len, kv_len)``. A causal mask is a
    prefix of ones for left-aligned inputs, so the visible length is exactly
    "last visible index + 1". Returns ``(batch, seq_len)``.
    """
    return visible.sum(dim=-1, dtype=torch.int64).squeeze(1)


def _chunk_blocks(n_blocks: int, batch_size: int, seq_len: int, n_heads: int) -> int:
    """Block chunk size that keeps the score buffer near the memory budget."""
    per_block = max(1, batch_size * seq_len * n_heads)
    return max(1, min(n_blocks, _SCORE_ELEMENT_BUDGET // per_block))


def _block_scores(q: torch.Tensor, block_key_states: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """ReLU-summed head scores for blocks ``[start, stop)``.

    Mirrors ``relu(q @ k.T).sum(-1) / sqrt(head_dim)`` from the reference impl.
    """
    keys = block_key_states[:, start:stop, :]
    # (B,S,H,dim) x (B,n,dim) -> (B,S,H,n)
    scores = torch.einsum("bshd,bnd->bshn", q.float(), keys.float())
    scores = torch.relu(scores).sum(dim=-2)
    return scores / math.sqrt(q.shape[-1])


def _topk_blocks(
    q: torch.Tensor,
    block_key_states: torch.Tensor,
    valid_counts: torch.Tensor,
    block_topk: int,
) -> torch.Tensor:
    """Top-``block_topk`` block ids per query, ordered by descending score."""
    n_blocks = block_key_states.shape[1]
    k = min(block_topk, n_blocks)
    chunk = _chunk_blocks(n_blocks, q.shape[0], q.shape[1], q.shape[2])
    block_ids = torch.arange(n_blocks, device=q.device)

    best_vals = best_idx = None
    for start in range(0, n_blocks, chunk):
        stop = min(start + chunk, n_blocks)
        scores = _block_scores(q, block_key_states, start, stop)
        inside = block_ids[start:stop].view(1, 1, -1) < valid_counts.unsqueeze(-1)
        scores = scores.masked_fill(~inside, float("-inf"))
        vals, idx = scores.topk(min(k, stop - start), dim=-1)
        if best_vals is None:
            best_vals, best_idx = vals, idx + start
        else:
            best_vals = torch.cat([best_vals, vals], dim=-1)
            best_idx = torch.cat([best_idx, idx + start], dim=-1)
            merge_k = min(k, best_vals.shape[-1])
            best_vals, order = best_vals.topk(merge_k, dim=-1)
            best_idx = best_idx.gather(-1, order)
    return best_idx


def _to_attention_mask(selected: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Turn per-query index rows into the final block mask."""
    kv_length = attention_mask.shape[-1]
    # The reference over-allocates one trailing column to absorb the -1 padding
    # sentinels and slices it away; keep the same trick so the mask is identical.
    full = torch.full_like(selected, kv_length, dtype=torch.long)
    scatter_indices = torch.where(selected >= 0, selected.long(), full)
    mask = torch.zeros((*selected.shape[:-1], kv_length + 1), device=attention_mask.device, dtype=torch.bool)
    mask = mask.scatter(-1, scatter_indices, True)[..., :kv_length].unsqueeze(1)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        mask = torch.where(mask, attention_mask.new_zeros(()), min_dtype)
    return mask


@torch.no_grad()
def qsa_indexer_forward(
    indexer: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values=None,
) -> torch.Tensor:
    """Drop-in replacement for ``Qwen4ExpTextQSAIndexer.forward``.

    The returned mask is used as an additive/broadcast attention mask and never
    needs gradients: the reference implementation builds it by scattering booleans
    into a fresh zeros tensor, so it already detaches (``grad_fn is None``). Running
    under ``no_grad`` therefore keeps results identical while dropping the useless
    autograd bookkeeping for the q/k projections, which matters during tuning when
    the full-attention blocks are the memory high-water mark.
    """
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    batch_size, seq_length, _ = hidden_states.shape
    head_dim = indexer.index_head_dim
    compress_ratio = indexer.compress_ratio
    block_topk = indexer.block_topk
    token_budget = indexer.token_budget
    device = hidden_states.device

    full_cos, full_sin = position_embeddings
    current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]

    qk = indexer.index_qk_proj(hidden_states)
    q, token_k = torch.split(qk, [indexer.index_n_heads * head_dim, indexer.index_kv_heads * head_dim], dim=-1)
    hidden_shape = (batch_size, seq_length, -1, head_dim)
    q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
    q = indexer.q_layernorm(q)
    q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, indexer.layer_idx)

    visible = _visible_mask(attention_mask)
    kv_length = attention_mask.shape[-1]
    lengths = _visible_lengths(visible)  # (B, S)
    n_complete = torch.div(lengths, compress_ratio, rounding_mode="floor")  # (B, S)

    width = token_budget + compress_ratio - 1
    selected = torch.full((batch_size, seq_length, width), _PAD_SENTINEL, dtype=torch.int32, device=device)

    # Blocks that can be complete for the longest query. Depends only on
    # (kv_length, compress_ratio), so the graph shape stays stable.
    n_blocks = max(0, kv_length // compress_ratio)

    if n_blocks > 0:
        used_keys = n_blocks * compress_ratio
        # Causal prefix => complete blocks always start at key 0, so the pooled
        # block keys are query-independent and computed once per batch.
        pooled = raw_keys[:, :used_keys].view(batch_size, n_blocks, compress_ratio, head_dim).float().mean(dim=2)
        pooled = indexer.k_layernorm(pooled.to(raw_keys.dtype))
        group_starts = torch.arange(0, used_keys, compress_ratio, device=device)
        block_key_states = apply_rotary_pos_emb(
            pooled.unsqueeze(2),
            cos=full_cos[:, group_starts, :],
            sin=full_sin[:, group_starts, :],
            unsqueeze_dim=2,
        ).squeeze(2)

        topk_idx = _topk_blocks(q, block_key_states, n_complete, block_topk)  # (B,S,k)
        k_sel = topk_idx.shape[-1]

        # Scores are non-increasing along the block id for a causal mask, so the
        # reference's "sort valid blocks first, drop -inf" compaction is exactly
        # "keep the leading ``valid`` entries".
        offsets = torch.arange(compress_ratio, device=device)
        block_tokens = (topk_idx * compress_ratio).unsqueeze(-1) + offsets.view(1, 1, 1, compress_ratio)
        block_tokens = block_tokens.reshape(batch_size, seq_length, k_sel * compress_ratio)
        block_valid = (topk_idx < n_complete.unsqueeze(-1)).unsqueeze(-1).expand(-1, -1, -1, compress_ratio)
        block_valid = block_valid.reshape(batch_size, seq_length, k_sel * compress_ratio)

        n_budget = max(0, width - (compress_ratio - 1))
        keep = min(n_budget, block_tokens.shape[-1])
        if keep > 0:
            slot = selected[:, :, :keep]
            selected[:, :, :keep] = torch.where(
                block_valid[:, :, :keep], block_tokens[:, :, :keep].to(torch.int32), slot
            )
            count = block_valid[:, :, :keep].sum(dim=-1, dtype=torch.int64)
        else:
            count = torch.zeros_like(n_complete)
    else:
        count = torch.zeros_like(n_complete)

    # Append the trailing incomplete block: keys [n_complete * R, length).
    tail = compress_ratio - 1
    if tail > 0:
        offsets = torch.arange(tail, device=device).view(1, 1, -1)
        positions = (count.unsqueeze(-1) + offsets).clamp(max=width - 1).long()
        tail_keys = n_complete.unsqueeze(-1) * compress_ratio + offsets
        tail_valid = (tail_keys < lengths.unsqueeze(-1)) & (count.unsqueeze(-1) + offsets < width)
        previous = selected.gather(-1, positions).long()
        selected.scatter_(-1, positions, torch.where(tail_valid, tail_keys, previous).to(torch.int32))

    return _to_attention_mask(selected, attention_mask)


def patch_qwen4_exp_indexer() -> bool:
    """Install the vectorized indexer on the ``transformers`` class.

    Set ``AR_DISABLE_QWEN4_QSA_VECTORIZE=1`` to keep the upstream per-token loop
    (useful for A/B measurements or as an escape hatch).

    Returns ``True`` when the patch is active.
    """
    import os

    if os.getenv("AR_DISABLE_QWEN4_QSA_VECTORIZE", "0").lower() in ("1", "true", "yes"):
        return False

    try:
        from transformers.models.qwen4_exp import modeling_qwen4_exp
    except ImportError:
        return False

    cls = getattr(modeling_qwen4_exp, "Qwen4ExpTextQSAIndexer", None)
    if cls is None:
        return False
    if getattr(cls.forward, "_auto_round_vectorized", False):
        return True

    original = cls.forward

    @wraps(original)
    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        return qsa_indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)

    forward._auto_round_vectorized = True
    forward._auto_round_original = original
    cls.forward = forward
    return True
