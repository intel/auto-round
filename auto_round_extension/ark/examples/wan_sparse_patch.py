# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

EXAMPLES_DIR = Path(__file__).resolve().parent
ARK_DIR = EXAMPLES_DIR.parent
KERNEL_DIR = ARK_DIR / "auto_round_kernel"
if str(ARK_DIR) not in sys.path:
    sys.path.insert(0, str(ARK_DIR))

import auto_round_kernel as ark
from auto_round_kernel.xpu_loader import ensure_xpu_lib
from diffusers.models.transformers.transformer_wan import WanAttention, _get_qkv_projections


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _parse_optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    parsed = int(value)
    return None if parsed == 0 else parsed


def ensure_ark_sparse_binding() -> None:
    ensure_xpu_lib(
        required_symbols=("sage_sparse", "sage_dynamic_quant_layout"),
        search_roots=(KERNEL_DIR, KERNEL_DIR / "xbuild", KERNEL_DIR / "xbuild_diffuser"),
    )


def _apply_rotary_emb(hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def _normalize_attention_mask(
    attn_mask: torch.Tensor | None,
    batch: int,
    seq_q: int,
    seq_kv: int,
    device: torch.device,
) -> torch.Tensor | None:
    if attn_mask is None:
        return None

    mask = attn_mask
    if mask.dtype == torch.bool:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        neg_inf = torch.full((), -1.0e9, dtype=torch.float32, device=device)
        mask = torch.where(mask, zero, neg_inf)
    else:
        mask = mask.to(torch.float32)

    if mask.ndim == 2:
        mask = mask.view(1, 1, seq_q, seq_kv)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError(f"Unsupported Wan attention mask rank: {mask.ndim}")

    if mask.shape[0] == 1 and batch != 1:
        mask = mask.expand(batch, -1, -1, -1)
    return mask.contiguous()


@dataclass
class WanSparseAttentionStats:
    self_attn_total: int = 0
    cross_attn_total: int = 0
    sparse_self_attn_calls: int = 0
    sparse_cross_attn_calls: int = 0
    cross_attn_fallbacks: int = 0
    cross_attn_policy_fallbacks: int = 0
    cross_attn_unsupported_fallbacks: int = 0
    cross_attn_runtime_fallbacks: int = 0
    unsupported_fallbacks: int = 0
    runtime_fallbacks: int = 0
    sparse_sparsity_sum: float = 0.0

    @property
    def avg_sparsity(self) -> float:
        sparse_calls = self.sparse_self_attn_calls + self.sparse_cross_attn_calls
        if sparse_calls == 0:
            return 0.0
        return self.sparse_sparsity_sum / sparse_calls


class WanSparseAttnProcessor:
    def __init__(
        self,
        original_processor,
        stats: WanSparseAttentionStats,
        *,
        smooth_k: bool,
        topk: float,
        attention_sink: bool,
        enable_cross_attention: bool,
        q_tile_override: int,
        sparse_q_block_tokens: int | None,
        sparse_k_block_tokens: int | None,
    ):
        self.original_processor = original_processor
        self.stats = stats
        self.smooth_k = smooth_k
        self.topk = topk
        self.attention_sink = attention_sink
        self.enable_cross_attention = enable_cross_attention
        self.q_tile_override = q_tile_override
        self.sparse_q_block_tokens = sparse_q_block_tokens
        self.sparse_k_block_tokens = sparse_k_block_tokens
        self._warned_runtime_fallback = False

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if attn.is_cross_attention:
            self.stats.cross_attn_total += 1
            if not self.enable_cross_attention:
                self.stats.cross_attn_fallbacks += 1
                self.stats.cross_attn_policy_fallbacks += 1
                return self.original_processor(
                    attn,
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask,
                    rotary_emb,
                    **kwargs,
                )
        else:
            self.stats.self_attn_total += 1

        if encoder_hidden_states is not None and not attn.is_cross_attention:
            self.stats.unsupported_fallbacks += 1
            return self.original_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                rotary_emb,
                **kwargs,
            )

        if hidden_states.device.type != "xpu" or hidden_states.dtype not in (torch.float16, torch.bfloat16):
            if attn.is_cross_attention:
                self.stats.cross_attn_fallbacks += 1
                self.stats.cross_attn_unsupported_fallbacks += 1
            self.stats.unsupported_fallbacks += 1
            return self.original_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                rotary_emb,
                **kwargs,
            )

        try:
            query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            query = query.unflatten(2, (attn.heads, -1)).contiguous()
            key = key.unflatten(2, (attn.heads, -1)).contiguous()
            value = value.unflatten(2, (attn.heads, -1)).contiguous()

            if rotary_emb is not None:
                query = _apply_rotary_emb(query, *rotary_emb)
                key = _apply_rotary_emb(key, *rotary_emb)

            mask = _normalize_attention_mask(
                attention_mask,
                batch=query.shape[0],
                seq_q=query.shape[1],
                seq_kv=key.shape[1],
                device=query.device,
            )

            attn_out, sparsity = ark.sparge_sage2_attn_meansim_topk_xpu(
                query,
                key,
                value,
                attn_mask=mask,
                is_causal=False,
                smooth_k=self.smooth_k,
                simthreshd1=-1.0,
                topk=self.topk,
                attention_sink=self.attention_sink,
                tensor_layout="NHD",
                q_tile_override=self.q_tile_override,
                sparse_q_block_tokens=self.sparse_q_block_tokens,
                sparse_k_block_tokens=self.sparse_k_block_tokens,
                return_sparsity=True,
            )
            if attn.is_cross_attention:
                self.stats.sparse_cross_attn_calls += 1
            else:
                self.stats.sparse_self_attn_calls += 1
            self.stats.sparse_sparsity_sum += float(sparsity)
            attn_out = attn_out.flatten(2, 3).type_as(query)
            attn_out = attn.to_out[0](attn_out)
            attn_out = attn.to_out[1](attn_out)
            return attn_out
        except Exception as exc:  # noqa: BLE001
            if attn.is_cross_attention:
                self.stats.cross_attn_fallbacks += 1
                self.stats.cross_attn_runtime_fallbacks += 1
            self.stats.runtime_fallbacks += 1
            if not self._warned_runtime_fallback:
                warnings.warn(
                    f"Wan sparse attention fell back to the original processor after a runtime error: {exc}",
                    stacklevel=2,
                )
                self._warned_runtime_fallback = True
            return self.original_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                rotary_emb,
                **kwargs,
            )


@contextlib.contextmanager
def patch_wan_sparse_attention(
    transformer,
    *,
    smooth_k: bool = True,
    topk: float = 0.75,
    attention_sink: bool = False,
    enable_cross_attention: bool = False,
    q_tile_override: int = 0,
    sparse_q_block_tokens: int | None = None,
    sparse_k_block_tokens: int | None = None,
):
    ensure_ark_sparse_binding()
    originals: list[tuple[WanAttention, object]] = []
    stats = WanSparseAttentionStats()

    for module in transformer.modules():
        if isinstance(module, WanAttention):
            original = module.processor
            module.set_processor(
                WanSparseAttnProcessor(
                    original,
                    stats,
                    smooth_k=smooth_k,
                    topk=topk,
                    attention_sink=attention_sink,
                    enable_cross_attention=enable_cross_attention,
                    q_tile_override=q_tile_override,
                    sparse_q_block_tokens=sparse_q_block_tokens,
                    sparse_k_block_tokens=sparse_k_block_tokens,
                )
            )
            originals.append((module, original))

    try:
        yield stats
    finally:
        for module, original in originals:
            module.set_processor(original)


def patch_wan_sparse_attention_from_env(transformer):
    return patch_wan_sparse_attention(
        transformer,
        smooth_k=_parse_bool_env("WAN_SPARSE_SMOOTH_K", True),
        topk=float(os.getenv("WAN_SPARSE_TOPK", "0.75")),
        attention_sink=_parse_bool_env("WAN_SPARSE_ATTENTION_SINK", False),
        enable_cross_attention=_parse_bool_env("WAN_SPARSE_ENABLE_CROSS_ATTN", False),
        q_tile_override=int(os.getenv("WAN_SPARSE_Q_TILE_OVERRIDE", "0")),
        sparse_q_block_tokens=_parse_optional_int_env("WAN_SPARSE_Q_BLOCK_TOKENS"),
        sparse_k_block_tokens=_parse_optional_int_env("WAN_SPARSE_K_BLOCK_TOKENS"),
    )
