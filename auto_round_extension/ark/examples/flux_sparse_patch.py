import contextlib
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

ARK_DIR = Path(__file__).resolve().parent.parent
if str(ARK_DIR) not in sys.path:
    sys.path.insert(0, str(ARK_DIR))

from wan_sparse_patch import ensure_ark_sparse_binding, _parse_bool_env, _parse_optional_int_env

import auto_round_kernel as ark
from diffusers.models.transformers.transformer_flux import FluxAttention, _get_qkv_projections, apply_rotary_emb


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
        raise ValueError(f"Unsupported Flux attention mask rank: {mask.ndim}")

    if mask.shape[0] == 1 and batch != 1:
        mask = mask.expand(batch, -1, -1, -1)
    return mask.contiguous()


@dataclass
class FluxSparseAttentionStats:
    total_calls: int = 0
    single_stream_calls: int = 0
    joint_stream_calls: int = 0
    sparse_calls: int = 0
    unsupported_fallbacks: int = 0
    runtime_fallbacks: int = 0
    sparse_sparsity_sum: float = 0.0
    timed_calls: int = 0
    processor_time_ms_sum: float = 0.0
    processor_time_ms_min: float = float("inf")
    processor_time_ms_max: float = 0.0

    @property
    def avg_sparsity(self) -> float:
        if self.sparse_calls == 0:
            return 0.0
        return self.sparse_sparsity_sum / self.sparse_calls

    @property
    def avg_processor_time_ms(self) -> float:
        if self.timed_calls == 0:
            return 0.0
        return self.processor_time_ms_sum / self.timed_calls


class FluxSparseAttnProcessor:
    def __init__(
        self,
        original_processor,
        stats: FluxSparseAttentionStats,
        *,
        smooth_k: bool,
        topk: float,
        attention_sink: bool,
        debug_timing: bool,
        q_tile_override: int,
        sparse_q_block_tokens: int | None,
        sparse_k_block_tokens: int | None,
    ):
        self.original_processor = original_processor
        self.stats = stats
        self.smooth_k = smooth_k
        self.topk = topk
        self.attention_sink = attention_sink
        self.debug_timing = debug_timing
        self.q_tile_override = q_tile_override
        self.sparse_q_block_tokens = sparse_q_block_tokens
        self.sparse_k_block_tokens = sparse_k_block_tokens
        self._warned_runtime_fallback = False

    @staticmethod
    def _synchronize_for_timing(device: torch.device) -> None:
        if device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        self.stats.total_calls += 1
        if encoder_hidden_states is None:
            self.stats.single_stream_calls += 1
        else:
            self.stats.joint_stream_calls += 1

        call_kind = "single" if encoder_hidden_states is None else "joint"
        call_status = "sparse"
        start_time = None
        if self.debug_timing:
            self._synchronize_for_timing(hidden_states.device)
            start_time = time.perf_counter()

        try:
            if hidden_states.device.type != "xpu" or hidden_states.dtype not in (torch.float16, torch.bfloat16):
                self.stats.unsupported_fallbacks += 1
                call_status = "fallback_unsupported_device_or_dtype"
                return self.original_processor(
                    attn,
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask,
                    image_rotary_emb,
                    **kwargs,
                )

            if encoder_hidden_states is not None and attn.added_kv_proj_dim is None:
                self.stats.unsupported_fallbacks += 1
                call_status = "fallback_unsupported_joint_layout"
                return self.original_processor(
                    attn,
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask,
                    image_rotary_emb,
                    **kwargs,
                )

            query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
                attn,
                hidden_states,
                encoder_hidden_states,
            )

            query = query.unflatten(-1, (attn.heads, -1)).contiguous()
            key = key.unflatten(-1, (attn.heads, -1)).contiguous()
            value = value.unflatten(-1, (attn.heads, -1)).contiguous()

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            encoder_seq_len = 0
            if encoder_hidden_states is not None:
                encoder_seq_len = encoder_hidden_states.shape[1]
                encoder_query = encoder_query.unflatten(-1, (attn.heads, -1)).contiguous()
                encoder_key = encoder_key.unflatten(-1, (attn.heads, -1)).contiguous()
                encoder_value = encoder_value.unflatten(-1, (attn.heads, -1)).contiguous()

                encoder_query = attn.norm_added_q(encoder_query)
                encoder_key = attn.norm_added_k(encoder_key)

                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)

            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            mask = _normalize_attention_mask(
                attention_mask,
                batch=query.shape[0],
                seq_q=query.shape[1],
                seq_kv=key.shape[1],
                device=query.device,
            )
            hidden_states, sparsity = ark.sparge_sage2_attn_meansim_topk_xpu(
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
            self.stats.sparse_calls += 1
            self.stats.sparse_sparsity_sum += float(sparsity)

            hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

            if encoder_hidden_states is not None:
                encoder_hidden_states_out, hidden_states_out = hidden_states.split_with_sizes(
                    [encoder_seq_len, hidden_states.shape[1] - encoder_seq_len],
                    dim=1,
                )
                hidden_states_out = attn.to_out[0](hidden_states_out.contiguous())
                hidden_states_out = attn.to_out[1](hidden_states_out)
                encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out.contiguous())
                return hidden_states_out, encoder_hidden_states_out

            return hidden_states
        except Exception as exc:  # noqa: BLE001
            self.stats.runtime_fallbacks += 1
            call_status = "fallback_runtime"
            if not self._warned_runtime_fallback:
                warnings.warn(
                    f"Flux sparse attention fell back to the original processor after a runtime error: {exc}",
                    stacklevel=2,
                )
                self._warned_runtime_fallback = True
            return self.original_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
                **kwargs,
            )
        finally:
            if self.debug_timing and start_time is not None:
                self._synchronize_for_timing(hidden_states.device)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                self.stats.timed_calls += 1
                self.stats.processor_time_ms_sum += elapsed_ms
                self.stats.processor_time_ms_min = min(self.stats.processor_time_ms_min, elapsed_ms)
                self.stats.processor_time_ms_max = max(self.stats.processor_time_ms_max, elapsed_ms)
                print(
                    "[flux_sparse][timing]"
                    f" call={self.stats.timed_calls}"
                    f" kind={call_kind}"
                    f" status={call_status}"
                    f" hidden_shape={tuple(hidden_states.shape)}"
                    f" elapsed_ms={elapsed_ms:.3f}"
                )


@contextlib.contextmanager
def patch_flux_sparse_attention(
    transformer,
    *,
    smooth_k: bool = True,
    topk: float = 0.75,
    attention_sink: bool = False,
    debug_timing: bool = False,
    q_tile_override: int = 0,
    sparse_q_block_tokens: int | None = None,
    sparse_k_block_tokens: int | None = None,
):
    ensure_ark_sparse_binding()
    originals: list[tuple[FluxAttention, object]] = []
    stats = FluxSparseAttentionStats()

    for module in transformer.modules():
        if isinstance(module, FluxAttention):
            original = module.processor
            module.set_processor(
                FluxSparseAttnProcessor(
                    original,
                    stats,
                    smooth_k=smooth_k,
                    topk=topk,
                    attention_sink=attention_sink,
                    debug_timing=debug_timing,
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


def patch_flux_sparse_attention_from_env(transformer):
    return patch_flux_sparse_attention(
        transformer,
        smooth_k=_parse_bool_env("FLUX_SPARSE_SMOOTH_K", True),
        topk=float(os.getenv("FLUX_SPARSE_TOPK", "0.75")),
        attention_sink=_parse_bool_env("FLUX_SPARSE_ATTENTION_SINK", False),
        debug_timing=_parse_bool_env("FLUX_SPARSE_DEBUG_TIMING", False),
        q_tile_override=int(os.getenv("FLUX_SPARSE_Q_TILE_OVERRIDE", "0")),
        sparse_q_block_tokens=_parse_optional_int_env("FLUX_SPARSE_Q_BLOCK_TOKENS"),
        sparse_k_block_tokens=_parse_optional_int_env("FLUX_SPARSE_K_BLOCK_TOKENS"),
    )
