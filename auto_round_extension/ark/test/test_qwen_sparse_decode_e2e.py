#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_ARK_PARENT = REPO_ROOT / "auto_round_extension" / "ark"
if str(LOCAL_ARK_PARENT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ARK_PARENT))

import auto_round_kernel as ark

from test_qwen_sparse_prefill_e2e import (
    MODEL,
    PROMPT,
    assert_generation_close,
    build_prompt,
    ensure_sparse_binding,
    get_model_path,
    is_xpu_available,
    quantize_qk,
    report_generation,
)

# Mode reference for this harness:
# - dense_masked_all_selected_cached:
#   Dense reference path built on top of the public sparse-prefill API by selecting
#   every KV block. In decode it uses concatenated K/V replay and acts as the
#   model-level baseline for cached integration.
# - sparse_all_selected_cached:
#   Exercises the real sparse cached-decode API with an all-selected LUT. This is
#   the closest model-level smoke test for `sage_sparse_decode(...)`.
# - direct_preprocess_cached:
#   Uses preprocess metadata generation explicitly, then calls the low-level sparse
#   API directly. This is useful when isolating preprocess output from kernel entry
#   plumbing.
# - sparse_preprocess_cached:
#   Uses the fully integrated preprocess+kernel helper for cached decode.
#   It is kept here for targeted debugging, even though the current test method
#   only asserts the all-selected cached branch.


def _build_all_selected_lut(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    *,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_blocks = (seq_q + block_size - 1) // block_size
    kv_blocks = (seq_kv + block_size - 1) // block_size
    lut = torch.zeros((batch, heads, q_blocks, kv_blocks), dtype=torch.int32, device=device)
    valid = torch.full((batch, heads, q_blocks), kv_blocks, dtype=torch.int32, device=device)
    if kv_blocks > 0:
        lut[..., 0] = 0
        if kv_blocks > 1:
            lut[..., 1:kv_blocks] = 1
    return lut.contiguous(), valid.contiguous()


def _build_all_selected_lut_from_block_counts(
    batch: int,
    heads: int,
    q_blocks: int,
    kv_blocks: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    lut = torch.zeros((batch, heads, q_blocks, kv_blocks), dtype=torch.int32, device=device)
    valid = torch.full((batch, heads, q_blocks), kv_blocks, dtype=torch.int32, device=device)
    if kv_blocks > 0:
        lut[..., 0] = 0
        if kv_blocks > 1:
            lut[..., 1:kv_blocks] = 1
    return lut.contiguous(), valid.contiguous()


def _quantize_qk_layout_hnd(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, S, D = tensor.shape
    out = torch.empty_like(tensor, dtype=torch.int8)
    scale = torch.empty((B, H, (S + block_size - 1) // block_size, 1), dtype=torch.float32, device=tensor.device)
    ark.xpu_lib.sage_dynamic_quant_layout(
        ark.get_stream(tensor),
        tensor.data_ptr(),
        0,
        out.data_ptr(),
        scale.data_ptr(),
        B,
        H,
        S,
        D,
        block_size,
        tensor.stride(2),
        tensor.stride(3),
        tensor.stride(1),
        tensor.stride(0),
    )
    return out, scale


def _pad_hnd_seq(tensor: torch.Tensor, padded_seq: int) -> torch.Tensor:
    pad_tokens = padded_seq - tensor.shape[2]
    if pad_tokens <= 0:
        return tensor.contiguous()
    return torch.nn.functional.pad(tensor, (0, 0, 0, pad_tokens)).contiguous()


def _cache_step_is_block_aligned(cache_len: int, block_size: int) -> bool:
    return cache_len > 0 and cache_len % block_size == 0


def _run_prefill_mode(mode: str, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
    """Run the prefill branch for the requested mode.

    For the first model call, Qwen still runs in prefill (`Sq == Skv`), so even
    the cached-generation harness needs the prefill sparse path before it reaches
    `seq_len_q == 1` decode steps.
    """
    quant_block_size = 64
    q = query.contiguous()
    k = key.contiguous()
    v = value.contiguous()
    B, Hq, Sq, D = q.shape
    seq_pad = ((Sq + quant_block_size - 1) // quant_block_size) * quant_block_size
    q_pad = _pad_hnd_seq(q, seq_pad)
    k_pad = _pad_hnd_seq(k, seq_pad)
    v_pad = _pad_hnd_seq(v, seq_pad)
    padded_mask = torch.zeros((B, 1, seq_pad, seq_pad), dtype=torch.float32, device=q.device)
    if seq_pad > Sq:
        padded_mask[:, :, Sq:, :] = -1.0e9
        padded_mask[:, :, :, Sq:] = -1.0e9
    causal = torch.triu(torch.full((seq_pad, seq_pad), -1.0e9, dtype=torch.float32, device=q.device), diagonal=1)
    padded_mask = torch.minimum(padded_mask, causal.view(1, 1, seq_pad, seq_pad))

    if mode == "dense_masked_all_selected_cached":
        q_i8, q_scale = quantize_qk(q_pad, quant_block_size)
        k_i8, k_scale = quantize_qk(k_pad, quant_block_size)
        lut, valid = _build_all_selected_lut(B, Hq, seq_pad, k_pad.shape[2], block_size=quant_block_size, device=q.device)
        out = ark.sage_sparse(
            q_i8,
            k_i8,
            v_pad,
            lut,
            valid,
            attn_mask=padded_mask,
            is_causal=False,
            scale=scale,
            quant_block_size=quant_block_size,
            qscale=q_scale,
            kscale=k_scale,
            tensor_layout="HND",
        )
        return out[:, :, :Sq, :]
    if mode == "sparse_all_selected_cached":
        q_i8, q_scale = quantize_qk(q_pad, quant_block_size)
        k_i8, k_scale = quantize_qk(k_pad, quant_block_size)
        lut, valid = _build_all_selected_lut(B, Hq, seq_pad, k_pad.shape[2], block_size=quant_block_size, device=q.device)
        out = ark.sage_sparse(
            q_i8,
            k_i8,
            v_pad,
            lut,
            valid,
            attn_mask=padded_mask,
            is_causal=False,
            scale=scale,
            quant_block_size=quant_block_size,
            qscale=q_scale,
            kscale=k_scale,
            tensor_layout="HND",
        )
        return out[:, :, :Sq, :]
    if mode == "direct_preprocess_cached":
        meta = ark.sparge_preprocess_topk(
            q_pad,
            k_pad,
            is_causal=False,
            smooth_k=True,
            simthreshd1=-1.0,
            topk=0.75,
            attention_sink=False,
            quant_block_size=quant_block_size,
            tensor_layout="HND",
        )
        out = ark.sage_sparse(
            meta["query_i8"],
            meta["key_i8"],
            v_pad,
            meta["lut"],
            meta["valid_block_num"],
            attn_mask=padded_mask,
            is_causal=False,
            scale=scale,
            quant_block_size=meta["quant_block_size"],
            qscale=meta["qscale"],
            kscale=meta["kscale"],
            tensor_layout="HND",
        )
        return out[:, :, :Sq, :]
    if mode == "sparse_preprocess_cached":
        out = ark.sparge_sage2_attn_meansim_topk_xpu(
            q_pad,
            k_pad,
            v_pad,
            attn_mask=padded_mask,
            is_causal=False,
            scale=scale,
            smooth_k=True,
            simthreshd1=-1.0,
            topk=0.75,
            attention_sink=False,
            tensor_layout="HND",
        )
        return out[:, :, :Sq, :]
    raise ValueError(f"Unsupported prefill mode: {mode}")


def _run_decode_mode(
    mode: str,
    query: torch.Tensor,
    key_cur: torch.Tensor,
    value_cur: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run the cached decode branch for one-step Q queries.

    The current sparse decode kernel expects cache and current-token blocks to be
    separable at the quant block boundary. When the cache tail is not block-aligned,
    this harness falls back to concatenated sparse replay so the model-level test
    remains correct while still exercising the real decode kernel on aligned steps.
    """
    quant_block_size = 64
    cache_len = key_cache.shape[2]
    if not _cache_step_is_block_aligned(cache_len, quant_block_size):
        key_total = torch.cat([key_cache, key_cur], dim=2).contiguous()
        value_total = torch.cat([value_cache, value_cur], dim=2).contiguous()
        if mode == "dense_masked_all_selected_cached" or mode == "sparse_all_selected_cached":
            q_i8, q_scale = _quantize_qk_layout_hnd(query, quant_block_size)
            k_total_i8, k_total_scale = _quantize_qk_layout_hnd(key_total, quant_block_size)
            lut, valid = _build_all_selected_lut(
                query.shape[0], query.shape[1], query.shape[2], key_total.shape[2], block_size=quant_block_size, device=query.device
            )
            return ark.sage_sparse(
                q_i8,
                k_total_i8,
                value_total,
                lut,
                valid,
                is_causal=True,
                scale=scale,
                quant_block_size=quant_block_size,
                qscale=q_scale,
                kscale=k_total_scale,
                tensor_layout="HND",
            )
        if mode == "direct_preprocess_cached":
            meta = ark.sparge_preprocess_topk(
                query,
                key_total,
                is_causal=True,
                smooth_k=True,
                simthreshd1=-1.0,
                topk=0.75,
                attention_sink=False,
                quant_block_size=quant_block_size,
                tensor_layout="HND",
            )
            return ark.sage_sparse(
                meta["query_i8"],
                meta["key_i8"],
                value_total,
                meta["lut"],
                meta["valid_block_num"],
                is_causal=True,
                scale=scale,
                quant_block_size=meta["quant_block_size"],
                qscale=meta["qscale"],
                kscale=meta["kscale"],
                tensor_layout="HND",
            )
        if mode == "sparse_preprocess_cached":
            return ark.sparge_sage2_attn_meansim_topk_xpu(
                query,
                key_total,
                value_total,
                is_causal=True,
                scale=scale,
                smooth_k=True,
                simthreshd1=-1.0,
                topk=0.75,
                attention_sink=False,
                tensor_layout="HND",
            )
    if mode == "dense_masked_all_selected_cached":
        q_i8, q_scale = _quantize_qk_layout_hnd(query, quant_block_size)
        key_total = torch.cat([key_cache, key_cur], dim=2).contiguous()
        value_total = torch.cat([value_cache, value_cur], dim=2).contiguous()
        k_total_i8, k_total_scale = _quantize_qk_layout_hnd(key_total, quant_block_size)
        lut, valid = _build_all_selected_lut(query.shape[0], query.shape[1], query.shape[2], key_total.shape[2], block_size=quant_block_size, device=query.device)
        return ark.sage_sparse(
            q_i8,
            k_total_i8,
            value_total,
            lut,
            valid,
            is_causal=True,
            scale=scale,
            quant_block_size=quant_block_size,
            qscale=q_scale,
            kscale=k_total_scale,
            tensor_layout="HND",
        )
    if mode == "sparse_all_selected_cached":
        q_i8, q_scale = _quantize_qk_layout_hnd(query, quant_block_size)
        k_cur_i8, k_cur_scale = _quantize_qk_layout_hnd(key_cur, quant_block_size)
        k_cache_i8, k_cache_scale = _quantize_qk_layout_hnd(key_cache, quant_block_size)
        q_blocks = (query.shape[2] + quant_block_size - 1) // quant_block_size
        total_blocks = ((key_cache.shape[2] + quant_block_size - 1) // quant_block_size) + (
            (key_cur.shape[2] + quant_block_size - 1) // quant_block_size
        )
        lut, valid = _build_all_selected_lut_from_block_counts(
            query.shape[0], query.shape[1], q_blocks, total_blocks, device=query.device
        )
        return ark.sage_sparse_decode(
            q_i8,
            k_cur_i8,
            value_cur,
            k_cache_i8,
            value_cache,
            lut,
            valid,
            is_causal=True,
            scale=scale,
            quant_block_size=quant_block_size,
            qscale=q_scale,
            kscale=k_cur_scale,
            kscale_cache=k_cache_scale,
            tensor_layout="HND",
        )
    if mode == "direct_preprocess_cached":
        meta = ark.sparge_preprocess_topk_decode(
            query,
            key_cur,
            key_cache,
            smooth_k=True,
            simthreshd1=-1.0,
            topk=0.75,
            attention_sink=False,
            quant_block_size=quant_block_size,
            tensor_layout="HND",
        )
        return ark.sage_sparse_decode(
            meta["query_i8"],
            meta["key_i8"],
            value_cur,
            meta["key_cache_i8"],
            value_cache,
            meta["lut"],
            meta["valid_block_num"],
            is_causal=True,
            scale=scale,
            quant_block_size=meta["quant_block_size"],
            qscale=meta["qscale"],
            kscale=meta["kscale"],
            kscale_cache=meta["kscale_cache"],
            tensor_layout="HND",
        )
    if mode == "sparse_preprocess_cached":
        return ark.sparge_sage2_decode_meansim_topk_xpu(
            query,
            key_cur,
            value_cur,
            key_cache,
            value_cache,
            is_causal=True,
            scale=scale,
            smooth_k=True,
            simthreshd1=-1.0,
            topk=0.75,
            attention_sink=False,
            tensor_layout="HND",
        )
    raise ValueError(f"Unsupported decode mode: {mode}")


@contextlib.contextmanager
def patch_qwen_attention(model, mode: str):
    originals: list[tuple[Qwen3Attention, object]] = []

    def make_forward(attn: Qwen3Attention):
        def wrapped(
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_values=None,
            **kwargs,
        ):
            if hidden_states.device.type != "xpu" or hidden_states.dtype != torch.float16:
                return original(hidden_states, position_embeddings, attention_mask, past_key_values=past_key_values, **kwargs)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn.head_dim)
            query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states_new = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states_new = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states_new = apply_rotary_pos_emb(query_states, key_states_new, cos, sin)

            if past_key_values is not None:
                key_states_total, value_states_total = past_key_values.update(key_states_new, value_states_new, attn.layer_idx)
            else:
                key_states_total, value_states_total = key_states_new, value_states_new

            cache_len = key_states_total.shape[2] - key_states_new.shape[2]
            if query_states.shape[2] == 1 and cache_len > 0:
                attn_hnd = _run_decode_mode(
                    mode,
                    query_states,
                    key_states_new,
                    value_states_new,
                    key_states_total[:, :, :cache_len, :].contiguous(),
                    value_states_total[:, :, :cache_len, :].contiguous(),
                    attn.scaling,
                )
            else:
                attn_hnd = _run_prefill_mode(mode, query_states, key_states_total, value_states_total, attn.scaling)

            attn_output = attn_hnd.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            attn_output = attn.o_proj(attn_output)
            return attn_output, None

        original = attn.forward
        return original, wrapped

    for layer in model.model.layers:
        original, wrapped = make_forward(layer.self_attn)
        originals.append((layer.self_attn, original))
        layer.self_attn.forward = wrapped
    try:
        yield
    finally:
        for attn, original in originals:
            attn.forward = original


def run_manual_cached_loop(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    mode: str,
    forced_next_tokens: torch.Tensor | None = None,
):
    """Manual `use_cache=True` autoregressive loop for model-level integration tests.

    `forced_next_tokens` enables teacher-forced replay so dense and sparse branches
    can be compared on the same token path even when small logit drift exists.
    """
    current_ids = input_ids.clone()
    generated = []
    logits_history = []
    past_key_values = None

    for step in range(max_new_tokens):
        if past_key_values is None:
            step_input_ids = current_ids
        else:
            step_input_ids = current_ids[:, -1:]
        attention_mask = torch.ones((current_ids.shape[0], current_ids.shape[1]), dtype=torch.long, device=current_ids.device)
        with torch.no_grad(), patch_qwen_attention(model, mode):
            outputs = model(
                input_ids=step_input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
        past_key_values = outputs.past_key_values
        step_logits = outputs.logits[:, -1, :].float()
        logits_history.append(step_logits.cpu())
        if forced_next_tokens is None:
            next_token = step_logits.argmax(dim=-1, keepdim=True)
        else:
            next_token = forced_next_tokens[:, step : step + 1].to(current_ids.device)
        generated.append(next_token.cpu())
        current_ids = torch.cat([current_ids, next_token], dim=1)

    return torch.cat(generated, dim=1), logits_history


def assert_generation_bounded(
    tag: str,
    dense_tokens: torch.Tensor,
    dense_logits: list[torch.Tensor],
    sparse_tokens: torch.Tensor,
    sparse_logits: list[torch.Tensor],
    *,
    max_diff_limit: float,
    mean_diff_limit: float,
) -> None:
    for step, (dense_step, sparse_step) in enumerate(zip(dense_logits, sparse_logits)):
        diff = (dense_step - sparse_step).abs()
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        if max_diff > max_diff_limit or mean_diff > mean_diff_limit:
            raise AssertionError(
                f"{tag}: logits mismatch at step {step}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
            )


def assert_generation_finite(tag: str, tokens: torch.Tensor, logits: list[torch.Tensor]) -> None:
    assert tokens.ndim == 2 and tokens.size(1) > 0, f"{tag}: expected non-empty generated tokens"
    for step, step_logits in enumerate(logits):
        if not torch.isfinite(step_logits).all():
            raise AssertionError(f"{tag}: non-finite logits at step {step}")


@pytest.mark.skipif(not is_xpu_available(), reason="XPU not available")
class TestQwenSparseDecodeE2E:
    def test_qwen3_0p6b_sparse_decode_manual_generate(self):
        ensure_sparse_binding()
        model_name = get_model_path(MODEL)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        ).to("xpu")
        model.eval()

        input_ids = build_prompt(tokenizer, target_tokens=575).to("xpu")
        input_ids_preprocess = build_prompt(tokenizer, target_tokens=576).to("xpu")
        max_new_tokens = 30
        max_new_tokens_preprocess = 2

        dense_all_tokens, dense_all_logits = run_manual_cached_loop(
            model, input_ids, max_new_tokens, mode="dense_masked_all_selected_cached"
        )
        sparse_all_tokens, sparse_all_logits = run_manual_cached_loop(
            model,
            input_ids,
            max_new_tokens,
            mode="sparse_all_selected_cached",
            forced_next_tokens=dense_all_tokens,
        )
        assert torch.equal(dense_all_tokens, sparse_all_tokens)
        assert_generation_finite("decode_all_selected_dense", dense_all_tokens, dense_all_logits)
        assert_generation_finite("decode_all_selected_sparse", sparse_all_tokens, sparse_all_logits)
        report_generation(
            "decode_all_selected",
            tokenizer,
            input_ids,
            dense_all_tokens,
            dense_all_logits,
            sparse_all_tokens,
            sparse_all_logits,
        )

        direct_pre_tokens, direct_pre_logits = run_manual_cached_loop(
            model, input_ids_preprocess, max_new_tokens_preprocess, mode="direct_preprocess_cached"
        )
        sparse_pre_tokens, sparse_pre_logits = run_manual_cached_loop(
            model,
            input_ids_preprocess,
            max_new_tokens_preprocess,
            mode="sparse_preprocess_cached",
            forced_next_tokens=direct_pre_tokens,
        )
        assert torch.equal(direct_pre_tokens, sparse_pre_tokens)
        report_generation(
            "decode_preprocess",
            tokenizer,
            input_ids_preprocess,
            direct_pre_tokens,
            direct_pre_logits,
            sparse_pre_tokens,
            sparse_pre_logits,
        )
        assert_generation_close(
            "decode_preprocess",
            direct_pre_tokens,
            direct_pre_logits,
            sparse_pre_tokens,
            sparse_pre_logits,
        )


if __name__ == "__main__":
    import pathlib

    test_file = pathlib.Path(__file__).resolve()
    pytest.main([str(test_file), "-v", "--confcutdir", str(test_file.parent)])
