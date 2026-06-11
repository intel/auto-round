#!/usr/bin/env python
# -*- coding: utf-8 -*-

PROMPT = """
You are solving a long arithmetic worksheet. Read every example carefully and answer the final question with only the final number.
Example 1: 11 + 7 = 18.
Example 2: 14 + 9 = 23.
Example 3: 19 - 6 = 13.
Example 4: 25 + 8 = 33.
Example 5: 42 - 17 = 25.
Example 6: 13 + 13 = 26.
Example 7: 27 + 15 = 42.
Example 8: 30 - 11 = 19.
Example 9: 18 + 24 = 42.
Example 10: 44 - 19 = 25.
Example 11: 16 + 18 = 34.
Example 12: 21 + 22 = 43.
Example 13: 39 - 14 = 25.
Example 14: 17 + 26 = 43.
Example 15: 55 - 12 = 43.
Example 16: 23 + 20 = 43.
Example 17: 28 + 15 = 43.
Example 18: 47 - 4 = 43.
Example 19: 31 + 12 = 43.
Example 20: 60 - 17 = 43.
Example 21: 24 + 19 = 43.
Example 22: 34 + 9 = 43.
Example 23: 52 - 9 = 43.
Example 24: 18 + 25 = 43.
Example 25: 70 - 27 = 43.
Example 26: 29 + 14 = 43.
Example 27: 36 + 7 = 43.
Example 28: 58 - 15 = 43.
Example 29: 20 + 23 = 43.
Example 30: 48 - 5 = 43.
Example 31: 15 + 28 = 43.
Example 32: 63 - 20 = 43.
Example 33: 32 + 11 = 43.
Example 34: 50 - 7 = 43.
Example 35: 26 + 17 = 43.
Example 36: 46 - 3 = 43.
Example 37: 12 + 31 = 43.
Example 38: 67 - 24 = 43.
Example 39: 40 + 3 = 43.
Example 40: 74 - 31 = 43.
Example 41: 22 + 21 = 43.
Example 42: 54 - 11 = 43.
Example 43: 38 + 5 = 43.
Example 44: 49 - 6 = 43.
Example 45: 10 + 33 = 43.
Example 46: 61 - 18 = 43.
Example 47: 35 + 8 = 43.
Example 48: 57 - 14 = 43.
Example 49: 13 + 30 = 43.
Example 50: 72 - 29 = 43.
Example 51: 41 + 2 = 43.
Example 52: 69 - 26 = 43.
Example 53: 16 + 27 = 43.
Example 54: 65 - 22 = 43.
Example 55: 14 + 29 = 43.
Example 56: 59 - 16 = 43.
Example 57: 33 + 10 = 43.
Example 58: 62 - 19 = 43.
Example 59: 11 + 32 = 43.
Example 60: 53 - 10 = 43.
Example 61: 19 + 24 = 43.
Example 62: 66 - 23 = 43.
Example 63: 18 + 25 = 43.
Example 64: 64 - 21 = 43.
Example 65: 37 + 6 = 43.
Example 66: 56 - 13 = 43.
Example 67: 21 + 22 = 43.
Example 68: 68 - 25 = 43.
Example 69: 24 + 19 = 43.
Example 70: 51 - 8 = 43.
Example 71: 30 + 13 = 43.
Example 72: 71 - 28 = 43.
Example 73: 27 + 16 = 43.
Example 74: 45 - 2 = 43.
Example 75: 17 + 26 = 43.
Example 76: 73 - 30 = 43.
Example 77: 28 + 15 = 43.
Example 78: 52 - 9 = 43.
Example 79: 31 + 12 = 43.
Example 80: 58 - 15 = 43.
Example 81: 25 + 18 = 43.
Example 82: 47 - 4 = 43.
Example 83: 12 + 31 = 43.
Example 84: 60 - 17 = 43.
Example 85: 39 + 4 = 43.
Example 86: 55 - 12 = 43.
Example 87: 20 + 23 = 43.
Example 88: 63 - 20 = 43.
Example 89: 34 + 9 = 43.
Example 90: 50 - 7 = 43.
Example 91: 26 + 17 = 43.
Example 92: 46 - 3 = 43.
Example 93: 22 + 21 = 43.
Example 94: 54 - 11 = 43.
Example 95: 15 + 28 = 43.
Example 96: 49 - 6 = 43.
Example 97: 29 + 14 = 43.
Example 98: 57 - 14 = 43.
Example 99: 18 + 25 = 43.
Example 100: 64 - 21 = 43.
Example 101: 37 + 6 = 43.
Example 102: 56 - 13 = 43.
Example 103: 21 + 22 = 43.
Example 104: 68 - 25 = 43.
Example 105: 24 + 19 = 43.
Example 106: 51 - 8 = 43.
Example 107: 30 + 13 = 43.
Example 108: 71 - 28 = 43.
Example 109: 27 + 16 = 43.
Example 110: 45 - 2 = 43.
Example 111: 17 + 26 = 43.
Example 112: 73 - 30 = 43.
Example 113: 28 + 15 = 43.
Example 114: 52 - 9 = 43.
Example 115: 31 + 12 = 43.
Example 116: 58 - 15 = 43.
Example 117: 25 + 18 = 43.
Example 118: 47 - 4 = 43.
Example 119: 12 + 31 = 43.
Example 120: 60 - 17 = 43.
Final question: What is 17 + 26?
"""

import contextlib
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_ARK_PARENT = REPO_ROOT / "auto_round_extension" / "ark"
if str(LOCAL_ARK_PARENT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ARK_PARENT))

import auto_round_kernel as ark


PREPROCESS_TRACE: list[dict[str, float | int]] = []


def is_xpu_available():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def get_model_path(model_name: str) -> str:
    model_name = model_name.rstrip("/")
    candidates = [
        f"/tf_dataset/auto_round/models/{model_name}",
        f"/models/{model_name.split('/')[-1]}",
        f"/dataset/{model_name.split('/')[-1]}",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return model_name


def ensure_sparse_binding():
    if getattr(ark, "xpu_lib", None) is not None and hasattr(ark.xpu_lib, "sage_sparse"):
        return
    candidates = sorted((REPO_ROOT / "auto_round_extension" / "ark" / "auto_round_kernel" / "xbuild").glob("auto_round_kernel_xpu*.so"))
    if not candidates:
        raise RuntimeError("Unable to locate built XPU extension with sage_sparse in xbuild/")
    ext_path = candidates[-1]
    spec = importlib.util.spec_from_file_location("auto_round_kernel_xpu", ext_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load extension spec from {ext_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["auto_round_kernel_xpu"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "sage_sparse"):
        raise RuntimeError(f"Loaded extension does not expose sage_sparse: {ext_path}")
    ark.xpu_lib = module


def quantize_qk(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, seq_len, head_dim = tensor.shape
    num_rows = batch * heads * seq_len
    num_blocks = num_rows // block_size
    out = torch.empty_like(tensor, dtype=torch.int8)
    scale = torch.empty(num_blocks, dtype=torch.float32, device=tensor.device)
    lib = ark.get_lib(tensor)
    stream = ark.get_stream(tensor)
    lib.sage_dynamic_quant(
        stream,
        tensor.data_ptr(),
        0,
        out.data_ptr(),
        scale.data_ptr(),
        num_rows,
        head_dim,
        block_size,
    )
    return out, scale.reshape(batch, heads, seq_len // block_size, 1)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def normalize_attn_mask(attn_mask: torch.Tensor | None, batch: int, seq_q: int, seq_kv: int, device: torch.device) -> torch.Tensor:
    if attn_mask is None:
        return torch.zeros((batch, 1, seq_q, seq_kv), dtype=torch.float32, device=device)
    mask = attn_mask
    if mask.ndim == 2:
        mask = mask.view(1, 1, seq_q, seq_kv)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError(f"Unsupported attention mask rank: {mask.ndim}")
    if mask.shape[0] == 1 and batch != 1:
        mask = mask.expand(batch, -1, -1, -1)
    return mask.contiguous().to(torch.float32)


def build_prefill_sparse_metadata_and_mask(
    batch: int,
    num_heads_q: int,
    seq_len: int,
    valid_seq_len: int,
    head_dim: int,
    base_mask: torch.Tensor,
    *,
    block_size: int = 64,
    all_selected: bool = False,
    local_window_blocks: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_tile_tokens = 128 if head_dim == 64 else 256
    q_blocks = seq_len // block_size
    kv_blocks = q_blocks
    active_query_tiles = (seq_len + query_tile_tokens - 1) // query_tile_tokens

    lut = torch.zeros((batch, num_heads_q, q_blocks, kv_blocks), dtype=torch.int32, device=base_mask.device)
    valid = torch.zeros((batch, num_heads_q, q_blocks), dtype=torch.int32, device=base_mask.device)
    sparse_mask = base_mask.clone()
    sparse_mask.fill_(-1.0e9)

    if base_mask.numel() > 0:
        sparse_mask = torch.minimum(sparse_mask, base_mask)

    for qtile in range(active_query_tiles):
        q_start = qtile * query_tile_tokens
        q_end = min(q_start + query_tile_tokens, seq_len)
        if q_start >= valid_seq_len:
            continue
        if all_selected:
            selected_blocks = list(range(kv_blocks))
        else:
            tile_last_token = min(q_end, valid_seq_len) - 1
            last_real_block = tile_last_token // block_size
            keep_start_block = max(0, last_real_block - local_window_blocks + 1)
            selected_blocks = list(range(keep_start_block, last_real_block + 1))
            if keep_start_block > 0:
                selected_blocks = [0] + selected_blocks

        for qblk in range(q_start // block_size, ceil_div(min(q_end, valid_seq_len), block_size)):
            previous = 0
            for i, block_idx in enumerate(selected_blocks):
                lut[:, :, qblk, i] = block_idx if i == 0 else block_idx - previous
                previous = block_idx
            valid[:, :, qblk] = len(selected_blocks)

        for qpos in range(q_start, min(q_end, valid_seq_len)):
            q_block = qpos // block_size
            visible_blocks = [block_idx for block_idx in selected_blocks if block_idx <= q_block]
            for block_idx in visible_blocks:
                k_start = block_idx * block_size
                k_end = min(k_start + block_size, valid_seq_len)
                sparse_mask[:, :, qpos, k_start:k_end] = 0.0
            sparse_mask[:, :, qpos, :] = torch.minimum(sparse_mask[:, :, qpos, :], base_mask[:, :, qpos, :])

    return lut.contiguous(), valid.contiguous(), sparse_mask.contiguous()


@contextlib.contextmanager
def patch_model_sdpa(mode: str, trace_sink: list[dict[str, float | int]] | None = None):
    original = F.scaled_dot_product_attention

    def patched(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs):
        if (
            query.device.type != "xpu"
            or query.dtype != torch.float16
            or key.dtype != query.dtype
            or value.dtype != query.dtype
            or query.ndim != 4
            or key.ndim != 4
            or value.ndim != 4
            or query.shape[-2] != key.shape[-2]
            or key.shape != value.shape
            or query.shape[-1] not in (64, 128)
            or dropout_p != 0.0
        ):
            return original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
                **kwargs,
            )

        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
        batch, num_heads_q, seq_len, head_dim = q.shape
        _, num_heads_kv, _, _ = k.shape
        softmax_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
        quant_block_size = 64
        seq_pad = ceil_div(seq_len, quant_block_size) * quant_block_size
        pad_tokens = seq_pad - seq_len

        base_mask = normalize_attn_mask(attn_mask, batch, seq_len, seq_len, q.device)
        padded_mask = torch.zeros((batch, 1, seq_pad, seq_pad), dtype=torch.float32, device=q.device)
        padded_mask[:, :, seq_len:, :] = -1.0e9
        padded_mask[:, :, :, seq_len:] = -1.0e9
        padded_mask[:, :, :seq_len, :seq_len] = base_mask
        if is_causal:
            causal = torch.triu(
                torch.full((seq_pad, seq_pad), -1.0e9, dtype=torch.float32, device=q.device),
                diagonal=1,
            )
            padded_mask = torch.minimum(padded_mask, causal.view(1, 1, seq_pad, seq_pad))

        if pad_tokens:
            q = torch.nn.functional.pad(q, (0, 0, 0, pad_tokens))
            k = torch.nn.functional.pad(k, (0, 0, 0, pad_tokens))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad_tokens))

        if mode in ("sparse_preprocess", "dense_masked_preprocess"):
            preprocess = ark.sparge_preprocess_topk(
                q,
                k,
                is_causal=is_causal,
                smooth_k=True,
                simthreshd1=-1.0,
                topk=0.75,
                attention_sink=False,
                quant_block_size=quant_block_size,
                tensor_layout="HND",
            )
            if trace_sink is not None:
                stats = preprocess.get("stats", {})
                trace_sink.append(
                    {
                        "selected_ratio": float(stats.get("selected_ratio", 0.0)),
                        "selected_blocks_per_row": float(stats.get("selected_blocks_per_row", 0.0)),
                        "kernel_compatibility_added_blocks": int(preprocess.get("kernel_compatibility_added_blocks", 0)),
                    }
                )
            if mode == "dense_masked_preprocess":
                head_group = num_heads_q // num_heads_kv
                per_head_outputs = []
                for head_q in range(num_heads_q):
                    head_kv = head_q // head_group
                    per_head_mask = ark.sparge_block_map_to_mask(
                        preprocess["block_map"][:, head_q : head_q + 1],
                        quant_block_size=preprocess["quant_block_size"],
                        seq_len_q=seq_pad,
                        seq_len_kv=seq_pad,
                        is_causal=is_causal,
                    )
                    dense_mask = torch.minimum(per_head_mask, padded_mask)
                    per_head_out = ark.sage(
                        preprocess["query_i8"][:, head_q : head_q + 1],
                        preprocess["key_i8"][:, head_kv : head_kv + 1],
                        v[:, head_kv : head_kv + 1],
                        attn_mask=dense_mask,
                        is_causal=False,
                        scale=softmax_scale,
                        quant_block_size=preprocess["quant_block_size"],
                        qscale=preprocess["qscale"][:, head_q : head_q + 1],
                        kscale=preprocess["kscale"][:, head_kv : head_kv + 1],
                        tensor_layout="HND",
                    )
                    per_head_outputs.append(per_head_out)
                out = torch.cat(per_head_outputs, dim=1).contiguous()
                return out[:, :, :seq_len, :]
            out = ark.sage_sparse(
                preprocess["query_i8"],
                preprocess["key_i8"],
                v,
                preprocess["lut"],
                preprocess["valid_block_num"],
                attn_mask=padded_mask,
                is_causal=False,
                scale=softmax_scale,
                quant_block_size=preprocess["quant_block_size"],
                qscale=preprocess["qscale"],
                kscale=preprocess["kscale"],
                tensor_layout="HND",
            )
            return out[:, :, :seq_len, :]

        q_i8, q_scale = quantize_qk(q, quant_block_size)
        k_i8, k_scale = quantize_qk(k, quant_block_size)
        if mode in ("dense_masked_all_selected", "sparse_all_selected"):
            lut, valid, sparse_mask = build_prefill_sparse_metadata_and_mask(
                batch,
                num_heads_q,
                seq_pad,
                seq_len,
                head_dim,
                padded_mask,
                block_size=quant_block_size,
                all_selected=True,
            )
        else:
            lut, valid, sparse_mask = build_prefill_sparse_metadata_and_mask(
                batch,
                num_heads_q,
                seq_pad,
                seq_len,
                head_dim,
                padded_mask,
                block_size=quant_block_size,
                all_selected=False,
                local_window_blocks=2,
            )

        if mode in ("dense_masked", "dense_masked_all_selected"):
            out = ark.sage(
                q_i8,
                k_i8,
                v,
                attn_mask=sparse_mask,
                is_causal=False,
                scale=softmax_scale,
                quant_block_size=quant_block_size,
                qscale=q_scale,
                kscale=k_scale,
                tensor_layout="HND",
            )
            return out[:, :, :seq_len, :]
        if mode in ("sparse", "sparse_all_selected"):
            out = ark.sage_sparse(
                q_i8,
                k_i8,
                v,
                lut,
                valid,
                attn_mask=sparse_mask,
                is_causal=False,
                scale=softmax_scale,
                quant_block_size=quant_block_size,
                qscale=q_scale,
                kscale=k_scale,
                tensor_layout="HND",
            )
            return out[:, :, :seq_len, :]
        raise ValueError(f"Unsupported patch mode: {mode}")

    F.scaled_dot_product_attention = patched
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


def build_prompt(tokenizer, target_tokens: int = 256) -> torch.Tensor:
    # text = "The capital of France is Paris. " * 128
    text = PROMPT.strip().replace("\n", " ")
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :target_tokens]
    if ids.shape[1] < target_tokens:
        raise RuntimeError(f"Prompt tokenization produced only {ids.shape[1]} tokens, expected {target_tokens}")
    return ids


def run_manual_no_cache_step(model, input_ids: torch.Tensor, mode: str):
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    step_trace: list[dict[str, float | int]] = []
    with torch.no_grad(), patch_model_sdpa(mode, trace_sink=step_trace):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if step_trace:
        PREPROCESS_TRACE.append(
            {
                "selected_ratio_mean": float(sum(item["selected_ratio"] for item in step_trace) / len(step_trace)),
                "selected_blocks_per_row_mean": float(
                    sum(item["selected_blocks_per_row"] for item in step_trace) / len(step_trace)
                ),
                "kernel_compatibility_added_blocks_sum": int(
                    sum(int(item["kernel_compatibility_added_blocks"]) for item in step_trace)
                ),
            }
        )
    return outputs.logits[:, -1, :].float()


def run_manual_no_cache_loop(model, input_ids: torch.Tensor, max_new_tokens: int, mode: str):
    current_ids = input_ids.clone()
    generated = []
    logits_history = []
    trace_start = len(PREPROCESS_TRACE)

    for _ in range(max_new_tokens):
        step_logits = run_manual_no_cache_step(model, current_ids, mode=mode)
        logits_history.append(step_logits.cpu())
        next_token = step_logits.argmax(dim=-1, keepdim=True)
        generated.append(next_token.cpu())
        current_ids = torch.cat([current_ids, next_token], dim=1)

    return torch.cat(generated, dim=1), logits_history, PREPROCESS_TRACE[trace_start:]


def report_generation(tag: str, tokenizer, input_ids: torch.Tensor, dense_tokens, dense_logits, sparse_tokens, sparse_logits):
    assert sparse_tokens.shape == dense_tokens.shape
    for step, sparse_step in enumerate(sparse_logits):
        assert torch.isfinite(sparse_step).all(), f"Non-finite sparse logits at step {step} for {tag}"

    for step, (dense_step, sparse_step) in enumerate(zip(dense_logits, sparse_logits)):
        diff = (dense_step - sparse_step).abs()
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        print(
            f"[qwen_sparse_prefill][{tag}][step={step}] dense_token={int(dense_tokens[0, step])} "
            f"sparse_token={int(sparse_tokens[0, step])} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}"
        )

    prompt_text = tokenizer.decode(input_ids[0].cpu(), skip_special_tokens=True)
    dense_gen_text = tokenizer.decode(dense_tokens[0], skip_special_tokens=True)
    sparse_gen_text = tokenizer.decode(sparse_tokens[0], skip_special_tokens=True)
    dense_full_text = prompt_text + dense_gen_text
    sparse_full_text = prompt_text + sparse_gen_text
    # print(f"[qwen_sparse_prefill][{tag}][prompt_text] {prompt_text}")
    print(f"[qwen_sparse_prefill][{tag}][dense_gen_text] {dense_gen_text}")
    print(f"[qwen_sparse_prefill][{tag}][sparse_gen_text] {sparse_gen_text}")
    # print(f"[qwen_sparse_prefill][{tag}][dense_full_text] {dense_full_text}")
    # print(f"[qwen_sparse_prefill][{tag}][sparse_full_text] {sparse_full_text}")


def assert_generation_close(
    tag: str,
    dense_tokens: torch.Tensor,
    dense_logits: list[torch.Tensor],
    sparse_tokens: torch.Tensor,
    sparse_logits: list[torch.Tensor],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    assert torch.equal(dense_tokens, sparse_tokens), f"{tag}: token mismatch"
    assert len(dense_logits) == len(sparse_logits), f"{tag}: logits length mismatch"
    for step, (dense_step, sparse_step) in enumerate(zip(dense_logits, sparse_logits)):
        if not torch.allclose(dense_step, sparse_step, atol=atol, rtol=rtol):
            diff = (dense_step - sparse_step).abs()
            raise AssertionError(
                f"{tag}: logits mismatch at step {step}, max_diff={float(diff.max()):.6f} mean_diff={float(diff.mean()):.6f}"
            )


def report_preprocess_trace(tag: str, trace: list[dict[str, float | int]]) -> None:
    for step, stats in enumerate(trace):
        print(
            f"[qwen_sparse_prefill][{tag}][step={step}] "
            f"selected_ratio_mean={stats['selected_ratio_mean']:.6f} "
            f"selected_blocks_per_row_mean={stats['selected_blocks_per_row_mean']:.6f} "
            f"kernel_compatibility_added_blocks_sum={stats['kernel_compatibility_added_blocks_sum']}"
        )

MODEL = "Qwen/Qwen3-0.6B"

@pytest.mark.skipif(not is_xpu_available(), reason="XPU not available")
class TestQwenSparsePrefillE2E:
    def test_qwen3_0p6b_sparse_prefill_manual_generate(self):
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
        max_new_tokens = 30

        dense_all_tokens, dense_all_logits, _ = run_manual_no_cache_loop(
            model, input_ids, max_new_tokens, mode="dense_masked_all_selected"
        )
        sparse_all_tokens, sparse_all_logits, _ = run_manual_no_cache_loop(
            model, input_ids, max_new_tokens, mode="sparse_all_selected"
        )
        assert sparse_all_tokens.shape == (1, max_new_tokens)
        assert_generation_close(
            "all_selected",
            dense_all_tokens,
            dense_all_logits,
            sparse_all_tokens,
            sparse_all_logits,
        )

        dense_partial_tokens, dense_partial_logits, _ = run_manual_no_cache_loop(
            model, input_ids, max_new_tokens, mode="dense_masked"
        )
        sparse_partial_tokens, sparse_partial_logits, _ = run_manual_no_cache_loop(
            model, input_ids, max_new_tokens, mode="sparse"
        )
        preprocess_partial_tokens, preprocess_partial_logits, preprocess_partial_trace = run_manual_no_cache_loop(
            model, input_ids, max_new_tokens, mode="sparse_preprocess"
        )
        preprocess_dense_tokens, preprocess_dense_logits, _ = run_manual_no_cache_loop(
            model, input_ids, max_new_tokens, mode="dense_masked_preprocess"
        )
        assert sparse_partial_tokens.shape == (1, max_new_tokens)
        assert preprocess_partial_tokens.shape == (1, max_new_tokens)
        assert preprocess_dense_tokens.shape == (1, max_new_tokens)
        # report_generation(
        #     "partial_sparse",
        #     tokenizer,
        #     input_ids,
        #     dense_partial_tokens,
        #     dense_partial_logits,
        #     sparse_partial_tokens,
        #     sparse_partial_logits,
        # )
        report_generation(
            "partial_sparse_preprocess",
            tokenizer,
            input_ids,
            dense_partial_tokens,
            dense_partial_logits,
            preprocess_partial_tokens,
            preprocess_partial_logits,
        )
        report_preprocess_trace("partial_sparse_preprocess", preprocess_partial_trace)
        report_generation(
            "preprocess_dense_reference",
            tokenizer,
            input_ids,
            preprocess_dense_tokens,
            preprocess_dense_logits,
            preprocess_partial_tokens,
            preprocess_partial_logits,
        )
        assert_generation_close(
            "preprocess_dense_reference",
            preprocess_dense_tokens,
            preprocess_dense_logits,
            preprocess_partial_tokens,
            preprocess_partial_logits,
        )


if __name__ == "__main__":
    import pathlib

    test_file = pathlib.Path(__file__).resolve()
    pytest.main([str(test_file), "-v", "--confcutdir", str(test_file.parent)])
