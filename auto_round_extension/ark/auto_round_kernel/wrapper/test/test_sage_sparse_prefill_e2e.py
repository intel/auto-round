# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import math
import sys
from pathlib import Path

import torch

REPO_PARENT = Path(__file__).resolve().parents[3]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import auto_round_kernel as ark
from auto_round_kernel.xpu_loader import ensure_xpu_lib


def ensure_sparse_binding(*, required_symbols: tuple[str, ...] = ("sage_sparse",)) -> None:
    search_roots = (
        REPO_PARENT / "auto_round_kernel",
        REPO_PARENT / "auto_round_kernel" / "xbuild_diffuser",
        REPO_PARENT / "auto_round_kernel" / "xbuild",
        REPO_PARENT / "auto_round_kernel" / "xbuild_bf16_v2",
        REPO_PARENT / "auto_round_kernel" / "ark-xbuild",
    )
    ensure_xpu_lib(
        required_symbols=required_symbols,
        search_roots=search_roots,
    )


def quantize_qk(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, seq_len, head_dim = tensor.shape
    num_rows = batch * heads * seq_len
    num_blocks = num_rows // block_size
    q_i8 = torch.empty_like(tensor, dtype=torch.int8)
    scale = torch.empty(num_blocks, dtype=torch.float32, device=tensor.device)
    lib = ark.get_lib(tensor)
    stream = ark.get_stream(tensor)
    lib.sage_dynamic_quant(
        stream,
        tensor.data_ptr(),
        0,
        q_i8.data_ptr(),
        scale.data_ptr(),
        num_rows,
        head_dim,
        block_size,
    )
    return q_i8, scale.reshape(batch, heads, seq_len // block_size, 1)


def build_sparse_metadata_and_mask(
    batch: int,
    heads: int,
    seq_len_q: int,
    quant_block_size: int,
    query_tile_tokens: int,
    per_query_tile_selection: list[list[int]],
    device: torch.device,
    is_causal: bool = False,
    sparse_q_block_tokens: int | None = None,
    sparse_k_block_tokens: int | None = None,
    seq_len_kv: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len_kv = seq_len_q if seq_len_kv is None else seq_len_kv
    q_block_tokens = quant_block_size if sparse_q_block_tokens is None else sparse_q_block_tokens
    k_block_tokens = quant_block_size if sparse_k_block_tokens is None else sparse_k_block_tokens
    q_blocks = (seq_len_q + q_block_tokens - 1) // q_block_tokens
    kv_blocks = (seq_len_kv + k_block_tokens - 1) // k_block_tokens
    active_query_tiles = (seq_len_q + query_tile_tokens - 1) // query_tile_tokens
    assert len(per_query_tile_selection) == active_query_tiles

    lut = torch.zeros((batch, heads, q_blocks, kv_blocks), dtype=torch.int32, device=device)
    valid = torch.zeros((batch, heads, q_blocks), dtype=torch.int32, device=device)
    mask = torch.full((batch, 1, seq_len_q, seq_len_kv), -1.0e9, dtype=torch.float32, device=device)

    q_blocks_per_query_tile = max(1, query_tile_tokens // q_block_tokens)
    for qblk in range(q_blocks):
        qtile = min(qblk // q_blocks_per_query_tile, active_query_tiles - 1)
        selected_blocks = per_query_tile_selection[qtile]
        previous = 0
        for i, selected in enumerate(selected_blocks):
            lut[..., qblk, i] = selected if i == 0 else (selected - previous)
            previous = selected
        valid[..., qblk] = len(selected_blocks)

    for qtile, selected_blocks in enumerate(per_query_tile_selection):
        q_start = qtile * query_tile_tokens
        q_end = min(q_start + query_tile_tokens, seq_len_q)
        for qt in range(q_start, q_end):
            for selected in selected_blocks:
                k_start = selected * k_block_tokens
                k_end = min(k_start + k_block_tokens, seq_len_kv)
                if not is_causal:
                    mask[:, :, qt : qt + 1, k_start:k_end] = 0.0
                else:
                    visible_end = min(k_end, qt + 1)
                    if visible_end > k_start:
                        mask[:, :, qt : qt + 1, k_start:visible_end] = 0.0

    return lut.contiguous(), valid.contiguous(), mask.contiguous()


def _clamp_selected_blocks(kv_blocks: int, per_query_tile_selection: list[list[int]]) -> list[list[int]]:
    clamped: list[list[int]] = []
    for selected_blocks in per_query_tile_selection:
        clamped_blocks = sorted({min(max(block, 0), kv_blocks - 1) for block in selected_blocks})
        clamped.append(clamped_blocks)
    return clamped


def bf16_sparse_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    *,
    scale: float,
    enable_gqa: bool,
) -> torch.Tensor:
    # Run the reference on CPU: torch-xpu's softmax is numerically broken on large
    # masked tensors (sums to <1, drops visible entries), which corrupts the reference.
    q = query.float().cpu()
    k = key.float().cpu()
    v = value.float().cpu()
    mask = None if attn_mask is None else attn_mask.float().cpu()
    if enable_gqa and q.shape[1] != k.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores + mask
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v).to(query.dtype).to(query.device)


def run_case(
    head_dim: int,
    block_size: int = 64,
    is_causal: bool = False,
    *,
    seq_len: int = 256,
    num_heads_q: int = 4,
    num_heads_kv: int | None = None,
    q_tile_override: int = 0,
    sparse_q_block_tokens: int | None = None,
    sparse_k_block_tokens: int | None = None,
) -> None:
    device = torch.device("xpu")
    batch = 1
    num_heads_kv = num_heads_q if num_heads_kv is None else num_heads_kv
    scale = 1.0 / math.sqrt(head_dim)
    query_tile_tokens = q_tile_override or (128 if head_dim == 64 else 256)
    kv_blocks = seq_len // block_size
    active_query_tiles = (seq_len + query_tile_tokens - 1) // query_tile_tokens
    if not is_causal:
        if head_dim == 64:
            per_query_tile_selection = [[0, 1], [1, 3]]
        elif query_tile_tokens == 64:
            per_query_tile_selection = [[0, 1], [1, 3], [0, 2, 3], [2, 3]]
        else:
            per_query_tile_selection = [[0, 2, 3], [2, 5, 7]][:active_query_tiles]
        case_name = "python_prefill"
    else:
        if head_dim == 64:
            per_query_tile_selection = [[0, 1, 2], [1, 3]]
        elif query_tile_tokens == 64:
            per_query_tile_selection = [[0, 1], [1, 2], [1, 2, 3], [2, 3]]
        else:
            per_query_tile_selection = [[0, 2, 3], [2, 6, 7]][:active_query_tiles]
        case_name = "python_prefill_causal"
    per_query_tile_selection = _clamp_selected_blocks(kv_blocks, per_query_tile_selection)

    torch.manual_seed(2026 + head_dim + num_heads_q + (num_heads_kv * 11))
    query = torch.randn((batch, num_heads_q, seq_len, head_dim), dtype=torch.float16, device=device)
    key = torch.randn((batch, num_heads_kv, seq_len, head_dim), dtype=torch.float16, device=device)
    value = torch.randn((batch, num_heads_kv, seq_len, head_dim), dtype=torch.float16, device=device)

    q_i8, q_scale = quantize_qk(query, block_size)
    k_i8, k_scale = quantize_qk(key, block_size)
    effective_sparse_q_block_tokens = block_size if sparse_q_block_tokens is None else sparse_q_block_tokens
    effective_sparse_k_block_tokens = block_size if sparse_k_block_tokens is None else sparse_k_block_tokens
    lut, valid, dense_mask = build_sparse_metadata_and_mask(
        batch,
        num_heads_q,
        seq_len,
        block_size,
        query_tile_tokens,
        per_query_tile_selection,
        device,
        is_causal=is_causal,
        sparse_q_block_tokens=effective_sparse_q_block_tokens,
        sparse_k_block_tokens=effective_sparse_k_block_tokens,
    )

    dense_out = ark.sage(
        q_i8,
        k_i8,
        value,
        attn_mask=dense_mask,
        is_causal=False,
        scale=scale,
        quant_block_size=block_size,
        qscale=q_scale,
        kscale=k_scale,
        tensor_layout="HND",
    )
    sparse_out = ark.sage_sparse(
        q_i8,
        k_i8,
        value,
        lut,
        valid,
        is_causal=is_causal,
        scale=scale,
        quant_block_size=block_size,
        qscale=q_scale,
        kscale=k_scale,
        q_tile_override=q_tile_override,
        sparse_q_block_tokens=sparse_q_block_tokens,
        sparse_k_block_tokens=sparse_k_block_tokens,
        tensor_layout="HND",
    )
    torch.xpu.synchronize()

    diff = (dense_out.float() - sparse_out.float()).abs()
    max_diff = float(diff.max().cpu())
    mean_diff = float(diff.mean().cpu())
    print(
        f"[sage_sparse][{case_name}] D={head_dim} Hq={num_heads_q} Hkv={num_heads_kv} "
        f"qtile={query_tile_tokens} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}"
    )
    if max_diff > 5e-3 or mean_diff > 5e-4:
        raise RuntimeError(
            f"sage_sparse python prefill mismatch for D={head_dim}, Hq={num_heads_q}, "
            f"Hkv={num_heads_kv}, causal={is_causal}"
        )


def run_case_sdpa(
    dtype: torch.dtype,
    head_dim: int,
    *,
    seq_len_q: int = 256,
    seq_len_kv: int = 256,
    num_heads_q: int = 4,
    num_heads_kv: int | None = None,
    is_causal: bool = False,
    q_tile_override: int = 0,
    sparse_q_block_tokens: int | None = None,
    sparse_k_block_tokens: int | None = None,
) -> None:
    """Validate the independent native-precision sparse-SDPA path (bf16/fp16)
    against a dense reference built from the same block selection."""
    ensure_sparse_binding(required_symbols=("sage_sparse", "sage_sparse_sdpa"))
    device = torch.device("xpu")
    batch = 1
    num_heads_kv = num_heads_q if num_heads_kv is None else num_heads_kv
    scale = 1.0 / math.sqrt(head_dim)
    query_tile_tokens = q_tile_override or (64 if head_dim == 64 else 256)
    k_block_tokens = 64 if sparse_k_block_tokens is None else sparse_k_block_tokens
    kv_blocks = (seq_len_kv + k_block_tokens - 1) // k_block_tokens
    active_query_tiles = (seq_len_q + query_tile_tokens - 1) // query_tile_tokens
    if not is_causal:
        per_query_tile_selection = [list(range(max(1, min(kv_blocks, 3)))) for _ in range(active_query_tiles)]
        if active_query_tiles > 1 and kv_blocks > 3:
            per_query_tile_selection[-1] = list(range(kv_blocks - 3, kv_blocks))
        case_name = f"{dtype}_prefill"
    else:
        per_query_tile_selection = [list(range(min(kv_blocks, idx + 1))) for idx in range(active_query_tiles)]
        case_name = f"{dtype}_prefill_causal"
    per_query_tile_selection = _clamp_selected_blocks(kv_blocks, per_query_tile_selection)

    torch.manual_seed(9026 + head_dim + num_heads_q + (num_heads_kv * 13) + seq_len_q + seq_len_kv)
    query = torch.randn((batch, num_heads_q, seq_len_q, head_dim), dtype=dtype, device=device)
    key = torch.randn((batch, num_heads_kv, seq_len_kv, head_dim), dtype=dtype, device=device)
    value = torch.randn((batch, num_heads_kv, seq_len_kv, head_dim), dtype=dtype, device=device)

    effective_sparse_q_block_tokens = 64 if sparse_q_block_tokens is None else sparse_q_block_tokens
    effective_sparse_k_block_tokens = 64 if sparse_k_block_tokens is None else sparse_k_block_tokens
    lut, valid, dense_mask = build_sparse_metadata_and_mask(
        batch,
        num_heads_q,
        seq_len_q,
        64,
        query_tile_tokens,
        per_query_tile_selection,
        device,
        is_causal=is_causal,
        sparse_q_block_tokens=effective_sparse_q_block_tokens,
        sparse_k_block_tokens=effective_sparse_k_block_tokens,
        seq_len_kv=seq_len_kv,
    )

    dense_out = bf16_sparse_reference(
        query,
        key,
        value,
        dense_mask,
        scale=scale,
        enable_gqa=num_heads_q != num_heads_kv,
    )
    sparse_out = ark.sage_sparse_sdpa(
        query,
        key,
        value,
        lut,
        valid,
        is_causal=is_causal,
        scale=scale,
        q_tile_override=q_tile_override,
        sparse_q_block_tokens=sparse_q_block_tokens,
        sparse_k_block_tokens=sparse_k_block_tokens,
        tensor_layout="HND",
    )
    torch.xpu.synchronize()

    diff = (dense_out.float() - sparse_out.float()).abs()
    max_diff = float(diff.max().cpu())
    mean_diff = float(diff.mean().cpu())
    print(
        f"[sage_sparse_sdpa][{case_name}] D={head_dim} Hq={num_heads_q} Hkv={num_heads_kv} "
        f"Sq={seq_len_q} Skv={seq_len_kv} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}"
    )
    if max_diff > 2e-2 or mean_diff > 2e-3:
        raise RuntimeError(
            f"sage_sparse_sdpa mismatch for dtype={dtype}, D={head_dim}, Hq={num_heads_q}, Hkv={num_heads_kv}, "
            f"Sq={seq_len_q}, Skv={seq_len_kv}, causal={is_causal}"
        )


def run_case_sdpa_full(
    dtype: torch.dtype,
    head_dim: int,
    *,
    seq_len_q: int = 256,
    seq_len_kv: int = 256,
    num_heads_q: int = 4,
    q_tile_override: int = 0,
    sparse_q_block_tokens: int | None = None,
    sparse_k_block_tokens: int | None = None,
) -> None:
    """Dense gate: selecting every KV block must make sparse-SDPA match the dense reference."""
    ensure_sparse_binding(required_symbols=("sage_sparse", "sage_sparse_sdpa"))
    device = torch.device("xpu")
    batch = 1
    scale = 1.0 / math.sqrt(head_dim)
    query_tile_tokens = q_tile_override or (64 if head_dim == 64 else 256)
    k_block_tokens = 64 if sparse_k_block_tokens is None else sparse_k_block_tokens
    kv_blocks = (seq_len_kv + k_block_tokens - 1) // k_block_tokens
    active_query_tiles = (seq_len_q + query_tile_tokens - 1) // query_tile_tokens
    per_query_tile_selection = [list(range(kv_blocks)) for _ in range(active_query_tiles)]

    torch.manual_seed(77 + head_dim + num_heads_q)
    query = torch.randn((batch, num_heads_q, seq_len_q, head_dim), dtype=dtype, device=device)
    key = torch.randn((batch, num_heads_q, seq_len_kv, head_dim), dtype=dtype, device=device)
    value = torch.randn((batch, num_heads_q, seq_len_kv, head_dim), dtype=dtype, device=device)

    lut, valid, dense_mask = build_sparse_metadata_and_mask(
        batch,
        num_heads_q,
        seq_len_q,
        64,
        query_tile_tokens,
        per_query_tile_selection,
        device,
        is_causal=False,
        sparse_q_block_tokens=64 if sparse_q_block_tokens is None else sparse_q_block_tokens,
        sparse_k_block_tokens=k_block_tokens,
        seq_len_kv=seq_len_kv,
    )

    dense_out = bf16_sparse_reference(query, key, value, dense_mask, scale=scale, enable_gqa=False)
    sparse_out = ark.sage_sparse_sdpa(
        query,
        key,
        value,
        lut,
        valid,
        is_causal=False,
        scale=scale,
        q_tile_override=q_tile_override,
        sparse_q_block_tokens=sparse_q_block_tokens,
        sparse_k_block_tokens=sparse_k_block_tokens,
        tensor_layout="HND",
    )
    torch.xpu.synchronize()

    diff = (dense_out.float() - sparse_out.float()).abs()
    max_diff = float(diff.max().cpu())
    mean_diff = float(diff.mean().cpu())
    print(f"[sage_sparse_sdpa][{dtype}_all_selected] D={head_dim} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    if max_diff > 2e-2 or mean_diff > 2e-3:
        raise RuntimeError(f"sage_sparse_sdpa all-selected mismatch for dtype={dtype}, D={head_dim}")


def run_multi_row_tile_case() -> None:
    device = torch.device("xpu")
    batch = 1
    heads = 4
    head_dim = 128
    seq_len = 256
    block_size = 64
    scale = 1.0 / math.sqrt(head_dim)
    query_tile_tokens = 64
    per_query_tile_selection = [
        [0, 1],
        [1, 3],
        [0, 2, 3],
        [2, 3],
    ]

    torch.manual_seed(4026)
    query = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device=device)
    key = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device=device)
    value = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device=device)

    q_i8, q_scale = quantize_qk(query, block_size)
    k_i8, k_scale = quantize_qk(key, block_size)
    lut, valid, dense_mask = build_sparse_metadata_and_mask(
        batch, heads, seq_len, block_size, query_tile_tokens, per_query_tile_selection, device, is_causal=False
    )

    dense_out = ark.sage(
        q_i8,
        k_i8,
        value,
        attn_mask=dense_mask,
        is_causal=False,
        scale=scale,
        quant_block_size=block_size,
        qscale=q_scale,
        kscale=k_scale,
        tensor_layout="HND",
    )
    sparse_out = ark.sage_sparse(
        q_i8,
        k_i8,
        value,
        lut,
        valid,
        is_causal=False,
        scale=scale,
        quant_block_size=block_size,
        qscale=q_scale,
        kscale=k_scale,
        tensor_layout="HND",
    )
    torch.xpu.synchronize()

    diff = (dense_out.float() - sparse_out.float()).abs()
    max_diff = float(diff.max().cpu())
    mean_diff = float(diff.mean().cpu())
    print(f"[sage_sparse][python_prefill_multi_row_tile] D=128 max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    if max_diff > 5e-3 or mean_diff > 5e-4:
        raise RuntimeError("sage_sparse multi-row tile mismatch for D=128")


def run_all_selected_case(head_dim: int, block_size: int = 64) -> None:
    device = torch.device("xpu")
    batch = 1
    heads = 4
    seq_len = 256
    scale = 1.0 / math.sqrt(head_dim)
    query_tile_tokens = 128 if head_dim == 64 else 256
    kv_blocks = seq_len // block_size
    active_query_tiles = (seq_len + query_tile_tokens - 1) // query_tile_tokens
    per_query_tile_selection = [list(range(kv_blocks)) for _ in range(active_query_tiles)]

    torch.manual_seed(3026 + head_dim)
    query = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device=device)
    key = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device=device)
    value = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device=device)

    q_i8, q_scale = quantize_qk(query, block_size)
    k_i8, k_scale = quantize_qk(key, block_size)
    lut, valid, dense_mask = build_sparse_metadata_and_mask(
        batch, heads, seq_len, block_size, query_tile_tokens, per_query_tile_selection, device, is_causal=False
    )

    dense_out = ark.sage(
        q_i8,
        k_i8,
        value,
        attn_mask=dense_mask,
        is_causal=False,
        scale=scale,
        quant_block_size=block_size,
        qscale=q_scale,
        kscale=k_scale,
        tensor_layout="HND",
    )
    sparse_out = ark.sage_sparse(
        q_i8,
        k_i8,
        value,
        lut,
        valid,
        is_causal=False,
        scale=scale,
        quant_block_size=block_size,
        qscale=q_scale,
        kscale=k_scale,
        tensor_layout="HND",
    )
    torch.xpu.synchronize()

    diff = (dense_out.float() - sparse_out.float()).abs()
    max_diff = float(diff.max().cpu())
    mean_diff = float(diff.mean().cpu())
    print(f"[sage_sparse][python_prefill_all_selected] D={head_dim} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    if max_diff > 5e-3 or mean_diff > 5e-4:
        raise RuntimeError(f"sage_sparse python all-selected mismatch for D={head_dim}")


def main() -> None:
    ensure_sparse_binding()
    if not torch.xpu.is_available():
        raise RuntimeError("XPU device is required")
    run_all_selected_case(64)
    run_all_selected_case(128)
    run_case(64)
    run_case(128)
    run_multi_row_tile_case()
    run_case(128, num_heads_q=32, num_heads_kv=8, q_tile_override=64)
    run_case(
        128,
        num_heads_q=32,
        num_heads_kv=8,
        seq_len=512,
        q_tile_override=256,
        sparse_q_block_tokens=256,
        sparse_k_block_tokens=64,
    )
    run_case(64, is_causal=True)
    run_case(128, is_causal=True)
    # Independent sparse-SDPA path (bf16 + fp16)
    for dtype in (torch.bfloat16, torch.float16):
        run_case_sdpa(dtype, 64)
        run_case_sdpa(dtype, 128, q_tile_override=64, num_heads_q=32, num_heads_kv=8)
        run_case_sdpa(
            dtype,
            128,
            seq_len_q=512,
            seq_len_kv=768,
            num_heads_q=32,
            num_heads_kv=8,
            q_tile_override=256,
            sparse_q_block_tokens=256,
            sparse_k_block_tokens=64,
        )
        run_case_sdpa(dtype, 128, is_causal=True, q_tile_override=64)
        run_case_sdpa_full(dtype, 64)
        run_case_sdpa_full(dtype, 128, q_tile_override=64)


if __name__ == "__main__":
    main()
