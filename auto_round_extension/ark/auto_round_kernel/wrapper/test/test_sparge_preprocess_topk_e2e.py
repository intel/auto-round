# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import importlib.util
import math
import sys
from pathlib import Path

import torch

REPO_PARENT = Path(__file__).resolve().parents[3]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import auto_round_kernel as ark


def ensure_sparse_binding() -> None:
    if getattr(ark, "xpu_lib", None) is not None and hasattr(ark.xpu_lib, "sage_sparse"):
        return
    candidates = sorted((REPO_PARENT / "auto_round_kernel" / "xbuild").glob("auto_round_kernel_xpu*.so"))
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


def _to_layout(tensor: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    if tensor_layout == "HND":
        return tensor.contiguous()
    return tensor.permute(0, 2, 1, 3).contiguous()


def _assert_metadata_matches(actual: dict, reference: dict) -> None:
    for key in ("query_i8", "key_i8", "lut", "valid_block_num", "block_map", "raw_block_map", "tile_block_map"):
        assert torch.equal(actual[key], reference[key]), key
    for key in ("sim_qblocks", "sim_kblocks"):
        assert torch.equal(actual[key], reference[key]), key
    for key in ("qscale", "kscale"):
        assert torch.allclose(actual[key], reference[key], atol=2.0e-5, rtol=0.0), key
    assert actual["query_tile_tokens"] == reference["query_tile_tokens"]
    assert actual["quant_block_size"] == reference["quant_block_size"]
    assert actual["sparse_q_block_tokens"] == reference["sparse_q_block_tokens"]
    assert actual["sparse_k_block_tokens"] == reference["sparse_k_block_tokens"]
    assert actual["kernel_compatibility_added_blocks"] == reference["kernel_compatibility_added_blocks"]
    assert actual["stats"] == reference["stats"]


def run_case(
    head_dim: int,
    *,
    topk: float,
    is_causal: bool,
    tensor_layout: str,
    query_tile_tokens: int | None = None,
    q_tile_override: int = 0,
    sparse_q_block_tokens: int | None = None,
    sparse_k_block_tokens: int | None = None,
    seq_len_q: int = 256,
    seq_len_kv: int = 256,
    num_heads_q: int = 1,
    num_heads_kv: int = 1,
) -> None:
    device = torch.device("xpu")
    batch = 1
    scale = 1.0 / math.sqrt(head_dim)
    effective_query_tile_tokens = query_tile_tokens or (q_tile_override or None)
    effective_q_tile_override = q_tile_override if q_tile_override != 0 else (query_tile_tokens or 0)

    seed = (
        5200
        + head_dim
        + int(topk * 100)
        + (7 if is_causal else 0)
        + (13 if tensor_layout == "NHD" else 0)
        + seq_len_q
        + (seq_len_kv * 3)
        + (num_heads_q * 5)
        + (num_heads_kv * 11)
    )
    torch.manual_seed(seed)
    query_hnd = torch.randn((batch, num_heads_q, seq_len_q, head_dim), dtype=torch.float16, device=device)
    key_hnd = torch.randn((batch, num_heads_kv, seq_len_kv, head_dim), dtype=torch.float16, device=device)
    value_hnd = torch.randn((batch, num_heads_kv, seq_len_kv, head_dim), dtype=torch.float16, device=device)

    query = _to_layout(query_hnd, tensor_layout)
    key = _to_layout(key_hnd, tensor_layout)
    value = _to_layout(value_hnd, tensor_layout)

    preprocess_meta = ark.sparge_preprocess_topk(
        query,
        key,
        is_causal=is_causal,
        smooth_k=True,
        simthreshd1=-1.0,
        topk=topk,
        attention_sink=False,
        tensor_layout=tensor_layout,
        query_tile_tokens=effective_query_tile_tokens,
        sparse_q_block_tokens=sparse_q_block_tokens,
        sparse_k_block_tokens=sparse_k_block_tokens,
    )
    torch_meta = ark._sparge_preprocess_topk_torch(
        query,
        key,
        is_causal=is_causal,
        smooth_k=True,
        simthreshd1=-1.0,
        topk=topk,
        attention_sink=False,
        tensor_layout=tensor_layout,
        query_tile_tokens=effective_query_tile_tokens,
        sparse_q_block_tokens=sparse_q_block_tokens,
        sparse_k_block_tokens=sparse_k_block_tokens,
    )
    _assert_metadata_matches(preprocess_meta, torch_meta)

    sparse_out, meta = ark.sparge_sage2_attn_meansim_topk_xpu(
        query,
        key,
        value,
        is_causal=is_causal,
        scale=scale,
        smooth_k=True,
        simthreshd1=-1.0,
        topk=topk,
        attention_sink=False,
        tensor_layout=tensor_layout,
        query_tile_tokens=effective_query_tile_tokens,
        q_tile_override=q_tile_override,
        sparse_q_block_tokens=sparse_q_block_tokens,
        sparse_k_block_tokens=sparse_k_block_tokens,
        return_metadata=True,
    )

    assert meta["backend"] in {"torch", "triton_xpu"}
    _assert_metadata_matches(meta, preprocess_meta)
    assert tuple(meta["query_i8"].shape) == tuple(query.shape)
    assert tuple(meta["key_i8"].shape) == tuple(key.shape)
    assert meta["valid_block_num"].dtype == torch.int32
    assert meta["lut"].dtype == torch.int32

    direct_sparse = ark.sage_sparse(
        meta["query_i8"],
        meta["key_i8"],
        value,
        meta["lut"],
        meta["valid_block_num"],
        is_causal=is_causal,
        scale=scale,
        quant_block_size=meta["quant_block_size"],
        qscale=meta["qscale"],
        kscale=meta["kscale"],
        q_tile_override=effective_q_tile_override,
        sparse_q_block_tokens=meta["sparse_q_block_tokens"],
        sparse_k_block_tokens=meta["sparse_k_block_tokens"],
        tensor_layout=tensor_layout,
    )
    torch_sparse = ark.sage_sparse(
        torch_meta["query_i8"],
        torch_meta["key_i8"],
        value,
        torch_meta["lut"],
        torch_meta["valid_block_num"],
        is_causal=is_causal,
        scale=scale,
        quant_block_size=torch_meta["quant_block_size"],
        qscale=torch_meta["qscale"],
        kscale=torch_meta["kscale"],
        q_tile_override=effective_q_tile_override,
        sparse_q_block_tokens=torch_meta["sparse_q_block_tokens"],
        sparse_k_block_tokens=torch_meta["sparse_k_block_tokens"],
        tensor_layout=tensor_layout,
    )
    torch.xpu.synchronize()

    case_name = (
        f"topk_{topk:.2f}_{'causal' if is_causal else 'noncausal'}_{tensor_layout.lower()}"
        f"_qtile{query_tile_tokens or 'default'}_kqtile{effective_q_tile_override or 'default'}"
    )
    direct_diff = (direct_sparse.float() - sparse_out.float()).abs()
    direct_max_diff = float(direct_diff.max().cpu())
    direct_mean_diff = float(direct_diff.mean().cpu())
    print(
        f"[sparge_preprocess][{case_name}] D={head_dim} "
        f"wrapper_max_diff={direct_max_diff:.6f} wrapper_mean_diff={direct_mean_diff:.6f} backend={meta['backend']}"
    )
    if direct_max_diff > 5e-3 or direct_mean_diff > 5e-4:
        raise RuntimeError(f"sparge preprocess wrapper mismatch for {case_name}, D={head_dim}")

    torch_diff = (torch_sparse.float() - sparse_out.float()).abs()
    torch_max_diff = float(torch_diff.max().cpu())
    torch_mean_diff = float(torch_diff.mean().cpu())
    print(
        f"[sparge_preprocess][{case_name}] D={head_dim} "
        f"torch_replay_max_diff={torch_max_diff:.6f} torch_replay_mean_diff={torch_mean_diff:.6f}"
    )
    if torch_max_diff > 5e-3 or torch_mean_diff > 5e-4:
        raise RuntimeError(f"sparge preprocess torch replay mismatch for {case_name}, D={head_dim}")

    if topk == 1.0:
        dense_mask = ark.sparge_block_map_to_mask(
            meta["block_map"],
            quant_block_size=meta["quant_block_size"],
            q_block_tokens=meta["sparse_q_block_tokens"],
            k_block_tokens=meta["sparse_k_block_tokens"],
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            is_causal=is_causal,
        )
        dense_out = ark.sage(
            meta["query_i8"],
            meta["key_i8"],
            value,
            attn_mask=dense_mask,
            is_causal=False,
            scale=scale,
            quant_block_size=meta["quant_block_size"],
            qscale=meta["qscale"],
            kscale=meta["kscale"],
            tensor_layout=tensor_layout,
        )
        diff = (dense_out.float() - sparse_out.float()).abs()
        max_diff = float(diff.max().cpu())
        mean_diff = float(diff.mean().cpu())
        print(
            f"[sparge_preprocess][{case_name}] D={head_dim} dense_max_diff={max_diff:.6f} dense_mean_diff={mean_diff:.6f}"
        )
        if max_diff > 5e-3 or mean_diff > 5e-4:
            raise RuntimeError(f"sparge preprocess dense-reference mismatch for {case_name}, D={head_dim}")

    if topk < 1.0:
        assert 0.0 <= meta["stats"]["selected_ratio"] <= 1.0
        if seq_len_q == seq_len_kv:
            assert meta["stats"]["selected_ratio"] < 1.0
    assert meta["kernel_compatibility_added_blocks"] == 0
    assert torch.equal(meta["block_map"], meta["raw_block_map"])
    assert torch.isfinite(sparse_out).all()

    if topk < 1.0 and not is_causal:
        compatibility_seed = seed + 101
        torch.manual_seed(compatibility_seed)
        crafted_query_hnd = torch.randn((batch, num_heads_q, seq_len_q, head_dim), dtype=torch.float16, device=device)
        crafted_key_hnd = torch.randn((batch, num_heads_kv, seq_len_kv, head_dim), dtype=torch.float16, device=device)
        crafted_query = _to_layout(crafted_query_hnd, tensor_layout)
        crafted_key = _to_layout(crafted_key_hnd, tensor_layout)
        crafted_meta = ark._sparge_preprocess_topk_torch(
            crafted_query,
            crafted_key,
            is_causal=False,
            smooth_k=True,
            simthreshd1=-1.0,
            topk=topk,
            attention_sink=False,
            tensor_layout=tensor_layout,
            query_tile_tokens=effective_query_tile_tokens,
            sparse_q_block_tokens=sparse_q_block_tokens,
            sparse_k_block_tokens=sparse_k_block_tokens,
        )
        assert crafted_meta["kernel_compatibility_added_blocks"] == 0
        assert torch.equal(crafted_meta["block_map"], crafted_meta["raw_block_map"])


def main() -> None:
    ensure_sparse_binding()
    if not torch.xpu.is_available():
        raise RuntimeError("XPU device is required")
    for tensor_layout in ("HND", "NHD"):
        run_case(64, topk=1.0, is_causal=False, tensor_layout=tensor_layout)
        run_case(128, topk=1.0, is_causal=False, tensor_layout=tensor_layout)
        run_case(64, topk=0.5, is_causal=False, tensor_layout=tensor_layout)
        run_case(128, topk=0.5, is_causal=False, tensor_layout=tensor_layout)
        run_case(64, topk=0.5, is_causal=True, tensor_layout=tensor_layout)
        run_case(128, topk=0.5, is_causal=True, tensor_layout=tensor_layout)
        run_case(64, topk=0.25, is_causal=False, tensor_layout=tensor_layout)
        run_case(128, topk=0.25, is_causal=False, tensor_layout=tensor_layout)
        run_case(128, topk=0.5, is_causal=False, tensor_layout=tensor_layout, query_tile_tokens=256)
        run_case(128, topk=0.5, is_causal=False, tensor_layout=tensor_layout, q_tile_override=256)
        run_case(
            128,
            topk=0.5,
            is_causal=False,
            tensor_layout=tensor_layout,
            q_tile_override=256,
            sparse_q_block_tokens=256,
            sparse_k_block_tokens=64,
            seq_len_q=512,
            seq_len_kv=512,
            num_heads_q=2,
            num_heads_kv=2,
        )
        run_case(
            128,
            topk=0.5,
            is_causal=False,
            tensor_layout=tensor_layout,
            q_tile_override=64,
            seq_len_q=256,
            seq_len_kv=256,
            num_heads_q=32,
            num_heads_kv=8,
        )
        run_case(
            128,
            topk=0.5,
            is_causal=False,
            tensor_layout=tensor_layout,
            q_tile_override=256,
            sparse_q_block_tokens=256,
            sparse_k_block_tokens=64,
            seq_len_q=512,
            seq_len_kv=512,
            num_heads_q=32,
            num_heads_kv=8,
        )
        run_case(
            128,
            topk=0.5,
            is_causal=False,
            tensor_layout=tensor_layout,
            seq_len_q=256,
            seq_len_kv=128,
            num_heads_q=2,
            num_heads_kv=1,
        )


if __name__ == "__main__":
    main()
