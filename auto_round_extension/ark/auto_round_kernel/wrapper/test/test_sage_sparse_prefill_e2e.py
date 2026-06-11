import math
import sys
import importlib.util
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
    seq_len: int,
    block_size: int,
    query_tile_tokens: int,
    per_query_tile_selection: list[list[int]],
    device: torch.device,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_blocks = seq_len // block_size
    kv_blocks = q_blocks
    active_query_tiles = (seq_len + query_tile_tokens - 1) // query_tile_tokens
    assert len(per_query_tile_selection) == active_query_tiles

    lut = torch.zeros((batch, heads, q_blocks, kv_blocks), dtype=torch.int32, device=device)
    valid = torch.zeros((batch, heads, q_blocks), dtype=torch.int32, device=device)
    mask = torch.full((batch, 1, seq_len, seq_len), -1.0e9, dtype=torch.float32, device=device)

    for qblk in range(q_blocks):
        qtile = min(qblk, active_query_tiles - 1)
        selected_blocks = per_query_tile_selection[qtile]
        previous = 0
        for i, selected in enumerate(selected_blocks):
            lut[..., qblk, i] = selected if i == 0 else (selected - previous)
            previous = selected
        valid[..., qblk] = len(selected_blocks)

    for qtile, selected_blocks in enumerate(per_query_tile_selection):
        q_start = qtile * query_tile_tokens
        q_end = min(q_start + query_tile_tokens, seq_len)
        for qt in range(q_start, q_end):
            for selected in selected_blocks:
                k_start = selected * block_size
                k_end = min(k_start + block_size, seq_len)
                if not is_causal:
                    mask[:, :, qt : qt + 1, k_start:k_end] = 0.0
                else:
                    visible_end = min(k_end, qt + 1)
                    if visible_end > k_start:
                        mask[:, :, qt : qt + 1, k_start:visible_end] = 0.0

    return lut.contiguous(), valid.contiguous(), mask.contiguous()


def run_case(head_dim: int, block_size: int = 64, is_causal: bool = False) -> None:
    device = torch.device("xpu")
    batch = 1
    heads = 4
    seq_len = 256
    scale = 1.0 / math.sqrt(head_dim)
    query_tile_tokens = 128 if head_dim == 64 else 256
    kv_blocks = seq_len // block_size
    if not is_causal:
        per_query_tile_selection = [[0, 1], [1, 3]] if head_dim == 64 else [[0, 2]]
        case_name = "python_prefill"
    else:
        per_query_tile_selection = [[0, 1, 2], [1, 3]] if head_dim == 64 else [[0, 2, 3]]
        case_name = "python_prefill_causal"

    torch.manual_seed(2026 + head_dim)
    query = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device=device)
    key = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device=device)
    value = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device=device)

    q_i8, q_scale = quantize_qk(query, block_size)
    k_i8, k_scale = quantize_qk(key, block_size)
    lut, valid, dense_mask = build_sparse_metadata_and_mask(
        batch, heads, seq_len, block_size, query_tile_tokens, per_query_tile_selection, device, is_causal=is_causal
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
        tensor_layout="HND",
    )
    torch.xpu.synchronize()

    diff = (dense_out.float() - sparse_out.float()).abs()
    max_diff = float(diff.max().cpu())
    mean_diff = float(diff.mean().cpu())
    print(f"[sage_sparse][{case_name}] D={head_dim} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    if max_diff > 5e-3 or mean_diff > 5e-4:
        raise RuntimeError(f"sage_sparse python prefill mismatch for D={head_dim}, causal={is_causal}")


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
    run_case(64, is_causal=True)
    run_case(128, is_causal=True)


if __name__ == "__main__":
    main()
