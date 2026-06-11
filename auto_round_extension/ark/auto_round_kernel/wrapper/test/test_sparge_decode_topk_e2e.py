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
    if getattr(ark, "xpu_lib", None) is not None and hasattr(ark.xpu_lib, "sage_sparse_decode"):
        return
    candidates = sorted((REPO_PARENT / "auto_round_kernel" / "xbuild").glob("auto_round_kernel_xpu*.so"))
    if not candidates:
        raise RuntimeError("Unable to locate built XPU extension with sage_sparse_decode in xbuild/")
    ext_path = candidates[-1]
    spec = importlib.util.spec_from_file_location("auto_round_kernel_xpu", ext_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load extension spec from {ext_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["auto_round_kernel_xpu"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "sage_sparse_decode"):
        raise RuntimeError(f"Loaded extension does not expose sage_sparse_decode: {ext_path}")
    ark.xpu_lib = module


def _to_layout(tensor: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    if tensor_layout == "HND":
        return tensor.contiguous()
    return tensor.permute(0, 2, 1, 3).contiguous()


def _quantize_qk(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    hnd = tensor if tensor.ndim == 4 and tensor.shape[1] <= tensor.shape[2] else tensor
    B, H, S, D = hnd.shape
    out = torch.empty_like(hnd, dtype=torch.int8)
    scale = torch.empty((B, H, (S + block_size - 1) // block_size, 1), dtype=torch.float32, device=hnd.device)
    ark.xpu_lib.sage_dynamic_quant_layout(
        ark.get_stream(hnd),
        hnd.data_ptr(),
        0,
        out.data_ptr(),
        scale.data_ptr(),
        B,
        H,
        S,
        D,
        block_size,
        hnd.stride(2),
        hnd.stride(3),
        hnd.stride(1),
        hnd.stride(0),
    )
    return out, scale


def _make_block_map(batch: int, heads: int, total_blocks: int, selected: list[int], device: torch.device) -> torch.Tensor:
    block_map = torch.zeros((batch, heads, 1, total_blocks), dtype=torch.bool, device=device)
    block_map[..., selected] = True
    return block_map


def _run_low_level_case(head_dim: int, *, selected_blocks: list[int], tensor_layout: str) -> None:
    device = torch.device("xpu")
    batch = 1
    heads = 2
    seq_q = 1
    seq_cache = 256
    seq_cur = 1
    total_seq = seq_cache + seq_cur
    block = 64
    scale = 1.0 / math.sqrt(head_dim)

    seed = 8100 + head_dim + len(selected_blocks) + (13 if tensor_layout == "NHD" else 0)
    torch.manual_seed(seed)
    query_hnd = torch.randn((batch, heads, seq_q, head_dim), dtype=torch.float16, device=device)
    key_cache_hnd = torch.randn((batch, heads, seq_cache, head_dim), dtype=torch.float16, device=device)
    value_cache_hnd = torch.randn((batch, heads, seq_cache, head_dim), dtype=torch.float16, device=device)
    key_hnd = torch.randn((batch, heads, seq_cur, head_dim), dtype=torch.float16, device=device)
    value_hnd = torch.randn((batch, heads, seq_cur, head_dim), dtype=torch.float16, device=device)

    query_i8_hnd, qscale = _quantize_qk(query_hnd, block)
    key_cache_i8_hnd, kscale_cache = _quantize_qk(key_cache_hnd, block)
    key_i8_hnd, kscale = _quantize_qk(key_hnd, block)
    key_total_i8_hnd = torch.cat([key_cache_i8_hnd, key_i8_hnd], dim=2)
    value_total_hnd = torch.cat([value_cache_hnd, value_hnd], dim=2)
    kscale_total = torch.cat([kscale_cache, kscale], dim=2)

    query_i8 = _to_layout(query_i8_hnd, tensor_layout)
    key_i8 = _to_layout(key_i8_hnd, tensor_layout)
    value = _to_layout(value_hnd, tensor_layout)
    key_cache_i8 = _to_layout(key_cache_i8_hnd, tensor_layout)
    value_cache = _to_layout(value_cache_hnd, tensor_layout)

    total_blocks = (total_seq + block - 1) // block
    block_map = _make_block_map(batch, heads, total_blocks, selected_blocks, device)
    lut, valid = ark._block_map_lut_torch(block_map)
    sparse_out = ark.sage_sparse_decode(
        query_i8,
        key_i8,
        value,
        key_cache_i8,
        value_cache,
        lut,
        valid,
        is_causal=False,
        scale=scale,
        quant_block_size=block,
        qscale=qscale,
        kscale=kscale,
        kscale_cache=kscale_cache,
        tensor_layout=tensor_layout,
    )

    dense_out = ark.sage_sparse(
        _to_layout(query_i8_hnd, tensor_layout),
        _to_layout(key_total_i8_hnd, tensor_layout),
        _to_layout(value_total_hnd, tensor_layout),
        lut,
        valid,
        is_causal=False,
        scale=scale,
        quant_block_size=block,
        qscale=qscale,
        kscale=kscale_total,
        tensor_layout=tensor_layout,
    )
    torch.xpu.synchronize()
    assert torch.isfinite(dense_out).all()
    assert torch.isfinite(sparse_out).all()
    diff = (dense_out.float() - sparse_out.float()).abs()
    max_diff = float(diff.max().cpu())
    mean_diff = float(diff.mean().cpu())
    case_name = "all_selected" if len(selected_blocks) == total_blocks else "partial"
    print(f"[sparge_decode][{case_name}_{tensor_layout.lower()}] D={head_dim} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    if max_diff > 3e-2 or mean_diff > 7e-3:
        raise RuntimeError(f"low-level sparse decode mismatch for {case_name}, D={head_dim}, layout={tensor_layout}")


def _run_preprocess_case(head_dim: int, *, tensor_layout: str) -> None:
    device = torch.device("xpu")
    batch = 1
    heads = 2
    seq_q = 1
    seq_cache = 256
    seq_cur = 1
    scale = 1.0 / math.sqrt(head_dim)

    seed = 9200 + head_dim + (19 if tensor_layout == "NHD" else 0)
    torch.manual_seed(seed)
    query_hnd = torch.randn((batch, heads, seq_q, head_dim), dtype=torch.float16, device=device)
    key_cache_hnd = torch.randn((batch, heads, seq_cache, head_dim), dtype=torch.float16, device=device)
    value_cache_hnd = torch.randn((batch, heads, seq_cache, head_dim), dtype=torch.float16, device=device)
    key_hnd = torch.randn((batch, heads, seq_cur, head_dim), dtype=torch.float16, device=device)
    value_hnd = torch.randn((batch, heads, seq_cur, head_dim), dtype=torch.float16, device=device)

    query = _to_layout(query_hnd, tensor_layout)
    key = _to_layout(key_hnd, tensor_layout)
    value = _to_layout(value_hnd, tensor_layout)
    key_cache = _to_layout(key_cache_hnd, tensor_layout)
    value_cache = _to_layout(value_cache_hnd, tensor_layout)

    sparse_out, meta = ark.sparge_sage2_decode_meansim_topk_xpu(
        query,
        key,
        value,
        key_cache,
        value_cache,
        is_causal=True,
        scale=scale,
        smooth_k=True,
        simthreshd1=-1.0,
        topk=0.5,
        attention_sink=False,
        tensor_layout=tensor_layout,
        return_metadata=True,
    )
    direct_out = ark.sage_sparse_decode(
        meta["query_i8"],
        meta["key_i8"],
        value,
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
        tensor_layout=tensor_layout,
    )
    torch.xpu.synchronize()
    assert torch.isfinite(direct_out).all()
    assert torch.isfinite(sparse_out).all()
    diff = (direct_out.float() - sparse_out.float()).abs()
    max_diff = float(diff.max().cpu())
    mean_diff = float(diff.mean().cpu())
    print(f"[sparge_decode][preprocess_{tensor_layout.lower()}] D={head_dim} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    if max_diff > 3e-2 or mean_diff > 5e-3:
        raise RuntimeError(f"preprocess sparse decode mismatch for D={head_dim}, layout={tensor_layout}")


def main() -> None:
    ensure_sparse_binding()
    if not torch.xpu.is_available():
        raise RuntimeError("XPU device is required")
    for tensor_layout in ("HND", "NHD"):
        _run_low_level_case(64, selected_blocks=[0, 1, 2, 3, 4], tensor_layout=tensor_layout)
        _run_low_level_case(128, selected_blocks=[0, 1, 2, 3, 4], tensor_layout=tensor_layout)
        _run_low_level_case(64, selected_blocks=[2, 3, 4], tensor_layout=tensor_layout)
        _run_low_level_case(128, selected_blocks=[2, 3, 4], tensor_layout=tensor_layout)
        _run_preprocess_case(64, tensor_layout=tensor_layout)
        _run_preprocess_case(128, tensor_layout=tensor_layout)


if __name__ == "__main__":
    main()
