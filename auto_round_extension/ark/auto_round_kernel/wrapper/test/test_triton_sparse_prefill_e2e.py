import math
import os
import sys
import importlib.util
from pathlib import Path

import torch


REPO_PARENT = Path(__file__).resolve().parents[3]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import auto_round_kernel as ark
from auto_round_kernel.triton_sparse_attention_xpu import triton_sparse_prefill_attention


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


def run_case(head_dim: int, *, topk: float, tensor_layout: str) -> None:
    device = torch.device("xpu")
    batch = 1
    num_heads = 4
    seq_len = 256
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(8100 + head_dim + int(topk * 100) + (17 if tensor_layout == "NHD" else 0))
    query_hnd = torch.randn((batch, num_heads, seq_len, head_dim), dtype=torch.bfloat16, device=device)
    key_hnd = torch.randn((batch, num_heads, seq_len, head_dim), dtype=torch.bfloat16, device=device)
    value_hnd = torch.randn((batch, num_heads, seq_len, head_dim), dtype=torch.bfloat16, device=device)

    query = _to_layout(query_hnd, tensor_layout)
    key = _to_layout(key_hnd, tensor_layout)
    value = _to_layout(value_hnd, tensor_layout)
    meta = ark.sparge_preprocess_topk(
        query,
        key,
        is_causal=False,
        smooth_k=True,
        simthreshd1=-1.0,
        topk=topk,
        attention_sink=False,
        tensor_layout=tensor_layout,
    )

    sycl_out = ark.sage_sparse(
        meta["query_i8"],
        meta["key_i8"],
        value,
        meta["lut"],
        meta["valid_block_num"],
        is_causal=False,
        scale=scale,
        quant_block_size=meta["quant_block_size"],
        qscale=meta["qscale"],
        kscale=meta["kscale"],
        tensor_layout=tensor_layout,
    )
    triton_out_hnd = triton_sparse_prefill_attention(
        ark._to_hnd(meta["query_i8"], tensor_layout),
        ark._to_hnd(meta["key_i8"], tensor_layout),
        ark._to_hnd(value, tensor_layout),
        meta["lut"],
        meta["valid_block_num"],
        qscale=meta["qscale"],
        kscale=meta["kscale"],
        scale=scale,
        quant_block_size=meta["quant_block_size"],
    )
    triton_out = ark._from_hnd(triton_out_hnd, tensor_layout)

    previous_backend = os.environ.get("SAGE_ATTN_XPU_SPARSE_KERNEL_BACKEND")
    os.environ["SAGE_ATTN_XPU_SPARSE_KERNEL_BACKEND"] = "triton_xpu_kernel"
    try:
        wrapper_out = ark.sparge_sage2_attn_meansim_topk_xpu(
            query,
            key,
            value,
            is_causal=False,
            scale=scale,
            smooth_k=True,
            simthreshd1=-1.0,
            topk=topk,
            attention_sink=False,
            tensor_layout=tensor_layout,
        )
    finally:
        if previous_backend is None:
            os.environ.pop("SAGE_ATTN_XPU_SPARSE_KERNEL_BACKEND", None)
        else:
            os.environ["SAGE_ATTN_XPU_SPARSE_KERNEL_BACKEND"] = previous_backend

    torch.xpu.synchronize()

    case_name = f"D{head_dim}_topk{topk}_{tensor_layout.lower()}"
    diff = (sycl_out.float() - triton_out.float()).abs()
    max_diff = float(diff.max().cpu())
    mean_diff = float(diff.mean().cpu())
    print(f"[triton_sparse_prefill][{case_name}] sycl_vs_triton max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    if max_diff > 2e-2 or mean_diff > 2e-3:
        raise RuntimeError(f"Triton sparse prefill mismatch for {case_name}")

    wrapper_diff = (wrapper_out.float() - triton_out.float()).abs()
    wrapper_max_diff = float(wrapper_diff.max().cpu())
    wrapper_mean_diff = float(wrapper_diff.mean().cpu())
    print(
        f"[triton_sparse_prefill][{case_name}] wrapper_vs_triton "
        f"max_diff={wrapper_max_diff:.6f} mean_diff={wrapper_mean_diff:.6f}"
    )
    if wrapper_max_diff > 2e-2 or wrapper_mean_diff > 2e-3:
        raise RuntimeError(f"Triton sparse wrapper mismatch for {case_name}")


def main() -> None:
    ensure_sparse_binding()
    if not torch.xpu.is_available():
        raise RuntimeError("XPU device is required")
    for tensor_layout in ("HND", "NHD"):
        run_case(64, topk=1.0, tensor_layout=tensor_layout)
        run_case(128, topk=1.0, tensor_layout=tensor_layout)
        run_case(64, topk=0.5, tensor_layout=tensor_layout)
        run_case(128, topk=0.5, tensor_layout=tensor_layout)


if __name__ == "__main__":
    main()
