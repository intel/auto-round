# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import importlib
import os
import sys
import time
from importlib import metadata
from pathlib import Path

import auto_round_kernel as ark
import torch
from ut_utils import gen_weis8

M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
N = 16384
K = 4096
BLOCKSIZE = 32
DTYPE = torch.float16
DEVICE = "xpu"
COMPUTE_TYPE = "int8"
WEIGHT_TYPE = "int4"
SCALE_TYPE = "fp16"
ASYM = False
WARMUP_LIMIT = 1000


def _print_config_types():
    config_values = {
        "M_VALUES": M_VALUES,
        "N": N,
        "K": K,
        "BLOCKSIZE": BLOCKSIZE,
        "DTYPE": DTYPE,
        "DEVICE": DEVICE,
        "COMPUTE_TYPE": COMPUTE_TYPE,
        "WEIGHT_TYPE": WEIGHT_TYPE,
        "SCALE_TYPE": SCALE_TYPE,
    }
    print("\n=== Config types ===")
    for name, value in config_values.items():
        print(f"{name}: {value}")

def _sync_xpu():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def _env_enabled(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def _ark_expected_route(m, n=N, k=K, blocksize=BLOCKSIZE):
    if m <= 1:
        return "bestla_s4_gemv"
    if not _env_enabled("ARK_WOQ_DPAS_S4", True):
        return "woqgemm_s8(unpack_s4_to_s8)"
    if m > 128:
        return "woqgemm_s8(unpack_s4_to_s8)"
    if n % 64 != 0 or (k & 1) != 0 or blocksize <= 0:
        return "woqgemm_s8(unpack_s4_to_s8)"
    if k % blocksize != 0:
        return "woqgemm_s8(unpack_s4_to_s8)"
    if blocksize < 32 or blocksize > 4096 or not _is_power_of_two(blocksize):
        return "woqgemm_s8(unpack_s4_to_s8)"
    return "woq_s4_dpas"


def _has_torch_int4_op():
    return hasattr(torch.ops, "_xpu_C") and hasattr(torch.ops._xpu_C, "int4_gemm_w4a16")


def _prepare_xpu_kernel_import_path():
    repo_root = Path(__file__).resolve().parents[4]
    xpu_kernel_src = repo_root / "vllm-xpu-kernels"
    package_dir = xpu_kernel_src / "vllm_xpu_kernels"
    if not package_dir.exists():
        return

    try:
        local_src = xpu_kernel_src.resolve()
    except OSError:
        return

    if any(package_dir.glob("_xpu_C*.so")):
        local_src_str = str(local_src)
        if local_src_str not in sys.path:
            sys.path.insert(0, local_src_str)
        return

    sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != local_src]


def _installed_xpu_kernel_extensions():
    try:
        files = metadata.files("vllm-xpu-kernels") or []
    except metadata.PackageNotFoundError:
        return "package metadata not found"

    extensions = sorted(str(file) for file in files if str(file).endswith((".so", ".pyd")) or "_xpu_C" in str(file))
    return ", ".join(extensions[:16]) if extensions else "no extension files found"


def _register_xpu_ops():
    if _has_torch_int4_op():
        return True

    _prepare_xpu_kernel_import_path()
    import_errors = []
    for module_name in ("vllm_xpu_kernels._xpu_C", "vllm.platforms.xpu", "vllm._xpu_ops"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            import_errors.append(f"{module_name}: {exc}")
        if _has_torch_int4_op():
            return True

    print("\n[torch int4_gemm_w4a16] skip: cannot register torch.ops._xpu_C.int4_gemm_w4a16")
    for error in import_errors:
        print(f"  {error}")
    print(f"  vllm-xpu-kernels extension files: {_installed_xpu_kernel_extensions()}")
    return False


def _rand_packed_int4(size, dtype=torch.int32, device=DEVICE):
    rand = torch.randint(-128, 128, [size // 2], device=device).to(torch.int8)
    return rand.view(dtype=dtype)


def _torch_int4_case(n=N, k=K, blocksize=BLOCKSIZE, dtype=DTYPE, device=DEVICE):
    weight = _rand_packed_int4(k * n, torch.int32, device).reshape(k // 8, n)
    weight_nt = weight.transpose(0, 1).contiguous().transpose(0, 1)
    scales = torch.rand([k // blocksize, n], device=device, dtype=dtype) / 300 + 0.002
    zero_points = torch.tensor([8], device=device, dtype=torch.int8)
    bias = torch.randn(n, device=device, dtype=dtype)
    return weight_nt, scales, zero_points, bias


def _ark_case(m, n=N, k=K, blocksize=BLOCKSIZE, dtype=DTYPE, device=DEVICE):
    torch.manual_seed(0)
    raw_s8_wei = gen_weis8(WEIGHT_TYPE, device, k, n)
    scales = torch.rand(k // blocksize, n, dtype=dtype, device=device) / 300 + 0.002
    bias = torch.randn(1, n, dtype=dtype, device=device)
    zp = torch.Tensor()

    packw = ark.repack_quantized_weight(raw_s8_wei, scales, zp, blocksize, COMPUTE_TYPE, WEIGHT_TYPE, SCALE_TYPE, ASYM)
    revert_wei_t = ark.unpack_weight(packw, dtype, n, k, blocksize, COMPUTE_TYPE, WEIGHT_TYPE, SCALE_TYPE, ASYM)
    revert_wei = revert_wei_t.t()
    ref_weight = raw_s8_wei.to(dtype) * scales.repeat_interleave(repeats=blocksize, dim=0)
    assert torch.allclose(revert_wei, ref_weight)

    activation = torch.randn(m, k, dtype=dtype, device=device) - 0.5
    ref_c = torch.matmul(activation, revert_wei) + bias
    return activation, packw, bias, ref_c


def _runs_for_m(m):
    return 1000


def _batch_for_m(m):
    return 64 if m == 1 else 8


def _warmup_for_runs(runs):
    return min(runs, WARMUP_LIMIT)


def _repeat_ark_blob(packw, batch):
    return packw.unsqueeze(0).repeat(batch, 1)


def _repeat_activation(activation, batch):
    return activation.unsqueeze(0).repeat(batch, 1, 1)


def _repeat_nt_weight(weight, batch):
    weight_nk = weight.transpose(0, 1).contiguous()
    return weight_nk.unsqueeze(0).repeat(batch, 1, 1).transpose(1, 2)


def _memory_bytes(m, n=N, k=K, blocksize=BLOCKSIZE, dtype=DTYPE):
    element_size = torch.empty((), dtype=dtype).element_size()
    return m * k * element_size + m * n * element_size + n * k // 2 + (k // blocksize) * n * element_size


def _benchmark_loop(call, runs, warmup):
    output = None
    for i in range(warmup):
        output = call(i)
    _sync_xpu()

    start = time.perf_counter()
    for i in range(runs):
        output = call(i)
    _sync_xpu()
    return output, (time.perf_counter() - start) / runs


def _print_perf(m, batch, warmup, runs, op_name, dur, route=None):
    ops = m * N * K * 2
    memsize = _memory_bytes(m)
    route_text = f", route={route}" if route is not None else ""
    print(
        f"\n m={m}, n={N}, k={K}, blocksize={BLOCKSIZE}, batch={batch}, warmup={warmup}, runs={runs}, op={op_name}{route_text}"
    )
    print(f"[Performance] Time: {dur * 1000:.4f} ms")
    print(f"              GFLOPS: {ops / dur / 1e9:.2f}")
    print(f"              Bandwidth: {memsize / dur / 1e9:.2f} GB/s")


def run_ark_woqgemm():
    print("\n=== ARK woqgemm ===")
    print("Timed loops use the same warmup/runs/batch policy as the torch oneDNN path.")
    for m in M_VALUES:
        route = _ark_expected_route(m)
        print(f"\n[ARK route] m={m}: {route}")
        activation, packw, bias, ref_c = _ark_case(m)
        runs = _runs_for_m(m)
        batch = _batch_for_m(m)
        warmup = _warmup_for_runs(runs)
        activation_set = _repeat_activation(activation, batch)
        packw_set = _repeat_ark_blob(packw, batch)
        output_set = torch.empty(batch, m, N, dtype=activation.dtype, device=activation.device)

        def call(i):
            idx = i % batch
            return ark.woqgemm(
                activation_set[idx],
                packw_set[idx],
                bias,
                N,
                K,
                BLOCKSIZE,
                COMPUTE_TYPE,
                WEIGHT_TYPE,
                SCALE_TYPE,
                ASYM,
                out=output_set[idx],
            )

        output, dur = _benchmark_loop(call, runs, warmup)
        diff = abs(ref_c - output)
        print(
            f"  Max Diff: {diff.max().item():.6f}, Mean Diff: {diff.mean().item():.6f}, "
            f"ref mean:{ref_c.mean():.6f}, OUT mean:{output.mean():.6f}"
        )
        assert torch.allclose(output, ref_c, rtol=0.1, atol=2.0)
        _print_perf(m, batch, warmup, runs, "ark.woqgemm", dur, route)


def run_torch_int4_gemm_w4a16():
    print("\n=== torch.ops._xpu_C.int4_gemm_w4a16 ===")
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        print("[torch int4_gemm_w4a16] skip: no XPU device")
        return
    if not _register_xpu_ops():
        return

    torch.manual_seed(0)
    weight, scales, zero_points, bias = _torch_int4_case()
    for m in M_VALUES:
        activation = torch.randn(m, K, dtype=DTYPE, device=DEVICE) - 0.5
        runs = _runs_for_m(m)
        batch = _batch_for_m(m)
        warmup = _warmup_for_runs(runs)
        activation_set = _repeat_activation(activation, batch)
        weight_set = _repeat_nt_weight(weight, batch)

        def call(i):
            idx = i % batch
            return torch.ops._xpu_C.int4_gemm_w4a16(
                activation_set[idx], weight_set[idx], bias, scales, zero_points, BLOCKSIZE, None
            )

        try:
            _, dur = _benchmark_loop(call, runs, warmup)
        except Exception as exc:
            print(f"\n[torch int4_gemm_w4a16] m={m} skip: {exc}")
            continue

        _print_perf(m, batch, warmup, runs, "torch.ops._xpu_C.int4_gemm_w4a16", dur, "oneDNN_w4a16_int4")


if __name__ == "__main__":
    _print_config_types()
    run_ark_woqgemm()
    run_torch_int4_gemm_w4a16()
