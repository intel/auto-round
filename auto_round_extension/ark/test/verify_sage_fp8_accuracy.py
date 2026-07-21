# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify accuracy of ARK Sage FP8 attention kernel against PyTorch reference."""

import argparse
import math
import sys
from pathlib import Path

import torch

_script_dir = Path(__file__).resolve().parent
_ark_root = _script_dir.parent
if str(_ark_root) not in sys.path:
    sys.path.insert(0, str(_ark_root))

import auto_round_kernel as ark


def sync_stage(name):
    print(f"[stage] {name}", flush=True)
    torch.xpu.synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Verify ARK Sage FP8 attention accuracy")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len-qo", type=int, default=256)
    parser.add_argument("--seq-len-kv", type=int, default=256)
    parser.add_argument("--num-heads-q", type=int, default=4)
    parser.add_argument("--num-heads-kv", type=int, default=4)
    parser.add_argument("--head-dim", type=int, choices=(64, 96, 128, 192), default=64)
    parser.add_argument("--causal", action="store_true", help="Enable causal masking")
    parser.add_argument("--iterations", type=int, default=10, help="Number of test iterations")
    parser.add_argument("--atol", type=float, default=0.08, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=0.08, help="Relative tolerance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--trace", action="store_true", help="Print Sage FP8 intermediate error statistics")
    parser.add_argument("--compare-sagev1", action="store_true", help="Compare FP8 output directly with SageV1")
    return parser.parse_args()


def compute_reference(query, key, value, scale, is_causal, enable_gqa):
    """Compute attention using PyTorch's reference implementation (bf16)."""
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=scale,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )


def compute_ark(query, key, value, scale, is_causal, enable_gqa):
    """Compute attention using ARK Sage FP8 kernel with stage-level synchronization.

    Uses the high-level sage_fp8 API which handles K/V smoothing and V mean
    restoration internally for correct FP8 attention computation.
    """
    fused_backend = hasattr(ark.xpu_lib, "sage_fp8_fused")
    print(
        "[stage] ARK FP8 backend: "
        + ("native fused smooth+quant with device scales" if fused_backend else "Python preprocess fallback"),
        flush=True,
    )
    output = ark.sage_fp8(
        query,
        key,
        value,
        scale=scale,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
        smooth_k=True,
        smooth_v=True,
    )
    sync_stage("ARK FP8 attention")
    return output


def analyze_error(actual, expected):
    """Analyze error between actual and expected outputs."""
    actual_float = actual.float()
    diff = actual_float - expected
    abs_diff = diff.abs()
    rel_diff = torch.where(expected.abs() > 1e-8, abs_diff / expected.abs(), torch.zeros_like(abs_diff))

    return {
        "max_abs_err": abs_diff.max().item(),
        "mean_abs_err": abs_diff.mean().item(),
        "max_rel_err": rel_diff.max().item(),
        "mean_rel_err": rel_diff.mean().item(),
        "num_finite": actual_float.isfinite().sum().item(),
        "total": actual_float.numel(),
    }


def print_error_stats(name, actual, expected):
    actual_float = actual.float()
    abs_diff = (actual_float - expected.float()).abs()
    max_abs_err, mean_abs_err, actual_absmax = torch.stack(
        (abs_diff.max(), abs_diff.mean(), actual_float.abs().max())
    ).cpu().tolist()
    print(
        f"[trace] {name}: max_abs={max_abs_err:.7g} "
        f"mean_abs={mean_abs_err:.7g} "
        f"actual_absmax={actual_absmax:.7g}",
        flush=True,
    )


def trace_intermediates(query, key, value, scale):
    """Compare every host-visible Sage FP8 intermediate with its FP32 reference."""
    print("[trace] begin", flush=True)
    key_smoothed, key_mean = ark._smooth_sequence_mean(key, "HND")
    value_smoothed, value_mean = ark._smooth_sequence_mean(value, "HND")
    sync_stage("trace smoothing")
    key_mean_seq = key_mean.unsqueeze(2)
    value_mean_seq = value_mean.unsqueeze(2)
    print_error_stats("K sequence mean", key_mean_seq, key.float().mean(dim=2, keepdim=True))
    print_error_stats("V sequence mean", value_mean_seq, value.float().mean(dim=2, keepdim=True))
    print_error_stats("K smoothing", key_smoothed, key.float() - key_mean_seq)
    print_error_stats("V smoothing", value_smoothed, value.float() - value_mean_seq)

    query_fp8, qscale = ark.quantize_fp8(query)
    key_fp8, kscale = ark.quantize_fp8(key_smoothed)
    value_fp8, vscale = ark.quantize_fp8(value_smoothed)
    query_dequant = query_fp8.float() * qscale
    key_dequant = key_fp8.float() * kscale
    value_dequant = value_fp8.float() * vscale
    sync_stage("trace quantization")
    print(f"[trace] scales: q={qscale.item():.7g} k={kscale.item():.7g} v={vscale.item():.7g}", flush=True)
    print_error_stats("Q dequantization", query_dequant, query)
    print_error_stats("K dequantization", key_dequant, key_smoothed)
    print_error_stats("V dequantization", value_dequant, value_smoothed)

    logits = torch.matmul(query.float(), key_smoothed.float().transpose(-1, -2)) * scale
    logits_fp8 = torch.matmul(query_dequant, key_dequant.transpose(-1, -2)) * scale
    probabilities = torch.softmax(logits, dim=-1)
    probabilities_fp8 = torch.softmax(logits_fp8, dim=-1)
    pv_reference = torch.matmul(probabilities, value_smoothed.float())
    pv_fp8_reference = torch.matmul(probabilities_fp8, value_dequant)
    sync_stage("trace FP32 QK and PV references")
    print_error_stats("QK logits after FP8", logits_fp8, logits)
    print_error_stats("QK softmax after FP8", probabilities_fp8, probabilities)
    print_error_stats("PV after FP8", pv_fp8_reference, pv_reference)

    print("[stage] launch raw fp8_fa2", flush=True)
    kernel_raw = ark.fp8_fa2(
        query_fp8,
        key_fp8,
        value_fp8,
        qscale=qscale,
        kscale=kscale,
        vscale=vscale,
        scale=scale,
        tensor_layout="HND",
    ).float()
    sync_stage("trace raw fp8_fa2")
    print_error_stats("kernel raw output vs FP8 PV reference", kernel_raw, pv_fp8_reference)
    print_error_stats("kernel raw output vs FP32 PV reference", kernel_raw, pv_reference)
    print("[stage] launch fused V mean restore", flush=True)
    kernel_fused = ark.fp8_fa2(
        query_fp8,
        key_fp8,
        value_fp8,
        qscale=qscale,
        kscale=kscale,
        vscale=vscale,
        vmean=value_mean,
        scale=scale,
        tensor_layout="HND",
    ).float()
    sync_stage("trace fused V mean restore")
    print_error_stats("kernel fused output", kernel_fused, pv_fp8_reference + value_mean_seq)
    print_error_stats("fused vs raw plus V mean", kernel_fused, kernel_raw + value_mean_seq)
    print("[trace] end", flush=True)
    return kernel_fused


def test_configuration(args, desc):
    """Test a specific configuration and return results."""
    torch.manual_seed(args.seed)

    query = torch.randn(
        args.batch, args.num_heads_q, args.seq_len_qo, args.head_dim,
        dtype=torch.bfloat16, device="xpu"
    ) * 0.5
    sync_stage("allocated query")
    key = torch.randn(
        args.batch, args.num_heads_kv, args.seq_len_kv, args.head_dim,
        dtype=torch.bfloat16, device="xpu"
    ) * 0.5
    sync_stage("allocated key")
    value = torch.randn_like(key) * 0.5
    sync_stage("allocated value")

    scale = 1.0 / math.sqrt(args.head_dim)
    enable_gqa = args.num_heads_q != args.num_heads_kv

    # Compute reference and ARK outputs
    expected = compute_reference(query, key, value, scale, args.causal, enable_gqa)
    sync_stage("PyTorch SDPA reference")
    traced_fp8_oracle = None
    if args.trace:
        if args.causal or enable_gqa:
            print("[trace] Intermediate QK/PV reference trace supports the non-causal, non-GQA configuration only.")
        else:
            traced_fp8_oracle = trace_intermediates(query, key, value, scale)
    actual_fp8 = compute_ark(query, key, value, scale, args.causal, enable_gqa)
    if traced_fp8_oracle is not None:
        print_error_stats("native fused preprocess vs traced FP8 oracle", actual_fp8, traced_fp8_oracle)
    sagev1_output = None
    if args.compare_sagev1:
        if not hasattr(ark.xpu_lib, "sagev1"):
            print("[sagev1] skipped: the native XPU extension does not export a SageV1 kernel", flush=True)
        elif args.head_dim not in (64, 128):
            print("[sagev1] skipped: SageV1 supports head dimensions 64 and 128 only")
        else:
            sagev1_output = ark.sagev1(
                query, key, value, scale=scale, is_causal=args.causal,
                enable_gqa=enable_gqa, smooth_k=True,
            )
            sync_stage("SageV1 attention")
            print_error_stats("Sage FP8 vs SageV1", actual_fp8, sagev1_output.float())

    # Convert both outputs to float32 for accurate comparison
    actual = actual_fp8.float()
    expected_fp32 = expected.float()

    # Analyze error
    stats = analyze_error(actual, expected_fp32)

    # Check if within tolerance
    try:
        torch.testing.assert_close(actual, expected_fp32, atol=args.atol, rtol=args.rtol)
        passed = True
        error_msg = None
    except AssertionError as e:
        passed = False
        error_msg = str(e)

    return {
        "desc": desc,
        "stats": stats,
        "passed": passed,
        "error_msg": error_msg,
        "sagev1_stats": analyze_error(actual, sagev1_output.float()) if sagev1_output is not None else None,
    }


def main():
    args = parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError("XPU device is required")
    if ark.xpu_lib is None:
        raise RuntimeError("ARK extension not loaded")
    if not hasattr(ark, "sage_fp8"):
        raise RuntimeError("ARK does not export sage_fp8 function")

    print("=" * 70)
    print("ARK Sage FP8 Attention Accuracy Test")
    print("=" * 70)
    print(f"Test configuration:")
    print(f"  Batch size:       {args.batch}")
    print(f"  Seq len (Q):      {args.seq_len_qo}")
    print(f"  Seq len (KV):     {args.seq_len_kv}")
    print(f"  Num heads (Q):   {args.num_heads_q}")
    print(f"  Num heads (KV):  {args.num_heads_kv}")
    print(f"  Head dim:        {args.head_dim}")
    print(f"  Causal:          {args.causal}")
    print(f"  GQA enabled:     {args.num_heads_q != args.num_heads_kv}")
    print(f"  Tolerance:       atol={args.atol}, rtol={args.rtol}")
    print(f"  Iterations:      {args.iterations}")
    print(f"  Seed:            {args.seed}")
    print("=" * 70)

    # Run multiple iterations for robustness
    all_results = []
    for i in range(args.iterations):
        desc = f"Iteration {i + 1}/{args.iterations}"
        result = test_configuration(args, desc)
        all_results.append(result)

    # Print summary
    print("\nResults Summary:")
    print("-" * 70)
    passed_count = sum(1 for r in all_results if r["passed"])
    failed_count = len(all_results) - passed_count

    for result in all_results:
        status = "PASS" if result["passed"] else "FAIL"
        stats = result["stats"]
        print(f"\n{result['desc']}: {status}")
        print(f"  Max Abs Error:  {stats['max_abs_err']:.6f}")
        print(f"  Mean Abs Error: {stats['mean_abs_err']:.6f}")
        print(f"  Max Rel Error:  {stats['max_rel_err']:.6f}")
        print(f"  Mean Rel Error: {stats['mean_rel_err']:.6f}")
        print(f"  Finite values:  {stats['num_finite']}/{stats['total']}")
        if result["sagev1_stats"] is not None:
            sagev1_stats = result["sagev1_stats"]
            print(f"  vs SageV1 max/mean abs: {sagev1_stats['max_abs_err']:.6f}/{sagev1_stats['mean_abs_err']:.6f}")

        if not result["passed"]:
            print(f"  Error: {result['error_msg'][:200]}...")

    # Final verdict
    print("\n" + "=" * 70)
    if failed_count == 0:
        print(f"ALL TESTS PASSED ({passed_count}/{len(all_results)})")
        return 0
    else:
        print(f"TESTS FAILED ({failed_count}/{len(all_results)})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
