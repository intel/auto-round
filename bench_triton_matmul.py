"""
Benchmark: fused dequant+matmul Triton kernel vs. the reference
"dequant-then-cuBLAS" path in auto_round_extension/triton/triton_utils/dequant.py.

Run:
    python bench_triton_matmul.py
    python bench_triton_matmul.py --dtype fp16 --sym
    python bench_triton_matmul.py --bits 4 --group_size 128
"""
import argparse

import torch

from auto_round_extension.triton.triton_utils.dequant import (
    quant_matmul_248 as ref_dequant_matmul,  # dequant -> cuBLAS (NOT fused)
    dequant248,
)
from auto_round_extension.triton.triton_utils.kernels import (
    quant_matmul_248 as ref_fused_matmul,    # existing fused kernel (w/ g_idx)
)
from auto_round_extension.triton.triton_utils.fused_matmul import (
    fused_quant_matmul,
    fused_quant_gemv,
    auto_quant_matmul,
    fast_dequant248,
)


def make_fake_qweight(K, N, bits, group_size, sym, device, dtype, sym_zp=None):
    pack = 32 // bits
    maxq = (1 << bits) - 1
    G = K // group_size
    if sym and sym_zp is None:
        sym_zp = 1 << (bits - 1)

    w_int = torch.randint(0, maxq + 1, (K, N), device=device, dtype=torch.int32)
    qweight = torch.zeros((K // pack, N), dtype=torch.int32, device=device)
    for i in range(pack):
        qweight |= (w_int[i::pack] & maxq) << (i * bits)

    scales = (torch.rand((G, N), device=device, dtype=dtype) * 0.02 + 0.005)

    if sym:
        z_int = torch.full((G, N), sym_zp, dtype=torch.int32, device=device)
    else:
        z_int = torch.randint(0, maxq + 1, (G, N), dtype=torch.int32, device=device)

    qzeros = torch.zeros((G, N // pack), dtype=torch.int32, device=device)
    for i in range(pack):
        qzeros |= (z_int[:, i::pack] & maxq) << (i * bits)

    g_idx = (torch.arange(K, device=device, dtype=torch.int32) // group_size)
    return qweight, scales, qzeros, g_idx, sym_zp


def bench_fn(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


import sys
import traceback

_ERR_PRINTED = set()


def _safe_bench(label, fn):
    """Return (median_ms, status, output_or_None). status in {'OK','ERR'}.

    Prints the full traceback to stderr the FIRST time each (label, error)
    combination is seen, so failures are diagnosable without overwhelming
    the table.
    """
    try:
        out = fn()
        ms = bench_fn(fn)
        return ms, "OK", out
    except Exception as e:
        msg_first_line = str(e).splitlines()[0][:60] if str(e) else type(e).__name__
        key = (label, msg_first_line)
        if key not in _ERR_PRINTED:
            _ERR_PRINTED.add(key)
            print(f"\n[BENCH ERR] label={label}: {type(e).__name__}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("", file=sys.stderr)
        return float("nan"), f"ERR:{msg_first_line}", None


def run_case(M, K, N, bits, group_size, sym, dtype, device, check=True):
    qweight, scales, qzeros, g_idx, sym_zp = make_fake_qweight(
        K, N, bits, group_size, sym, device, dtype
    )
    x = torch.randn((M, K), device=device, dtype=dtype) * 0.1
    maxq = (1 << bits) - 1

    def deq_call():
        return ref_dequant_matmul(x, qweight, scales, qzeros, g_idx, bits)

    def fused_old_call():
        return ref_fused_matmul(x, qweight, scales, qzeros, g_idx, bits, maxq)

    if sym:
        def fused_new_call():
            return fused_quant_matmul(
                x, qweight, scales, qzeros=None,
                bits=bits, group_size=group_size, sym=True, sym_zp=sym_zp,
            )

        def hybrid_call():
            return auto_quant_matmul(
                x, qweight, scales, qzeros=None, g_idx=g_idx,
                bits=bits, group_size=group_size, sym=True, sym_zp=sym_zp,
            )
    else:
        def fused_new_call():
            return fused_quant_matmul(
                x, qweight, scales, qzeros,
                bits=bits, group_size=group_size, sym=False,
            )

        def hybrid_call():
            return auto_quant_matmul(
                x, qweight, scales, qzeros=qzeros, g_idx=g_idx,
                bits=bits, group_size=group_size, sym=False,
            )

    w_fp16 = dequant248(qweight, scales, qzeros, g_idx, bits, input_dtype=dtype)

    def fp16_call():
        return x @ w_fp16

    # fast dequant + cuBLAS (the optimised prefill path)
    def fast_deq_call():
        w = fast_dequant248(
            qweight, scales,
            qzeros=qzeros if not sym else None,
            bits=bits, group_size=group_size,
            sym=sym, sym_zp=sym_zp,
            out_dtype=dtype,
        )
        return x @ w

    # bench each path independently
    t_deq, st_deq, y_deq = _safe_bench("deq", deq_call)
    t_fdq, st_fdq, y_fdq = _safe_bench("fdq", fast_deq_call)
    t_old, st_old, y_old = _safe_bench("old", fused_old_call)
    t_new, st_new, y_new = _safe_bench("new", fused_new_call)
    t_hyb, st_hyb, y_hyb = _safe_bench("hyb", hybrid_call)
    t_fp16, st_fp16, _ = _safe_bench("fp16", fp16_call)

    # correctness against deq path (when both are OK)
    err_old = err_new = err_fdq = float("nan")
    if check and y_deq is not None:
        atol, rtol = (1e-2, 1e-2) if dtype == torch.float16 else (5e-2, 5e-2)
        if y_old is not None:
            err_old = (y_deq - y_old).abs().max().item()
            ok = torch.allclose(y_deq, y_old, atol=atol, rtol=rtol)
            st_old = "OK" if (st_old == "OK" and ok) else (st_old if st_old != "OK" else "DIFF")
        if y_new is not None:
            err_new = (y_deq - y_new).abs().max().item()
            ok = torch.allclose(y_deq, y_new, atol=atol, rtol=rtol)
            st_new = "OK" if (st_new == "OK" and ok) else (st_new if st_new != "OK" else "DIFF")
        if y_fdq is not None:
            err_fdq = (y_deq - y_fdq).abs().max().item()
            ok = torch.allclose(y_deq, y_fdq, atol=atol, rtol=rtol)
            st_fdq = "OK" if (st_fdq == "OK" and ok) else (st_fdq if st_fdq != "OK" else "DIFF")

    flops = 2.0 * M * N * K
    def tf(t):
        return float("nan") if (t != t) else flops / (t * 1e-3) / 1e12
    def spd(num, den):
        if num != num or den != den:
            return float("nan")
        return num / den

    return dict(
        M=M, K=K, N=N,
        deq_ms=t_deq, fdq_ms=t_fdq, old_ms=t_old, new_ms=t_new,
        hyb_ms=t_hyb, fp16_ms=t_fp16,
        spd_vs_deq=spd(t_deq, t_new),
        spd_vs_old=spd(t_old, t_new),
        spd_hyb_vs_deq=spd(t_deq, t_hyb),
        spd_fdq_vs_deq=spd(t_deq, t_fdq),
        deq_tflops=tf(t_deq),
        fdq_tflops=tf(t_fdq),
        old_tflops=tf(t_old),
        new_tflops=tf(t_new),
        hyb_tflops=tf(t_hyb),
        fp16_tflops=tf(t_fp16),
        status_old=st_old, status_new=st_new, status_hyb=st_hyb, status_fdq=st_fdq,
        err_old=err_old, err_new=err_new, err_fdq=err_fdq,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--sym", action="store_true")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = args.device

    shapes = [
        ("llama7b-qkv  decode",   1, 4096, 4096),
        ("llama7b-down decode",   1, 11008, 4096),
        ("llama7b-up   decode",   1, 4096, 11008),
        ("llama7b-qkv  bs16",    16, 4096, 4096),
        ("llama7b-down bs16",    16, 11008, 4096),
        ("llama7b-up   bs16",    16, 4096, 11008),
        ("llama7b-qkv  bs64",    64, 4096, 4096),
        ("llama7b-down bs64",    64, 11008, 4096),
        ("llama7b-up   bs64",    64, 4096, 11008),
        ("llama7b-qkv  pref",   512, 4096, 4096),
        ("llama7b-down pref",   512, 11008, 4096),
        ("llama7b-up   pref",   512, 4096, 11008),
        ("llama7b-qkv  bs2k",  2048, 4096, 4096),
        ("llama7b-down bs2k",  2048, 11008, 4096),
        ("llama7b-up   bs2k",  2048, 4096, 11008),
        ("llama70b-qkv decode",   1, 8192, 8192),
        ("llama70b-up  decode",   1, 8192, 28672),
        ("llama70b-qkv bs64",    64, 8192, 8192),
        ("llama70b-up  bs64",    64, 8192, 28672),
        ("llama70b-qkv bs512",  512, 8192, 8192),
        ("llama70b-up  bs512",  512, 8192, 28672),
    ]

    print(f"=== bits={args.bits}  group_size={args.group_size}  "
          f"dtype={args.dtype}  sym={args.sym}  device={device} ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    header = (
        f"{'shape':22s} {'M':>5s} {'K':>6s} {'N':>6s} | "
        f"{'deq ms':>8s} {'fdq ms':>8s} {'old ms':>8s} {'new ms':>8s} {'hyb ms':>8s} {'fp16 ms':>8s} | "
        f"{'fdq/deq':>8s} {'vs deq':>7s} {'vs old':>7s} {'hyb/deq':>8s} | "
        f"{'fp16TF':>6s} | "
        f"{'fdq':>4s} {'old':>4s} {'new':>4s} {'errN':>9s}"
    )
    print(header)
    print("-" * len(header))

    geo_deq, n_deq = 1.0, 0
    geo_old, n_old = 1.0, 0
    geo_hyb, n_hyb = 1.0, 0
    geo_fdq, n_fdq = 1.0, 0
    n = 0
    for name, M, K, N in shapes:
        if K % args.group_size != 0:
            continue
        try:
            r = run_case(M, K, N, args.bits, args.group_size, args.sym,
                         dtype, device, check=not args.no_check)
        except Exception as e:
            print(f"{name:22s}  ERROR: {e}")
            continue
        def fmt_ms(v):
            return f"{v:>8.3f}" if v == v else f"{'n/a':>8s}"
        def fmt_spd(v):
            return f"{v:>6.2f}x" if v == v else f"{'n/a':>7s}"
        def fmt_tf(v):
            return f"{v:>6.2f}" if v == v else f"{'n/a':>6s}"
        print(
            f"{name:22s} {r['M']:>5d} {r['K']:>6d} {r['N']:>6d} | "
            f"{fmt_ms(r['deq_ms'])} {fmt_ms(r['fdq_ms'])} {fmt_ms(r['old_ms'])} "
            f"{fmt_ms(r['new_ms'])} {fmt_ms(r['hyb_ms'])} {fmt_ms(r['fp16_ms'])} | "
            f"{fmt_spd(r['spd_fdq_vs_deq'])} {fmt_spd(r['spd_vs_deq'])} "
            f"{fmt_spd(r['spd_vs_old'])} {fmt_spd(r['spd_hyb_vs_deq'])} | "
            f"{fmt_tf(r['fp16_tflops'])} | "
            f"{r['status_fdq'][:4]:>4s} {r['status_old'][:4]:>4s} "
            f"{r['status_new'][:4]:>4s} {r['err_new']:>9.2e}"
        )
        if r['spd_vs_deq'] == r['spd_vs_deq']:
            geo_deq *= r['spd_vs_deq']; n_deq += 1
        if r['spd_vs_old'] == r['spd_vs_old']:
            geo_old *= r['spd_vs_old']; n_old += 1
        if r['spd_hyb_vs_deq'] == r['spd_hyb_vs_deq']:
            geo_hyb *= r['spd_hyb_vs_deq']; n_hyb += 1
        if r['spd_fdq_vs_deq'] == r['spd_fdq_vs_deq']:
            geo_fdq *= r['spd_fdq_vs_deq']; n_fdq += 1
        n += 1

    if n:
        print("-" * len(header))
        if n_fdq:
            print(f"geomean over {n_fdq} shapes:  fast_deq+cuBLAS vs deq+cuBLAS = {geo_fdq ** (1.0/n_fdq):.2f}x   (dequant-only optimisation)")
        if n_deq:
            print(f"geomean over {n_deq} shapes:  new (fused)   vs deq+cuBLAS    = {geo_deq ** (1.0/n_deq):.2f}x")
        if n_hyb:
            print(f"geomean over {n_hyb} shapes:  hybrid        vs deq+cuBLAS    = {geo_hyb ** (1.0/n_hyb):.2f}x")
        if n_old:
            print(f"geomean over {n_old} shapes:  new (fused)   vs existing-fused= {geo_old ** (1.0/n_old):.2f}x")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()

