#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import auto_round_kernel
import pytest
import torch

ark = auto_round_kernel


def main_op(m, k, n, dt, batch_size, runs, has_bias, record_property, device, op=None, op_name="matmul"):
    op = op or ark.matmul
    print(f"\n  op={op_name}, m={m}, k={k}, n={n}, dt={dt}, batch={batch_size}")
    torch.manual_seed(0)
    cdt = dt
    if device == "cpu":
        cpu_cores = os.cpu_count()
        print(f"cpu_cores: {cpu_cores}", f"torch threads: {torch.get_num_threads()}")
        cdt = torch.float32
    try:
        activation = torch.rand(m, k, dtype=dt, device=device) - 0.5
        wei = torch.rand(n, k, dtype=dt, device=device) - 0.5
        bias = torch.rand(1, n, dtype=cdt, device=device) if has_bias else torch.Tensor()

        act_set = activation.unsqueeze(0).repeat(batch_size, 1, 1)
        wei_set = wei.unsqueeze(0).repeat(batch_size, 1, 1)

    except RuntimeError as e:
        if record_property:
            pytest.fail(f"OOM during basic init: {e}")
        else:
            print(f"OOM during basic init: {e}")
            return

    tar_dst = op(activation, wei, bias)
    ref_dst = torch.matmul(activation, wei.T).to(cdt)
    if has_bias:
        ref_dst = ref_dst + bias
    diff = abs(tar_dst - ref_dst)
    print(f"  Max Diff: {diff.max().item():.5f}, Mean Diff: {diff.mean().item():.5f}", end="", flush=True)
    if dt == torch.float32:
        atol = 0.001
        rtol = 0.03
    if dt == torch.float16:
        atol = 0.1
        rtol = 0.06
    if dt == torch.bfloat16:
        atol = 0.6
        rtol = 0.1
    assert torch.allclose(tar_dst, ref_dst, atol=atol, rtol=rtol), "Verification Failed!"
    print("  -> Verification PASS")

    try:
        act_set = activation.unsqueeze(0).repeat(batch_size, 1, 1)
        wei_set = wei.unsqueeze(0).repeat(batch_size, 1, 1)
        dst_set = tar_dst.unsqueeze(0).repeat(batch_size, 1, 1)
    except RuntimeError as e:
        if record_property:
            pytest.fail(f"OOM during basic init: {e}")
        else:
            print(f"OOM during basic init: {e}")
            return

    for i in range(runs):
        idx = i % batch_size
        tar_dst = op(act_set[idx], wei_set[idx], bias)
    if device == "xpu":
        torch.xpu.synchronize()

    st = time.perf_counter()
    for i in range(runs):
        idx = i % batch_size
        tar_dst = op(act_set[idx], wei_set[idx], bias)
    if device == "xpu":
        torch.xpu.synchronize()
    dur = (time.perf_counter() - st) / runs

    # GFLOPS
    ops = m * n * k * 2
    gflops = ops / dur / 1e9

    # Bandwidth
    elem_size = activation.element_size()
    total_bytes = (m * k + k * n + m * n) * elem_size
    bandwidth = total_bytes / dur / 1e9

    if record_property:
        record_property("Time_ms", round(dur * 1000, 4))
        record_property("GFLOPS", round(gflops, 2))
        record_property("Bandwidth_GBs", round(bandwidth, 2))

    print(f"[Performance] Time: {dur*1000:.4f} ms, GFLOPS: {gflops:.2f}, Bandwidth: {bandwidth:.2f} GB/s")


def woqgemm(m, k, n, dt, batch_size, runs, record_property, device):
    print(f"\n  m={m}, k={k}, n={n}, dt={dt}, batch={batch_size}")
    torch.manual_seed(0)

    A = torch.rand(batch_size, m, k, dtype=dt, device=device) - 0.5
    bias = torch.rand(1, n, dtype=dt, device=device) + 2
    B = torch.randint(-128, 127, (batch_size, n, k), dtype=torch.int8, device=device)
    scaleB = torch.rand(n, 1, dtype=dt, device=device) / 100
    C = ark.woqgemm_s8(A[0], B[0], scaleB, bias)

    maxA = abs(A[0]).max()
    scaleA = maxA / 127
    ratio = (1 / scaleA).to(dt)
    qA = torch.round(A[0] * ratio).to(torch.int8)
    DA = qA.to(dt) * scaleA
    DB = B[0].to(dt) * scaleB
    ref = torch.matmul(DA, DB.T)
    ref = ref + bias
    diff = abs(C - ref)
    print(f"  Max Diff: {diff.max().item():.5f}, Mean Diff: {diff.mean().item():.5f}", end="", flush=True)
    # QDQ tolerance
    assert torch.allclose(C, ref, atol=1, rtol=0.1), "Verification Failed!"
    print("  -> Verification PASS")
    for i in range(runs):
        idx = i % batch_size
        tar_dst = ark.woqgemm_s8(A[idx], B[idx], scaleB, bias)
    if device == "xpu":
        torch.xpu.synchronize()

    st = time.perf_counter()
    for i in range(runs):
        idx = i % batch_size
        tar_dst = ark.woqgemm_s8(A[idx], B[idx], scaleB, bias)
    if device == "xpu":
        torch.xpu.synchronize()

    dur = (time.perf_counter() - st) / runs

    # GFLOPS
    ops = m * n * k * 2
    gflops = ops / dur / 1e9

    # Bandwidth
    elem_size = A.element_size()
    total_bytes = (m * k + m * n) * elem_size + k * n
    bandwidth = total_bytes / dur / 1e9

    if record_property:
        record_property("Time_ms", round(dur * 1000, 4))
        record_property("GFLOPS", round(gflops, 2))
        record_property("Bandwidth_GBs", round(bandwidth, 2))

    print(f"[Performance] Time: {dur*1000:.4f} ms, GFLOPS: {gflops:.2f}, Bandwidth: {bandwidth:.2f} GB/s")


@pytest.mark.parametrize("m", [1, 8, 16, 32, 128, 256, 1024, 2048, 3072, 4096])
@pytest.mark.parametrize("k, n", [(4096, 4096)])
@pytest.mark.parametrize("dt", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("runs", [1000])
def test_igemm_s8s8_joint_matrix_vs_sycl_tla(m, k, n, dt, runs, record_property):
    if not torch.xpu.is_available():
        pytest.skip("No XPU Device")
    if not hasattr(ark, "dyn_quant_s8"):
        pytest.skip("Low-level igemm benchmark APIs are not available")

    torch.manual_seed(0)
    A = torch.rand(m, k, dtype=dt, device="xpu") - 0.5
    B = torch.randint(-128, 127, (n, k), dtype=torch.int8, device="xpu")
    scaleB = torch.rand(n, 1, dtype=dt, device="xpu") / 100
    bias = torch.rand(1, n, dtype=dt, device="xpu") + 2

    qA, scaleA = ark.dyn_quant_s8(A)
    torch.xpu.synchronize()

    out_joint = torch.empty(m, n, dtype=dt, device="xpu")
    out_tla = torch.empty(m, n, dtype=dt, device="xpu")

    ark.igemm_s8s8_joint_matrix(qA, B, scaleA, scaleB, bias, out_joint)
    ark.igemm_s8s8_sycl_tla(qA, B, scaleA, scaleB, bias, out_tla)
    torch.xpu.synchronize()

    diff = (out_joint - out_tla).abs()
    print(f"\n  Max Diff igemm joint_matrix vs sycl_tla: {diff.max().item():.5f}, Mean Diff: {diff.mean().item():.5f}")
    assert torch.allclose(out_tla, out_joint, atol=1, rtol=0.1)

    def bench(op, out):
        warmup = 200
        for _ in range(warmup):
            op(qA, B, scaleA, scaleB, bias, out)
        torch.xpu.synchronize()

        st = time.perf_counter()
        for _ in range(runs):
            op(qA, B, scaleA, scaleB, bias, out)
        torch.xpu.synchronize()
        return (time.perf_counter() - st) / runs

    joint_dur = min(bench(ark.igemm_s8s8_joint_matrix, out_joint) for _ in range(3))
    tla_dur = min(bench(ark.igemm_s8s8_sycl_tla, out_tla) for _ in range(3))

    ops = m * n * k * 2
    joint_tflops = ops / joint_dur / 1e12
    tla_tflops = ops / tla_dur / 1e12
    speedup = joint_dur / tla_dur

    print(f"  [igemm joint_matrix] : {joint_dur * 1000:8.4f} ms  {joint_tflops:7.3f} TFLOPS")
    print(f"  [igemm sycl_tla]     : {tla_dur * 1000:8.4f} ms  {tla_tflops:7.3f} TFLOPS  speedup={speedup:5.2f}x")

    record_property("igemm_joint_matrix_time_ms", round(joint_dur * 1000, 4))
    record_property("igemm_sycl_tla_time_ms", round(tla_dur * 1000, 4))
    record_property("igemm_speedup", round(speedup, 4))


@pytest.mark.parametrize("m", [1024, 2048, 3072, 4096])
@pytest.mark.parametrize("k, n", [(4096, 4096)])
@pytest.mark.parametrize("dt", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("runs", [1000])
def test_sycl_tla_igemm_accum_vs_dequant_perf(m, k, n, dt, runs, record_property):
    if not torch.xpu.is_available():
        pytest.skip("No XPU Device")
    if not hasattr(ark, "igemm_s8s8_sycl_tla_accum"):
        pytest.skip("SYCL-TLA accum-only igemm API is not available")

    torch.manual_seed(0)
    A = torch.rand(m, k, dtype=dt, device="xpu") - 0.5
    B = torch.randint(-128, 127, (n, k), dtype=torch.int8, device="xpu")
    scaleB = torch.rand(n, 1, dtype=dt, device="xpu") / 100
    bias = torch.rand(1, n, dtype=dt, device="xpu") + 2

    qA, scaleA = ark.dyn_quant_s8(A)
    torch.xpu.synchronize()

    out_accum = torch.empty(m, n, dtype=torch.int32, device="xpu")
    out_dequant = torch.empty(m, n, dtype=dt, device="xpu")

    ark.igemm_s8s8_sycl_tla_accum(qA, B, out_accum)
    ark.igemm_s8s8_sycl_tla(qA, B, scaleA, scaleB, bias, out_dequant)
    torch.xpu.synchronize()

    def bench_accum():
        warmup = 200
        for _ in range(warmup):
            ark.igemm_s8s8_sycl_tla_accum(qA, B, out_accum)
        torch.xpu.synchronize()

        st = time.perf_counter()
        for _ in range(runs):
            ark.igemm_s8s8_sycl_tla_accum(qA, B, out_accum)
        torch.xpu.synchronize()
        return (time.perf_counter() - st) / runs

    def bench_dequant():
        warmup = 200
        for _ in range(warmup):
            ark.igemm_s8s8_sycl_tla(qA, B, scaleA, scaleB, bias, out_dequant)
        torch.xpu.synchronize()

        st = time.perf_counter()
        for _ in range(runs):
            ark.igemm_s8s8_sycl_tla(qA, B, scaleA, scaleB, bias, out_dequant)
        torch.xpu.synchronize()
        return (time.perf_counter() - st) / runs

    accum_dur = min(bench_accum() for _ in range(3))
    dequant_dur = min(bench_dequant() for _ in range(3))

    ops = m * n * k * 2
    accum_tflops = ops / accum_dur / 1e12
    dequant_tflops = ops / dequant_dur / 1e12
    epilogue_ms = (dequant_dur - accum_dur) * 1000

    print(f"\n  [sycl_tla accum-only] : {accum_dur * 1000:8.4f} ms  {accum_tflops:7.3f} TFLOPS")
    print(f"  [sycl_tla dequant]    : {dequant_dur * 1000:8.4f} ms  {dequant_tflops:7.3f} TFLOPS")
    print(f"  [delta]               : {epilogue_ms:8.4f} ms")

    record_property("sycl_tla_accum_time_ms", round(accum_dur * 1000, 4))
    record_property("sycl_tla_dequant_time_ms", round(dequant_dur * 1000, 4))
    record_property("sycl_tla_epilogue_delta_ms", round(epilogue_ms, 4))


@pytest.mark.parametrize("m", [1, 8, 32, 128, 1024, 2048, 3072, 4096])
@pytest.mark.parametrize("k, n", [(4096, 4096)])
@pytest.mark.parametrize("dt", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("runs", [1000])
def test_woqgemm_s8_joint_matrix_vs_sycl_tla(m, k, n, dt, runs, record_property):
    if not torch.xpu.is_available():
        pytest.skip("No XPU Device")
    if not hasattr(ark, "woqgemm_s8_joint_matrix") or not hasattr(ark, "woqgemm_s8_sycl_tla"):
        pytest.skip("Both woqgemm_s8 backends are not available in this build")

    torch.manual_seed(0)
    A = torch.rand(m, k, dtype=dt, device="xpu") - 0.5
    B = torch.randint(-128, 127, (n, k), dtype=torch.int8, device="xpu")
    scaleB = torch.rand(n, 1, dtype=dt, device="xpu") / 100
    bias = torch.rand(1, n, dtype=dt, device="xpu") + 2

    out_joint = ark.woqgemm_s8_joint_matrix(A, B, scaleB, bias)
    out_tla = ark.woqgemm_s8_sycl_tla(A, B, scaleB, bias)
    torch.xpu.synchronize()

    diff = (out_joint - out_tla).abs()
    print(f"\n  Max Diff joint_matrix vs sycl_tla: {diff.max().item():.5f}, Mean Diff: {diff.mean().item():.5f}")
    assert torch.allclose(out_tla, out_joint, atol=1, rtol=0.1)

    def bench(op):
        warmup = 200
        for _ in range(warmup):
            _ = op(A, B, scaleB, bias)
        torch.xpu.synchronize()

        st = time.perf_counter()
        for _ in range(runs):
            _ = op(A, B, scaleB, bias)
        torch.xpu.synchronize()
        return (time.perf_counter() - st) / runs

    joint_dur = min(bench(ark.woqgemm_s8_joint_matrix) for _ in range(3))
    tla_dur = min(bench(ark.woqgemm_s8_sycl_tla) for _ in range(3))

    ops = m * n * k * 2
    joint_tflops = ops / joint_dur / 1e12
    tla_tflops = ops / tla_dur / 1e12
    speedup = joint_dur / tla_dur

    print(f"  [joint_matrix] : {joint_dur * 1000:8.4f} ms  {joint_tflops:7.3f} TFLOPS")
    print(f"  [sycl_tla]     : {tla_dur * 1000:8.4f} ms  {tla_tflops:7.3f} TFLOPS  speedup={speedup:5.2f}x")

    record_property("joint_matrix_time_ms", round(joint_dur * 1000, 4))
    record_property("sycl_tla_time_ms", round(tla_dur * 1000, 4))
    record_property("speedup", round(speedup, 4))


@pytest.mark.parametrize("m", [4096])
@pytest.mark.parametrize("k, n", [(4096, 4096)])
@pytest.mark.parametrize("dt", [torch.float16, torch.float32])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("runs", [1])
def test_woqgemm_xpu(m, k, n, dt, batch_size, runs, record_property):
    if not torch.xpu.is_available():
        pytest.skip("No XPU Device")
    with torch.no_grad():
        woqgemm(m, k, n, dt, batch_size, runs, record_property, "xpu")
        torch.xpu.empty_cache()


@pytest.mark.parametrize("m", [1, 8, 16, 32, 64, 128, 256, 1024])
@pytest.mark.parametrize("k, n", [(4096, 4096)])
@pytest.mark.parametrize("dt", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("runs", [1])
def test_xpu(m, k, n, dt, batch_size, runs, record_property):
    if not torch.xpu.is_available():
        pytest.skip("No XPU Device")
    with torch.no_grad():
        main_op(m, k, n, dt, batch_size, runs, True, record_property, "xpu")
        torch.xpu.empty_cache()


@pytest.mark.parametrize("m", [1, 8, 16, 32, 64, 128, 256, 1024])
@pytest.mark.parametrize("k, n", [(4096, 4096)])
@pytest.mark.parametrize("dt", [torch.float32])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("runs", [1])
def test_cpu(m, k, n, dt, batch_size, runs, record_property):
    main_op(m, k, n, dt, batch_size, runs, True, record_property, "cpu")


@pytest.mark.parametrize("m", [1, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096])
@pytest.mark.parametrize("k, n", [(4096, 4096)])
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("runs", [1])
def test_xpu_sycl_tla(m, k, n, dt, batch_size, runs, record_property):
    if not torch.xpu.is_available():
        pytest.skip("No XPU Device")
    if not hasattr(ark, "matmul_sycl_tla"):
        pytest.skip("Python wrapper for matmul_sycl_tla is not available")
    if getattr(ark, "xpu_lib", None) is None or not hasattr(ark.xpu_lib, "matmul_sycl_tla"):
        pytest.skip("SYCL-TLA matmul is not available in this build")

    main_op(
        m,
        k,
        n,
        dt,
        batch_size,
        runs,
        True,
        record_property,
        "xpu",
        op=ark.matmul_sycl_tla,
        op_name="matmul_sycl_tla",
    )

    torch.xpu.empty_cache()


@pytest.mark.parametrize("m", [1, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096])
@pytest.mark.parametrize("k, n", [(4096, 4096)])
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("runs", [1])
def test_xpu_sycl_tla_no_bias(m, k, n, dt, batch_size, runs, record_property):
    if not torch.xpu.is_available():
        pytest.skip("No XPU Device")
    if not hasattr(ark, "matmul_sycl_tla"):
        pytest.skip("Python wrapper for matmul_sycl_tla is not available")
    if getattr(ark, "xpu_lib", None) is None or not hasattr(ark.xpu_lib, "matmul_sycl_tla"):
        pytest.skip("SYCL-TLA matmul is not available in this build")

    main_op(
        m,
        k,
        n,
        dt,
        batch_size,
        runs,
        False,
        record_property,
        "xpu",
        op=ark.matmul_sycl_tla,
        op_name="matmul_sycl_tla_no_bias",
    )

    torch.xpu.empty_cache()


# pytest -vs auto_round_extension/ark/test/test_matmul.py -k compare_dnnl_vs_sycl_tla
@pytest.mark.parametrize("m", [1, 8, 16, 32, 128, 1024, 2048, 4096])
@pytest.mark.parametrize("k, n", [(4096, 4096)])
@pytest.mark.parametrize("dt", [torch.float32, torch.float16, torch.bfloat16])
def test_xpu_compare_dnnl_vs_sycl_tla(m, k, n, dt):
    warmup = 100
    runs = 1000

    compare_matmul_backends(m, k, n, dt, warmup, runs, "xpu")
    torch.xpu.empty_cache()


@pytest.mark.parametrize("m", [1, 8, 16, 32, 128, 1024, 2048, 4096])
@pytest.mark.parametrize("k, n", [(4096, 4096)])
@pytest.mark.parametrize("dt", [torch.float32, torch.float16, torch.bfloat16])
def test_xpu_compare_dnnl_vs_sycl_tla_no_bias(m, k, n, dt):
    warmup = 100
    runs = 1000

    compare_matmul_backends(m, k, n, dt, warmup, runs, "xpu", has_bias=False)
    torch.xpu.empty_cache()


def compare_matmul_backends(m, k, n, dt, warmup, runs, device="xpu", has_bias=True):
    if device != "xpu":
        raise ValueError("compare_matmul_backends only supports XPU")

    def _matmul_tolerance(dt):
        if dt == torch.float32:
            return 0.1, 0.06
        if dt == torch.float16:
            return 0.1, 0.06
        if dt == torch.bfloat16:
            return 0.6, 0.1
        raise ValueError(f"Unsupported dtype: {dt}")

    def _benchmark_op(op, A, B, bias, runs, warmup):
        for _ in range(warmup):
            _ = op(A, B, bias)
        if A.device.type == "xpu":
            torch.xpu.synchronize()

        st = time.perf_counter()
        for _ in range(runs):
            _ = op(A, B, bias)
        if A.device.type == "xpu":
            torch.xpu.synchronize()

        return (time.perf_counter() - st) / runs

    torch.manual_seed(0)

    activation = torch.rand(m, k, dtype=dt, device=device) - 0.5
    wei = torch.rand(n, k, dtype=dt, device=device) - 0.5
    bias = torch.rand(1, n, dtype=dt, device=device) - 0.5 if has_bias else torch.Tensor()

    dnnl_out = ark.matmul(activation, wei, bias)
    tla_out = ark.matmul_sycl_tla(activation, wei, bias)

    atol, rtol = _matmul_tolerance(dt)
    ok = torch.allclose(dnnl_out, tla_out, atol=atol, rtol=rtol)

    assert ok, "oneDNN vs sycl-tla mismatch"

    dnnl_dur = _benchmark_op(ark.matmul, activation, wei, bias, runs=runs, warmup=warmup)
    tla_dur = _benchmark_op(ark.matmul_sycl_tla, activation, wei, bias, runs=runs, warmup=warmup)

    ops = m * n * k * 2
    print(f"\n  [oneDNN]                  : {dnnl_dur*1000:8.3f} ms   {ops / dnnl_dur / 1e12:7.3f} TFLOPS")
    print(
        f"  [matmul_sycl_tla]         : {tla_dur*1000:8.3f} ms   {ops / tla_dur / 1e12:7.3f} TFLOPS  speedup={dnnl_dur / tla_dur:5.2f}x"
    )


# LOCAL_TEST: python test_matmul.py
# CI pytest: pytest -v test_matmul.py -k 'test_xpu'
# pytest -s -v test_matmul.py -k 'test_xpu' --csv results_matmul_all.csv --csv-columns file,function,parameters_as_columns,properties_as_columns
if __name__ == "__main__":

    def mock_record(key, value):
        pass

    # woqgemm(512, 4096, 4096, torch.float16, batch_size=2, runs=10, record_property=mock_record, device='cpu')
    # woqgemm(512, 4096, 4096, torch.float32, batch_size=2, runs=10, record_property=mock_record, device='cpu')

    test_cpu(1024, 4096, 4096, torch.float32, batch_size=2, runs=10, record_property=mock_record)
    # test_cpu(1024, 4096, 4096, torch.float16, batch_size=2, runs=10, record_property=mock_record)
    # test_cpu(1024, 4096, 4096, torch.bfloat16, batch_size=2, runs=10, record_property=mock_record)

    woqgemm(4096, 4096, 4096, torch.float16, batch_size=2, runs=100, record_property=mock_record, device="xpu")
    woqgemm(4096, 4096, 4096, torch.float32, batch_size=2, runs=100, record_property=mock_record, device="xpu")
    test_xpu(4096, 4096, 8192, torch.float32, batch_size=2, runs=100, record_property=mock_record)
    test_xpu(4096, 4096, 8192, torch.float16, batch_size=2, runs=100, record_property=mock_record)
    test_xpu(4096, 4096, 8192, torch.bfloat16, batch_size=2, runs=100, record_property=mock_record)
