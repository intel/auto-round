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

ark = auto_round_kernel.ARK()


def main_op(m, k, n, dt, batch_size, runs, has_bias, record_property, device):
    print(f"\n  m={m}, k={k}, n={n}, dt={dt}, batch={batch_size}")
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

    tar_dst = ark.matmul(activation, wei, bias)
    ref_dst = torch.matmul(activation, wei.T).to(cdt)
    if has_bias:
        ref_dst = ref_dst + bias
    diff = abs(tar_dst - ref_dst)
    # print(f"  abs(ref_dst).max(): {abs(ref_dst).max()}, abs(ref_dst).mean(): {abs(ref_dst).mean()}, abs(tar_dst).max(): {abs(tar_dst).max()}, abs(tar_dst).mean(): {abs(tar_dst).mean()}")
    print(f"  Max Diff: {diff.max().item():.6f}, Mean Diff: {diff.mean().item():.6f}")
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
        tar_dst = ark.matmul(act_set[idx], wei_set[idx], bias)
    if device == "xpu":
        torch.xpu.synchronize()

    st = time.perf_counter()
    for i in range(runs):
        idx = i % batch_size
        tar_dst = ark.matmul(act_set[idx], wei_set[idx], bias)
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

    print(f"[Performance] Time: {dur*1000:.4f} ms")
    print(f"              GFLOPS: {gflops:.2f}")
    print(f"              Bandwidth: {bandwidth:.2f} GB/s")


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
    dff = abs(C - ref)
    print(f"  Max Diff: {dff.max().item():.6f}, Mean Diff: {dff.mean().item():.6f}")
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

    print(f"[Performance] Time: {dur*1000:.4f} ms")
    print(f"              GFLOPS: {gflops:.2f}")
    print(f"              Bandwidth: {bandwidth:.2f} GB/s")


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
