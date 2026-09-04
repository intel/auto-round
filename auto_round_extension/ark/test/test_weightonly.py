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
from ut_utils import (
    decode_fp8_to_float,
    gen_weis8,
    get_torch_dt,
    make_fp8_consistent_weight,
    sample_valid_fp8,
    sample_valid_fp8_e8m0_xpu_safe,
)

ark = auto_round_kernel


def main_op(m, n, k, blocksize, compute_type, weight_type, scale_type, asym, device, is_ci=False):
    print(
        f"\n m={m}, n={n}, k={k}, blocksize={blocksize}, compute_type={compute_type}, "
        f"weight_type={weight_type}, scale_type={scale_type}, asym={asym}, device={device}"
    )
    torch.manual_seed(0)
    if weight_type == "int8" and compute_type == "int8" and asym:
        pytest.skip("NO A8W8C8")
    stype = get_torch_dt(scale_type)
    if compute_type == "int8":
        dtype = stype
        ctype = torch.int8
    else:
        dtype = get_torch_dt(compute_type)
        ctype = dtype
    if device == "cpu":
        cpu_cores = os.cpu_count()
        print(f"cpu_cores per numa node: {cpu_cores}")
        # ark.set_threads(cpu_cores)
        dtype = torch.float32
    if weight_type.startswith("fp8_"):
        if asym:
            pytest.skip("FP8 weight-only does not support asym in this test")
        if scale_type not in {"fp16", "fp32", "fp8_e8m0"}:
            pytest.skip("FP8 test supports scale_type fp16/fp32/fp8_e8m0")
        if compute_type not in {"fp16", "fp32"}:
            pytest.skip("FP8 test supports compute_type fp16/fp32")

        if device == "xpu" and scale_type == "fp8_e8m0":
            # For fp16 compute, keep magnitudes small to avoid overflow to inf/nan.
            fp8_exp_range = None
            if compute_type == "fp16":
                fp8_exp_range = (0, 12) if weight_type == "fp8_e4m3" else (10, 21)
            raw_qweight = sample_valid_fp8_e8m0_xpu_safe((k, n), weight_type, device, exp_range=fp8_exp_range)
        else:
            raw_qweight = sample_valid_fp8((k, n), weight_type, device)
        decoded = decode_fp8_to_float(raw_qweight.cpu(), weight_type).to(dtype).to(device)
        if scale_type == "fp8_e8m0":
            raw_qweight, scale, ref_dst = make_fp8_consistent_weight(
                k, n, blocksize, weight_type, scale_type, dtype, device
            )
        else:
            scale = torch.randn(k // blocksize, n, dtype=stype, device=device) / 100 + 0.01
            scale_re = scale.repeat_interleave(repeats=blocksize, dim=0)
            ref_dst = decoded * scale_re.to(dtype)
        bias = torch.randn(1, n, dtype=dtype, device=device)
        zp = torch.Tensor()
        qweight_for_pack = raw_qweight
    else:
        if scale_type == "fp8_e8m0":
            pytest.skip("fp8_e8m0 scale is only supported for fp8 weights")
        raw_s8_wei = gen_weis8(weight_type, device, k, n)
        scale = torch.rand(k // blocksize, n, dtype=stype, device=device) / 300 + 0.002
        bias = torch.randn(1, n, dtype=dtype, device=device)

        scale_re = scale.repeat_interleave(repeats=blocksize, dim=0)
        if asym:
            zp = gen_weis8(weight_type, device, k // blocksize, n)
            zp_re = zp.repeat_interleave(repeats=blocksize, dim=0)
            ref_dst = (raw_s8_wei.to(dtype) - zp_re.to(dtype)) * scale_re
        else:
            zp = torch.Tensor()
            ref_dst = raw_s8_wei.to(dtype) * scale_re
        qweight_for_pack = raw_s8_wei

    packw = ark.repack_quantized_weight(
        qweight_for_pack, scale, zp, blocksize, compute_type, weight_type, scale_type, asym
    )

    revert_wei_t = ark.unpack_weight(packw, dtype, n, k, blocksize, compute_type, weight_type, scale_type, asym)
    revert_wei = revert_wei_t.t()
    assert torch.allclose(revert_wei, ref_dst)

    tar_activation = torch.randn(m, k, dtype=dtype, device=device) - 0.5
    if device == "xpu" and weight_type.startswith("fp8_") and scale_type == "fp8_e8m0" and compute_type == "fp16":
        tar_activation = tar_activation * 0.125
    ref_c = torch.matmul(tar_activation, revert_wei)
    ref_c = ref_c + bias
    runs = 400
    if m == 1:
        runs = 4000
    if is_ci:
        runs = 1
    if device == "xpu" and weight_type.startswith("fp8_") and scale_type == "fp8_e8m0":
        # Avoid large batch repeats on CI for fp8_e8m0.
        for i in range(runs):
            tar_dst = ark.woqgemm(
                tar_activation, packw, bias, n, k, blocksize, compute_type, weight_type, scale_type, asym
            )

    else:
        batch = 8
        if m == 1:
            batch = 64
        if is_ci:
            batch = 1
        packw_set = torch.unsqueeze(packw, 0).repeat(batch, 1)
        tar_activation = torch.unsqueeze(tar_activation, 0).repeat(batch, 1, 1)
        for i in range(runs):
            tar_dst = ark.woqgemm(
                tar_activation[0], packw_set[0], bias, n, k, blocksize, compute_type, weight_type, scale_type, asym
            )

    if device == "xpu":
        torch.xpu.synchronize()
    diff = abs(ref_c - tar_dst)
    print(
        f"  Max Diff: {diff.max().item():.6f}, Mean Diff: {diff.mean().item():.6f}, ref mean:{ref_c.mean():.6f}, OUT mean:{tar_dst.mean():.6f}"
    )
    rtol = 0.1
    atol = 2.0
    if weight_type.startswith("fp8_"):
        if weight_type == "fp8_e5m2":
            rtol = 0.2
            atol = 24.0
            if compute_type == "fp16":
                rtol = 0.25
                atol = 24.0
    # Use equal_nan for fp16+fp8_e8m0 only to tolerate rare backend NaN divergence
    if weight_type.startswith("fp8_") and scale_type == "fp8_e8m0" and compute_type == "fp16":
        assert torch.allclose(tar_dst, ref_c, rtol=rtol, atol=atol, equal_nan=True)
    else:
        assert torch.allclose(tar_dst, ref_c, rtol=rtol, atol=atol)
    st = time.perf_counter()
    if device == "xpu" and weight_type.startswith("fp8_") and scale_type == "fp8_e8m0":
        for i in range(runs):
            tar_dst = ark.woqgemm(
                tar_activation, packw, bias, n, k, blocksize, compute_type, weight_type, scale_type, asym
            )
            if (i + 1) % 100 == 0:
                torch.xpu.empty_cache()
    else:
        for i in range(runs):
            idx = i % batch
            tar_dst = ark.woqgemm(
                tar_activation[idx], packw_set[idx], bias, n, k, blocksize, compute_type, weight_type, scale_type, asym
            )
    if device == "xpu":
        torch.xpu.synchronize()
    dur = time.perf_counter() - st
    dur /= runs
    ops = m * n * k * 2
    memsize = tar_activation.element_size() * m * k + tar_dst.element_size() * m * n
    blks = k // blocksize
    if weight_type == "int4":
        memsize += n * k // 2
    if weight_type == "int2":
        memsize += n * k // 4
    if weight_type in {"int8", "fp8_e4m3", "fp8_e5m2"}:
        memsize += n * k
    memsize += blks * n * tar_activation.element_size()

    gflops = ops / dur / 1e9
    bandwidth = memsize / dur / 1e9
    print(f"[Performance] Time: {dur*1000:.4f} ms")
    print(f"              GFLOPS: {gflops:.2f}")
    print(f"              Bandwidth: {bandwidth:.2f} GB/s")
    # record_property("GFLOPS", round(gflops, 2))
    # record_property("Bandwidth_GBs", round(bandwidth, 2))
    # record_property("Latency_ms", round(dur * 1000, 4))


@pytest.mark.parametrize("m", [1, 8, 16, 1024])
@pytest.mark.parametrize("n, k", [(1024, 768)])
@pytest.mark.parametrize("blocksize", [32, 128])
@pytest.mark.parametrize("weight_type", ["int2", "int4", "int8"])
@pytest.mark.parametrize("scale_type", ["fp32"])
@pytest.mark.parametrize("compute_type", ["int8", "bf16", "fp32"])
@pytest.mark.parametrize("asym", [True, False])
def test_cpu(m, n, k, blocksize, compute_type, weight_type, scale_type, asym):
    main_op(m, n, k, blocksize, compute_type, weight_type, scale_type, asym, "cpu", True)


@pytest.mark.parametrize("m", [1, 8, 16, 32, 64, 128, 256, 1024])
@pytest.mark.parametrize("n, k", [(1024, 768)])
@pytest.mark.parametrize("blocksize", [32, 128])
@pytest.mark.parametrize("weight_type", ["int2", "int4", "int8"])
@pytest.mark.parametrize("scale_type, compute_type", [("fp16", "fp16"), ("fp32", "fp32"), ("fp16", "int8")])
@pytest.mark.parametrize("asym", [True, False])
def test_xpu(m, n, k, blocksize, compute_type, weight_type, scale_type, asym):
    if not torch.xpu.is_available():
        pytest.skip("No XPU Device")
    with torch.no_grad():
        main_op(m, n, k, blocksize, compute_type, weight_type, scale_type, asym, "xpu", True)
        torch.xpu.empty_cache()


@pytest.mark.parametrize("m, n, k", [(1280, 256, 256), (1281, 288, 256)])
def test_xpu_s8_large_tile(m, n, k):
    if not torch.xpu.is_available():
        pytest.skip("No XPU Device")
    with torch.no_grad():
        main_op(m, n, k, 128, "int8", "int8", "fp16", False, "xpu", True)
        torch.xpu.empty_cache()


@pytest.mark.skipif(
    os.environ.get("ARK_DNNL", "0") not in ["1", "ON", "true", "True"],
    reason="Skipped because ARK_DNNL is not enabled in environment variables",
)
def test_xpu_woqgemm_graph_capture_uses_live_queue():
    if not torch.xpu.is_available():
        pytest.skip("No XPU Device")
    required = ("CUDAGraph", "Stream", "graph", "stream")
    if any(not hasattr(torch.xpu, name) for name in required):
        pytest.skip("torch.xpu graph APIs are not available")
    xpu_lib = getattr(ark, "xpu_lib", None)
    debug_required = (
        "_debug_xpu_device_context_key",
        "_debug_xpu_pool_scratch_ptr",
        "_debug_xpu_dnnl_engine_ptr",
        "_debug_xpu_dnnl_scratch_ptr",
    )
    if xpu_lib is None or any(not hasattr(xpu_lib, name) for name in debug_required):
        pytest.skip("XPU debug probes for context-aware scratch/engine isolation are unavailable")

    with torch.no_grad():
        default_stream = torch.xpu.current_stream()
        capture_stream = torch.xpu.Stream()

        def _on_stream(stream, fn, *args):
            with torch.xpu.stream(stream):
                return fn(torch.xpu.current_stream().sycl_queue, *args)

        default_key = int(_on_stream(default_stream, xpu_lib._debug_xpu_device_context_key))
        capture_key = int(_on_stream(capture_stream, xpu_lib._debug_xpu_device_context_key))
        if default_key == capture_key:
            pytest.skip("Unable to create two distinct SYCL contexts for this XPU runtime")

        def _assert_pool_slot_isolated(slot, small_bytes, large_bytes):
            ptr_a0 = int(_on_stream(default_stream, xpu_lib._debug_xpu_pool_scratch_ptr, small_bytes, slot))
            ptr_a1 = int(_on_stream(default_stream, xpu_lib._debug_xpu_pool_scratch_ptr, small_bytes, slot))
            assert ptr_a0 != 0 and ptr_a0 == ptr_a1
            _ = _on_stream(capture_stream, xpu_lib._debug_xpu_pool_scratch_ptr, large_bytes, slot)
            ptr_a2 = int(_on_stream(default_stream, xpu_lib._debug_xpu_pool_scratch_ptr, small_bytes, slot))
            assert ptr_a2 == ptr_a0

        for slot, small, large in ((8, 4, 64), (9, 4096, 8192), (10, 4096, 8192)):
            _assert_pool_slot_isolated(slot, small, large)

        eng_a0 = int(_on_stream(default_stream, xpu_lib._debug_xpu_dnnl_engine_ptr))
        eng_a1 = int(_on_stream(default_stream, xpu_lib._debug_xpu_dnnl_engine_ptr))
        eng_b = int(_on_stream(capture_stream, xpu_lib._debug_xpu_dnnl_engine_ptr))
        assert eng_a0 != 0 and eng_a0 == eng_a1
        assert eng_b != 0 and eng_b != eng_a0

        dnnl_scratch_a0 = int(_on_stream(default_stream, xpu_lib._debug_xpu_dnnl_scratch_ptr, 4096, 0))
        _ = _on_stream(capture_stream, xpu_lib._debug_xpu_dnnl_scratch_ptr, 8192, 0)
        dnnl_scratch_a1 = int(_on_stream(default_stream, xpu_lib._debug_xpu_dnnl_scratch_ptr, 4096, 0))
        assert dnnl_scratch_a0 != 0 and dnnl_scratch_a1 == dnnl_scratch_a0

        m, n, k = 8, 256, 256
        blocksize = 32
        compute_type = "fp16"
        weight_type = "int4"
        scale_type = "fp16"
        asym = False

        raw_s8_wei = gen_weis8(weight_type, "xpu", k, n)
        scale = torch.rand(k // blocksize, n, dtype=torch.float16, device="xpu") / 300 + 0.002
        bias = torch.randn(1, n, dtype=torch.float16, device="xpu")
        zp = torch.empty(0, device=raw_s8_wei.device)
        packw = ark.repack_quantized_weight(
            raw_s8_wei, scale, zp, blocksize, compute_type, weight_type, scale_type, asym
        )

        def _woq_call(x):
            return ark.woqgemm(x, packw, bias, n, k, blocksize, compute_type, weight_type, scale_type, asym)

        eager_input = torch.randn(m, k, dtype=torch.float16, device="xpu") - 0.5
        replay_input = torch.randn(m, k, dtype=torch.float16, device="xpu") + 0.25

        # Warm up on the default queue to reproduce stale stream cache regressions.
        _ = _woq_call(eager_input)
        torch.xpu.synchronize()
        graph_input = eager_input.clone()
        graph = torch.xpu.CUDAGraph()

        try:
            with torch.xpu.stream(capture_stream):
                with torch.xpu.graph(graph):
                    graph_output = _woq_call(graph_input)
        except RuntimeError as err:
            pytest.skip(f"XPU graph capture is unavailable in this environment: {err}")

        outputs = []
        for current in (eager_input, replay_input):
            graph_input.copy_(current)
            graph.replay()
            torch.xpu.synchronize()
            replayed = graph_output.clone()
            expected = _woq_call(current)
            torch.xpu.synchronize()
            assert torch.allclose(replayed, expected, rtol=0.1, atol=2.0)
            outputs.append(replayed)

        assert not torch.allclose(outputs[0], outputs[1], rtol=1e-3, atol=1e-3)
        torch.xpu.empty_cache()


@pytest.mark.parametrize("m", [1, 1024])
@pytest.mark.parametrize("n, k", [(1024, 768)])
@pytest.mark.parametrize("blocksize", [32, 128])
@pytest.mark.parametrize("weight_type", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize(
    "scale_type, compute_type", [("fp16", "fp16"), ("fp32", "fp32"), ("fp8_e8m0", "fp16"), ("fp8_e8m0", "fp32")]
)
@pytest.mark.parametrize("asym", [False])
def test_xpu_fp8(m, n, k, blocksize, compute_type, weight_type, scale_type, asym):
    if not torch.xpu.is_available():
        pytest.skip("No XPU Device")
    with torch.no_grad():
        main_op(m, n, k, blocksize, compute_type, weight_type, scale_type, asym, "xpu", True)
        torch.xpu.empty_cache()


# LOCAL_TEST: python test_weightonly.py
# CI pytest: pytest -v test_weightonly.py -k 'test_'
# pytest -v test_weightonly.py -k 'test_' --csv results_woq.csv --csv-columns file,function,parameters_as_columns,properties_as_columns
if __name__ == "__main__":
    # main_op(1, 1024, 768, 32, "int8", "int2", "fp32", False, "cpu")
    # main_op(1, 4096, 4096, 128, "fp32", "int4", "fp32", False, "cpu")
    # main_op(1, 4096, 4096, 128, "int8", "int4", "fp32", False, "cpu")
    main_op(1024, 4096, 4096, 32, "fp32", "int8", "fp32", False, "cpu")
    main_op(1024, 4096, 4096, 32, "fp32", "int8", "fp32", False, "cpu")
    main_op(1024, 4096, 4096, 128, "int8", "int8", "fp32", False, "cpu")

    # main_op(1, 4096, 4096, 32, "fp32", "int4", "fp32", False, "xpu")
    # main_op(1, 4096, 4096, 32, "fp16", "int4", "fp16", False, "xpu")
    # main_op(1, 4096, 4096, 128, "fp16", "int4", "fp16", False, "xpu")
    # main_op(1, 4096, 4096, 32, "fp32", "int8", "fp32", False, "xpu")
    # main_op(1, 4096, 4096, 32, "fp16", "int8", "fp16", False, "xpu")
    # main_op(1, 4096, 4096, 128, "fp16", "int8", "fp16", False, "xpu")
    # main_op(1, 4096, 4096, 32, "fp32", "fp8_e4m3", "fp32", False, "xpu")
    # main_op(1, 4096, 4096, 32, "fp16", "fp8_e4m3", "fp16", False, "xpu")
    # main_op(1, 4096, 4096, 32, "fp32", "fp8_e5m2", "fp32", False, "xpu")
    # main_op(1, 4096, 4096, 32, "fp16", "fp8_e5m2", "fp16", False, "xpu")
    # main_op(1, 4096, 4096, 32, "fp32", "fp8_e4m3", "fp8_e8m0", False, "xpu")
    # main_op(1, 4096, 4096, 32, "fp16", "fp8_e4m3", "fp8_e8m0", False, "xpu")
    # main_op(1, 4096, 4096, 32, "fp16", "fp8_e5m2", "fp8_e8m0", False, "xpu")

    # main_op(4096, 4096, 4096, 32, "fp16", "int4", "fp16", False, "xpu")
    # main_op(4096, 4096, 4096, 4096, "int8", "int4", "fp32", False, "cpu")
    # main_op(4096, 4096, 4096, 32, "auto", "int4", "fp32", False, "cpu")
    # main_op(1, 1024, 768, 32, "int8", "int2", "fp32", False, "cpu")
    main_op(1024, 768, 768, 64, "int8", "int2", "fp32", False, "xpu")
    main_op(1024, 768, 768, 64, "int8", "int4", "fp32", False, "xpu")
    main_op(1024, 768, 3072, 64, "int8", "int4", "fp32", False, "xpu")
    main_op(1024, 3072, 768, 64, "int8", "int4", "fp32", False, "xpu")
    # main_op(4096, 4096, 4096, 4096, "int8", "int4", "fp16", False, "xpu")
    # main_op(4096, 4096, 4096, 32, "int8", "int4", "fp16", False, "xpu")
    # main_op(4096, 4096, 4096, 32, "fp16", "int4", "fp16", False, "xpu")
    # main_op(4096, 4096, 4096, 32, "fp16", "int8", "fp16", False, "xpu")
    # main_op(4096, 4096, 4096, 32, "fp32", "fp8_e4m3", "fp8_e8m0", False, "xpu")
    # main_op(4096, 4096, 4096, 32, "fp16", "fp8_e5m2", "fp8_e8m0", False, "xpu")

    # main_op(2048, 4096, 4096, 32, "fp32", "int4", "fp32", False, True, "xpu")
    # main_op(4096, 4096, 4096, 32, "fp16", "int4", "fp16", False, True, "xpu")
    # main_op(4096, 4096, 4096, 128, "fp16", "int4", "fp16", False, True, "xpu")

    # main_op(1024, 1024, 1024, 32, "fp32", "int8", "fp32", False, True, "xpu")
    # main_op(1024, 1024, 1024, 32, "fp16", "int8", "fp16", False, True, "xpu")
    # main_op(1024, 1024, 1024, 128, "fp16", "int8", "fp16", False, True, "xpu")

    # print("XPU FP8 TEST")

    # main_op(1024, 4096, 4096, 32, "fp32", "fp8_e4m3", "fp32", False, "xpu")
    # main_op(1024, 4096, 4096, 32, "fp16", "fp8_e4m3", "fp16", False, "xpu")

    # main_op(1024, 4096, 4096, 32, "fp32", "fp8_e5m2", "fp32", False, "xpu")
    # main_op(1024, 4096, 4096, 32, "fp16", "fp8_e5m2", "fp16", False, "xpu")
    # main_op(4096, 4096, 4096, 32, "fp32", "fp8_e5m2", "fp32", False, "xpu")
    # main_op(4096, 4096, 4096, 32, "fp16", "fp8_e5m2", "fp16", False, "xpu")
