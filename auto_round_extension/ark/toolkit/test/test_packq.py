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
from ut_utils import *

ark = auto_round_kernel.ARK()


def main_op(k, n, weight_type, scale_type, asym, blocksize, device, compute_type):
    print(
        f"\n k={k}, n={n}, blocksize={blocksize}, compute_type={compute_type}, "
        f"weight_type={weight_type}, scale_type={scale_type}, asym={asym}, device={device}"
    )

    torch.manual_seed(0)
    if device == "cpu":
        cpu_cores = os.cpu_count()
        print(f"cpu_cores per numa node: {cpu_cores}")
        # ark.set_threads(cpu_cores)
    if device == "xpu" and compute_type is None:
        compute_type = scale_type
    sdt = get_torch_dt(compute_type)
    if device == "cpu":
        sdt = torch.float32
    scale_dt = get_scale_torch_dt(scale_type)
    if scale_type == "fp8_e8m0":
        # int-valued exponents; actual scale factor is 2**exp
        scale = torch.randint(-8, 8, (k // blocksize, n), dtype=torch.int32, device=device).to(scale_dt)
        scale_re = torch.pow(torch.tensor(2.0, device=device, dtype=torch.float32), scale.to(torch.float32))
        scale_re = scale_re.repeat_interleave(repeats=blocksize, dim=0).to(sdt)
    else:
        scale = torch.randn(k // blocksize, n, dtype=scale_dt, device=device) / 100 + 0.01
        scale_re = scale.repeat_interleave(repeats=blocksize, dim=0).to(sdt)
    odt = sdt
    if scale_dt == "fp8_e8m0":
        odt = torch.float32
    if weight_type.startswith("fp8_"):
        if device == "cpu":
            pytest.skip("FP8 packq test is for XPU")
        if asym:
            pytest.skip("FP8 packq does not support asym in this test")
        raw_qweight = sample_valid_fp8((k, n), weight_type, device)
        decoded = decode_fp8_to_float(raw_qweight.cpu(), weight_type).to(odt).to(device)
        zp = torch.Tensor()
        ref_dst = decoded * scale_re.to(odt)
        qweight_for_pack = raw_qweight
    else:
        if scale_type == "fp8_e8m0":
            pytest.skip("fp8_e8m0 scale is only supported for fp8 weights")
        raw_s8_wei = gen_weis8(weight_type, device, k, n)
        if asym:
            zp = gen_weis8(weight_type, device, k // blocksize, n)
            zp_re = zp.repeat_interleave(repeats=blocksize, dim=0)
            ref_dst = (raw_s8_wei.to(sdt) - zp_re.to(sdt)) * scale_re
        else:
            zp = torch.Tensor()
            ref_dst = raw_s8_wei.to(sdt) * scale_re
        qweight_for_pack = raw_s8_wei

    packw = ark.repack_quantized_weight(
        qweight_for_pack, scale, zp, blocksize, compute_type, weight_type, scale_type, asym
    )
    revert_wei = ark.unpack_weight(packw, odt, n, k, blocksize, compute_type, weight_type, scale_type, asym)
    ref_dst = ref_dst.T
    compare2(revert_wei, ref_dst)
    atol = 0.001
    assert torch.allclose(revert_wei, ref_dst, atol=atol)


@pytest.mark.parametrize("k, n", [(1024, 1024), (512, 2048)])
@pytest.mark.parametrize("blocksize", [16, 32, 128])
@pytest.mark.parametrize("weight_type", ["int2", "int4", "int8", "fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize(
    "scale_type, compute_type", [("fp16", "fp16"), ("fp32", "fp32"), ("fp8_e8m0", "fp16"), ("fp8_e8m0", "fp32")]
)
@pytest.mark.parametrize("asym", [False, True])
def test_xpu(k, n, weight_type, scale_type, asym, blocksize, compute_type):
    if not torch.xpu.is_available():
        pytest.skip("No XPU Device")
    with torch.no_grad():
        main_op(k, n, weight_type, scale_type, asym, blocksize, "xpu", compute_type)
        torch.xpu.empty_cache()


@pytest.mark.parametrize("k, n", [(1024, 1024), (512, 2048)])
@pytest.mark.parametrize("blocksize", [32, 128])
@pytest.mark.parametrize("weight_type", ["int2", "int4", "int8"])
@pytest.mark.parametrize("scale_type", ["fp32"])
@pytest.mark.parametrize("compute_type", ["auto", "int8", "bf16", "fp32"])
@pytest.mark.parametrize("asym", [True, False])
def test_cpu(k, n, weight_type, scale_type, asym, blocksize, compute_type):
    main_op(k, n, weight_type, scale_type, asym, blocksize, "cpu", compute_type)


# LOCAL_TEST: python test_packq.py
# CI pytest: pytest -v test_packq.py -k 'test_'
if __name__ == "__main__":
    # main_op(1024, 1024, 'int2', 'fp32', True, 32, 'cpu', 'fp32')
    # main_op(1024, 1024, 'int2', 'fp32', False, 32, 'cpu', 'fp32')
    # main_op(1024, 1024, 'int4', 'fp32', False, 32, 'cpu', 'fp32')
    # main_op(1024, 1024, 'int4', 'fp32', False, 32, 'cpu', 'bf16')
    # main_op(1024, 1024, 'int4', 'fp32', False, 32, 'cpu', 'int8')
    # main_op(1024, 1024, 'int8', 'fp32', False, 128, 'cpu', 'int8')

    # main_op(512, 2048, "int2", "fp32", False, 128, "cpu", "fp32")
    main_op(1024, 1024, "int2", "fp32", True, 32, "xpu", None)
    main_op(1024, 1024, "int4", "fp32", True, 32, "xpu", None)
    # main_op(1024, 1024, "int8", "fp32", False, 32, "xpu", None)
    # main_op(1024, 1024, "int4", "fp16", False, 32, "xpu", None)

    # main_op(1024, 1024, "fp8_e4m3", "fp16", False, 32, "xpu", None)
    # main_op(1024, 1024, "fp8_e4m3", "fp32", False, 32, "xpu", None)
    # main_op(1024, 1024, "fp8_e4m3", "fp8_e8m0", False, 32, "xpu", None)
    # main_op(1024, 1024, "fp8_e5m2", "fp16", False, 32, "xpu", None)
    # main_op(1024, 1024, "fp8_e5m2", "fp32", False, 32, "xpu", None)
