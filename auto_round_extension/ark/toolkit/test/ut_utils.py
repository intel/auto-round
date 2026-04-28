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

import torch
import inspect
import time
import pytest
from functools import wraps
import auto_round_kernel


def is_xpu_available():
    return torch.xpu.is_available()


def capture_args(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(f)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        arg_strs = []
        for name, value in bound_args.arguments.items():
            arg_strs.append(f"{name}={value}")
        result = ", ".join(arg_strs)
        print(result)
        return f(*args, **kwargs)

    return wrapper


def get_torch_dt(dtstr):
    if dtstr == "fp16":
        return torch.float16
    if dtstr == "bf16":
        return torch.bfloat16
    return torch.float32


def get_scale_torch_dt(scale_type: str):
    if scale_type in {"fp16", "bf16"}:
        return get_torch_dt(scale_type)
    # fp8_e8m0 scales are passed as float32 exponents (int-valued)
    return torch.float32


def compare2(a, b):
    diff = abs(a - b)
    print(diff.max(), diff.mean(), a.max(), a.mean())


def gen_weis8(weight_type, device, k, n):
    if weight_type == "int4":
        raw_s8_wei = torch.randint(-8, 7, (k, n), dtype=torch.int8, device=device)
    if weight_type == "int2":
        raw_s8_wei = torch.randint(-2, 1, (k, n), dtype=torch.int8, device=device)
    if weight_type == "int8":
        raw_s8_wei = torch.randint(-128, 127, (k, n), dtype=torch.int8, device=device)
    return raw_s8_wei


def decode_fp8_to_float(raw_bytes: torch.Tensor, fmt: str) -> torch.Tensor:
    """Decode FP8 E4M3/E5M2 payloads into float32 tensors"""
    x = raw_bytes.to(torch.uint8).to(torch.int32)
    sign = (x >> 7) & 0x1
    if fmt == "fp8_e4m3":
        exp = (x >> 3) & 0xF
        mant = x & 0x7
        exp_f = exp.to(torch.float32)
        mant_f = mant.to(torch.float32)
        val = torch.zeros_like(exp_f, dtype=torch.float32)
        exp_max = 0xF
        bias = 7
        sub_mask = (exp == 0) & (mant != 0)
        nan_mask = (exp == exp_max) & (mant == 7)
        norm_mask = (exp != 0) & (exp != exp_max)
        max_mask = (exp == exp_max) & (mant != 7)
        val = torch.where(sub_mask, mant_f * 0.001953125, val)
        norm_val = torch.pow(2.0, exp_f - float(bias)) * (1.0 + mant_f * 0.125)
        val = torch.where(norm_mask, norm_val, val)
        max_base = 2.0 ** (exp_max - bias)
        max_val = max_base * (1.0 + mant_f * 0.125)
        val = torch.where(max_mask, max_val, val)
        val = torch.where(nan_mask, torch.tensor(float("nan"), dtype=torch.float32), val)
    else:
        exp = (x >> 2) & 0x1F
        mant = x & 0x3
        exp_f = exp.to(torch.float32)
        mant_f = mant.to(torch.float32)
        val = torch.zeros_like(exp_f, dtype=torch.float32)
        exp_max = 0x1F
        bias = 15
        sub_mask = (exp == 0) & (mant != 0)
        inf_mask = (exp == exp_max) & (mant == 0)
        nan_mask = (exp == exp_max) & (mant != 0)
        norm_mask = (exp != 0) & (exp != exp_max)
        val = torch.where(sub_mask, mant_f * 0.0000152587890625, val)
        norm_val = torch.pow(2.0, exp_f - float(bias)) * (1.0 + mant_f * 0.25)
        val = torch.where(norm_mask, norm_val, val)
        val = torch.where(inf_mask, torch.tensor(float("inf"), dtype=torch.float32), val)
        val = torch.where(nan_mask, torch.tensor(float("nan"), dtype=torch.float32), val)
    sign_mask = torch.where(sign.bool(), -torch.ones_like(val), torch.ones_like(val))
    return val * sign_mask


def sample_valid_fp8(shape, fmt: str, device):
    """Generate FP8 payload bytes that avoid NaN encodings."""
    if isinstance(device, str):
        device = torch.device(device)
    sign = torch.randint(0, 2, shape, dtype=torch.int32, device=device)
    if fmt == "fp8_e4m3":
        exp = torch.randint(0, 15, shape, dtype=torch.int32, device=device)
        mant = torch.randint(0, 8, shape, dtype=torch.int32, device=device)
        nan_mask = (exp == 15) & (mant == 7)
        mant = torch.where(nan_mask, torch.randint(0, 7, shape, dtype=torch.int32, device=device), mant)
        payload = (sign << 7) | (exp << 3) | mant
    else:
        exp = torch.randint(1, 31, shape, dtype=torch.int32, device=device)
        mant = torch.randint(0, 4, shape, dtype=torch.int32, device=device)
        nan_mask = (exp == 31) & (mant != 0)
        mant = torch.where(nan_mask, torch.zeros_like(mant), mant)
        payload = (sign << 7) | (exp << 2) | mant
    payload = torch.where(payload >= 128, payload - 256, payload)
    return payload.to(torch.int8)


def sample_valid_fp8_e8m0_xpu_safe(shape, fmt: str, device, *, exp_range=None):
    """XPU-safe FP8 payload generator for fp8_e8m0 CI.

    CI B580 is sensitive to large temporary allocations on XPU when generating
    int32 random tensors; generate on CPU with uint8 intermediates and copy.
    """
    if isinstance(device, str):
        device = torch.device(device)
    cpu = torch.device("cpu")
    sign = torch.randint(0, 2, shape, dtype=torch.uint8, device=cpu)
    if fmt == "fp8_e4m3":
        low, high = exp_range if exp_range is not None else (0, 15)
        exp = torch.randint(low, high, shape, dtype=torch.uint8, device=cpu)
        mant = torch.randint(0, 8, shape, dtype=torch.uint8, device=cpu)
        nan_mask = (exp == 15) & (mant == 7)
        if nan_mask.any():
            mant[nan_mask] = torch.randint(0, 7, (int(nan_mask.sum().item()),), dtype=torch.uint8, device=cpu)
        payload_u8 = (sign << 7) | (exp << 3) | mant
    else:
        low, high = exp_range if exp_range is not None else (1, 31)
        exp = torch.randint(low, high, shape, dtype=torch.uint8, device=cpu)  # avoid exp==31
        mant = torch.randint(0, 4, shape, dtype=torch.uint8, device=cpu)
        payload_u8 = (sign << 7) | (exp << 2) | mant
    return payload_u8.view(torch.int8).to(device)
