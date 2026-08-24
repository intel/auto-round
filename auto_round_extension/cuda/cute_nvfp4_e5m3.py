# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional CuTe DSL dispatch for NVFP4 E5M3 activation QDQ."""

from functools import lru_cache
from importlib.util import find_spec
from typing import Optional

import torch

from auto_round.logger import logger

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_GROUP_SIZE = 16
_THREADS_PER_BLOCK = 256
_FAILED_KERNEL_KEYS = set()
_FAILED_WEIGHT_DQ_KERNEL_KEYS = set()


@lru_cache(maxsize=1)
def is_cute_dsl_available() -> bool:
    """Return whether the optional NVIDIA CuTe DSL package is installed."""
    return find_spec("cutlass") is not None


def can_use_cute_nvfp4_v2_qdq(activation: torch.Tensor, group_size: int) -> bool:
    """Check whether an activation can use the CuTe QDQ kernel."""
    if not is_cute_dsl_available() or not activation.is_cuda:
        return False
    if group_size != _GROUP_SIZE or activation.dtype not in _SUPPORTED_DTYPES:
        return False
    if not activation.is_contiguous() or activation.shape[-1] % group_size:
        return False
    return torch.cuda.get_device_capability(activation.device)[0] >= 8


def _make_qdq_kernel():
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir_helpers import math

    @cute.kernel
    def qdq_kernel(input_tensor: cute.Tensor, output_tensor: cute.Tensor):
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        group_idx = block_idx * _THREADS_PER_BLOCK + thread_idx

        if group_idx < input_tensor.shape[0]:
            amax = cutlass.Float32(0.0)
            for column_idx in range(_GROUP_SIZE):
                value = cutlass.Float32(input_tensor[group_idx, column_idx])
                amax = cute.arch.fmax(amax, math.abs(value))

            # This reproduces the E5M3 scale domain used by nvfp4_v2 without
            # materializing scale or intermediate FP32 tensors in global memory.
            scale = amax / cutlass.Float32(6.0)
            if scale > cutlass.Float32(0.0):
                exponent = math.floor(math.log2(scale)) + cutlass.Float32(1.0)
                mantissa = scale / math.exp2(exponent)
                mantissa_bits = math.roundeven((mantissa - cutlass.Float32(0.5)) * cutlass.Float32(16.0))
                mantissa_bits = cute.arch.fmin(cute.arch.fmax(mantissa_bits, 0.0), 7.0)
                scale = (cutlass.Float32(1.0) + mantissa_bits / cutlass.Float32(8.0)) * math.exp2(
                    exponent - cutlass.Float32(1.0)
                )

            for column_idx in range(_GROUP_SIZE):
                value = cutlass.Float32(input_tensor[group_idx, column_idx])
                scaled = value / scale if scale > cutlass.Float32(0.0) else cutlass.Float32(0.0)
                magnitude = math.abs(scaled)
                quantized = cutlass.Float32(0.0)
                if magnitude < cutlass.Float32(2.0):
                    quantized = math.roundeven(magnitude * cutlass.Float32(2.0)) / cutlass.Float32(2.0)
                elif magnitude < cutlass.Float32(4.0):
                    quantized = math.roundeven(magnitude)
                else:
                    quantized = cutlass.Float32(2.0) * math.roundeven(magnitude / cutlass.Float32(2.0))
                quantized = cute.arch.fmin(cute.arch.fmax(quantized, 0.0), 6.0)
                if scaled < cutlass.Float32(0.0):
                    quantized = -quantized
                output_tensor[group_idx, column_idx] = (quantized * scale).to(output_tensor.element_type)

    @cute.jit
    def launch_qdq(input_tensor: cute.Tensor, output_tensor: cute.Tensor):
        groups = input_tensor.shape[0]
        qdq_kernel(input_tensor, output_tensor).launch(
            grid=((groups + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK, 1, 1),
            block=(_THREADS_PER_BLOCK, 1, 1),
        )

    return launch_qdq


def _make_weight_dq_kernel():
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir_helpers import math

    @cute.kernel
    def weight_dq_kernel(weight_packed: cute.Tensor, weight_scale: cute.Tensor, output_tensor: cute.Tensor):
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        group_idx = block_idx * _THREADS_PER_BLOCK + thread_idx

        if group_idx < output_tensor.shape[0]:
            scale_byte = cutlass.Float32(weight_scale[group_idx, 0])
            exponent = math.floor(scale_byte / cutlass.Float32(8.0))
            mantissa = scale_byte - exponent * cutlass.Float32(8.0)
            scale = (cutlass.Float32(1.0) + mantissa / cutlass.Float32(8.0)) * math.exp2(
                exponent - cutlass.Float32(15.0)
            )
            if exponent == cutlass.Float32(0.0):
                scale = mantissa * math.exp2(cutlass.Float32(-17.0))

            for column_idx in range(_GROUP_SIZE):
                packed = cutlass.Float32(weight_packed[group_idx, column_idx // 2])
                packed_high = math.floor(packed / cutlass.Float32(16.0))
                nibble = packed - packed_high * cutlass.Float32(16.0)
                if column_idx % 2:
                    nibble = packed_high

                sign = cutlass.Float32(1.0)
                if nibble >= cutlass.Float32(8.0):
                    sign = cutlass.Float32(-1.0)
                    nibble = nibble - cutlass.Float32(8.0)

                value = cutlass.Float32(0.0)
                if nibble == cutlass.Float32(1.0):
                    value = cutlass.Float32(0.5)
                elif nibble == cutlass.Float32(2.0):
                    value = cutlass.Float32(1.0)
                elif nibble == cutlass.Float32(3.0):
                    value = cutlass.Float32(1.5)
                elif nibble == cutlass.Float32(4.0):
                    value = cutlass.Float32(2.0)
                elif nibble == cutlass.Float32(5.0):
                    value = cutlass.Float32(3.0)
                elif nibble == cutlass.Float32(6.0):
                    value = cutlass.Float32(4.0)
                elif nibble == cutlass.Float32(7.0):
                    value = cutlass.Float32(6.0)

                output_tensor[group_idx, column_idx] = (sign * value * scale).to(output_tensor.element_type)

    @cute.jit
    def launch_weight_dq(weight_packed: cute.Tensor, weight_scale: cute.Tensor, output_tensor: cute.Tensor):
        groups = output_tensor.shape[0]
        weight_dq_kernel(weight_packed, weight_scale, output_tensor).launch(
            grid=((groups + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK, 1, 1),
            block=(_THREADS_PER_BLOCK, 1, 1),
        )

    return launch_weight_dq


@lru_cache(maxsize=None)
def _get_compiled_qdq_kernel(device_index: int, dtype: torch.dtype):
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    input_tensor = torch.empty((1, _GROUP_SIZE), device=f"cuda:{device_index}", dtype=dtype)
    output_tensor = torch.empty_like(input_tensor)
    return cute.compile(
        _make_qdq_kernel(),
        from_dlpack(input_tensor).mark_layout_dynamic(),
        from_dlpack(output_tensor).mark_layout_dynamic(),
    )


@lru_cache(maxsize=None)
def _get_compiled_weight_dq_kernel(device_index: int, dtype: torch.dtype):
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    weight_packed = torch.empty((2, _GROUP_SIZE // 2), device=f"cuda:{device_index}", dtype=torch.uint8)
    weight_scale = torch.empty((2, 1), device=f"cuda:{device_index}", dtype=torch.uint8)
    output_tensor = torch.empty((2, _GROUP_SIZE), device=f"cuda:{device_index}", dtype=dtype)
    return cute.compile(
        _make_weight_dq_kernel(),
        from_dlpack(weight_packed).mark_layout_dynamic(),
        from_dlpack(weight_scale).mark_layout_dynamic(),
        from_dlpack(output_tensor).mark_layout_dynamic(),
    )


def try_cute_nvfp4_v2_qdq(activation: torch.Tensor, group_size: int) -> Optional[torch.Tensor]:
    """Run a CuTe DSL group-size-16 FP4 QDQ kernel when eligible."""
    if not can_use_cute_nvfp4_v2_qdq(activation, group_size):
        return None
    if torch.cuda.is_current_stream_capturing():
        return None

    kernel_key = (activation.device.index, activation.dtype)
    if kernel_key in _FAILED_KERNEL_KEYS:
        return None

    try:
        groups = activation.numel() // group_size
        output = torch.empty_like(activation)
        compiled_qdq = _get_compiled_qdq_kernel(*kernel_key)
        compiled_qdq(activation.reshape(groups, group_size), output.reshape(groups, group_size))
        return output
    except Exception as error:
        _FAILED_KERNEL_KEYS.add(kernel_key)
        logger.warning_once("CuTe NVFP4 E5M3 QDQ failed; falling back to PyTorch reference: %s", error)
        return None


def try_cute_nvfp4_e5m3_weight_dq(
    weight_packed: torch.Tensor, weight_scale: torch.Tensor, dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """Dequantize packed FP4 E5M3 weights with CuTe."""
    if (
        not is_cute_dsl_available()
        or not weight_packed.is_cuda
        or weight_packed.dtype != torch.uint8
        or weight_scale.dtype != torch.uint8
        or weight_packed.ndim != 2
        or weight_scale.shape != (weight_packed.shape[0], weight_packed.shape[1] // (_GROUP_SIZE // 2))
        or dtype not in _SUPPORTED_DTYPES
        or torch.cuda.get_device_capability(weight_packed.device)[0] < 8
        or torch.cuda.is_current_stream_capturing()
    ):
        return None

    kernel_key = (weight_packed.device.index, dtype)
    if kernel_key in _FAILED_WEIGHT_DQ_KERNEL_KEYS:
        return None

    try:
        out_features = weight_packed.shape[0]
        in_features = weight_packed.shape[1] * 2
        groups = weight_scale.numel()
        packed_groups = weight_packed.reshape(groups, _GROUP_SIZE // 2)
        scale_groups = weight_scale.reshape(groups, 1)
        output = torch.empty((groups, _GROUP_SIZE), device=weight_packed.device, dtype=dtype)
        compiled_weight_dq = _get_compiled_weight_dq_kernel(*kernel_key)
        compiled_weight_dq(packed_groups, scale_groups, output)
        return output.reshape(out_features, in_features)
    except Exception as error:
        _FAILED_WEIGHT_DQ_KERNEL_KEYS.add(kernel_key)
        logger.warning_once("CuTe NVFP4 E5M3 weight DQ failed; falling back to PyTorch reference: %s", error)
        return None


def try_cute_nvfp4_e5m3_linear(
    activation: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Reserved second-stage dispatch point for fused QDQ, unpack, and GEMM.

    Returning ``None`` keeps the reference Linear path active until the packed-weight
    mainloop has been validated against the existing E5M3 checkpoint format.
    """
    del activation, weight_packed, weight_scale, bias
    fused_output: Optional[torch.Tensor] = None
    return fused_output
