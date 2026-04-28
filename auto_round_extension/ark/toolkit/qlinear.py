# Copyright (c) 2025 Intel Corporation
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

import math

import torch
import torch.nn as nn

from auto_round.utils import convert_dtype_torch2str, logger

try:
    from auto_round_extension.ark import auto_round_kernel as ark

    ARK_INSTALLED = True
except:
    ARK_INSTALLED = False

BITS_DTYPE_MAPPING = {
    2: "int2",
    4: "int4",
    8: "int8",
}

FP8_DTYPE_MAPPING = {
    "fp8_e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
}

AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device="cpu")

    # unpacking columnwise
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    iweights = iweights.view(iweights.shape[0], -1)

    # unpacking columnwise
    if qzeros is not None:
        izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
            torch.int8  # smallest dtype available
        )
        izeros = izeros.view(izeros.shape[0], -1)
    else:
        izeros = qzeros

    return iweights, izeros


def reverse_awq_order(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    reverse_order_tensor = torch.arange(
        iweights.shape[-1],
        dtype=torch.int32,
        device="cpu",
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]
    return iweights, izeros


def convert_dtype_torch2str(dtype):
    if dtype == torch.int8:
        return "int8"
    elif dtype == torch.float:
        return "fp32"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif isinstance(dtype, str) and dtype in ["int8", "fp32", "fp16", "bf16"]:
        return dtype
    else:
        assert False, "Unsupported pytorch dtype {} to str dtype".format(dtype)


class QuantLinearAWQ(nn.Module):
    QUANT_TYPE = "ark_awq"

    def __init__(self, w_bit, group_size, in_features, out_features, bias, zero_point, dev):
        super().__init__()
        assert ARK_INSTALLED, "Please install auto_round_kernel package."

        self.use_bf16 = ark.check_isa_supported("AMX")

        if w_bit not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2, 3, 4, 8 bits are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.zero_point = zero_point
        self.scale_dtype = torch.float32

        # quick sanity check (make sure alignment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        self.pack_num = 32 // self.w_bit
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // self.pack_num),
                dtype=torch.int8,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.bfloat16 if self.use_bf16 else torch.float32,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((out_features), dtype=torch.bfloat16 if self.use_bf16 else torch.float32, device=dev),
            )
        else:
            self.register_buffer(
                "bias",
                None,
            )
        qweight = torch.zeros((in_features, out_features // self.pack_num), dtype=torch.int32, device=dev)
        self.register_buffer("qweight", qweight)

    def post_init(self):
        assert self.qweight.device.type == "cpu"

        intweight, zeros = unpack_awq(self.qweight, self.qzeros, self.w_bit)  # weight: k x n zeros: k / group_size x n
        intweight, zeros = reverse_awq_order(intweight, zeros, self.w_bit)  # weight: k x n zeros: k / group_size x n
        if self.zero_point:  ## asym has accuracy issue, have not root caused yet
            intweight = torch.bitwise_and(intweight, (2**self.w_bit) - 1) - (2 ** (self.w_bit - 1))
            zeros = torch.bitwise_and(zeros, (2**self.w_bit) - 1) - (2 ** (self.w_bit - 1))
        else:
            ##symmetric, our default zp is 8
            intweight = torch.bitwise_and(intweight, (2**self.w_bit) - 1) - (2 ** (self.w_bit - 1))
        g_idx = torch.empty(0, dtype=torch.int32)
        self.qweight = ark.repack_quantized_weight(
            intweight,
            self.scales.float(),
            zeros,
            g_idx,
            BITS_DTYPE_MAPPING[self.w_bit],
            convert_dtype_torch2str(self.scale_dtype),
            convert_dtype_torch2str(self.scales.dtype),
            self.zero_point,
            self.group_size,
        )

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None, has_zero_points=False):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            has_zero_points,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        raise NotImplementedError("Only inference is supported for Exllama kernels")

    @torch.no_grad()
    def forward(self, x):
        assert ARK_INSTALLED, "ARK kernels could not be loaded. "

        input_dtype = x.dtype
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.view(-1, x.shape[-1])  # convert xd to 2d
        out_2d_shape = x.shape[:-1] + (self.out_features,)

        outputs = torch.zeros(out_2d_shape, dtype=input_dtype)
        bias = (
            self.bias
            if self.bias is not None
            else torch.empty(0, dtype=torch.bfloat16 if self.use_bf16 else torch.float32)
        )

        ark.woq_linear(
            x,
            self.qweight,
            bias,
            outputs,
            convert_dtype_torch2str(input_dtype),
            BITS_DTYPE_MAPPING[self.w_bit],
            convert_dtype_torch2str(self.scale_dtype),
            True,
        )

        return outputs.view(out_shape)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.w_bit,
            self.group_size,
        )


class QuantLinear(nn.Module):
    QUANT_TYPE = "ark_gptq_nozp"
    ZP_BIAS = 0

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        kernel_switch_threshold=128,
        trainable=False,
        weight_dtype=torch.bfloat16,
        **kwargs,
    ):
        super().__init__()

        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2, 4,8 bits are supported for ARK.")
        assert ARK_INSTALLED, "Please install auto_round_kernel."

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1
        self.weight_dtype = weight_dtype
        self.asym = True
        if hasattr(ark, "set_threads"):
            ark.set_threads(torch.get_num_threads())
        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(infeatures / self.group_size),
                    outfeatures // 32 * self.bits,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(infeatures / self.group_size), outfeatures),
                dtype=weight_dtype,
            ),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float))
        else:
            self.bias = None

        self.kernel_switch_threshold = kernel_switch_threshold
        self.trainable = trainable

    def post_init(self):
        assert self.qweight.device.type in ["cpu", "xpu"]
        # intweight: k x n, zeros: k / group_size x n
        intweight, zeros = unpack_to_8bit_signed(self.qweight, self.qzeros, self.bits, self.ZP_BIAS)
        if zeros is None:
            zeros = torch.empty(0, dtype=torch.int8)
            self.asym = False
        else:
            # change it to int8 with offset 128
            if self.bits == 8:
                zeros = (zeros.to(torch.int32) - (2 ** (self.bits - 1))).to(torch.int8)
            else:
                zeros -= 2 ** (self.bits - 1)
        if self.qweight.device.type != "cpu":
            assert not self.asym
        if not self.asym:
            intweight -= 2 ** (self.bits - 1)
        intweight = intweight.to(torch.uint8 if self.asym else torch.int8)
        # due to asym return torch.uint8 but backend request int8,
        # change it to int8 with offset 128
        if self.asym:
            intweight = (intweight.to(torch.int32) - (2 ** (self.bits - 1))).to(torch.int8)

        logger.debug(
            f"ARK repack quantized weight: K:{intweight.shape[0]}, N:{intweight.shape[1]}, weight_dtype:{BITS_DTYPE_MAPPING[self.bits]}, scale_dtype:fp32, compute_dtype:fp32, group_size:{self.group_size}"
        )

        if self.qweight.device.type == "xpu":
            self.sdt = "fp16"
            self.cdt = "fp16"
            scales = self.scales.to(torch.float16).contiguous()
        else:
            self.sdt = "fp32"
            self.cdt = "fp32"
            scales = self.scales.float().contiguous()
        self.wdt = BITS_DTYPE_MAPPING[self.bits]

        self.qweight = ark.repack_quantized_weight(
            intweight.contiguous(),
            scales,
            zeros.contiguous(),
            torch.Tensor(),
            # compute_dtype
            self.cdt,
            # weight_dtype
            self.wdt,
            # scale_dtype
            self.sdt,
            self.asym,
            self.group_size,
        )

        # self.revert_wei = torch.zeros(self.infeatures, self.outfeatures, dtype=scales.dtype, device=self.qweight.device)
        # # print(packw, packw.device, packw.dtype)
        # ark.dequantize_packed_weight(
        #     self.qweight, self.revert_wei, False, self.cdt, self.wdt, self.sdt, self.group_size, self.outfeatures, self.infeatures)
        # free mem
        self.qzeros = torch.empty(0)
        self.scales = torch.empty(0)
        if self.bias is not None:
            if self.bias.device.type == "cpu":
                self.bias = self.bias.to(torch.float32)
            else:
                self.bias = self.bias.to(torch.float16)

    def forward(self, x: torch.Tensor):
        raw_input_dtype = x.dtype
        if x.device.type == "cpu":
            odt = torch.float32
            if raw_input_dtype != torch.float32:
                x = x.to(torch.float32)
        else:
            odt = x.dtype

        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.view(-1, x.shape[-1])  # convert xd to 2d
        out_2d_shape = x.shape[:-1] + (self.outfeatures,)
        outputs = torch.empty(out_2d_shape, device=x.device, dtype=odt)
        if self.bias is None:
            bias = torch.empty(0, device=x.device, dtype=odt)
        else:
            bias = self.bias
            if bias.device != x.device:
                bias = bias.to(x.device)
            if bias.dtype != odt:
                bias = bias.to(odt)

        ark.woq_linear(
            x,
            self.qweight,
            bias,
            outputs,
            self.cdt,  # compute_dtype
            self.wdt,  # weight_dtype
            self.sdt,  # scale_dtype
            self.asym,
            self.group_size,
        )
        return outputs.to(raw_input_dtype).view(out_shape)


class QuantLinearGPTQ(QuantLinear):
    QUANT_TYPE = "ark_gptq"
    ZP_BIAS = 1


@torch.no_grad()
def unpack_to_8bit_signed(qweight, qzeros, bits, gptq_bias=1):
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32, device=qweight.device).unsqueeze(0)
    zeros = None
    if not torch.all(torch.eq(qzeros, 2004318071 if bits == 4 else 0b01111111011111110111111101111111)):
        zp_shape = list(qzeros.shape)
        zp_shape[1] = zp_shape[1] * (32 // bits)

        zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)).to(
            torch.int16 if bits == 8 else torch.int8
        )
        torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)
        if bits == 8:
            zeros = zeros.to(torch.uint8)
        zeros += gptq_bias
        try:
            zeros = zeros.reshape(zp_shape)
        except:
            # zeros and scales have different iteam numbers.
            # remove 1 (due to 0 + 1 in line 252)
            zeros = zeros[zeros != 1]
            zeros = zeros.reshape(zp_shape)
    weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)).to(
        torch.int16 if bits == 8 else torch.int8
    )
    weight.bitwise_and_((2**bits) - 1)
    weight = weight.view(-1, weight.shape[-1])

    return weight, zeros


# Copied from qlinear_marlin.py
@torch.no_grad()
def dequantize_weight(qweight, qzeros, scales, bits):
    unpacked_qweight, unpacked_qzeros = unpack_to_8bit_signed(qweight, qzeros, bits)
    group_size = unpacked_qweight.shape[0] // scales.shape[0]
    scales = scales.repeat_interleave(group_size, dim=0)
    if unpacked_qzeros is not None:
        unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
    else:
        unpacked_qzeros = torch.full_like(scales, 8 if bits == 4 else 128, dtype=torch.int32, device=qweight.device)
    unpacked_qweight = (unpacked_qweight - unpacked_qzeros) * scales

    return unpacked_qweight, unpacked_qzeros


def ark_post_init(model):
    for _, submodule in model.named_modules():
        if isinstance(submodule, (QuantLinear, QuantLinearFP8)):
            submodule.post_init()

    return model


class QuantLinearFP8(nn.Module):
    """Quantized Linear layer using FP8 format with ARK kernel."""

    QUANT_TYPE = "ark_fp8"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        kernel_switch_threshold=128,
        trainable=False,
        weight_dtype=torch.float16,
        data_type="fp8_e4m3",
        **kwargs,
    ):
        super().__init__()

        data_type = str(data_type)
        self.original_data_type = data_type
        self.is_mx = "mx_fp" in data_type.lower()
        # Normalize to the underlying FP8 weight format. MXFP8 stores weights in
        # FP8 (E4M3/E5M2) but uses E8M0 exponent-only scales.
        if self.is_mx:
            if "e5m2" in data_type.lower():
                data_type = "fp8_e5m2"
            else:
                data_type = "fp8_e4m3"
        else:
            # Support both fp8_e4m3 and fp8_e5m2 format names
            if data_type not in ["fp8_e4m3", "fp8_e5m2", "fp8"]:
                data_type = "fp8_e4m3"  # Default to E4M3
            if data_type == "fp8":
                data_type = "fp8_e4m3"

        assert ARK_INSTALLED, "Please install auto_round_kernel."

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = 8  # FP8 is always 8-bit
        self.group_size = group_size if group_size != -1 else infeatures
        self.weight_dtype = weight_dtype
        self.data_type = data_type
        if hasattr(ark, "set_threads"):
            ark.set_threads(torch.get_num_threads())

        # FP8 weights - load from safetensors with correct FP8 dtype
        # Shape is [outfeatures, infeatures] to match standard Linear layer
        # IMPORTANT: Must use float8 dtype to match safetensors, not uint8
        fp8_dtype = torch.float8_e5m2 if data_type == "fp8_e5m2" else torch.float8_e4m3fn
        self.register_buffer(
            "weight",
            torch.zeros((outfeatures, infeatures), dtype=fp8_dtype),
        )

        # Scales for dequantization - load from safetensors as "weight_scale"
        # Shape is [outfeatures, num_groups] to match per-channel quantization
        # where num_groups = ceil(infeatures / group_size)
        scale_storage_dtype = torch.uint8 if self.is_mx else weight_dtype
        self.register_buffer(
            "weight_scale",
            torch.zeros(
                (outfeatures, math.ceil(infeatures / self.group_size)),
                dtype=scale_storage_dtype,
            ),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float))
        else:
            self.bias = None

        self.kernel_switch_threshold = kernel_switch_threshold
        self.trainable = trainable

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Override to handle weight_scale shape mismatch between checkpoint and model."""
        weight_scale_key = prefix + "weight_scale"
        if weight_scale_key in state_dict:
            loaded_scale = state_dict[weight_scale_key]
            expected_shape = self.weight_scale.shape  # [outfeatures, num_groups]
            out_features, num_groups = expected_shape
            expected_numel = out_features * num_groups

            if loaded_scale.ndim == 2 and loaded_scale.shape == (num_groups, out_features):
                state_dict[weight_scale_key] = loaded_scale.t().contiguous()
                loaded_scale = state_dict[weight_scale_key]

            # If loaded scale is 1D but we expect 2D, reshape or broadcast it.
            if loaded_scale.ndim == 1 and len(expected_shape) == 2:
                numel = loaded_scale.numel()
                if numel == expected_numel:
                    state_dict[weight_scale_key] = loaded_scale.view(expected_shape)
                elif numel == out_features:
                    expanded = loaded_scale.view(out_features, 1).expand(-1, num_groups)
                    state_dict[weight_scale_key] = expanded.to(self.weight_scale.dtype)
                elif numel == num_groups:
                    expanded = loaded_scale.view(1, num_groups).expand(out_features, -1)
                    state_dict[weight_scale_key] = expanded.to(self.weight_scale.dtype)
                elif numel % out_features == 0:
                    inferred_num_groups = numel // out_features
                    inferred_shape = (out_features, inferred_num_groups)
                    state_dict[weight_scale_key] = loaded_scale.view(inferred_shape)
                    self.weight_scale = torch.zeros(
                        inferred_shape, dtype=self.weight_scale.dtype, device=self.weight_scale.device
                    )

                else:
                    logger.error(
                        f"Cannot reshape scale! loaded_scale.numel()={numel}, expected={expected_numel}, out_features={out_features}"
                    )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def post_init(self):
        target_fp8_dtype = torch.float8_e5m2 if self.data_type == "fp8_e5m2" else torch.float8_e4m3fn
        weight_tensor = self.weight
        converted_via_cpu = False
        if weight_tensor.dtype != target_fp8_dtype:
            weight_cpu = weight_tensor.detach().to("cpu")
            fp8_cpu = weight_cpu.to(target_fp8_dtype)
            converted_via_cpu = True
        else:
            fp8_cpu = weight_tensor.detach()
            if weight_tensor.device.type != "cpu":
                fp8_cpu = fp8_cpu.to("cpu")
                converted_via_cpu = True
        fp8_cpu = fp8_cpu.contiguous().view(self.outfeatures, self.infeatures)

        fp8_uint8_cpu = fp8_cpu.view(torch.uint8)
        if converted_via_cpu and self.weight.device.type != "cpu":
            fp8_uint8 = fp8_uint8_cpu.to(self.weight.device)
        else:
            fp8_uint8 = fp8_uint8_cpu
        fp8_weight = fp8_uint8.view(self.outfeatures, self.infeatures).t().contiguous()
        zeros = torch.empty(0, dtype=torch.uint8, device=fp8_weight.device)  # No zero points for FP8

        # Compute type selection (independent of how scales are stored).
        if self.weight.device.type == "xpu":
            self.cdt = "fp16"
        else:
            self.cdt = "fp32"

        if self.is_mx:
            # MXFP8 stores per-group scales as E8M0 exponents in uint8 with bias 127.
            # ARK expects fp8_e8m0 scales as float32 exponents (int-valued), which will
            # be packed to bestla::utils::f8 (int8 exponent) internally.
            self.sdt = "fp8_e8m0"
            if self.weight_scale.dtype == torch.uint8:
                exponents = self.weight_scale.to(torch.int16) - 127
            else:
                # Fallback: allow feeding exponents directly.
                exponents = self.weight_scale.to(torch.int16)
            exponents = exponents.clamp(-127, 127)
            scales = exponents.t().to(torch.float32).contiguous()  # [num_groups,out]
        else:
            if self.weight.device.type == "xpu":
                self.sdt = "fp16"
                scales = self.weight_scale.t().to(torch.float16).contiguous()  # [num_groups,out]
            else:
                self.sdt = "fp32"
                scales = self.weight_scale.t().float().contiguous()  # [num_groups,out]

        self.wdt = FP8_DTYPE_MAPPING[self.data_type]
        # Repack weights using ARK kernel
        packed_weight = ark.repack_quantized_weight(
            fp8_weight,
            scales,
            zeros.contiguous(),
            torch.empty(0, dtype=torch.int32, device=fp8_weight.device),
            self.cdt,  # compute_dtype
            self.wdt,  # weight_dtype (fp8_e4m3 or fp8_e5m2)
            self.sdt,  # scale_dtype
            False,  # asym = False for FP8
            self.group_size,
        )
        # Replace weight buffer with packed weight
        self.weight = packed_weight
        # Free original weight_scale
        self.weight_scale = torch.empty(0)
        if self.bias is not None:
            if self.bias.device.type == "cpu":
                self.bias = self.bias.to(torch.float32)
            else:
                self.bias = self.bias.to(torch.float16)

    def forward(self, x: torch.Tensor):
        raw_input_dtype = x.dtype
        if x.device.type == "cpu":
            odt = torch.float32
            if raw_input_dtype != torch.float32:
                x = x.to(torch.float32)
        else:
            odt = x.dtype
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.view(-1, x.shape[-1])  # convert xd to 2d
        out_2d_shape = x.shape[:-1] + (self.outfeatures,)
        outputs = torch.empty(out_2d_shape, device=x.device, dtype=odt)
        if self.bias is None:
            bias = torch.empty(0, device=x.device, dtype=odt)
        else:
            bias = self.bias
            if bias.device != x.device:
                bias = bias.to(x.device)
            if bias.dtype != odt:
                bias = bias.to(odt)
        ark.woq_linear(
            x,
            self.weight,
            bias,
            outputs,
            self.cdt,  # compute_dtype
            self.wdt,  # weight_dtype
            self.sdt,  # scale_dtype
            False,  # asym
            self.group_size,
        )
        return outputs.to(raw_input_dtype).view(out_shape)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, bits={}, " "group_size={}, data_type={}".format(
            self.infeatures,
            self.outfeatures,
            self.bias is not None,
            self.bits,
            self.group_size,
            self.data_type,
        )


__all__ = ["QuantLinear", "QuantLinearGPTQ", "QuantLinearAWQ", "QuantLinearFP8"]
