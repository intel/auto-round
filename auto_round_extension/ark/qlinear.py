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

from auto_round.export.export_to_awq.utils import dequantize_gemm, reverse_awq_order, unpack_awq
from auto_round.utils import convert_dtype_torch2str, logger

try:
    import auto_round_kernel as ark

    ARK_INSTALLED = True
except:
    ARK_INSTALLED = False

BITS_DTYPE_MAPPING = {
    2: "int2",
    4: "int4",
    8: "int8",
}


class QuantLinearAWQ(nn.Module):
    QUANT_TYPE = "ark_awq"

    def __init__(self, w_bit, group_size, in_features, out_features, bias, zero_point, dev):
        super().__init__()
        if not ARK_INSTALLED:
            raise ModuleNotFoundError(
                "The 'auto_round_kernel' module is required but not installed. "
                "Please install the 'auto-round-kernel' package, for example:\n"
                "  pip install auto-round-kernel"
            )

        self.use_bf16 = ark.check_isa_supported("AMX")

        if w_bit not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2, 3, 4, 8 bits are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.zero_point = zero_point
        self.scale_dtype = torch.float32
        ark.set_threads(torch.get_num_threads())

        if not self.in_features % self.group_size == 0:
            raise NotImplementedError("")

        self.pack_num = 32 // self.w_bit
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // self.pack_num),
                dtype=torch.int32,
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
        self.asym = self.qzeros.min() != self.qzeros.max()
        intweight, zeros = unpack_awq(self.qweight, self.qzeros, self.w_bit)  # weight: k x n zeros: k / group_size x n
        intweight, zeros = reverse_awq_order(intweight, zeros, self.w_bit)  # weight: k x n zeros: k / group_size x n
        intweight = torch.bitwise_and(intweight, (2**self.w_bit) - 1)
        zeros = torch.bitwise_and(zeros, (2**self.w_bit) - 1)

        intweight = intweight - (2 ** (self.w_bit - 1))  # from uint8_t to int8_t
        zeros = zeros - (2 ** (self.w_bit - 1))  # from uint8_t to int8_t

        if self.qweight.device.type == "xpu":
            self.sdt = "fp16"
            self.cdt = "fp16"
            scales = self.scales.to(torch.float16).contiguous()
        else:
            self.sdt = "fp32"
            self.cdt = "auto"
            scales = self.scales.float().contiguous()
        self.wdt = BITS_DTYPE_MAPPING[self.w_bit]
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
        if self.bias is not None:
            if self.bias.device.type == "cpu":
                self.bias = self.bias.to(torch.float32)
            else:
                self.bias = self.bias.to(torch.float16)
        else:
            self.bias = torch.Tensor()

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
        raw_input_dtype = x.dtype
        if x.device.type == "cpu":
            odt = torch.float32
            self.bias = self.bias.to(torch.float32)
            if raw_input_dtype != torch.float32:
                x = x.to(torch.float32)
        else:
            odt = x.dtype

        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.view(-1, x.shape[-1])  # convert xd to 2d
        out_2d_shape = x.shape[:-1] + (self.out_features,)
        outputs = torch.empty(out_2d_shape, device=x.device, dtype=odt)

        ark.woq_linear(
            x,
            self.qweight,
            self.bias,
            outputs,
            self.cdt,  # compute_dtype
            self.wdt,  # weight_dtype
            self.sdt,  # scale_dtype
            self.asym,
            self.group_size,
        )
        if x.device.type == "xpu":
            outputs = outputs + self.bias
        return outputs.to(raw_input_dtype).view(out_shape)

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
        if not ARK_INSTALLED:
            raise ModuleNotFoundError("Please install auto_round_kernel.")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1
        self.weight_dtype = weight_dtype
        self.asym = True
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
        if self.qweight.device.type not in ["cpu", "xpu"]:
            raise NotImplementedError("Only for CPU/XPU")
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
        if self.qweight.device.type != "cpu" and self.asym:
            raise NotImplementedError()
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
            self.cdt = "auto"
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

        # free mem
        self.qzeros = torch.empty(0)
        self.scales = torch.empty(0)
        if self.bias is not None:
            if self.bias.device.type == "cpu":
                self.bias = self.bias.to(torch.float32)
            else:
                self.bias = self.bias.to(torch.float16)
        else:
            self.bias = torch.Tensor()

    def forward(self, x: torch.Tensor):
        raw_input_dtype = x.dtype
        if x.device.type == "cpu":
            odt = torch.float32
            self.bias = self.bias.to(torch.float32)
            if raw_input_dtype != torch.float32:
                x = x.to(torch.float32)
        else:
            odt = x.dtype

        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.view(-1, x.shape[-1])  # convert xd to 2d
        out_2d_shape = x.shape[:-1] + (self.outfeatures,)
        outputs = torch.empty(out_2d_shape, device=x.device, dtype=odt)

        ark.woq_linear(
            x,
            self.qweight,
            self.bias,
            outputs,
            self.cdt,  # compute_dtype
            self.wdt,  # weight_dtype
            self.sdt,  # scale_dtype
            self.asym,
            self.group_size,
        )
        if x.device.type == "xpu":
            outputs = outputs + self.bias
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
        if isinstance(submodule, QuantLinear):
            submodule.post_init()

    return model


__all__ = ["QuantLinear", "QuantLinearGPTQ", "QuantLinearAWQ"]
