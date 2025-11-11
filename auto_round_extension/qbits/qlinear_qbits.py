# Copyright (c) 2024 Intel Corporation
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

import numpy as np
import torch
import torch.nn as nn

from auto_round.utils import convert_dtype_torch2str, logger

QBITS_AVAILABLE = True

BITS_DTYPE_MAPPING = {
    2: "int2_clip",
    4: "int4_clip",
    8: "int8",
}


class QuantLinear(nn.Module):
    QUANT_TYPE = "qbits"

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
            raise NotImplementedError("Only 2, 4,8 bits are supported for QBits.")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1
        self.weight_dtype = weight_dtype
        self.asym = True
        self.qbits = None

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

    def req_check(self):
        torch_version = str(torch.__version__)
        if QBITS_AVAILABLE:
            pass
            # import intel_extension_for_transformers
            # itrex_version = str(intel_extension_for_transformers.__version__)
            # version_match_map = {"1.4": "2.2.0+cpu",
            #                      "1.4.1": "2.2.0+cpu", "1.4.2": "2.3.0+cpu"}
            # if itrex_version in version_match_map:
            #     if torch_version != version_match_map[itrex_version]:
            #         logger.warning(
            #             f"Please install torch {version_match_map[itrex_version]} by command 'pip install torch=={version_match_map[itrex_version]} --extra-index-url https://download.pytorch.org/whl/cpu' as Intel Extension for Transformers {itrex_version} is not compatible with current torch.")
        else:
            logger.error(
                "Please install Intel Extension for Transformers by running 'pip install intel-extension-for-transformers' as qbits linear requirements checking fail. "
            )
            exit(1)

    def post_init(self):
        import intel_extension_for_transformers

        self.qbits = intel_extension_for_transformers.qbits
        assert self.qweight.device.type == "cpu"
        if self.bias is not None:
            self.bias = self.bias.to(dtype=torch.float32)

        # intweight: k x n, zeros: k / group_size x n
        intweight, zeros = unpack_to_8bit_signed(self.qweight, self.qzeros, self.bits)
        if zeros is None:
            zeros = torch.empty(0, dtype=torch.int8)
            self.asym = False
        else:
            # change it to int8 with offset 128
            if self.bits == 8:
                zeros = (zeros.to(torch.int32) - (2 ** (self.bits - 1))).to(torch.int8)
            else:
                zeros -= 2 ** (self.bits - 1)

        if not self.asym:
            intweight -= 2 ** (self.bits - 1)
        intweight = intweight.to(torch.uint8 if self.asym else torch.int8)
        # due to asym return torch.uint8 but backend request int8,
        # change it to int8 with offset 128
        if self.asym:
            intweight = (intweight.to(torch.int32) - (2 ** (self.bits - 1))).to(torch.int8)

        scales = self.scales

        logger.debug(
            f"QBits repack quantized weight: K:{intweight.shape[0]}, N:{intweight.shape[1]}, weight_dtype:{BITS_DTYPE_MAPPING[self.bits]}, scale_dtype:fp32, compute_dtype:fp32, group_size:{self.group_size}"
        )
        self.qweight = self.qbits.repack_quantized_weight(
            intweight.contiguous(),
            scales.float().contiguous(),
            zeros.contiguous(),
            torch.empty(0),
            # weight_dtype
            BITS_DTYPE_MAPPING[self.bits],
            # scale_dtype
            "fp32",
            # TODO(zhe): consider dynamic-set cmpt for better perf?
            "fp32",
            self.asym,
            self.group_size,
        )
        # free mem
        self.qzeros = torch.empty(0)
        self.scales = torch.empty(0)

    def forward(self, x: torch.Tensor):
        raw_input_dtype = x.dtype
        if raw_input_dtype != torch.float32:
            x = x.to(torch.float32)
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.view(-1, x.shape[-1])  # convert xd to 2d
        out_2d_shape = x.shape[:-1] + (self.outfeatures,)

        outputs = torch.zeros(out_2d_shape, device=x.device, dtype=torch.float)
        bias = self.bias if self.bias is not None else torch.empty(0, dtype=torch.float)

        self.qbits.woq_linear(
            x,
            self.qweight,
            bias,
            outputs,
            convert_dtype_torch2str(torch.float),  # compute_dtype
            BITS_DTYPE_MAPPING[self.bits],  # weight_dtype
            "fp32",  # scale_dtype
            self.asym,
        )
        return outputs.to(raw_input_dtype).view(out_shape)


@torch.no_grad()
def unpack_to_8bit_signed(qweight, qzeros, bits):
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
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
        unpacked_qzeros = torch.full_like(scales, 8 if bits == 4 else 128, dtype=torch.int32)
    unpacked_qweight = (unpacked_qweight - unpacked_qzeros) * scales

    return unpacked_qweight, unpacked_qzeros


__all__ = ["QuantLinear"]
