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

import torch
import torch.nn as nn

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


try:
    from intel_extension_for_transformers import qbits  # with QBits kernels ()

    QBITS_INSTALLED = True
except:
    QBITS_INSTALLED = False

BITS_DTYPE_MAPPING = {
    4: "int4_clip",
    8: "int8",
}


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


class QuantLinear(nn.Module):
    QUANT_TYPE = "qbits_awq"

    def __init__(self, w_bit, group_size, in_features, out_features, bias, zero_point, dev):
        super().__init__()
        assert (
            QBITS_INSTALLED
        ), "Please install ITREX qbits package with `pip install intel-extension-for-transformers`."

        self.use_bf16 = qbits.check_isa_supported("AMX")

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
        self.qweight = qbits.repack_quantized_weight(
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
        assert QBITS_INSTALLED, (
            "QBits kernels could not be loaded. "
            "Please install with `pip install intel-extension-for-transformers` and "
            "refer to the detail https://github.com/intel/intel-extension-for-transformers/blob/main/docs/qbits.md"
        )

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

        qbits.woq_linear(
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


def qbits_post_init(model):
    for _, submodule in model.named_modules():
        if isinstance(submodule, QuantLinear):
            submodule.post_init()

    return model
