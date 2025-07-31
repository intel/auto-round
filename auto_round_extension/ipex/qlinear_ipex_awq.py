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


class QuantLinear(nn.Module):
    QUANT_TYPE = "ipex_awq"

    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        assert w_bit == 4, "Only 4 bit are supported for now."
        self.compute_dtype = torch.float16 if torch.xpu.is_available() else torch.bfloat16
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.scale_dtype = torch.float32

        # quick sanity check (make sure alignment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
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
                dtype=self.compute_dtype,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((out_features), dtype=self.compute_dtype, device=dev),
            )
        else:
            self.register_buffer(
                "bias",
                None,
            )
        qweight = torch.zeros((in_features, out_features // self.pack_num), dtype=torch.int32, device=dev)
        self.register_buffer("qweight", qweight)

    def post_init(self):
        assert self.qweight.device.type == "cpu" or self.qweight.device.type == "xpu"
        import intel_extension_for_pytorch as ipex

        self.ipex_linear = ipex.llm.quantization.IPEXWeightOnlyQuantizedLinear.from_weight(
            self.qweight,
            self.scales,
            self.qzeros,
            self.in_features,
            self.out_features,
            None,
            self.bias,
            self.group_size,
            None,
            1,
            0,
        )

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        raise NotImplementedError("Only inference is supported for IPEX kernels")

    @torch.no_grad()
    def forward(self, x):

        outputs = self.ipex_linear(x)

        return outputs

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.w_bit,
            self.group_size,
        )
