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

# Copyright (c) 2023 MIT HAN Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gc
import torch
import warnings
import torch.nn as nn
from torch.autograd import Function


def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

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

AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

def reverse_awq_order(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    reverse_order_tensor = torch.arange(
        iweights.shape[-1],
        dtype=torch.int32,
        device=izeros.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]

    return iweights, izeros


def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    # Reverse the order of the iweight and izeros tensors
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    # overflow checks
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    # fp16 weights
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    iweight = (iweight - izeros) * scales

    return iweight


def get_best_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"


class WQLinearMMFunction(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(
        ctx,
        x,
        qweight,
        qzeros,
        scales,
        w_bit=4,
        group_size=128,
        bias=None,
        out_features=0,
    ):
        # The forward pass can use ctx.
        ctx.save_for_backward(x, qweight, qzeros, scales, bias)
        ctx.out_features = out_features

        out_shape = x.shape[:-1] + (out_features,)
        x = x.to(torch.float16)

        out = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
        out = torch.matmul(x, out)

        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)

        # always want 3D tensor if tensor is 2D
        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        return out


class WQLinear_GEMM(nn.Module):
    def __init__(
        self, w_bit, group_size, in_features, out_features, bias, dev, training=False
    ):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.training = training

        # quick sanity check (make sure alignment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
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

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit

        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[idx // group_size])
                    / awq_linear.scales[idx // group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)

        best_device = get_best_device()

        # Avoid: The operator 'aten::__lshift__.Scalar' is not currently implemented for the MPS device
        if "mps" in best_device:
            intweight = intweight.to("cpu")

        qweight = torch.zeros(
            (intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=intweight.device,
        )

        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32, device=best_device)

        if "mps" in best_device:
            zeros = zeros.to("cpu")

        qzeros = torch.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=zeros.device,
        )

        for col in range(zeros.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros

        return awq_linear

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)

        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()

        if self.training:
            out = WQLinearMMFunction.apply(
                x,
                self.qweight,
                self.qzeros,
                self.scales,
                self.w_bit,
                self.group_size,
                self.bias,
                self.out_features,
            )
        else:
            with torch.no_grad():
                out = WQLinearMMFunction.apply(
                    x,
                    self.qweight,
                    self.qzeros,
                    self.scales,
                    self.w_bit,
                    self.group_size,
                    self.bias,
                    self.out_features,
                )

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )


def clear_memory(weight=None):
    if weight is not None:
        del weight
    gc.collect()
    torch.cuda.empty_cache()


MODEL_LAYERS_BY_MODEL_TYPE = {
    'aquila': 'model.layers',
    'baichuan': 'model.layers',
    'bloom': 'transformer.h',
    'cohere': 'model.layers',
    'deepseek_v2': 'model.layers',
    'falcon': 'transformer.h',
    'gemma': 'model.layers',
    'gemma2': 'model.layers',
    'gpt2': 'transformer.h',
    'gpt_bigcode': 'transformer.h',
    'gpt_neo': 'transformer.h',
    'gpt_neox': 'gpt_neox.layers',
    'gptj': 'transformer.h',
    'internlm2': 'model.layers',
    'llama': 'model.layers',
    'llava': 'language_model.model.layers',
    'llava_next': 'language_model.model.layers',
    'minicpm': 'model.layers',
    'mistral': 'model.layers',
    'mixtral': 'model.layers',
    'mpt': 'transformer.blocks',
    'opt': 'model.decoder.layers',
    'phi': 'model.layers',
    'phi3': 'model.layers',
    'qwen': 'transformer.h',
    'qwen2': 'model.layers',
    'roberta': 'roberta.encoder',
    'stablelm': 'model.layers',
    'starcoder2': 'model.layers',
    'xlm-roberta': 'roberta.encoder',
    'yi': 'model.layers',
}

def get_self_modules(model):
    def _get(model, self_modules):
        module = model
        for m in self_modules.split('.'):
            module = getattr(module, m)
        return module

    model_type = model.config.model_type
    if model_type not in MODEL_LAYERS_BY_MODEL_TYPE:
        for tmp_self_modules in set(MODEL_LAYERS_BY_MODEL_TYPE.values()):
            try:
                return _get(model, tmp_self_modules)
            except:
                continue
        warnings.warn(f"Model type {model_type} may be not supported by auto_awq,"
                      " please checkout the auto_awq repository.")
        return None
    self_modules = MODEL_LAYERS_BY_MODEL_TYPE.get(model_type)
    return _get(model, self_modules)
