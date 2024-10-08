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

# Adapted from turboderp exllama: https://github.com/turboderp/exllamav2
# MIT License
#
# Copyright (c) 2023 潘其威(William)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math
from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger(__name__)

try:
    from autoround_exllamav2_kernels import gemm_half_q_half, make_q_matrix
except ImportError as e:
    exllama_v2_import_exception = e


    def error_raiser_exllama(*args, **kwargs):
        raise ValueError(
            f"Trying to use the exllama v2 backend, but could not import the C++/CUDA dependencies with the following "
            f"error: {exllama_v2_import_exception}"
        )


    make_q_matrix = error_raiser_exllama
    gemm_half_q_half = error_raiser_exllama

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


def _torch_device(idx):
    if idx == -1:
        return "cpu"
    return f"cuda:{idx}"


def ext_gemm_half_q_half(x, q_handle, q4_width, force_cuda):
    """Matrix multiplication, returns x @ q4"""
    output_shape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.half, device=x.device)
    gemm_half_q_half(x, q_handle, output, force_cuda)
    return output.view(output_shape)


def ext_make_q_matrix(w: dict, temp_dq, key: str = None):
    """
    Create Q matrix
    """
    # EXL2
    # won't work as the moment because the tensors are not the same.
    if "q_weight" in w:
        w["q_scale_max"] /= 256
        w["q_perm"] = w["q_perm"].short()
        w["q_invperm"] = w["q_invperm"].short()
        return make_q_matrix(
            w["q_weight"],
            w["q_perm"],
            w["q_invperm"],
            w["q_scale"],
            w["q_scale_max"],
            w["q_groups"],
            none_tensor,
            none_tensor,
            none_tensor,
            temp_dq,
        )
    # GPTQ
    elif "qweight" in w:
        if w["scales"].dtype == torch.float:
            w["scales"] = w["scales"].half()

        # GPTQ with g_idx (act_order)
        if "g_idx" in w and not (w["g_idx"] == 0).all().item():
            w["q_perm"] = torch.empty(
                (w["qweight"].shape[0] * 8,),
                dtype=torch.short,
                device=w["qweight"].device,
            )
            w["q_invperm"] = torch.empty_like(w["q_perm"])
            # make_q4 segfaults if g_idx is not on cpu in the act-order case. In the non act-order case, None needs
            # to be passed for g_idx.
            return make_q_matrix(
                w["qweight"],
                w["q_perm"],
                w["q_invperm"],
                none_tensor,
                none_tensor,
                none_tensor,
                w["qzeros"],
                w["scales"],
                w["g_idx"].cpu(),
                temp_dq,
            )
        # GPTQ without g_idx
        else:
            return make_q_matrix(
                w["qweight"],
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                w["qzeros"],
                w["scales"],
                none_tensor,
                temp_dq,
            )


class QuantLinear(nn.Module):
    QUANT_TYPE = "exllamav2"

    """Linear layer implementation with per-group 4-bit quantization of the weights"""

    def __init__(self, bits, group_size, infeatures, outfeatures, bias, trainable=False, **kwargs):
        super().__init__()
        if bits != 4:
            raise ValueError(
                f"Exllamav2 kernel supports only bits=4, requested bits={bits}. Something is wrong in the model "
                f"initialization."
            )
        if trainable:
            raise NotImplementedError("Exllamav2 kernel does not support training.")

        self.q_handle = None
        self.q_tensors = None

        self.padding = -outfeatures % 32
        self.outfeatures = outfeatures + self.padding
        outfeatures = self.outfeatures

        self.infeatures = infeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.trainable = trainable
        self.maxq = 2 ** self.bits - 1

        assert infeatures % 32 == 0
        assert infeatures % self.group_size == 0
        assert outfeatures % 32 == 0

        # I need to register the tensors, otherwise, we won't be able to load them easily using transformers ...
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
                dtype=torch.float16,
            ),
        )

        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None
        self.clip = kwargs.get("clip", False)

    def post_init(self, temp_dq):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None
        self.q_tensors = {
            "qweight": self.qweight,
            "qzeros": self.qzeros,
            "scales": self.scales,
        }
        temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
        self.q_handle = ext_make_q_matrix(self.q_tensors, temp_dq)

    def forward(self, x, force_cuda=False):
        orig_dtype = x.dtype
        if x.dtype != torch.float16:
            logger.warning_once(
                f"The exllama v2 kernel for GPTQ requires a float16 input activation, while {x.dtype} was passed. "
                f"Casting to float16.\nMake sure you loaded your model with torch_dtype=torch.float16, that the model "
                f"definition does not inadvertently cast to float32, or disable AMP Autocast that may produce float32 "
                f"intermediate activations in the model."
            )

            x = x.half()

        output = ext_gemm_half_q_half(x, self.q_handle, self.outfeatures, force_cuda)
        if orig_dtype != torch.float16:
            output = output.to(orig_dtype)
        if self.bias is not None:
            output.add_(self.bias)
        if self.clip:
            output = torch.clip(output, -65504, 65504)
        return output

    def temp_dq_size(self):
        return self.infeatures * self.outfeatures * 2 + 128

    def temp_fwd_size(self, max_input_len, max_batch_size):
        return self.outfeatures * max_input_len * max_batch_size * 4 + 128

    def scratch_space_fixed(self, max_input_len=2048, max_batch_size=8):
        return self.temp_dq_size() + self.temp_fwd_size(max_input_len, max_batch_size)


class ExLlamaV2DeviceTensors:
    device_idx: int
    scratch_bytes: int
    scratch_idx: int
    scratch: torch.tensor = None

    def __init__(self, device_idx, scratch_bytes):
        self.device_idx = device_idx
        self.scratch_bytes = scratch_bytes

    def prepare(self):
        self.scratch = torch.empty(
            (self.scratch_bytes // 2,),
            dtype=torch.half,
            device=_torch_device(self.device_idx),
        )

    def get_scratch_slice(self, size_bytes):
        if self.scratch is None:
            self.prepare()

        size_bytes = ((size_bytes + 127) // 128) * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, 0, size_half)
        return scratch_slice
