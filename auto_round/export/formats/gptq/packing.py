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

import inspect

import torch
import torch.nn as nn
import transformers

from auto_round.export.common.packing import get_autogptq_packing_qlinear, release_layer_safely
from auto_round.utils import SUPPORTED_LAYER_TYPES, get_module, set_module


def pack_layer(name, model, backend, device=None):
    layer = get_module(model, name)

    if type(layer) not in SUPPORTED_LAYER_TYPES:  # already packed
        return

    orig_device = layer.weight.device  # must place after 74
    bits = layer.bits
    if bits > 8:
        return

    group_size = layer.group_size
    sym = layer.sym

    QuantLinear = get_autogptq_packing_qlinear(backend, bits, group_size, sym)

    if type(layer) == nn.Linear:
        in_features = layer.in_features
        out_features = layer.out_features
    elif type(layer) == nn.Conv2d:
        in_features = layer.in_channels
        out_features = layer.out_channels
    elif type(layer) == transformers.pytorch_utils.Conv1D:
        in_features = layer.weight.shape[0]
        out_features = layer.weight.shape[1]

    bias = layer.bias is not None
    ##bias = True  ## if using the above, llama3 lambada RTN will be NAN , TODO why?
    qlayer = QuantLinear(bits, group_size, in_features, out_features, bias, g_idx=True)  ##pylint: disable=E1123

    qlayer.device = orig_device
    scale = layer.scale
    zero = layer.zp
    # so far can only pack layer on CPU
    qlayer.to("cpu")
    ##force to float32 to be compatible with torch 2.0
    layer, scale = layer.to("cpu"), scale.to("cpu")
    if isinstance(zero, torch.Tensor):
        # Packing runs on CPU, so the zero point has to follow the weight and
        # the scale there, whether or not the layer is symmetric.
        zero = zero.to("cpu")
        if sym:
            zero = int(zero.flatten()[0])
    if isinstance(zero, torch.Tensor) and zero.dtype == torch.bfloat16:
        zero = zero.float()
    sig = inspect.signature(qlayer.pack)
    param_count = len(sig.parameters)
    if param_count == 2:
        qlayer.pack(layer, scale, device)
    else:
        qlayer.pack(layer, scale, zero, None, device)
    qlayer.to(orig_device)
    set_module(model, name, qlayer)
    # Note: release weight and bias explicitly, in case they are referenced elsewhere
    release_layer_safely(layer)
