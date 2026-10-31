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

import torch

from auto_round.export.common.packing import release_layer_safely
from auto_round.export.formats.awq.utils import WQLinear_GEMM, awq_gemm_kernel_supported, awq_layer_user_forced
from auto_round.logger import logger
from auto_round.utils import SUPPORTED_LAYER_TYPES, get_module, set_module


def pack_layer(name, model, backend, device=None):
    layer = get_module(model, name)

    if type(layer) not in SUPPORTED_LAYER_TYPES:  ##already packed
        return

    bits = getattr(layer, "bits", None)
    if bits is None or bits > 8:
        return

    group_size = getattr(layer, "group_size", None)
    in_features = getattr(layer, "in_features", None)
    out_features = getattr(layer, "out_features", None)
    if group_size is None or in_features is None or out_features is None:
        # Layers missing AWQ packing attributes (e.g. Conv1D) stay in full precision.
        logger.warning_once(f"skipping {name}: its shape cannot be served by the AWQ GEMM kernel")
        return
    if not awq_gemm_kernel_supported(in_features, out_features, bits, group_size):
        if awq_layer_user_forced(layer):
            # The user explicitly configured this layer for quantization;
            # honor it and pack anyway.
            logger.warning_once(
                f"{name}: quantized per explicit `layer_config` but its shape "
                "cannot be served by the AWQ GEMM kernel; packing as configured"
            )
        else:
            # The GEMM kernel cannot serve this shape; keep it in full precision.
            logger.warning_once(f"skipping {name}: its shape cannot be served by the AWQ GEMM kernel")
            return
    sym = layer.sym
    linear_layer = get_module(model, name)
    scale, zp = linear_layer.scale, linear_layer.zp
    scale = scale.t().contiguous()
    if isinstance(zp, torch.Tensor):
        zp = zp.t().contiguous().to(torch.float32)
        if sym:
            zp = int(zp.flatten()[0])
    q_linear = WQLinear_GEMM.from_linear(
        linear=linear_layer,
        w_bit=bits,
        group_size=group_size,
        init_only=False,
        scales=scale,
        zeros=zp,
        device=device,
    )
    set_module(model, name, q_linear)
    # Note: release weight and bias explicitly, in case they are referenced elsewhere
    release_layer_safely(layer)
