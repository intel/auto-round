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
"""RRQ output format: exports residual planes in auto_round:rrq format."""

from typing import Any, Union

import torch

from auto_round.export.formats.base import OutputFormat
from auto_round.logger import logger


@OutputFormat.register("auto_round:rrq")
class RRQFormat(OutputFormat):
    """Output format for RRQ residual model export.

    The residual model contains 3 INT2 residual planes (planes 1..3)
    packed in standard INT2 format. The base plane (plane 0) is exported
    separately via the standard auto_round format.

    Format name: auto_round:rrq
    """

    support_schemes = ["W2A16"]
    format_name = "auto_round:rrq"

    def __init__(self, format: str, scheme, ctx: Any):
        super().__init__(format, scheme, ctx)
        self.output_format = "auto_round:rrq"
        self.backend = None  # RRQ doesn't delegate to a sub-backend

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        """No-op: residual planes are already packed by the quantizer.

        ``RRQRTNQuantizer`` packs each residual plane into the standard W2A16
        INT2 layout (``qweight`` / ``scales`` / ``qzeros``) during quantization,
        so there is nothing to pack per-layer here.  The buffer rename
        (``rrq_*_k`` -> ``*_k``) and the ``quantization_config`` attach happen in
        :meth:`save_quantized`.
        """
        return None

    def save_quantized(
        self,
        output_dir,
        model=None,
        tokenizer=None,
        layer_config=None,
        inplace=True,
        device="cpu",
        serialization_dict=None,
        **kwargs,
    ):
        """Save the RRQ residual model.

        Args:
            output_dir: Output directory.
            model: The model with RRQ-quantized layers.
            tokenizer: Tokenizer (not used for RRQ residual export).
            layer_config: Per-layer config (may be used to extract group_size/sym).
            inplace: Whether to modify the model in place.
            device: Device for computation.
            serialization_dict: Serialization config from the compressor.
            **kwargs: Additional kwargs (e.g. safe_serialization, max_shard_size).
        """
        from auto_round.export.export_to_autoround.export_to_rrq import save_quantized_rrq

        # group_size / sym / num_planes are read from the model's RRQ layers
        # (set during quantize).  Pass None so the model values win.
        safe_serialization = kwargs.pop("safe_serialization", True)

        save_quantized_rrq(
            output_dir=output_dir,
            model=model,
            group_size=None,
            sym=None,
            num_planes=None,
            device=device,
            safe_serialization=safe_serialization,
            **kwargs,
        )
        return model

    def _get_num_planes(self, model, layer_name) -> int:
        """Get the number of planes for a layer."""
        from auto_round.utils import get_module

        layer = get_module(model, layer_name)
        if hasattr(layer, "rrq_total_planes"):
            return layer.rrq_total_planes
        return 4  # Default for Phase 1
