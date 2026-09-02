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

"""Inference backend backed by the `humming <https://github.com/inclusionAI/humming>`_ GEMM kernels.

humming is a JIT-compiled quantized-GEMM library that supports **any** weight
bit-width from 1 to 8 bits (SM75+), which makes it the natural kernel for
AutoRound's 5/6/7-bit checkpoints -- the Triton/torch fallbacks either reject
those bit-widths or dequantize them in Python.

The checkpoint layouts AutoRound writes map one-to-one onto humming's schemas:

===============================  =================================  ==================
packing_format                   humming schema                     zero point
===============================  =================================  ==================
``auto_round`` / ``:gptqmodel``  ``GPTQWeightSchema``               stored as-is
``auto_round:auto_gptq``         ``GPTQWeightSchema``               ``zp - 1`` (sym only)
``auto_round:auto_awq``          ``AWQWeightSchema``                stored as-is
===============================  =================================  ==================

For symmetric quantization humming ignores ``qzeros`` entirely and implicitly
dequantizes with ``(value - 2**(bits-1)) * scale`` (see
``humming/utils/weight.py::dequantize_weight``), which is exactly AutoRound's
full-range-symmetric convention -- so both the ``zp`` and ``zp - 1`` GPTQ
variants land on the same (correct) numerics.
"""

import math
from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger(__name__)

# humming's weight transform asserts ``padded_shape_n % 64 == 0`` and
# ``padded_shape_k % (2 * 256 // act_bits) == 0`` (32 for fp16/bf16 activations).
PAD_N_TO_MULTIPLE = 64
PAD_K_TO_MULTIPLE = 32

SUPPORTED_BITS = (2, 3, 4, 5, 6, 7, 8)


def is_humming_available() -> bool:
    """Whether the humming kernels can be imported and used on this machine."""
    try:
        import humming  # noqa: F401  # pylint: disable=E0401
    except Exception:  # pragma: no cover - depends on the local environment
        return False
    return torch.cuda.is_available()


class QuantLinear(nn.Module):
    """Weight-only quantized linear served by humming.

    Buffers are registered with the *checkpoint* layout so that HuggingFace's
    state-dict loader can fill them directly. ``post_init()`` then hands them to
    humming, which repacks them into its kernel-native layout and releases the
    originals.
    """

    QUANT_TYPE = "humming"
    # packing_format string handed to humming's AutoRoundWeightSchema
    PACKING_FORMAT = "auto_round"
    # AWQ stores qweight transposed relative to GPTQ
    AWQ_LAYOUT = False

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        sym=True,
        weight_dtype=torch.float16,
        **kwargs,
    ):
        super().__init__()
        if bits not in SUPPORTED_BITS:
            raise NotImplementedError(f"humming backend supports {SUPPORTED_BITS} bits, got {bits}")
        if infeatures % 32 != 0:
            raise ValueError(f"humming requires in_features ({infeatures}) to be a multiple of 32")
        if outfeatures % 32 != 0:
            raise ValueError(f"humming requires out_features ({outfeatures}) to be a multiple of 32")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.sym = bool(sym)
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1
        self.compute_dtype = weight_dtype if weight_dtype in (torch.float16, torch.bfloat16) else torch.float16

        num_groups = math.ceil(infeatures / self.group_size)
        packed_out = outfeatures * self.bits // 32

        if self.AWQ_LAYOUT:
            qweight_shape = (infeatures, packed_out)
        else:
            qweight_shape = (infeatures * self.bits // 32, outfeatures)

        self.register_buffer("qweight", torch.zeros(qweight_shape, dtype=torch.int32))
        self.register_buffer("qzeros", torch.zeros((num_groups, packed_out), dtype=torch.int32))
        self.register_buffer("scales", torch.zeros((num_groups, outfeatures), dtype=self.compute_dtype))
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures,), dtype=self.compute_dtype))
        else:
            self.bias = None

        self.humming_layer = None

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def _build_schema(self):
        from humming.schema.autoround import AutoRoundWeightSchema  # pylint: disable=E0401

        return AutoRoundWeightSchema(
            bits=self.bits,
            group_size=self.group_size,
            sym=self.sym,
            data_type="int",
            packing_format=self.PACKING_FORMAT,
            act_bits=16,
        )

    def post_init(self):
        """Hand the loaded checkpoint buffers to humming and repack them."""
        if self.humming_layer is not None:
            return

        try:
            from humming.layer import HummingLayer  # pylint: disable=E0401
        except Exception as e:  # pragma: no cover - depends on the local environment
            raise ImportError(
                "The humming backend requires the humming kernels, install them with: "
                "`pip install git+https://github.com/inclusionAI/humming.git`"
            ) from e

        device = self.qweight.device
        if device.type != "cuda":
            raise RuntimeError(f"humming kernels require a CUDA device, but the layer lives on {device}")

        layer = HummingLayer(
            shape_n=self.outfeatures,
            shape_k=self.infeatures,
            weight_config=self._build_schema(),
            has_bias=self.bias is not None,
            pad_n_to_multiple=PAD_N_TO_MULTIPLE,
            pad_k_to_multiple=PAD_K_TO_MULTIPLE,
            torch_dtype=self.compute_dtype,
        )

        tensors = {
            "qweight": self.qweight,
            "qzeros": self.qzeros,
            "scales": self.scales.to(self.compute_dtype),
        }
        if self.bias is not None:
            tensors["bias"] = self.bias.to(self.compute_dtype)
        layer.load_from_tensors(tensors)
        layer = layer.to(device)
        # JIT-compiles / selects the kernel and rewrites the weights in place.
        layer.transform()

        # Release the checkpoint-layout buffers; humming owns the weights now.
        for name in ("qweight", "qzeros", "scales", "bias"):
            if name in self._buffers:
                self._buffers.pop(name)
        self.bias = None
        self.humming_layer = layer

    def pack(self, linear, scales, zeros, g_idx=None, device=None):
        raise NotImplementedError(
            "the humming backend is inference-only; export with the `auto_round`, "
            "`auto_round:auto_gptq` or `auto_round:auto_awq` format instead"
        )

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, x):
        if self.humming_layer is None:
            self.post_init()

        out_shape = x.shape[:-1] + (self.outfeatures,)
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1]).to(self.compute_dtype)
        out = self.humming_layer(x)
        # humming may pad the N dimension up to a multiple of 64.
        if out.shape[-1] != self.outfeatures:
            out = out[..., : self.outfeatures]
        return out.reshape(out_shape).to(x_dtype)


class QuantLinearGPTQ(QuantLinear):
    """``auto_round:auto_gptq`` packing (``qzeros`` stores ``zp - 1``)."""

    QUANT_TYPE = "humming_gptq"
    PACKING_FORMAT = "auto_round:auto_gptq"
    AWQ_LAYOUT = False


class QuantLinearAWQ(QuantLinear):
    """``auto_round:auto_awq`` packing (interleaved, transposed qweight)."""

    QUANT_TYPE = "humming_awq"
    PACKING_FORMAT = "auto_round:auto_awq"
    AWQ_LAYOUT = True


__all__ = [
    "QuantLinear",
    "QuantLinearAWQ",
    "QuantLinearGPTQ",
    "SUPPORTED_BITS",
    "is_humming_available",
]

