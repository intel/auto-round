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

from auto_round.algorithms.quantization.config import QuantizationConfig


class AWQConfig(QuantizationConfig):
    """Configuration for AWQ (Activation-Aware Weight Quantization).

    AWQ is a **pre-processing** algorithm (``role="preprocess"``).  It
    protects salient weight channels by analyzing activation patterns and
    applying channel-wise scaling to reduce quantization error.  After
    smoothing, a separate ``block_quantizer`` (RTN, SignRound, …) performs
    the actual weight compression.

    The quantization parameters (``bits``, ``group_size``, ``sym``,
    ``data_type``, …) on this config are used *only* for the internal grid
    search loss calculation (quantize-dequantize during scale selection).
    The definitive quantization parameters for the final weight compression
    step come from the pipeline's ``block_quantizer`` config.
    """

    def __init__(
        self,
        *,
        duo_scaling: bool | str = True,
        n_grid: int = 20,
        seqlen: int = 2048,
        nsamples: int = 128,
        batch_size: int = 8,
        apply_smooth: bool = True,
        mappings: list[dict] | None = None,
        **kwargs,
    ):
        """Initialize an AWQ configuration.

        Args:
            duo_scaling: Whether AWQ should use activation-aware and
                weight-aware scaling together. Use True to always enable
                duo scaling, False to use activation-only scaling, or
                "both" to search both modes and keep the better result.
            n_grid: Number of grid-search points used when searching the
                AWQ scaling ratio.
            seqlen: Calibration sequence length retained for compatibility
                with standalone AWQ entry points.
            nsamples: Number of calibration samples retained for compatibility
                with standalone AWQ entry points.
            batch_size: Batch size retained for compatibility with standalone
                AWQ entry points.
            apply_smooth: Whether to apply AWQ smoothing before the downstream
                block quantizer.
            mappings: Optional explicit AWQ smooth/balance mappings. Each
                item should contain ``smooth_layer`` and
                ``balance_layers`` entries. If None, mappings are inferred
                automatically from the model structure.
            **kwargs: Common quantization arguments forwarded to
                QuantizationConfig, such as bits, group_size, sym,
                data_type, and activation quantization fields.
        """
        super().__init__(**kwargs)

        if isinstance(duo_scaling, str) and duo_scaling != "both":
            raise ValueError(f"duo_scaling must be True, False, or 'both', got '{duo_scaling!r}'")
        self.duo_scaling = duo_scaling
        self.n_grid = n_grid
        self.seqlen = seqlen
        self.nsamples = nsamples
        self.batch_size = batch_size
        self.apply_smooth = apply_smooth
        self.mappings = mappings
        self.infer_bs_coeff = 1
        self.batch_dim = None
        # NOTE: enable_quanted_input is NOT set here.  It belongs to the
        # block_quantizer (RTN/AutoRound), not to AWQ.  See §3.7.1.

    def __repr__(self) -> str:
        return (
            f"AWQConfig(duo_scaling={self.duo_scaling!r}, n_grid={self.n_grid}, "
            f"bits={self.bits}, group_size={self.group_size}, sym={self.sym}, "
            f"mappings={'<explicit>' if self.mappings else 'auto'})"
        )
