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

    AWQ protects salient weight channels by analyzing activation patterns and
    applying channel-wise scaling to reduce quantization error.  The scaling
    factors are computed offline via a grid search over calibration data.
    After smoothing, standard RTN quantization is applied to the adjusted weights.

    Args:
        duo_scaling: Use both activations and weights for the scaling factor.
            True: always use duo scaling. False: only activation scaling.
            "both": search both modes and pick the best.
        n_grid: Number of grid points for the scaling-ratio search.
        seqlen: Calibration sequence length.
        nsamples: Number of calibration samples.  Grid search time scales
            linearly with nsamples (each batch triggers one parent forward
            per grid point).
        batch_size: Batch size for calibration forward passes.
        mappings: Explicit AWQ mappings.  Each mapping is a dict with keys
            ``smooth_layer`` (str) and ``balance_layers`` (list[str]).
            If None, mappings are inferred automatically from the model structure.
        **kwargs: Forwarded to ``QuantizationConfig`` (bits, group_size, sym, …).
    """

    _alg_cls = "AWQQuantizer"

    def __init__(
        self,
        *,
        duo_scaling: bool | str = True,
        n_grid: int = 20,
        seqlen: int = 2048,
        nsamples: int = 128,
        batch_size: int = 8,
        mappings: list[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(duo_scaling, str) and duo_scaling != "both":
            raise ValueError(f"duo_scaling must be True, False, or 'both', got '{duo_scaling}'")
        self.duo_scaling = duo_scaling
        self.n_grid = n_grid
        self.seqlen = seqlen
        self.nsamples = nsamples
        self.batch_size = batch_size
        self.mappings = mappings

        # TODO: These infrastructure attrs are expected by BaseQuantizer / RTNQuantizer.
        # They should be derived from the algorithm config rather than set here directly.
        self.infer_bs_coeff = 1
        self.batch_dim = None
        self.enable_quanted_input = False  # AWQ doesn't cascade quantized block outputs
        # AWQ uses plain RTN (no iterative optimization) for the quantization step.
        self.disable_opt_rtn = True
        self.orig_disable_opt_rtn = True
