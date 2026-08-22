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
from auto_round.logger import logger


def _unique_opt_rtn_value(values) -> bool | None:
    values = [value for value in values if value is not None]
    unique_values = []
    for value in values:
        if not any(value == existing for existing in unique_values):
            unique_values.append(value)
    if len(unique_values) > 1:
        raise ValueError("Conflicting AWQ disable_opt_rtn values. Use one AWQ config or the same opt-RTN policy.")
    return unique_values[0] if unique_values else None


def awq_disable_opt_rtn(configs) -> bool | None:
    """Return AWQ's resolved opt-RTN policy from normalized config objects."""
    return _unique_opt_rtn_value(config.disable_opt_rtn for config in configs if isinstance(config, AWQConfig))


def direct_opt_rtn_kwargs(config_kwargs) -> dict:
    """Return explicit opt-RTN kwargs from direct entry/CLI arguments."""
    if config_kwargs.get("enable_opt_rtn"):
        return {"enable_opt_rtn": True}
    if config_kwargs.get("disable_opt_rtn") is not None:
        return {"disable_opt_rtn": config_kwargs["disable_opt_rtn"]}
    return {}


def rtn_inherited_opt_kwargs(config_kwargs, awq_disable_opt_rtn_value) -> dict:
    """Return RTN kwargs where explicit user policy wins over AWQ inheritance."""
    opt_kwargs = direct_opt_rtn_kwargs(config_kwargs)
    if not opt_kwargs and awq_disable_opt_rtn_value is not None:
        opt_kwargs["disable_opt_rtn"] = awq_disable_opt_rtn_value
    return opt_kwargs


def sync_rtn_opt_rtn_from_awq(configs) -> None:
    """Apply AWQ's opt-RTN policy to RTN configs that did not set one explicitly."""
    from auto_round.algorithms.quantization.rtn.config import RTNConfig

    awq_value = awq_disable_opt_rtn(configs)
    if awq_value is None:
        return
    for config in configs:
        if isinstance(config, RTNConfig) and getattr(config, "orig_disable_opt_rtn", None) is None:
            config.disable_opt_rtn = awq_value
            config.orig_disable_opt_rtn = awq_value


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
        smooth_iters: int = 1,
        apply_clip: bool = False,
        clip_as_init: bool = False,
        clip_n_grid: int = 20,
        clip_max_shrink: float = 0.5,
        clip_n_sample_token: int = 512,
        awq_seqlen: int | None = None,
        smooth_batch_size: int | None = None,
        disable_opt_rtn: bool | None = True,
        skip_moe: bool = True,
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
            smooth_iters: Number of times the per-block smooth (grid search +
                scale apply) is repeated. Repeating refines the smoothing scale
                because the mx max_scale search and the AWQ alpha search affect
                each other; each extra pass re-derives the max_scale from the
                freshly-smoothed weights and re-searches alpha, accumulating the
                resulting scales. ``1`` reproduces the original single-pass AWQ.
            apply_clip: Whether to search and apply AWQ weight clipping after
                smoothing. When True, AWQ searches a per-group clipping
                range for each balance layer (minimizing output MSE) and
                hard-clamps the weights in place. Symmetric quantization uses a
                symmetric ``[-max_val, max_val]`` range; asymmetric quantization
                searches separate ``min_val`` and ``max_val`` bounds.
                This is a pure weight transformation, so it composes with any
                downstream block quantizer: a SignRound/SignRoundV2 quantizer
                re-derives its ``weight_min``/``weight_max`` from the clamped
                weights (the AWQ clip becomes the initialization) and then tunes
                ``min_scale``/``max_scale`` on top of it. Set False to keep the
                original smooth-only behavior (the block quantizer performs its
                own clip/min-max tuning).
            clip_as_init: Selects how the searched clip threshold is consumed
                (only relevant when ``apply_clip`` is True). When False
                (default), the weights are hard-clamped in place. When True,
                the weights are left untouched and the per-group clip range
                is instead stored (on the model context and on each balance
                layer) so the downstream block quantizer uses it to *initialize*
                its tunable weight range: SignRound caps its
                ``weight_min``/``weight_max`` to the clip and tunes
                ``min_scale``/``max_scale`` on top, while SignRoundV2 clamps the
                weight before its scale search. This works for both symmetric
                and asymmetric quantization and any data type, analogous to
                ``enable_minmax_tuning``.
            clip_n_grid: Number of shrink steps used when searching the AWQ
                clipping threshold.
            clip_max_shrink: Maximum fraction by which the per-group max value
                may be shrunk during the clip search (e.g. ``0.5`` searches down
                to 50% of the original per-group max).
            clip_n_sample_token: Maximum number of calibration tokens used per
                balance layer when searching the clip threshold (subsampled to
                bound memory).
            awq_seqlen: Maximum sequence length (number of tokens) used per
                calibration sample during AWQ calibration, including activation
                statistics, smoothing scale search, and clip-search input
                features. Defaults to ``512``. Set a larger positive integer to
                use longer sequences, or a value ``<= 0`` to disable truncation
                entirely.
            smooth_batch_size: Optional microbatch size used when replaying AWQ
                parent modules during scale grid search. Smaller values reduce
                peak VRAM while preserving the AWQ parent-output loss, at the
                cost of more parent forward calls. ``None`` or ``<= 0`` replays
                the cached calibration batch as-is.
            disable_opt_rtn: Whether AWQ's internal weight QDQ should use the
                plain RTN path. Defaults to True because AWQ calibration has not
                shown accuracy or cost benefits from optimized RTN in benchmarks.
                ``enable_opt_rtn=True`` may also be passed to force the
                optimized path.
            skip_moe: Whether to exclude routed MoE experts from AWQ smoothing.
                When True, balance layers belonging to routed experts (module
                names matching ``.experts.<N>.``) are dropped from the resolved
                mappings, so AWQ only smooths attention and dense/shared paths
                and leaves routed experts to the downstream block quantizer. This
                has no effect on dense models and is ignored when explicit
                ``mappings`` are provided.
            mappings: Optional explicit AWQ smooth/balance mappings. Each
                item should contain ``smooth_layer`` and
                ``balance_layers`` entries. If None, mappings are inferred
                automatically from the model structure.
            **kwargs: Common quantization arguments forwarded to
                QuantizationConfig, such as bits, group_size, sym,
                data_type, and activation quantization fields.
        """
        enable_opt_rtn = kwargs.pop("enable_opt_rtn", None)
        if enable_opt_rtn:
            disable_opt_rtn = False
        super().__init__(**kwargs)

        if awq_seqlen is None:
            awq_seqlen = 512

        if isinstance(duo_scaling, str) and duo_scaling != "both":
            raise ValueError(f"duo_scaling must be True, False, or 'both', got '{duo_scaling!r}'")
        self.duo_scaling = duo_scaling
        self.n_grid = n_grid
        self.seqlen = seqlen
        self.nsamples = nsamples
        self.batch_size = batch_size
        self.apply_smooth = apply_smooth
        if smooth_iters is None or smooth_iters < 1:
            raise ValueError(f"`smooth_iters` must be a positive integer, got {smooth_iters!r}")
        self.smooth_iters = smooth_iters
        self.apply_clip = apply_clip
        self.clip_as_init = clip_as_init
        if self.clip_as_init and not self.apply_clip:
            raise ValueError("`clip_as_init=True` requires `apply_clip=True`.")
        if clip_n_grid is None or clip_n_grid < 1:
            raise ValueError(f"`clip_n_grid` must be a positive integer, got {clip_n_grid!r}")
        if not (0.0 < clip_max_shrink < 1.0):
            raise ValueError(f"`clip_max_shrink` must be in (0, 1), got {clip_max_shrink!r}")
        if clip_n_sample_token is None or clip_n_sample_token < 1:
            raise ValueError(f"`clip_n_sample_token` must be a positive integer, got {clip_n_sample_token!r}")
        if smooth_batch_size is not None and smooth_batch_size < 0:
            raise ValueError(f"`smooth_batch_size` must be a non-negative integer or None, got {smooth_batch_size!r}")
        self._disable_opt_rtn = disable_opt_rtn
        self.clip_n_grid = clip_n_grid
        self.clip_max_shrink = clip_max_shrink
        self.clip_n_sample_token = clip_n_sample_token
        self.awq_seqlen = awq_seqlen
        self.smooth_batch_size = smooth_batch_size
        self.skip_moe = skip_moe
        self.mappings = mappings
        self.infer_bs_coeff = 1
        self.batch_dim = None

    @property
    def disable_opt_rtn(self) -> bool | None:
        """Whether AWQ's internal QDQ uses plain RTN instead of optimized RTN."""
        return self._disable_opt_rtn

    @disable_opt_rtn.setter
    def disable_opt_rtn(self, value: bool | None) -> None:
        self._disable_opt_rtn = value

    def finalize_scheme(self) -> None:
        """Adjust AWQ state that depends on the resolved run scheme."""
        data_type = self.data_type
        is_gguf_double_quant = bool(data_type) and (data_type.endswith("_dq") or data_type.endswith("float_zp"))
        if self.apply_clip and is_gguf_double_quant:
            logger.warning(
                "AWQ weight clipping (apply_clip=True) is not supported for GGUF "
                "double-quant schemes; disabling clipping and proceeding with AWQ "
                "smoothing only."
            )
            self.apply_clip = False

    def __repr__(self) -> str:
        return (
            f"AWQConfig(duo_scaling={self.duo_scaling!r}, n_grid={self.n_grid}, "
            f"smooth_iters={self.smooth_iters}, "
            f"apply_clip={self.apply_clip}, clip_as_init={self.clip_as_init}, "
            f"awq_seqlen={self.awq_seqlen}, smooth_batch_size={self.smooth_batch_size}, "
            f"disable_opt_rtn={self.disable_opt_rtn}, "
            f"skip_moe={self.skip_moe}, "
            f"bits={self.bits}, group_size={self.group_size}, sym={self.sym}, "
            f"mappings={'<explicit>' if self.mappings else 'auto'})"
        )
