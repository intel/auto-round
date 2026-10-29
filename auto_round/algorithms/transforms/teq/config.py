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

from auto_round.algorithms.config import AlgorithmParameterRegistry
from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.algorithms.registry import register_algorithm


class TEQConfig(QuantizationConfig):
    """Configuration for TEQ (Trainable Equivalent Transformation).

    TEQ is a pre-processing algorithm. It trains per-channel equivalent
    transform scales on calibration samples, folds the inverse scale into the
    upstream smooth layer, scales downstream weights, and then delegates final
    compression to the block quantizer, such as SignRoundV2/ARV2.
    """

    _OPTIMIZATION_MODES = frozenset({"staged", "staged_refine", "frozen_ar_refine", "joint"})

    @classmethod
    def register_args(cls, registry: AlgorithmParameterRegistry) -> None:
        registry.add_argument("--teq-iters", field="iters", default=20, type=int)
        registry.add_argument("--teq-lr", field="lr", default=1e-3, type=float)
        registry.add_argument(
            "--teq-optimizer",
            field="optimizer",
            default="adam",
            choices=("adam", "lion"),
            help="Optimizer for the pre-AutoRound TEQ scale optimization.",
        )
        registry.add_argument("--teq-min-scale", field="min_scale", default=1e-5, type=float)
        registry.add_argument("--teq-max-scale", field="max_scale", default=10.0, type=float)
        registry.add_argument("--teq-sqrt-w-init", field="sqrt_w_init", default=False, action="store_true")
        registry.add_argument("--teq-awq-init", field="awq_init", default=False, action="store_true")
        registry.add_argument("--teq-awq-init-n-grid", field="awq_init_n_grid", default=20, type=int)
        registry.add_argument("--teq-nsamples", field="nsamples", default=None, type=int)
        registry.add_argument("--teq-batch-size", field="batch_size", default=None, type=int)
        registry.add_argument("--teq-sample-seqlen", field="sample_seqlen", default=512, type=int)
        registry.add_argument(
            "--teq-optimization-mode",
            field="optimization_mode",
            default="staged",
            choices=sorted(cls._OPTIMIZATION_MODES),
        )
        registry.add_argument("--teq-refine-iters", field="refine_iters", default=10, type=int)
        registry.add_argument(
            "--teq-joint-lr",
            "--teq-refine-lr",
            field="joint_lr",
            default=1e-4,
            type=float,
        )
        registry.add_argument(
            "--teq-refine-optimizer",
            field="refine_optimizer",
            default="adam",
            choices=("adam", "lion"),
            help="Optimizer for the TEQ-only phase of staged_refine.",
        )
        registry.add_argument(
            "--teq-ar-first-iters",
            field="ar_first_iters",
            default=None,
            type=int,
            help="AR iterations before TEQ refinement in staged_refine mode.",
        )
        registry.add_argument(
            "--teq-experimental",
            field="experimental",
            default=False,
            action="store_true",
            help="Enable experimental TEQ research modes.",
        )

    def __init__(
        self,
        *,
        iters: int = 20,
        lr: float = 1e-3,
        optimizer: str = "adam",
        min_scale: float = 1e-5,
        max_scale: float = 10.0,
        sqrt_w_init: bool = False,
        awq_init: bool = False,
        awq_init_n_grid: int = 20,
        nsamples: int | None = None,
        batch_size: int | None = None,
        sample_seqlen: int | None = 512,
        skip_moe: bool = True,
        optimization_mode: str = "staged",
        refine_iters: int = 10,
        joint_lr: float = 1e-4,
        refine_optimizer: str = "adam",
        ar_first_iters: int | None = None,
        experimental: bool = False,
        mappings: list[dict] | None = None,
        **kwargs,
    ):
        """Initialize TEQ.

        Args:
            iters: Optimization steps per smooth-balance mapping.
            lr: Adam learning rate for trainable transform scales.
            optimizer: Optimizer for the pre-AutoRound TEQ scale optimization.
                ``lion`` uses a lightweight sign-momentum update and does not
                require an external package.
            min_scale: Lower clamp for transform scales.
            max_scale: Upper clamp for transform scales.
            sqrt_w_init: Initialize scales from inverse sqrt channel weight
                magnitude instead of ones.
            awq_init: Initialize scales from an AWQ-style activation/weight
                grid search before TEQ gradient fine-tuning.
            awq_init_n_grid: Number of grid points used by AWQ-style
                initialization.
            nsamples: Optional maximum number of calibration samples replayed by
                TEQ per mapping. ``None`` reuses all captured samples.
            batch_size: Optional TEQ replay micro-batch size. ``None`` preserves
                the calibration batch size.
            sample_seqlen: Optional sequence length cap for cached parent
                replay samples. ``None`` or ``<=0`` disables truncation.
            skip_moe: Exclude routed MoE expert mappings when automatic AWQ
                mappings are used.
            optimization_mode: ``staged`` trains TEQ before AutoRound;
                ``staged_refine`` trains TEQ before AutoRound, then refines
                TEQ scales between two AR phases;
                ``frozen_ar_refine`` starts from plain AutoRound and refines
                only TEQ scales; ``joint`` refines TEQ and AutoRound parameters
                together after the plain-AutoRound phase.
            refine_iters: Number of guarded refinement iterations used by the
                experimental modes.
            joint_lr: Adam learning rate for TEQ scales during refinement.
            refine_optimizer: Optimizer for the TEQ-only phase of
                ``staged_refine``. ``lion`` uses a lightweight sign-momentum
                update and does not require an external package.
            ar_first_iters: AR iterations in the first phase of
                ``staged_refine``. The remaining configured AR iterations form
                the second phase.
            experimental: Required opt-in for non-staged research modes.
            mappings: Optional explicit smooth/balance mappings. The format is
                compatible with ``AWQConfig.mappings``.
            **kwargs: Common quantization arguments used to mirror the target
                block quantizer during TEQ's fake quantization loss.
        """
        super().__init__(**kwargs)
        if iters < 0:
            raise ValueError(f"`iters` must be non-negative, got {iters!r}")
        if lr <= 0:
            raise ValueError(f"`lr` must be positive, got {lr!r}")
        if optimizer not in {"adam", "lion"}:
            raise ValueError(f"`optimizer` must be 'adam' or 'lion', got {optimizer!r}")
        if min_scale <= 0:
            raise ValueError(f"`min_scale` must be positive, got {min_scale!r}")
        if max_scale <= min_scale:
            raise ValueError(
                f"`max_scale` must be greater than `min_scale`, got min_scale={min_scale!r}, "
                f"max_scale={max_scale!r}"
            )
        if awq_init_n_grid < 2:
            raise ValueError(f"`awq_init_n_grid` must be at least 2, got {awq_init_n_grid!r}")
        if nsamples is not None and nsamples <= 0:
            raise ValueError(f"`nsamples` must be positive when set, got {nsamples!r}")
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive when set, got {batch_size!r}")
        if optimization_mode not in self._OPTIMIZATION_MODES:
            raise ValueError(
                f"`optimization_mode` must be one of {sorted(self._OPTIMIZATION_MODES)}, got {optimization_mode!r}"
            )
        if optimization_mode in {"staged_refine", "frozen_ar_refine", "joint"} and not experimental:
            raise ValueError("Experimental TEQ modes require `experimental=True`.")
        if refine_iters < 0:
            raise ValueError(f"`refine_iters` must be non-negative, got {refine_iters!r}")
        if joint_lr <= 0:
            raise ValueError(f"`joint_lr` must be positive, got {joint_lr!r}")
        if refine_optimizer not in {"adam", "lion"}:
            raise ValueError(f"`refine_optimizer` must be 'adam' or 'lion', got {refine_optimizer!r}")
        if ar_first_iters is not None and ar_first_iters <= 0:
            raise ValueError(f"`ar_first_iters` must be positive when set, got {ar_first_iters!r}")

        self.teq_iters = iters
        self.teq_lr = lr
        self.teq_optimizer = optimizer
        self.teq_min_scale = min_scale
        self.teq_max_scale = max_scale
        self.teq_sqrt_w_init = sqrt_w_init
        self.teq_awq_init = awq_init
        self.teq_awq_init_n_grid = awq_init_n_grid
        self.teq_nsamples = nsamples
        self.teq_batch_size = batch_size
        self.teq_sample_seqlen = sample_seqlen
        self.teq_skip_moe = skip_moe
        self.teq_optimization_mode = optimization_mode
        self.teq_refine_iters = refine_iters
        self.teq_joint_lr = joint_lr
        self.teq_refine_optimizer = refine_optimizer
        self.teq_ar_first_iters = ar_first_iters
        self.teq_experimental = experimental
        self.mappings = mappings
        self.infer_bs_coeff = 1
        self.batch_dim = None

    def __repr__(self) -> str:
        return (
            f"TEQConfig(iters={self.teq_iters}, lr={self.teq_lr}, optimizer={self.teq_optimizer!r}, "
            f"min_scale={self.teq_min_scale}, "
            f"max_scale={self.teq_max_scale}, bits={self.bits}, group_size={self.group_size}, "
            f"sym={self.sym}, awq_init={self.teq_awq_init}, awq_init_n_grid={self.teq_awq_init_n_grid}, "
            f"nsamples={self.teq_nsamples}, batch_size={self.teq_batch_size}, "
            f"sample_seqlen={self.teq_sample_seqlen}, skip_moe={self.teq_skip_moe}, "
            f"optimization_mode={self.teq_optimization_mode!r}, refine_iters={self.teq_refine_iters}, "
            f"joint_lr={self.teq_joint_lr}, ar_first_iters={self.teq_ar_first_iters}, "
            f"refine_optimizer={self.teq_refine_optimizer!r}, "
            f"mappings={'<explicit>' if self.mappings else 'auto'})"
        )


register_algorithm(
    "teq",
    aliases=("teq",),
    config_factory=TEQConfig,
    summary="Trainable Equivalent Transformation before block quantization.",
)
