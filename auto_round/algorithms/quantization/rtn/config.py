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
from auto_round.logger import logger


class RTNConfig(QuantizationConfig):
    need_calib = False

    @classmethod
    def register_args(cls, registry: AlgorithmParameterRegistry) -> None:
        mutex = registry.add_mutually_exclusive_group()
        mutex.add_argument(
            "--disable_opt_rtn",
            field="disable_opt_rtn",
            default=None,
            dest="disable_opt_rtn",
            action="store_const",
            const=True,
            help="Force plain RTN (disable optimized path).",
        )
        mutex.add_argument(
            "--enable_opt_rtn",
            field="disable_opt_rtn",
            dest="disable_opt_rtn",
            action="store_const",
            const=False,
            help="Force optimized RTN path.",
        )
        registry.add_argument(
            "--enable_neuqi",
            field="enable_neuqi",
            action="store_true",
            help=(
                "Enable the NeUQI grid search (arXiv 2505.17595): the asymmetric zero-shot "
                "optimized-RTN path runs a joint (scale, integer zero-point) search instead of plain "
                "min/max, symmetric layers run the two-stage scale search, and SignRound tuning "
                "anchors its grid to the search result (frozen init)."
            ),
        )

    def __init__(
        self,
        *,
        disable_opt_rtn: bool = None,
        enable_opt_rtn: bool = None,
        enable_neuqi: bool = False,
        **kwargs,
    ) -> None:
        """Initialize an RTN configuration.

        Args:
            disable_opt_rtn: Whether to disable the optimized RTN path.
                ``None`` keeps the default heuristic, True forces plain
                RTN, and False forces the optimized implementation.
            enable_opt_rtn: Convenience alias for ``disable_opt_rtn=False``.
            enable_neuqi: Opt into the NeUQI grid search (arXiv 2505.17595):
                the asymmetric zero-shot optimized-RTN path runs a joint
                (scale, integer zero-point) search instead of plain min/max,
                symmetric layers run the two-stage scale search, and SignRound
                tuning (``iters > 0``) anchors its grid to the search result
                (frozen init). Requires the optimized path (``disable_opt_rtn``
                not forced True). The search grid sizes are tunable via
                ``AR_NEUQI_COARSE`` and ``AR_NEUQI_FINE``.
            **kwargs: Common quantization arguments forwarded to
                QuantizationConfig, such as bits, group_size, sym,
                data_type, and activation quantization fields.
        """
        super().__init__(**kwargs)

        if enable_opt_rtn:
            disable_opt_rtn = False
        self.orig_disable_opt_rtn = disable_opt_rtn

        if disable_opt_rtn is None:  # TODO wenhuach move to AR entry
            if self.bits and self.bits >= 8 and self.act_bits and self.act_bits >= 8 and self.data_type == "int":
                logger.warning("`disable_opt_rtn` is turned on for W8A16/W8A8 quantization to improve efficiency.")
                disable_opt_rtn = True
        if disable_opt_rtn is None:
            logger.info(
                "`enable_opt_rtn` is turned on, set `--disable_opt_rtn` for higher speed at the cost of accuracy."
            )
            disable_opt_rtn = False
        self.disable_opt_rtn = disable_opt_rtn

        if enable_neuqi and disable_opt_rtn:
            # validated AFTER the W8A16/W8A8 heuristic above so the auto-disable
            # cannot silently neutralize the search opt-in
            raise ValueError(
                "--enable_neuqi requires the optimized-RTN path: it replaces the plain min/max "
                "initialization with a search. Drop --disable_opt_rtn. Note "
                "that W8A16/W8A8 schemes auto-disable the optimized path for efficiency."
            )
        self.enable_neuqi = enable_neuqi


class OptimizedRTNConfig(RTNConfig):
    need_calib = True


register_algorithm(
    "rtn",
    aliases=("rtn",),
    config_factory=RTNConfig,
    summary="Round-To-Nearest quantization.",
)
