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
from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
from auto_round.logger import logger


class RTNConfig(QuantizationConfig):
    _alg_cls = "RTNQuantizer"

    def __init__(
        self,
        scheme="W4A16",
        layer_config=None,
        *,
        disable_opt_rtn: bool = None,
        # for opt-rtn
        seqlen: int = 2048,
        nsamples: int = 128,
        batch_size: int = 8,
        **kwargs,
    ):
        # pop before super().__init__ so it doesn't leak into QuantizationConfig as an unknown kwarg
        enable_opt_rtn = kwargs.pop("enable_opt_rtn", None)
        super().__init__(scheme=scheme, layer_config=layer_config, **kwargs)

        self.seqlen = seqlen
        self.nsamples = nsamples
        self.batch_size = batch_size

        # Some helpers
        self.infer_bs_coeff = 1
        self.batch_dim = None

        # Automatically adjust the disable_opt_rtn option if the user does not explicitly set it.
        # To avoid None issue, we keep a copy though it's a little ugly
        if enable_opt_rtn and disable_opt_rtn:
            raise ValueError("`enable_opt_rtn` and `disable_opt_rtn` are mutually exclusive; " "only one can be set.")
        if enable_opt_rtn:
            disable_opt_rtn = False
        self.orig_disable_opt_rtn = disable_opt_rtn

        if disable_opt_rtn is None:
            if isinstance(scheme, str) and scheme in ["W8A16", "W8A8"]:
                logger.warning("`disable_opt_rtn` is turned on for W8A16/W8A8 quantization to improve efficiency.")
                disable_opt_rtn = True
            if self.bits and self.bits >= 8 and self.act_bits and self.act_bits >= 8 and self.data_type == "int":
                logger.warning("`disable_opt_rtn` is turned on for W8A16/W8A8 quantization to improve efficiency.")
                disable_opt_rtn = True
        if disable_opt_rtn is None:
            logger.info(
                "`enable_opt_rtn` is turned on, set `--disable_opt_rtn` for higher speed at the cost of accuracy."
            )
            disable_opt_rtn = False
        self.disable_opt_rtn = disable_opt_rtn
        if not self.disable_opt_rtn:
            self._alg_cls = "OptimizedRTNQuantizer"
