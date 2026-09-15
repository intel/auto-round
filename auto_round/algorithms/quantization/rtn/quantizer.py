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

import time as _ptime

import torch

from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.logger import logger


def _rtn_phase_line(norm, quant, n, mean, max_):
    """Format the zero-shot RTN phase breakdown for AR_PERF_COUNTERS."""
    return "[perf] rtn phases: norm=%.2fs quant=%.2fs (n=%d, mean=%.3fs, max=%.3fs)" % (norm, quant, n, mean, max_)


from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
)


@register_pipeline_member(RTNConfig)
class RTNQuantizer(BaseQuantizer):

    def __init__(self, config: RTNConfig) -> None:
        BaseQuantizer.__init__(self, config)

    @torch.no_grad()
    def quantize_block(
        self,
        block,
        fp_inputs,
        input_others,
        fp_outputs,
        q_inputs,
        block_ctx,
        input_ids=None,
        **kwargs,
    ) -> dict:
        """Apply zero-shot RTN quantization to a block.

        Args:
            block: The transformer block module to quantize.
            fp_inputs: FP calibration inputs for this block (list[Tensor] or dict
                for diffusion models).
            input_others: Auxiliary kwargs passed to the block forward
                (e.g. attention_mask, position_ids).
            fp_outputs: FP reference outputs of the block used as quantization
                targets (list[Tensor]).
            q_inputs: Quantized inputs from the previous block, or ``None`` when
                cascaded quantized-input is disabled.
            block_ctx: Per-block pipeline context (BlockContext).
            input_ids: Raw token IDs from the tokenizer (unused in RTN).
            **kwargs: Reserved for forward-compatibility with future parameters.

        Returns:
            dict: Empty dict — zero-shot RTN has no tunable parameters to track.
        """

        for _name, m in block.named_modules():
            if check_to_quantized(m):
                self._quantize_layer_via_rtn(m, disable_opt_rtn=True)
        return {}


@register_pipeline_member(OptimizedRTNConfig)
class OptimizedRTNQuantizer(RTNQuantizer):

    def __init__(self, config: RTNConfig) -> None:
        super().__init__(config)

    def can_compile_block_forward(self):
        return False

    def register_fp_input_forward_hooks(self, block):
        """Register FP-input hooks: imatrix."""
        handles = super().register_fp_input_forward_hooks(block)
        handles.extend(self._register_imatrix_hooks(block, with_count=True))
        return handles

    def _register_imatrix_hooks(self, model, *, with_count: bool = False):
        def collect_imatrix(module, input, output):
            input = input[0] if isinstance(input, (tuple, list)) else input
            flattened = input.reshape(-1, input.shape[-1]).to(torch.float32)
            squared = torch.sum(torch.pow(flattened, 2), dim=0).to(torch.float32)

            if not hasattr(module, "imatrix"):
                module.imatrix = squared
                if with_count:
                    module.imatrix_cnt = input.shape[0]
                return
            module.imatrix += squared.to(module.imatrix.device)
            if with_count:
                module.imatrix_cnt += input.shape[0]

        handles = []
        for _, module in model.named_modules():
            if isinstance(module, SUPPORTED_LAYER_TYPES) and check_to_quantized(module):
                handles.append(module.register_forward_hook(collect_imatrix))
        return handles

    @torch.no_grad()
    def quantize_block(
        self,
        block,
        fp_inputs,
        input_others,
        fp_outputs,
        q_inputs,
        block_ctx,
        input_ids=None,
        **kwargs,
    ):
        """Apply imatrix-informed RTN quantization to a block.

        Args:
            block: The transformer block module to quantize.
            fp_inputs: FP calibration inputs for this block (list[Tensor] or dict
                for diffusion models).
            input_others: Auxiliary kwargs passed to the block forward
                (e.g. attention_mask, position_ids).
            fp_outputs: FP reference outputs of the block used as quantization
                targets (list[Tensor]).
            q_inputs: Quantized inputs from the previous block, or ``None`` when
                cascaded quantized-input is disabled.
            block_ctx: Per-block pipeline context (BlockContext).
            input_ids: Raw token IDs from the tokenizer (unused in RTN).
            **kwargs: Reserved for forward-compatibility with future parameters.
        """
        # Normalize imatrix first (cheap, serial), then quantize layers. The
        # optimized RTN scale search is weight-local (weight + imatrix), so on a
        # block whose layers sit on several devices the searches run in
        # parallel, one worker per device.
        import auto_round.envs as _envs
        from auto_round.algorithms.quantization import search_dispatch

        _perf = bool(getattr(_envs, "AR_PERF_COUNTERS", False))
        _t_norm = _ptime.perf_counter()
        for _name, m in block.named_modules():
            if hasattr(m, "imatrix"):
                m.imatrix /= m.imatrix_cnt
        _t_q0 = _ptime.perf_counter()
        _n = 0
        _q_max = 0.0
        work = [m for _name, m in block.named_modules() if hasattr(m, "global_name") and check_to_quantized(m)]

        def _quantize_counted(m):
            nonlocal _n, _q_max
            _l0 = _ptime.perf_counter() if _perf else 0.0
            self.quantize_layer_outside_block(m)
            if _perf:
                _q_max = max(_q_max, _ptime.perf_counter() - _l0)
            _n += 1

        def _quantize_staged(name, m):
            nonlocal _n, _q_max
            _l0 = _ptime.perf_counter() if _perf else 0.0
            w = self._quantize_layer_via_rtn(m, defer_search=True)
            if _perf:
                _q_max = max(_q_max, _ptime.perf_counter() - _l0)
            _n += 1
            return name, w

        if not search_dispatch.batched_search_disabled():
            from auto_round.algorithms.quantization.rtn.batched_search import run_batched_rtn_search

            staged = []
            for m in work:
                name, w = _quantize_staged(getattr(m, "global_name", None) or "", m)
                if w is not None:  # None = OOM fallback already finished it serially
                    staged.append((name, w))
            run_batched_rtn_search(self.model, staged)
            search_dispatch.log_engaged_once("batched rtn search")
        else:
            for m in work:
                _quantize_counted(m)
        if _perf:
            _t_q1 = _ptime.perf_counter()
            logger.info("%s", _rtn_phase_line(_t_q0 - _t_norm, _t_q1 - _t_q0, _n, (_t_q1 - _t_q0) / max(_n, 1), _q_max))
