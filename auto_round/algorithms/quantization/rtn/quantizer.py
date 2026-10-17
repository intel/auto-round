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
import torch.nn as nn

from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.logger import logger


def _rtn_phase_line(norm: float, quant: float, n: int, mean: float, max_: float) -> str:
    """Format the zero-shot RTN phase breakdown for AR_PERF_COUNTERS."""
    return "[perf] rtn phases: norm=%.2fs quant=%.2fs (n=%d, mean=%.3fs, max=%.3fs)" % (norm, quant, n, mean, max_)


from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
)
from auto_round.utils.device_manager import device_manager
from auto_round.utils.perf_counters import _PerfCounters


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

        _perf = _PerfCounters()
        _perf.start("norm")
        for _name, m in block.named_modules():
            if hasattr(m, "imatrix"):
                m.imatrix /= m.imatrix_cnt
        _perf.stop("norm")
        _perf.start("quant")
        work = [m for _name, m in block.named_modules() if hasattr(m, "global_name") and check_to_quantized(m)]

        def _quantize_counted(m):
            _l0 = _ptime.perf_counter()
            self.quantize_layer_outside_block(m)
            _perf.observe("quant_mod", _ptime.perf_counter() - _l0)

        def _quantize_staged(name, m):
            _l0 = _ptime.perf_counter()
            w = self._quantize_layer_via_rtn(m, defer_search=True)
            _perf.observe("quant_mod", _ptime.perf_counter() - _l0)
            return name, w

        _batching_off = search_dispatch.batched_search_disabled()
        if not _batching_off or not search_dispatch.cross_device_search_disabled():
            from auto_round.algorithms.quantization.rtn.batched_search import run_batched_rtn_search

            staged = []
            for m in work:
                name, w = _quantize_staged(getattr(m, "global_name", None) or "", m)
                if w is not None:  # None = OOM fallback already finished it serially
                    staged.append((name, w))
            if _batching_off:
                # batching disabled but idle-device spreading on: execute the
                # searches PER-MODULE on borrowed workers (max_batch=1 ->
                # singleton chunks, zero stacking -- same search per module as
                # the serial lane, placed on any viable device). The strict
                # serial baseline for A/B is AR_DISABLE_CROSS_DEVICE_SEARCH=1.
                run_batched_rtn_search(self.model, staged, max_batch=1)
                search_dispatch.log_engaged_once("per-module rtn search (batching disabled, idle devices)")
            else:
                run_batched_rtn_search(self.model, staged)
                search_dispatch.log_engaged_once("batched rtn search")
        else:
            for m in work:
                _quantize_counted(m)
        _perf.stop("quant")
        if _perf.enabled:
            _n = _perf.counts.get("quant_mod", 0)
            logger.info(
                "%s",
                _rtn_phase_line(
                    _perf.get("norm"),
                    _perf.get("quant"),
                    _n,
                    _perf.get("quant") / max(_n, 1),
                    _perf.maxima.get("quant_mod", 0.0),
                ),
            )
