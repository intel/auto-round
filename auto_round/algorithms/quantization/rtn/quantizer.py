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

import torch

from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.logger import logger
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


def _shard_rtn_searches(quantizer, block) -> tuple:
    """Run the per-layer RTN/OptRTN iters=0 searches, sharded across the DDP devices.

    Work-sharding on the engaged plan: layer i's search runs on
    ``plan.devices[i % world]`` (the layer already moves to its tuning device
    in the serial path, so this adds no extra weight traffic); the search
    itself is deterministic given (weight, imatrix), so sharded results are
    bit-identical to serial. Falls back to the serial single-device loop when
    no plan is engaged or the world is 1. Returns (total_seconds, n_layers,
    max_layer_seconds).
    """
    import time as _ptime

    from auto_round.utils import set_module as _set_module

    plan = getattr(quantizer, "_resolved_ddp_plan", None)
    targets = [(n, m) for n, m in block.named_modules() if hasattr(m, "global_name") and check_to_quantized(m)]
    if plan is None or getattr(plan, "world", 1) < 2 or len(targets) < 2:
        _tq, _mx = 0.0, 0.0
        for _n, m in targets:
            _t0 = _ptime.perf_counter()
            quantizer._quantize_layer_core(m)
            _d = _ptime.perf_counter() - _t0
            _tq += _d
            _mx = max(_mx, _d)
        return _tq, len(targets), _mx

    world = plan.world
    home = plan.devices[0]

    def _one(dev, mod):
        if dev.type == "cuda":
            with torch.cuda.device(dev):
                return quantizer._quantize_layer_core(mod, tuning_device=dev)
        return quantizer._quantize_layer_core(mod, tuning_device=dev)

    # round-robin: consecutive layers spread across the devices
    jobs = [(plan.devices[i % world], n, m) for i, (n, m) in enumerate(targets)]
    results: dict = {}

    def _run(idx):
        dev, n, m = jobs[idx]
        results[idx] = _one(dev, m)

    from auto_round.algorithms.quantization.sign_round.data_parallel import run_threaded_spawn

    _t0 = _ptime.perf_counter()
    run_threaded_spawn([lambda i=i: _run(i) for i in range(len(jobs))])
    _tq = _ptime.perf_counter() - _t0

    # place results home: back into the block and (when the quantizer carries
    # the global model) into the model, matching the serial path's placement
    model = getattr(quantizer, "model", None)
    for idx, (dev, n, m) in enumerate(jobs):
        q_layer = results[idx].to(home)
        _replace_module(block, n, q_layer)
        if isinstance(model, torch.nn.Module):
            _set_module(model, q_layer.global_name, q_layer)
    return _tq, len(targets), _tq / max(len(targets), 1)


def _replace_module(block, dotted_name, new_module):
    """Replace ``block.<dotted_name>`` with ``new_module``."""
    parts = dotted_name.split(".")
    parent = block
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


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
        # Normalize imatrix and quantize layers
        import time as _ptime

        _t0 = _ptime.perf_counter()
        _n_norm = 0
        for name, m in block.named_modules():
            if hasattr(m, "imatrix"):
                m.imatrix /= m.imatrix_cnt
                _n_norm += 1
        _tn = _ptime.perf_counter() - _t0
        _tq, _n, _mx = _shard_rtn_searches(self, block)
        from auto_round import envs as _envs

        if getattr(_envs, "AR_PERF_COUNTERS", False):
            logger.info(
                "[perf] rtn phases: norm=%.2fs (n=%d) quant=%.2fs (n=%d, mean=%.0fms max=%.0fms)",
                _tn,
                _n_norm,
                _tq,
                _n,
                1000 * _tq / max(_n, 1),
                1000 * _mx,
            )
