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

import re

import torch
import torch.nn as nn

from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.logger import logger
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
)
from auto_round.utils.device_manager import device_manager

_EXPERT_BATCH_MAX_ELEMS = 2**28  # ~1 GiB fp32 stacked weights per batched call

_EXPERT_RE = re.compile(r"^(?P<parent>.*)\.experts\.\d+\.(?P<proj>[^.]+)$")


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
        # Normalize imatrix (cheap elementwise divides), then quantize the
        # block's target modules - same-shape expert projections batched into
        # single stacked search calls when the search is active
        targets = []
        for name, m in block.named_modules():
            if hasattr(m, "imatrix"):
                m.imatrix /= m.imatrix_cnt
            if hasattr(m, "global_name") and check_to_quantized(m):
                targets.append(m)
        self._quantize_targets(targets)

    def _split_expert_batches(self, targets: list):
        """Partition same-shape expert projections into batchable groups.

        Experts of one layer share weight shapes (e.g. 192 x ``[1536, 4096]``
        gate/up/down projections). The per-group search is row-independent, so
        a whole group can be quantized in one call by stacking weights along
        the output dim - same results as per-module calls, one search's worth
        of per-module overhead instead of 192.
        """
        grouped = {}
        singles = []
        for m in targets:
            match = _EXPERT_RE.match(getattr(m, "global_name", "") or "")
            if (
                match is None
                or type(m) is not nn.Linear
                or not isinstance(getattr(m, "group_size", None), int)
                or getattr(m, "group_size", None) <= 0
                or getattr(m, "super_bits", None) is not None
                or getattr(m, "data_type", "int") != "int"
                or m.weight.shape[1] % m.group_size != 0  # not row-divisible: keep per-module
                or getattr(m, "act_bits", 16) <= 8  # act-quant layers need the per-module wrapper path
            ):
                singles.append(m)
                continue
            key = (
                match["parent"],
                match["proj"],
                tuple(m.weight.shape),
                m.bits,
                m.group_size,
                bool(getattr(m, "sym", False)),
            )
            grouped.setdefault(key, []).append(m)
        batches = []
        for g in grouped.values():
            if len(g) >= 2:
                batches.append(g)
            else:
                singles.extend(g)  # singleton groups must stay per-module, never dropped
        return batches, singles

    def _expert_search_active(self) -> bool:
        """Whether the optimized-RTN NeUQI search runs for expert modules.

        Mirrors the MoE heuristic in ``_quantize_layer_via_rtn``: experts skip
        the search unless explicitly forced on (``enable_opt_rtn``), so batching
        must not change that decision - only batch when the search would run.
        Both symmetry classes search under ``enable_neuqi`` (asym: joint
        (scale, zp); sym: two-stage scale), so both batch.
        """
        cfg = self.config
        if bool(getattr(cfg, "disable_opt_rtn", False)):
            return False
        if getattr(cfg, "orig_disable_opt_rtn", None) is None and getattr(self.model_context, "is_moe_model", False):
            return False
        return bool(getattr(cfg, "enable_neuqi", False))

    def _quantize_expert_batch(self, mods: list, device) -> list:
        """Quantize same-shape expert projections in one stacked search call.

        Returns the modules NOT quantized here (empty list = success); the
        caller falls back to per-module calls for them. Reproduces the
        per-module wrapper outputs: quantize-dequantized weights written in
        place, plus ``scale``/``zp``/``q_scale_thresh``/``data_type``
        attributes matching the unwrapper's conventions. On a mid-batch
        failure (e.g. OOM) already-written modules keep their results and
        only the remainder is returned -- per-module fallback must never
        re-quantize an already quantized-dequantized weight.
        """
        m0 = mods[0]
        bits, g = m0.bits, m0.group_size
        sym = bool(getattr(m0, "sym", False))
        # mirror the wrapper's threshold rule: fp32-scale modules use a tighter
        # q_scale_thresh than the fp16 default
        scale_dtype = getattr(m0, "scale_dtype", torch.float16)
        q_scale_thresh = 1e-8 if scale_dtype == torch.float32 else 1e-5
        resolved_dtype = "opt_rtn_int_sym_neuqi" if sym else "opt_rtn_int_asym"
        try:
            if sym:
                from auto_round.data_type.neuqi import quant_tensor_opt_rtn_sym_neuqi
            else:
                from auto_round.data_type.neuqi import quant_tensor_opt_rtn_asym
        except ImportError:
            return list(mods)

        max_elems = _EXPERT_BATCH_MAX_ELEMS
        per_call = max(1, max_elems // m0.weight.numel())
        # divisibility is guaranteed by _split_expert_batches; re-checked here
        # so a direct call can never half-write a chunk before bailing out
        if any(m.weight.shape[1] % g != 0 for m in mods):
            return list(mods)
        written = 0  # modules fully quantized so far (chunk writes are atomic)
        try:
            for s in range(0, len(mods), per_call):
                chunk = mods[s : s + per_call]
                dev = torch.device(device)
                weights = torch.cat([m.weight.data.reshape(m.weight.shape[0], -1) for m in chunk], dim=0).to(dev)
                imat_rows = []
                for m in chunk:
                    if hasattr(m, "imatrix"):
                        imat_rows.append(m.imatrix.to(dev).unsqueeze(0).expand(m.weight.shape[0], -1))
                if imat_rows and len(imat_rows) < len(chunk):  # mixed: fill gaps uniformly
                    it = iter(imat_rows)
                    imat_rows = [
                        next(it) if hasattr(m, "imatrix") else torch.ones(m.weight.shape, device=dev) for m in chunk
                    ]
                # uniform weighting needs no materialized tensor: qw=None is
                # bit-identical to a full ones imatrix and skips ~module-size
                # allocations per expert
                imat = torch.cat(imat_rows, dim=0) if imat_rows else None
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)

                if sym:
                    # returns follow the quant_tensor_opt_rtn_sym conventions: the
                    # zero point is the scalar nmax
                    qdq, scale, zp = quant_tensor_opt_rtn_sym_neuqi(
                        weights,
                        bits=bits,
                        group_size=g,
                        v=0.0,
                        q_scale_thresh=q_scale_thresh,
                        imatrix=imat,
                        scale_dtype=scale_dtype,
                    )
                else:
                    qdq, scale, zp = quant_tensor_opt_rtn_asym(
                        weights,
                        bits=bits,
                        group_size=g,
                        v=0.0,
                        q_scale_thresh=q_scale_thresh,
                        imatrix=imat,
                        scale_dtype=scale_dtype,
                    )

                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                row_w = row_g = 0
                with torch.no_grad():
                    for m in chunk:
                        out = m.weight.shape[0]
                        n_groups_m = out * (m.weight.shape[1] // g)
                        m.weight.data.copy_(qdq[row_w : row_w + out].reshape(m.weight.shape).to(m.weight.device))
                        m.scale = scale[row_g : row_g + n_groups_m].reshape(out, -1).to("cpu")
                        if sym:
                            m.zp = zp
                        else:
                            m.zp = zp[row_g : row_g + n_groups_m].reshape(out, -1).to("cpu")
                        m.q_scale_thresh = q_scale_thresh
                        m.data_type = resolved_dtype
                        m.weight.grad = None
                        row_w += out
                        row_g += n_groups_m
                written = s + len(chunk)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
        except torch.OutOfMemoryError:
            # fall back to per-module calls for the unwritten remainder only:
            # smaller transients, identical results, no double quantization
            pass
        return mods[written:]

    def _quantize_targets(self, targets: list) -> None:
        """Quantize a block's target modules; same-shape expert groups in one
        stacked search call each (row-independent search: results identical to
        per-module calls)."""
        if not targets:
            return
        batches = []
        if self._expert_search_active():
            batches, targets = self._split_expert_batches(targets)
            if batches:
                n_batched = sum(len(b) for b in batches)
                logger.info(
                    "[OptRTN] expert batching: %d expert modules in %d batched groups.",
                    n_batched,
                    len(batches),
                )
        device = device_manager.device
        for b in batches:
            for m in self._quantize_expert_batch(b, device):
                self.quantize_layer_outside_block(m)  # unwritten remainder: per-module fallback
        for m in targets:
            self.quantize_layer_outside_block(m)
