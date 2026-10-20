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
import copy
import math
import time as _ptime
from collections.abc import Callable
from contextlib import ExitStack, nullcontext
from typing import TYPE_CHECKING, Any

import torch
from torch import autocast

from auto_round.algorithms.block_runner import BlockForwardRunner
from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.quantization.sign_round.lfq import LFQMixin
from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD
from auto_round.algorithms.registry import register_quantizer
from auto_round.compressors.utils import (
    BestParamsSlot,
    IndexSampler,
    collect_best_params,
)
from auto_round.logger import logger
from auto_round.utils.perf_counters import _PerfCounters


def resolve_loss_device(loss_device, primary_device):
    """The tune loop's staging device: a concrete device, coalescing None.

    The placement pass may store ``None`` (and ``tensor.to(None)`` is a silent
    no-op), which would strand batch outputs on the block's exit device; the
    loop always stages on a concrete device, defaulting to the primary.
    """
    return loss_device or primary_device


def _log_tune_phases(perf: _PerfCounters, iters: int) -> None:
    """Derive the loop split and log the per-block tune phase line."""
    if not perf.enabled:
        return
    serial = perf.get("s_fwd") + perf.get("s_loss") + perf.get("s_bwd")
    rest = max(perf.get("loop") - perf.get("sampler") - perf.get("snap") - perf.get("step") - serial, 0.0)
    perf.set("lp_sampler", perf.get("sampler"))
    perf.set("lp_snap", perf.get("snap"))
    perf.set("lp_step", perf.get("step"))
    perf.set("lp_rest", rest)
    perf.set("lp_serial", serial)
    logger.info("%s", _tune_phase_line(perf.totals, iters))


def _tune_phase_line(phases: dict, iters: int) -> str:
    """Format the per-block tuning phase breakdown for AR_PERF_COUNTERS.

    ``wrap`` = wrapper_block (params init, quant-func resolve + optional
    per-wrapper torch.compile, SignRoundV2 init-scale search -- batched
    same-shape when eligible); ``prepare`` = tuning-param collection +
    optimizer/scheduler build + sampler setup; ``loop`` = the iteration
    try/finally; ``tail`` = best-params restore, clear_memory, unwrapping.
    The optional ``loop`` split appears when counters ran: ``sampler`` =
    index draw, ``snap`` = best-params snapshots, ``step`` = gradient
    sync + optimizer step, ``serial: fwd+loss+bwd`` = the inline forward,
    loss (incl. the .item() drain) and backward of every serial batch.
    """
    line = (
        f"[perf] tune phases (iters={iters}):"
        f" wrap={phases.get('wrap', 0.0):.2f}s prepare={phases.get('prepare', 0.0):.2f}s"
        f" loop={phases.get('loop', 0.0):.2f}s tail={phases.get('tail', 0.0):.2f}s"
    )
    if "lp_sampler" in phases:
        line += (
            f" (loop: sampler={phases['lp_sampler']:.2f}s snap={phases['lp_snap']:.2f}s"
            f" step={phases['lp_step']:.2f}s rest={phases['lp_rest']:.2f}s"
        )
        if phases.get("lp_serial", 0.0):
            line += f" serial: fwd+loss+bwd={phases['lp_serial']:.2f}s"
        line += ")"
    return line


from auto_round.utils import (
    htcore,
    is_hpex_available,
    mv_module_from_gpu,
)
from auto_round.utils.device import clear_memory_if_reached_threshold
from auto_round.utils.device_manager import device_manager
from auto_round.wrapper import WrapperLinear, unwrapper_layer, wrapper_block

if TYPE_CHECKING:
    from auto_round.algorithms.composer import BlockContext
    from auto_round.parallel.execution import ParallelCapabilities


# Output-element budget for one forward/backward while tuning an outside-block
# layer: a full-vocabulary lm_head at long seqlen cannot hold its fp32 logits
# plus backward transients on a single GPU, so the tune loop slices the
# sample's rows to stay within this many output elements per slice (64M = a
# 256MB fp32 logits tensor; the accumulated gradient is unchanged)
_OUTSIDE_TUNE_CHUNK_OUT_ELEMS = 2**26


def _valid_token_mask_rows(valid_token_mask, indices, device, rows_axis):
    """Concatenate per-sample valid-token masks into the input's row layout.

    Per-sample masks are ``[1, seq]``. For 3-D inputs (rows on dim 1) the
    result is ``[n, seq, 1]`` so a chunk slices ``[:, start:end]``; for 2-D
    inputs (rows concatenated on dim 0) it flattens to ``[total_rows, 1]`` so
    a chunk slices ``[start:end]``.
    """
    m = torch.cat([valid_token_mask[i] for i in indices], dim=0).to(device)
    if rows_axis == 1:
        return m.unsqueeze(-1)
    return m.reshape(-1, 1)


def _outside_tune_rows_per_chunk(out_features, sample_rows, budget=_OUTSIDE_TUNE_CHUNK_OUT_ELEMS):
    """Rows of one sample processed per forward/backward slice.

    Small-output layers (every decoder projection) get their whole sample in
    one slice; only huge-output layers (lm_head-class vocabularies) split."""
    if out_features is None or out_features <= 0:
        return sample_rows
    return max(1, min(sample_rows, budget // out_features))


@register_quantizer(SignRoundConfig)
class SignRoundQuantizer(LFQMixin, BaseQuantizer):

    def __init__(self, config: SignRoundConfig) -> None:
        super().__init__(config)
        self.iters = config.iters
        self.lr = config.lr
        self.minmax_lr = config.minmax_lr
        self.lr_scheduler = config.lr_scheduler
        self.momentum = config.momentum
        self.enable_minmax_tuning = config.enable_minmax_tuning
        self.enable_norm_bias_tuning = config.enable_norm_bias_tuning
        self.gradient_accumulate_steps = config.gradient_accumulate_steps

        self.enable_alg_ext = config.enable_alg_ext
        self.not_use_best_mse = config.not_use_best_mse
        self.enable_quanted_input = config.enable_quanted_input
        self.dynamic_max_gap = config.dynamic_max_gap
        self.enable_lfq = config.enable_lfq
        self.lfq_loss_type = getattr(config, "lfq_loss", "ce")
        self.lfq_top_k = getattr(config, "lfq_top_k", 32)
        self.lfq_temperature = getattr(config, "lfq_temperature", 1.0)
        self.lfq_num_blocks = getattr(config, "lfq_num_blocks", 1)
        self.lfq_mse_weight = getattr(config, "lfq_mse_weight", 0.0)
        self.enable_qad = getattr(config, "enable_qad", False)

        self.optimizer = self._get_optimizer(optimizer=config.optimizer)
        lora_rank = getattr(config, "qad_lora_rank", getattr(config, "lora_rank", 0))
        if (
            lora_rank > 0
            and self.iters > 0
            and getattr(config, "qad_enable_round_tuning", True)
            and isinstance(self.optimizer, type)
            and issubclass(self.optimizer, SignSGD)
        ):
            alternative = "--qad_optimizer adamw" if self.enable_qad else "--enable_adam"
            logger.warning_once(
                "LoRA rounding with SignSGD is not recommended: sign-only updates discard gradient magnitudes "
                "for the low-rank factors. Prefer Adam/AdamW and a suitable learning rate "
                f"(e.g. `{alternative}`). Adam/AdamW requires substantially more optimizer-state GPU memory "
                "than SignSGD because it stores two moment tensors per trainable parameter. "
                "The selected optimizer is unchanged."
            )
        self.wrapper_block = wrapper_block
        # Kept for per-layer (mixed-bit) lr resolution during tuning.
        self._config = config
        self.lr_is_auto = getattr(config, "lr_is_auto", False)
        self.minmax_lr_is_auto = getattr(config, "minmax_lr_is_auto", False)
        # Emit the low-bit lr notice at most once across all blocks/layers.
        self._logged_low_bit_lr = False

    def _configure_lora_scaling(self, module: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Bound the new SignSGD rounding offset by +/-4 over the tuning budget.

        This bounds s * B @ A, not any frozen offset from an earlier stage.
        Explicit alpha and non-SignSGD optimizers retain their existing scaling.

        Args:
            module (torch.nn.Module): Module containing LoRA-enabled rounding wrappers to update.
            optimizer (torch.optim.Optimizer): Optimizer whose learning-rate budget bounds the offset.
        """
        if not isinstance(optimizer, SignSGD) or self.iters <= 0:
            return
        wrappers = [
            m
            for m in module.modules()
            if isinstance(m, WrapperLinear) and hasattr(m, "lora_A") and m.lora_alpha is None
        ]
        if not wrappers:
            return
        if self.lr_scheduler is not None:
            logger.warning_once(
                "Automatic SignSGD LoRA scaling requires the default linear scheduler; "
                "keeping unit scaling. Set lora_alpha explicitly for a custom scheduler."
            )
            return
        # optimizer.step() precedes scheduler.step(): the T used learning rates
        # are lr * (1 - t/T), t=0,...,T-1, whose sum is lr * (T+1)/2.
        budgets = {
            id(p): float(group["lr"]) * (self.iters + 1) / 2
            for group in optimizer.param_groups
            for p in group["params"]
        }
        for wrapper in wrappers:
            budget_a = budgets.get(id(wrapper.lora_A), 0.0)
            budget_b = budgets.get(id(wrapper.lora_B), 0.0)
            # Each sign update moves a factor element by at most its lr.
            # Include initialization, rather than assuming both factors start
            # at zero (which would prevent either factor from learning).
            a0 = wrapper.lora_A.detach().abs().amax().item()
            b0 = wrapper.lora_B.detach().abs().amax().item()
            bound = wrapper.lora_rank * (a0 + budget_a) * (b0 + budget_b)
            if bound > 0 and math.isfinite(bound):
                wrapper.lora_scaling = 4.0 / bound
                wrapper.qad_lora_scaling = wrapper.lora_scaling

    def _maybe_log_low_bit_lr(self, bits) -> None:
        """Log once when low-bit (<=3) layers get the higher 2.0/iters lr."""
        if self._logged_low_bit_lr or not self.lr_is_auto:
            return
        if self.iters >= 1000 and bits is not None and bits <= 3:
            logger.info("using higher lr (2.0/iters) for <=3 bit layers to improve accuracy")
            self._logged_low_bit_lr = True

    def dispatch_block(self, block, input_ids, input_others):
        """Use the same placement contract for local, PP-placement and TP runs."""
        return super().dispatch_block(block, input_ids, input_others)

    @classmethod
    def parallel_capabilities(cls) -> "ParallelCapabilities":
        """Declare support for shard-local SignRound tuning.

        Returns:
            ParallelCapabilities: Capabilities enabling TP and pipeline placement.
        """
        from auto_round.parallel.execution import ParallelCapabilities

        return ParallelCapabilities(tensor_parallel=True)

    def _get_non_zero_cnt(self, tensor: list[torch.Tensor], indices: list[int]) -> int:
        current_tensors = [tensor[i] for i in indices]
        non_zero_cnt = 0
        for t in current_tensors:
            non_zero_cnt += torch.count_nonzero(t).item()
        return non_zero_cnt

    def _get_loss(
        self,
        pred_output: torch.Tensor,
        ref_output: torch.Tensor,
        indices: torch.Tensor,
        loss_func: Callable,
        device: str | torch.device = "cpu",
        valid_token_mask: torch.Tensor | None = None,
        input_ids=None,
    ):
        autocast_ctx = (
            nullcontext()
            if not self.model_context.amp
            else autocast(device_type=str(device).split(":")[0], dtype=self.model_context.amp_dtype)
        )
        if valid_token_mask:
            tmp_attention_mask = [valid_token_mask[i] for i in indices]
            tmp_attention_mask = torch.cat(tmp_attention_mask, dim=0).to(device)
            tmp_attention_mask.unsqueeze_(-1)

            with autocast_ctx:
                loss = loss_func(  # pylint: disable=not-callable
                    (pred_output * tmp_attention_mask).to(torch.float32),
                    (ref_output * tmp_attention_mask).to(torch.float32),
                )
        else:
            with autocast_ctx:
                loss = loss_func(  # pylint: disable=not-callable
                    pred_output.to(torch.float32), ref_output.to(torch.float32)
                )

        return loss

    def _count_samples(self, inputs: Any) -> int:
        if isinstance(inputs, dict):
            hs = inputs.get("hidden_states")
            return len(hs) if isinstance(hs, list) else hs.shape[self.calibration_context.batch_dim]
        elif isinstance(inputs, list):
            return len(inputs)
        else:
            return inputs.shape[self.calibration_context.batch_dim]

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
        """Apply the AutoRound optimization algorithm to a block.

        This is the pure-algorithm entry point.  All infrastructure concerns
        (device placement, act-max hook collection, DDP setup, memory cleanup,
        logging) are handled by the Compressor before and after this call.

        Args:
            block: The transformer block module to quantize.
            fp_inputs: FP calibration inputs for this block (list[Tensor] or dict
                for diffusion models).
            input_others: Auxiliary kwargs passed to the block forward
                (e.g. attention_mask, position_ids).
            fp_outputs: FP reference outputs of the block used as the optimization
                target for the sign-gradient descent loss (list[Tensor]).
            q_inputs: Quantized inputs from the previous block, or ``None`` when
                cascaded quantized-input is disabled.
            block_ctx: Per-block pipeline context (BlockContext).
            input_ids: Raw token IDs from the tokenizer (``[1, seq_len]`` per
                sample). Used to derive the valid-token loss mask once (result
                cached on ``self._cached_valid_token_mask`` for reuse across
                all blocks). ``None`` disables loss masking.
            **kwargs: Reserved for forward-compatibility with future parameters.

        Returns:
            dict: Best quantization parameters found during optimization, or an
                empty dict if no trainable parameters were found. Wrappers stay
                installed for the composer to finalize and collect outputs.
        """
        device = device_manager.device
        # None coalesces to the primary: an explicit None from the placement pass
        # must leave the loop with a concrete staging device; pred/ref and the
        # loss mask all stage here (card-0-risk relocation may move it off the
        # primary, and the masked loss multiplies pred by the mask)
        loss_device = resolve_loss_device(getattr(self, "_loss_device", None), device)
        card_0_in_high_risk = getattr(self, "_card_0_in_high_risk", False)
        mid_iter_mem_check = self.compress_context.low_gpu_mem_usage and card_0_in_high_risk

        valid_token_mask = None
        # Derive valid_token_mask from raw token IDs when not supplied by caller.
        # Result is cached on self so it is computed only once across all blocks.
        if input_ids is not None:
            if not hasattr(self, "_cached_valid_token_mask"):
                self._cached_valid_token_mask = self._compute_valid_token_mask(input_ids)
            valid_token_mask = self._cached_valid_token_mask

        # Use quantized inputs if available and enabled
        active_inputs = q_inputs if (q_inputs is not None and self.enable_quanted_input) else fp_inputs
        nsamples = len(active_inputs) if isinstance(active_inputs, list) else self._count_samples(active_inputs)

        _perf = _PerfCounters()
        _perf.start("wrap")
        quantized_layer_names, unquantized_layer_names = self.wrapper_block(
            block,
            self.enable_minmax_tuning,
            self.enable_norm_bias_tuning,
            enable_torch_compile=self.compress_context.enable_torch_compile,
            device=device,
            enable_neuqi=getattr(self.config, "enable_neuqi", False),
            lora_rank=getattr(self.config, "lora_rank", 0),
            lora_alpha=getattr(self.config, "lora_alpha", None),
        )
        _perf.stop("wrap")
        _perf.start("prepare")

        round_params = []
        minmax_params = []
        # Group parameters by their effective lr so that mixed-bit configs
        # (e.g. a 4-bit model with a few 2-bit layers) use a per-layer lr
        # derived from each layer's own bit-width.
        round_lr_groups: dict[float, list] = {}
        minmax_lr_groups: dict[float, list] = {}
        teq_params: list = []
        ar_params: list = []
        seen_teq_param_ids: set[int] = set()
        teq_mode = getattr(self, "teq_refinement_mode", None)
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                layer_bits = getattr(m.orig_layer, "bits", None)
                layer_lr = self._config.compute_lr(layer_bits)
                if layer_lr is None:
                    layer_lr = self.lr
                self._maybe_log_low_bit_lr(layer_bits)
                layer_minmax_lr = self._config.compute_minmax_lr(layer_bits)
                if layer_minmax_lr is None:
                    layer_minmax_lr = self.minmax_lr
                for key in m.params.keys():
                    if key == "teq_log_alpha":
                        param = m.params[key]
                        if id(param) not in seen_teq_param_ids:
                            seen_teq_param_ids.add(id(param))
                            teq_params.append(param)
                        continue
                    ar_params.append(m.params[key])
                    if "min" in key or "max" in key:
                        minmax_params.append(m.params[key])
                        minmax_lr_groups.setdefault(float(layer_minmax_lr), []).append(m.params[key])
                    else:
                        round_params.append(m.params[key])
                        round_lr_groups.setdefault(float(layer_lr), []).append(m.params[key])

        lr = torch.tensor(self.lr)
        minmax_lr = torch.tensor(self.minmax_lr)

        if teq_mode is not None and not teq_params:
            # No TEQ mapping is active in this block (e.g. every mapping was
            # skipped, or explicit mappings cover only some blocks). Building a
            # TEQ schedule here would allocate an empty optimizer and run extra
            # no-op iterations, so fall back to the plain AutoRound path.
            logger.warning_once(
                "TEQ: no trainable equivalent scale in this block; falling back to the standard AutoRound schedule."
            )
            teq_mode = None

        extra_kwargs = {} if self.momentum is None else {"momentum": self.momentum}

        if len(round_params) + len(minmax_params) + len(teq_params) <= 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block"
            )
            logger.info(dump_info)
            return {}

        # Build optimizer param groups with a per-layer lr for the rounding
        # parameters (and min-max parameters when enabled).
        params = [{"params": ps, "lr": torch.tensor(group_lr)} for group_lr, ps in round_lr_groups.items()]
        if self.enable_minmax_tuning:
            params += [{"params": ps, "lr": torch.tensor(group_lr)} for group_lr, ps in minmax_lr_groups.items()]
        if teq_params:
            params.append({"params": teq_params, "lr": torch.tensor(float(self.teq_joint_lr))})

        baseline_iters = self.iters
        ar_first_iters = None
        refine_end = None
        refine_best_loss = torch.finfo(torch.float).max
        refine_best_params = None
        second_ar_best_loss = torch.finfo(torch.float).max
        second_ar_best_params = None
        second_ar_best_iter = None
        if teq_mode == "staged_refine":
            ar_first_iters = int(getattr(self, "teq_ar_first_iters", 0))
            if not 0 < ar_first_iters < self.iters:
                raise ValueError(
                    "TEQ staged_refine requires `teq_ar_first_iters` to be in "
                    f"[1, {self.iters - 1}], got {ar_first_iters}."
                )
            baseline_iters = ar_first_iters
            refine_end = baseline_iters + int(getattr(self, "teq_refine_iters", 0))
        run_iters = self.iters
        if teq_mode == "staged_refine":
            # One loss-only evaluation selects the post-update TEQ scale before
            # the second AR phase. The two AR phases still receive their
            # requested counts: ar_first_iters and iters - ar_first_iters.
            run_iters += int(getattr(self, "teq_refine_iters", 0)) + 1
        elif teq_mode is not None:
            refine_iters = int(getattr(self, "teq_refine_iters", 0))
            run_iters += refine_iters + (1 if refine_iters > 0 else 0)

        optimizer = self.optimizer(
            params,
            lr=lr,
            weight_decay=0,
            **extra_kwargs,
        )

        if self.lr_scheduler is None:
            ar_schedule_iters = self.iters if teq_mode == "staged_refine" else baseline_iters
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=max(1, ar_schedule_iters)
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)

        self._configure_lora_scaling(block, optimizer)

        from auto_round.parallel.dp.spmd import current_rank

        rank = current_rank()
        if rank is not None:
            rank.synchronize_lora(block)

        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        num_elm = 1
        mse_reduction = "mean"
        # a masked loss is a SUM over the valid elements, normalized by the
        # valid-token count below (num_elm) -- also at accumulation == 1, where
        # a mean over all elements divided again by that count normalizes twice
        if self.gradient_accumulate_steps != 1 or valid_token_mask:
            mse_reduction = "sum"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)
        scaler = self._get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        best_params = {}
        baseline_loss = None
        baseline_params = None
        total_loss = 0
        batch_size = self.calibration_context.batch_size
        global_batch_size = batch_size * self.gradient_accumulate_steps
        global_batch_size = min(nsamples, global_batch_size)
        # Compute num_elm once before the loop (used to normalise the accumulated loss).
        # We assume the block input and output shape is same
        if self.gradient_accumulate_steps != 1 and not valid_token_mask:
            whole_indices = torch.arange(global_batch_size)
            if isinstance(active_inputs, list):  # dict for diffusion, tricky setting, not sure whether it's correct
                num_elm = sum(active_inputs[i.item()].numel() for i in whole_indices)

        block, sync_gradients = self.parallel.prepare_gradients(self, block)
        staged_selection_indices = list(range(global_batch_size)) if teq_mode == "staged_refine" else None
        tuning_session = self.tuning_session(
            block,
            active_inputs,
            input_others,
            fp_outputs,
            nsamples=nsamples,
            global_batch_size=global_batch_size,
            iters=run_iters,
            prefetch=teq_mode != "staged_refine",
            loss_device=loss_device,
            lfq_last_block=(
                getattr(self, "enable_lfq", False)
                and input_ids is not None
                and block_ctx.block_index == block_ctx.block_cnt - 1
                and self._is_text_decoder_block(block_ctx.block_name)
            ),
        )

        # LFQ with lfq_num_blocks > 1: a non-final KD block's output must pass
        # through the remaining (FP) blocks before reaching the final norm/lm_head.
        lfq_downstream_blocks = []
        lfq_teacher_final = None
        if (
            input_ids is not None
            and isinstance(fp_outputs, list)
            and isinstance(active_inputs, list)
            and self._lfq_is_kd_block(block_ctx)
        ):
            lfq_downstream_blocks = self._get_lfq_downstream_blocks(block_ctx, device)
            if lfq_downstream_blocks and self.lfq_loss_type in ("dtk", "rkl", "tk"):
                # The teacher path is fixed across iterations: precompute it once.
                lfq_teacher_final = []
                with torch.no_grad():
                    for s in range(0, nsamples, batch_size):
                        idx = list(range(s, min(s + batch_size, nsamples)))
                        ref = torch.cat([fp_outputs[j] for j in idx], dim=0)
                        out = self._forward_lfq_downstream(
                            ref, lfq_downstream_blocks, active_inputs, input_others, idx, device
                        )
                        lfq_teacher_final.extend(o.to(fp_outputs[0].device) for o in torch.split(out, 1, dim=0))

        _perf.stop("prepare")
        _perf.start("loop")
        # iters>0 hot-pool strategy (loop-amortized; the iters=0 lane streams
        # its pools once and never reaches these pulls): each pool moves in
        # bulk onto the device that reads it every iteration -- the input
        # pool onto the entry device (BlockForwardRunner.device, where the
        # forward gathers each batch), the fp reference pool onto the loss
        # device. Both gates charge ACTUAL data: exact tuning state from the
        # wrapper walker, and (input pull) the routed-buffer budget proven on
        # the mapped-streaming lane -- tokens x top_k hidden rows as in/out
        # accumulators plus the backward grad (x6), reproducing the loop
        # retention from shapes alone. Declined pulls keep the per-batch
        # gather.
        self.maybe_adapt_moe_implementation(block, active_inputs, batch_size)

        def _restore_staged_params(saved_params):
            for name, module in block.named_modules():
                if not hasattr(module, "params") or name not in saved_params:
                    continue
                for key, value in saved_params[name].items():
                    if key in module.params:
                        module.params[key].data.copy_(value.to(module.params[key].device))

        @torch.no_grad()
        def _evaluate_staged_candidate(saved_params):
            """Evaluate one checkpoint on the fixed staged selection slice."""
            _restore_staged_params(saved_params)
            selection_loss = 0.0
            selection_num_elm = 1
            if valid_token_mask:
                selection_num_elm = self._get_non_zero_cnt(valid_token_mask, staged_selection_indices)
            selection_num_elm = max(1, selection_num_elm)
            for batch_start in range(0, len(staged_selection_indices), batch_size):
                indices = staged_selection_indices[batch_start : batch_start + batch_size]
                _sel = [fp_outputs[index] for index in indices]
                _gather = BlockForwardRunner._gather_same_device(
                    _sel, str(loss_device) if loss_device is not None else str(_sel[0].device)
                )
                ref_output = torch.cat(_gather, dim=0).to(loss_device)
                pred_output = self.block_forward.forward(
                    block, active_inputs, input_others, indices, tuning_session.forward_device
                )
                if loss_device is not None:
                    pred_output = pred_output.to(loss_device)
                if (
                    block_ctx.block_index == block_ctx.block_cnt - 1
                    and self.enable_lfq
                    and input_ids is not None
                    and self._is_text_decoder_block(block_ctx.block_name)
                ):
                    loss = self.lfq_loss(pred_output, torch.cat([input_ids[index] for index in indices], dim=0))
                else:
                    loss = self._get_loss(pred_output, ref_output, indices, mse_loss, loss_device, valid_token_mask)
                selection_loss += loss.item() / selection_num_elm
            return selection_loss

        with ExitStack() as stack:
            # Close staging before releasing downstream blocks, including on errors.
            for module in reversed(lfq_downstream_blocks):
                stack.callback(mv_module_from_gpu, module)
            tuning = stack.enter_context(tuning_session)
            active_inputs, fp_outputs = tuning.pull_pools(self.pull_tuning_pool, batch_size)
            for i in tuning.iterations():
                is_final_refinement_candidate = (
                    teq_mode in {"joint", "frozen_ar_refine"} and i == run_iters - 1 and run_iters > baseline_iters
                )
                if teq_mode is not None and i == baseline_iters:
                    if teq_mode == "staged_refine":
                        refine_optimizer = getattr(self, "teq_refine_optimizer", "adam")
                        if refine_optimizer == "lion":
                            from auto_round.algorithms.transforms.teq.lion import Lion

                            optimizer_class = Lion
                            optimizer_kwargs = {"betas": (0.9, 0.99)}
                        else:
                            optimizer_class = torch.optim.Adam
                            optimizer_kwargs = {"betas": (0.9, 0.9)}
                        optimizer = optimizer_class(
                            teq_params,
                            lr=float(self.teq_joint_lr),
                            weight_decay=0.0,
                            **optimizer_kwargs,
                        )
                        logger.info(
                            "TEQ staged refinement: optimizer=%s iters=%d lr=%.3e",
                            refine_optimizer,
                            int(getattr(self, "teq_refine_iters", 0)),
                            float(self.teq_joint_lr),
                        )
                    else:
                        optimizer = self.optimizer(params, lr=lr, weight_decay=0, **extra_kwargs)
                    lr_schedule = torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=1.0,
                        end_factor=0.0,
                        total_iters=max(1, int(getattr(self, "teq_refine_iters", 0))),
                    )
                    if best_params:
                        _restore_staged_params(best_params)
                        if teq_mode == "staged_refine":
                            for param in teq_params:
                                param.data.zero_()
                if self.enable_alg_ext and self.scheme.data_type.endswith("dq"):
                    for n, m in block.named_modules():
                        m.cur_iter = i
                total_loss = 0
                _perf.start("sampler")
                global_indices = (
                    staged_selection_indices
                    if teq_mode == "staged_refine" and baseline_iters <= i <= refine_end
                    else tuning.next_batch()
                )
                if valid_token_mask:
                    num_elm = self._get_non_zero_cnt(valid_token_mask, global_indices)
                _perf.stop("sampler")

                # Per-block loss schedule (constant across batches/iters):
                #   lfq_kd_block  -> apply the KD logit-lens loss on this block
                #   lfq_mse_block -> add the MSE reconstruction loss on this block
                # QAD: every fused group is tuned against the final-output LM loss only
                # (no MSE); with the whole network fused this is the true final logits.
                # Non-QAD: KD on the last `lfq_num_blocks`, MSE elsewhere (and on KD
                # blocks only when `lfq_mse_weight` > 0).
                lfq_kd_block = input_ids is not None and self._lfq_is_kd_block(block_ctx)
                if self.enable_qad:
                    lfq_mse_block = False
                else:
                    lfq_mse_block = (not lfq_kd_block) or self.lfq_mse_weight > 0

                for batch_start in range(0, len(global_indices), batch_size):
                    indices = global_indices[batch_start : batch_start + batch_size]
                    _perf.start("s_fwd")
                    pred_output, ref_output = tuning.forward(indices)
                    if loss_device is not None:
                        pred_output = pred_output.to(loss_device)
                    _perf.stop("s_fwd")
                    _perf.start("s_loss")
                    if lfq_kd_block:
                        # For KD losses ("dtk"/"rkl"/"tk"), ref_output (FP block output) is the teacher.
                        teacher_output = ref_output if self.lfq_loss_type in ("dtk", "rkl", "tk") else None
                        lfq_student = pred_output
                        if lfq_downstream_blocks:
                            lfq_student = self._forward_lfq_downstream(
                                pred_output, lfq_downstream_blocks, active_inputs, input_others, indices, device
                            )
                            if lfq_teacher_final is not None:
                                teacher_output = torch.cat([lfq_teacher_final[j] for j in indices], dim=0).to(
                                    lfq_student.device
                                )
                        loss = self.lfq_loss(
                            lfq_student,
                            torch.cat([input_ids[i] for i in indices], dim=0),
                            teacher_hidden_state=teacher_output,
                            sample_indices=indices,
                        )
                        # Optionally combine with the MSE block-reconstruction loss so the
                        # block keeps its reconstruction fidelity while adding logit alignment.
                        if lfq_mse_block:
                            mse_term = self._get_loss(
                                pred_output, ref_output, indices, mse_loss, device, valid_token_mask
                            )
                            # In QAD a scheduled MSE checkpoint defaults to weight 1.0; outside
                            # QAD the explicit `lfq_mse_weight` controls the combination.
                            mse_w = self.lfq_mse_weight if self.lfq_mse_weight > 0 else 1.0
                            loss = loss + mse_w * mse_term
                    else:
                        # the mask must land on the same device pred/ref were staged
                        # on (loss_device, which card-0-risk relocation may move off
                        # the primary), not on the primary unconditionally
                        loss = self._get_loss(pred_output, ref_output, indices, mse_loss, loss_device, valid_token_mask)
                    num_elm = 1 if num_elm <= 0 else num_elm
                    total_loss += loss.item() / num_elm
                    _perf.stop("s_loss")
                    _perf.start("s_bwd")

                    if mid_iter_mem_check:
                        # clear memory to avoid OOM due to memory fragmentation
                        clear_memory_if_reached_threshold(threshold=0.5, device_list=device_manager.device_list)

                    self._scale_loss_and_backward(scaler, loss)
                    _perf.stop("s_bwd")

                    if mid_iter_mem_check:
                        # clear memory to avoid OOM due to memory fragmentation
                        clear_memory_if_reached_threshold(threshold=0.8, device_list=device_manager.device_list)

                total_loss = self.parallel.reduce_loss(total_loss, num_elm)
                if teq_mode is not None and i < baseline_iters:
                    for param in teq_params:
                        param.grad = None
                elif teq_mode == "frozen_ar_refine":
                    for param in ar_params:
                        param.grad = None
                elif teq_mode == "staged_refine":
                    if i < refine_end:
                        for param in ar_params:
                            param.grad = None
                    else:
                        for param in teq_params:
                            param.grad = None
                if is_final_refinement_candidate:
                    optimizer.zero_grad()
                if i == 0:
                    init_loss = total_loss
                current_lr = optimizer.param_groups[0]["lr"]
                logger.debug("iter %d loss: %.3e lr: %s", i, total_loss, current_lr)

                _perf.start("snap")
                if total_loss < best_loss:
                    best_loss = total_loss
                    if not self.not_use_best_mse:
                        # drop the previous snapshot BEFORE cloning the new one:
                        # holding both doubles the snapshot footprint for the
                        # duration of the clone (weight-sized fp32 rounding
                        # values make that window OOM-class for huge layers),
                        # and the fresh clone reuses the freed storage when
                        # shapes match, so the swap leaves no fragmentation
                        best_params = None
                        best_params = tuning.collect_best_params()
                        last_best_iter = i
                if teq_mode == "staged_refine" and baseline_iters <= i <= refine_end and total_loss < refine_best_loss:
                    refine_best_loss = total_loss
                    refine_best_params = tuning.collect_best_params()
                if teq_mode is not None and i + 1 == baseline_iters and not self.not_use_best_mse:
                    baseline_loss = best_loss
                    baseline_params = copy.deepcopy(best_params)
                    for saved_params in (best_params, baseline_params):
                        for layer_params in saved_params.values():
                            if "teq_log_alpha" in layer_params:
                                layer_params["teq_log_alpha"].zero_()
                if self.not_use_best_mse and i == run_iters - 1:
                    best_params = None
                    best_params = tuning.collect_best_params()

                _perf.stop("snap")
                _perf.start("step")
                if not self.not_use_best_mse:
                    if 0 < self.dynamic_max_gap <= i - last_best_iter and i + 1 != baseline_iters:
                        break
                if is_final_refinement_candidate:
                    break
                if teq_mode == "staged_refine" and i == refine_end:
                    if refine_best_params:
                        _restore_staged_params(refine_best_params)
                    ar_optimizer_params = [
                        {"params": ps, "lr": torch.tensor(group_lr)} for group_lr, ps in round_lr_groups.items()
                    ]
                    if self.enable_minmax_tuning:
                        ar_optimizer_params += [
                            {"params": ps, "lr": torch.tensor(group_lr)} for group_lr, ps in minmax_lr_groups.items()
                        ]
                    optimizer = self.optimizer(ar_optimizer_params, lr=lr, weight_decay=0, **extra_kwargs)
                    if self.lr_scheduler is None:
                        remaining_iters = self.iters - baseline_iters
                        lr_schedule = torch.optim.lr_scheduler.LinearLR(
                            optimizer,
                            start_factor=remaining_iters / self.iters,
                            end_factor=0.0,
                            total_iters=max(1, remaining_iters),
                        )
                    else:
                        lr_schedule = copy.deepcopy(self.lr_scheduler)
                    optimizer.zero_grad()
                    second_ar_best_loss = total_loss
                    second_ar_best_params = tuning.collect_best_params()
                    second_ar_best_iter = i
                    continue
                if teq_mode == "staged_refine" and i > refine_end and total_loss < second_ar_best_loss:
                    second_ar_best_loss = total_loss
                    second_ar_best_params = tuning.collect_best_params()
                    second_ar_best_iter = i
                sync_gradients()
                self._step(scaler, optimizer, lr_schedule)
                for param in teq_params:
                    with torch.no_grad():
                        param.clamp_(
                            min=math.log(getattr(param, "_teq_min_scale", 1e-5)),
                            max=math.log(getattr(param, "_teq_max_scale", 10.0)),
                        )
                _perf.stop("step")

        del lfq_teacher_final

        _perf.stop("loop")
        _perf.start("tail")

        last_loss = total_loss
        best_iter = run_iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        if teq_mode == "staged_refine" and second_ar_best_params is not None and baseline_params is not None:
            selection_baseline_loss = _evaluate_staged_candidate(baseline_params)
            selection_second_loss = _evaluate_staged_candidate(second_ar_best_params)
            if selection_second_loss <= selection_baseline_loss:
                best_loss = selection_second_loss
                best_params = second_ar_best_params
                last_loss = selection_second_loss
                best_iter = second_ar_best_iter
            else:
                best_loss = selection_baseline_loss
                best_params = baseline_params
                last_loss = selection_baseline_loss
                best_iter = max(0, baseline_iters - 1)
            logger.info(
                "TEQ staged fixed-slice selection: baseline_loss=%.3e second_ar_loss=%.3e selected=%s",
                selection_baseline_loss,
                selection_second_loss,
                "second_ar" if selection_second_loss <= selection_baseline_loss else "baseline",
            )
        elif baseline_loss is not None and best_loss > baseline_loss:
            best_loss = baseline_loss
            best_params = baseline_params
            last_loss = baseline_loss
            best_iter = max(0, baseline_iters - 1)
        if run_iters > 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block, loss iter 0: {init_loss:.3e} -> iter {best_iter}: {last_loss:.3e}"
            )
        else:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                "layers in the block"
            )

        self.compress_context.clear_memory()  # clear cached memory during training
        if len(unquantized_layer_names) != 0:
            logger.info(f"Unquantized layers: {unquantized_layer_names}")
        if baseline_loss is not None:
            selected_teq_delta = 0.0
            for layer_params in best_params.values():
                log_alpha = layer_params.get("teq_log_alpha")
                if log_alpha is not None:
                    selected_teq_delta = max(selected_teq_delta, float(log_alpha.abs().max().item()))
            self.teq_last_refinement = {
                "baseline_loss": float(baseline_loss),
                "selected_loss": float(best_loss),
                "improved": bool(best_loss < baseline_loss),
                "selected_teq_log_alpha_abs_max": selected_teq_delta,
            }
            logger.info(
                "TEQ refinement selection: baseline_loss=%.3e selected_loss=%.3e selected_log_alpha_abs_max=%.3e",
                baseline_loss,
                best_loss,
                selected_teq_delta,
            )

        _perf.stop("tail")
        _log_tune_phases(_perf, run_iters)

        logger.infoclean(dump_info)
        return best_params

    @staticmethod
    def _preallocate_tuning_grads_(optimizer, layer_name=""):
        """Create zero ``.grad`` buffers so autograd accumulates in place.

        The first backward would otherwise allocate a dense gradient for every
        tuning parameter mid-run (as large as the rounding parameter itself);
        allocating it up front keeps the peak inside the budget measured at
        wrapper-construction time and is gradient-identical (accumulate into
        zeros)."""
        total = 0
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None and param.requires_grad:
                    param.grad = torch.zeros_like(param)
                    total += param.grad.numel() * param.grad.element_size()
        if total:
            logger.debug(
                "pre-allocated tuning gradient buffers for %s (%.2fGiB)",
                layer_name,
                total / 1024**3,
            )

    def quantize_layer_outside_block(
        self,
        layer: "torch.nn.Module",
        fp_inputs: list[torch.Tensor] | None = None,
        q_inputs: list[torch.Tensor] | None = None,
        disable_opt_rtn: bool | None = None,
        input_ids: list[torch.Tensor] | None = None,
    ):
        """Quantize a single layer that lives outside a transformer block.

        When ``fp_inputs`` is provided the layer is tuned with the sign-gradient
        descent optimizer (same loss loop as block-level quantization).  When
        ``fp_inputs`` is ``None`` the method falls back to zero-shot RTN.

        Args:
            layer: The layer module to quantize.  Must have a ``global_name``
                attribute for model re-insertion and logging.
            fp_inputs: Per-sample FP activations fed into this layer, used as
                calibration inputs during optimization. ``None`` triggers RTN
                fallback.
            q_inputs: Per-sample quantized activations from the previous stage,
                used instead of ``fp_inputs`` during the forward pass when
                cascaded quantized-input is enabled. ``None`` means use
                ``fp_inputs`` for both reference and tuning forward.
            disable_opt_rtn: Override optimized-RTN; ``None`` defers to quantizer config.
            input_ids: Raw token IDs from the tokenizer (``[1, seq_len]`` per
                sample); used to derive the valid-token loss mask via
                ``_compute_valid_token_mask``. ``None`` disables loss masking.
        """

        layer_name = layer.global_name
        if fp_inputs is None:
            logger.info(f"using rtn to quantize {layer_name}")
            self._quantize_layer_via_rtn(
                layer,
                disable_opt_rtn=(
                    disable_opt_rtn if disable_opt_rtn is not None else getattr(self.config, "disable_opt_rtn", True)
                ),
            )
            return

        from auto_round.parallel.tp.runtime import is_dtensor

        if is_dtensor(layer.weight):
            raise NotImplementedError(
                "Calibrated outside-block tuning does not support DTensor weights yet; "
                "keep this layer unquantized or use RTN without calibration inputs."
            )

        # Derive valid_token_mask from raw token IDs when not supplied by caller.
        # Reuse the cached mask if already computed by a previous block.
        valid_token_mask = None
        if input_ids is not None:
            if not hasattr(self, "_cached_valid_token_mask"):
                self._cached_valid_token_mask = self._compute_valid_token_mask(input_ids)
            valid_token_mask = self._cached_valid_token_mask

        logger.info(f"quantizing layer {layer_name}")
        # Layer is already on the correct device (placed by the caller / AlgorithmComposer).
        device = layer.weight.device if hasattr(layer, "weight") else device_manager.device
        for i in range(len(fp_inputs)):
            fp_inputs[i] = fp_inputs[i].to(layer.weight.dtype)
            if q_inputs is not None:
                q_inputs[i] = q_inputs[i].to(layer.weight.dtype)

        # Inductor's backward for a 1B+-element value parameter materializes
        # an extra full-size tiled buffer (observed ~4.7GiB on a 248k-vocab
        # lm_head) on top of the value, its gradient, and the quantized
        # weight autograd saves - compiled eager autograd fits where the
        # compiled graph does not. Compiling one outside-block layer for a
        # handful of iterations buys no measurable speed, so skip it for
        # wrappers whose rounding parameter exceeds the chunking budget.
        value_elements = layer.weight.numel() if getattr(layer, "weight", None) is not None else 0
        compile_wrapper = self.compress_context.enable_torch_compile and value_elements <= _OUTSIDE_TUNE_CHUNK_OUT_ELEMS
        if self.compress_context.enable_torch_compile and not compile_wrapper:
            logger.info(
                "skipping torch.compile for %s (rounding parameter of %d elements; compiled backward "
                "would exceed the memory budget)",
                getattr(layer, "global_name", "layer"),
                value_elements,
            )
        wrapper_linear = WrapperLinear(
            layer,
            enable_minmax_tuning=self.enable_minmax_tuning,
            enable_torch_compile=compile_wrapper,
            device=device,
            enable_neuqi=getattr(self.config, "enable_neuqi", False),
            lora_rank=getattr(self.config, "lora_rank", 0),
            lora_alpha=getattr(self.config, "lora_alpha", None),
        ).to(device)
        round_params = []
        minmax_params = []
        for key in wrapper_linear.params.keys():
            if "min" in key or "max" in key:
                minmax_params.append(wrapper_linear.params[key])
            else:
                round_params.append(wrapper_linear.params[key])
        if len(round_params) + len(minmax_params) <= 0:
            dump_info = f"quantized {layer_name}"
            logger.info(dump_info)
            with torch.no_grad():
                unwrapper_layer(self.model, wrapper_linear, layer_name, {})
            mv_module_from_gpu(layer)

        lr = torch.tensor(self.lr)
        minmax_lr = torch.tensor(self.minmax_lr)
        # Use a lr derived from this layer's own bit-width so mixed-bit configs
        # (e.g. a 4-bit model with a few 2-bit layers) tune each layer correctly.
        layer_bits = getattr(layer, "bits", None)
        layer_lr = self._config.compute_lr(layer_bits)
        if layer_lr is not None:
            lr = torch.tensor(layer_lr)
        self._maybe_log_low_bit_lr(layer_bits)
        layer_minmax_lr = self._config.compute_minmax_lr(layer_bits)
        if layer_minmax_lr is not None:
            minmax_lr = torch.tensor(layer_minmax_lr)
        if self.enable_minmax_tuning:
            optimizer = self.optimizer(
                [{"params": round_params}, {"params": minmax_params, "lr": minmax_lr}], lr=lr, weight_decay=0
            )
        else:
            optimizer = self.optimizer(round_params, lr=lr, weight_decay=0)
        self._preallocate_tuning_grads_(optimizer, layer_name)

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)
        self._configure_lora_scaling(wrapper_linear, optimizer)
        nsamples = len(fp_inputs)
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        best_params = None
        # reserve the huge-layer snapshot BEFORE the loop: the row-window
        # machinery free-probes per forward, so an on-device reservation
        # shrinks the windows from iteration 0 instead of overflowing on the
        # first improving iteration
        snapshot_slot = BestParamsSlot(wrapper_linear) if value_elements > _OUTSIDE_TUNE_CHUNK_OUT_ELEMS else None
        scaler = self._get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None

        gradient_accumulate_steps = (
            self.calibration_context.batch_size * self.gradient_accumulate_steps
        )  # Force to low gpu

        total_loss = 0
        num_elm = 1
        mse_reduction = "mean"
        if gradient_accumulate_steps != 1:
            mse_reduction = "sum"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)
        mse_sum_loss = torch.nn.MSELoss(reduction="sum").to(device)
        batch_size = 1  # Force to low gpu
        global_batch_size = gradient_accumulate_steps
        global_batch_size = min(nsamples, global_batch_size)
        # Compute num_elm once before the loop.
        if gradient_accumulate_steps != 1:
            whole_indices = list(range(global_batch_size))
            if valid_token_mask:
                num_elm = self._get_non_zero_cnt(valid_token_mask, whole_indices)
            elif q_inputs is not None:
                num_elm = self._count_layer_input_elements(q_inputs, whole_indices)
            else:
                num_elm = self._count_layer_input_elements(fp_inputs, whole_indices)

        index_sampler = IndexSampler(nsamples, global_batch_size)

        for i in range(self.iters):
            total_loss = 0
            global_indices = index_sampler.next_batch()

            for batch_start in range(0, len(global_indices), batch_size):
                indices = global_indices[batch_start : batch_start + batch_size]
                if q_inputs is not None:
                    current_input = [q_inputs[i] for i in indices]
                    current_input = torch.cat(current_input, dim=0).to(device)
                    org_input = [fp_inputs[i] for i in indices]
                    org_input = torch.cat(org_input, dim=0).to(device)
                else:
                    current_input = [fp_inputs[i] for i in indices]
                    current_input = torch.cat(current_input, dim=0).to(device)
                    org_input = current_input
                autocast_ctx = (
                    nullcontext()
                    if not self.model_context.amp
                    else autocast(device_type=str(device).split(":")[0], dtype=self.model_context.amp_dtype)
                )
                # Huge-output layers (a full-vocabulary lm_head) cannot hold one
                # sample's fp32 logits plus the backward transients on a single
                # GPU; slice the sample's rows so each forward stays within the
                # output-element budget. Gradients accumulate across slices, so
                # the result is identical to the single-shot loss.
                rows_axis = 1 if current_input.dim() == 3 else 0
                sample_rows = current_input.shape[rows_axis]
                out_features = layer.weight.shape[0] if layer.weight.dim() == 2 else None
                # row-blocked wrappers keep only one block's graph alive, so the
                # position-chunk budget must size against the block width, not
                # the full output; they must also take the interleaved loop
                # even when one chunk covers all rows (a single-shot forward
                # would hold every block's graph at once)
                blockwise = wrapper_linear.row_block_active()
                block_bounds = wrapper_linear.row_block_bounds() if blockwise else None
                chunk_out_features = out_features
                if blockwise:
                    chunk_out_features = block_bounds[0][1] - block_bounds[0][0]
                chunk_rows = (
                    _outside_tune_rows_per_chunk(chunk_out_features, sample_rows, _OUTSIDE_TUNE_CHUNK_OUT_ELEMS)
                    if chunk_out_features is not None
                    else sample_rows
                )
                if chunk_rows >= sample_rows and not blockwise:
                    with torch.no_grad():
                        current_output = layer(org_input)
                    if valid_token_mask:
                        tmp_valid_mask = _valid_token_mask_rows(valid_token_mask, indices, device, rows_axis)

                        with autocast_ctx:
                            output_q = wrapper_linear(current_input)  # pylint: disable=not-callable
                            loss = mse_loss(  # pylint: disable=not-callable
                                (output_q * tmp_valid_mask).to(torch.float32),
                                (current_output * tmp_valid_mask).to(torch.float32),
                            )

                    else:
                        with autocast_ctx:
                            output_q = wrapper_linear(current_input)  # pylint: disable=not-callable
                            loss = mse_loss(  # pylint: disable=not-callable
                                output_q.to(torch.float32),
                                current_output.to(torch.float32),  # mul 1.0 will copy the output
                            )

                    num_elm = 1 if num_elm <= 0 else num_elm
                    total_loss += loss.item() / num_elm

                    self._scale_loss_and_backward(scaler, loss)
                else:
                    # chunk-sum losses divide by the same denominator the
                    # single-shot reduction uses, so both the accumulated
                    # gradient and the logged total stay unchanged
                    sum_denom = sample_rows * out_features if mse_reduction == "mean" else 1
                    cat_mask = None
                    if valid_token_mask:
                        cat_mask = _valid_token_mask_rows(valid_token_mask, indices, device, rows_axis)
                    # a row-blocked wrapper keeps its autograd graph per output
                    # block, so backward must run per block too: the MSE sum
                    # decomposes over output columns and gradients accumulate,
                    # making the result identical to one big backward
                    num_elm = 1 if num_elm <= 0 else num_elm
                    for start in range(0, sample_rows, chunk_rows):
                        end = min(start + chunk_rows, sample_rows)
                        chunk_input = current_input[:, start:end] if rows_axis == 1 else current_input[start:end]
                        chunk_ref_input = org_input[:, start:end] if rows_axis == 1 else org_input[start:end]
                        with torch.no_grad():
                            chunk_ref = layer(chunk_ref_input)
                        if cat_mask is not None:
                            # rows live on dim 1 for 3-D [batch, seq, out] inputs;
                            # the mask is [rows, 1] after unsqueeze, so slice the
                            # matching axis
                            chunk_mask = cat_mask[:, start:end] if rows_axis == 1 else cat_mask[start:end]
                        if not blockwise:
                            with autocast_ctx:
                                chunk_q = wrapper_linear(chunk_input)  # pylint: disable=not-callable
                            if cat_mask is not None:
                                loss = mse_sum_loss(
                                    (chunk_q * chunk_mask).to(torch.float32),
                                    (chunk_ref * chunk_mask).to(torch.float32),
                                )
                            else:
                                loss = mse_sum_loss(chunk_q.to(torch.float32), chunk_ref.to(torch.float32))
                            loss = loss / sum_denom
                            total_loss += loss.item() / num_elm
                            self._scale_loss_and_backward(scaler, loss)
                        else:
                            for b0, b1 in block_bounds:
                                # forward, loss, and backward complete per block
                                # so only one block's autograd graph is ever live
                                # (bias included: the reference forward keeps it)
                                _bias = getattr(layer, "bias", None)
                                with autocast_ctx:
                                    chunk_q_block = wrapper_linear.forward_rows(chunk_input, b0, b1, bias=_bias)
                                ref_block = chunk_ref[..., b0:b1]
                                if cat_mask is not None:
                                    loss = mse_sum_loss(
                                        (chunk_q_block * chunk_mask).to(torch.float32),
                                        (ref_block * chunk_mask).to(torch.float32),
                                    )
                                else:
                                    loss = mse_sum_loss(chunk_q_block.to(torch.float32), ref_block.to(torch.float32))
                                loss = loss / sum_denom
                                total_loss += loss.item() / num_elm
                                self._scale_loss_and_backward(scaler, loss)
            if i == 0:
                init_loss = total_loss
            current_lr = optimizer.param_groups[0]["lr"]
            logger.debug("iter %d loss: %.3e lr: %s", i, total_loss, current_lr)

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    best_params = (
                        snapshot_slot.refresh(wrapper_linear)
                        if snapshot_slot is not None
                        else collect_best_params(wrapper_linear, self.compress_context.cache_device)
                    )
                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_params = (
                    snapshot_slot.refresh(wrapper_linear)
                    if snapshot_slot is not None
                    else collect_best_params(wrapper_linear, self.compress_context.cache_device)
                )

            if not self.not_use_best_mse:
                if 0 < self.dynamic_max_gap <= i - last_best_iter:
                    break
            self._step(scaler, optimizer, lr_schedule)

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        # tuning is complete: the gradient buffers (as large as the rounding
        # parameter on huge layers, zeroed in place between steps) are dead
        # weight for the final quantize/dequantize transients that follow
        for group in optimizer.param_groups:
            for param in group["params"]:
                param.grad = None
        with torch.no_grad():
            unwrapper_layer(self.model, wrapper_linear, layer_name, best_params)
        mv_module_from_gpu(layer)
        dump_info = f"quantized {layer_name},  loss iter 0: {init_loss:.3e} -> iter {best_iter}: {last_loss:.3e}"
        logger.info(dump_info)

    def finalize_run(self) -> None:
        """Clear per-run caches (``_cached_valid_token_mask``, LFQ components)."""
        for attr in ("_cached_valid_token_mask", "_lm_head", "_post_block_modules"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _get_optimizer(self, optimizer: Any):
        """Returns the specified optimizer. In SignRound, we fix the optimizer.

        Args:
        optimizer: The optimizer to be used.

        Returns:
        The specified optimizer.
        """
        if optimizer is not None:
            logger.warning_once(
                "The optimizer setting in config will be ignored in AutoRound, using SignSGD as default."
            )
        return SignSGD

    def _count_layer_input_elements(self, input_ids, indices: list) -> int:
        return sum(input_ids[i].numel() for i in indices)

    def _get_scaler(self):
        """Returns scaler, in SignRound, no need to use scaler."""
        return

    def _scale_loss_and_backward(self, scaler: Any, loss: torch.Tensor) -> torch.Tensor:
        """Scales the loss and performs backward pass.

        Args:
        scaler: The scaler to be used.
        loss: The loss to be scaled.

        Returns:
        The scaled loss.
        """
        scale_loss = loss * 1000
        try:
            scale_loss.backward()
        except torch.OutOfMemoryError:
            from auto_round.algorithms.quantization.search_dispatch import dump_oom_tensor_census_

            dump_oom_tensor_census_("tune backward")
            raise
        if is_hpex_available():
            htcore.mark_step()
        return scale_loss

    def _step(self, scaler: Any, optimizer: Any, lr_schedule: Any):
        """Performs a step in the optimization process.

        Args:
        scaler: The scaler to be used.
        optimizer: The optimizer for the step.
        lr_schedule: The learning rate schedule.

        Returns:
        None
        """
        optimizer.step()
        # for hpu
        if is_hpex_available():
            htcore.mark_step()
        # keep the buffers: reallocating a dense gradient as large as the
        # rounding parameter would break the memory budget of huge layers
        # (outside-block lm_head tuning) in every iteration after the first
        optimizer.zero_grad(set_to_none=False)
        lr_schedule.step()
