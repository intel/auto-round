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
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from torch import autocast

from auto_round.algorithms.block_runner import _cat_device_safe
from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.compressors.utils import (
    IndexSampler,
    collect_best_params,
    shard_samplers,
)
from auto_round.logger import logger


def _tune_phase_line(phases: dict, iters: int) -> str:
    """Format the per-block tuning phase breakdown for AR_PERF_COUNTERS.

    ``wrap`` = wrapper_block (params init, quant-func resolve + optional
    per-wrapper torch.compile, SignRoundV2 init-scale search); ``prepare`` =
    DDP engagement + tuning-param collection + optimizer/scheduler build;
    ``loop`` = the iteration try/finally (overlaps the tune-ddp line at
    iters>0); ``tail`` = best-params restore, clear_memory, unwrapping.
    """
    return "[perf] tune phases (iters=%d): wrap=%.2fs prepare=%.2fs loop=%.2fs tail=%.2fs" % (
        iters,
        phases.get("wrap", 0.0),
        phases.get("prepare", 0.0),
        phases.get("loop", 0.0),
        phases.get("tail", 0.0),
    )


from auto_round.utils import (
    htcore,
    is_hpex_available,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
)
from auto_round.utils.device import clear_memory_if_reached_threshold
from auto_round.utils.device_manager import device_manager
from auto_round.utils.distributed import is_distributed, setup_ddp_if_needed_
from auto_round.wrapper import WrapperLinear, unwrapper_block, unwrapper_layer, wrapper_block

if TYPE_CHECKING:
    from auto_round.algorithms.composer import BlockContext


@register_pipeline_member(SignRoundConfig)
class SignRoundQuantizer(BaseQuantizer):

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

        self.optimizer = self._get_optimizer(optimizer=config.optimizer)
        self.wrapper_block = wrapper_block
        # Kept for per-layer (mixed-bit) lr resolution during tuning.
        self._config = config
        self.lr_is_auto = getattr(config, "lr_is_auto", False)
        self.minmax_lr_is_auto = getattr(config, "minmax_lr_is_auto", False)
        # Emit the low-bit lr notice at most once across all blocks/layers.
        self._logged_low_bit_lr = False

    def _maybe_log_low_bit_lr(self, bits) -> None:
        """Log once when low-bit (<=3) layers get the higher 2.0/iters lr."""
        if self._logged_low_bit_lr or not self.lr_is_auto:
            return
        if self.iters >= 1000 and bits is not None and bits <= 3:
            logger.info("using higher lr (2.0/iters) for <=3 bit layers to improve accuracy")
            self._logged_low_bit_lr = True

    def dispatch_block(self, block, input_ids, input_others):
        """Multi-GPU aware block dispatch for SignRound tuning.

        Stores _card_0_in_high_risk and _loss_device on self for use in quantize_block.
        Returns the block after device placement.
        """
        from auto_round.utils import is_auto_device_mapping

        if (
            is_auto_device_mapping(device_manager.device_map)
            and len(device_manager.device_list) > 1
            and not self.model_context.is_diffusion
        ):
            from auto_round.utils.device import set_auto_device_map_for_block_with_tuning

            card_0_in_high_risk, loss_device = set_auto_device_map_for_block_with_tuning(
                block,
                device_manager.device_list,
                input_ids,
                self.compress_context.low_gpu_mem_usage,
                self.calibration_context.batch_size,
                device_manager.device,
            )
            if len(device_manager.device_list) > 1:
                from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                for _n, _mod in block.named_modules():
                    if len(list(_mod.children())) != 0 or not hasattr(_mod, "tuning_device"):
                        continue
                    add_hook_to_module(_mod, AlignDevicesHook(_mod.tuning_device, io_same_device=True), True)
        else:
            from auto_round.utils.model import move_to_device_preserving_cpu_pinned, place_ngram_embeddings_for_tuning_

            # Honor AR_NGRAM_DEVICE even on a single GPU (default keeps the table on CPU and
            # logs a hint); this also surfaces the ngram size / placement info to the user.
            place_ngram_embeddings_for_tuning_(block)
            block = move_to_device_preserving_cpu_pinned(block, device_manager.device)
            card_0_in_high_risk, loss_device = False, device_manager.device

        self._card_0_in_high_risk = card_0_in_high_risk
        self._loss_device = loss_device
        return block

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
        device: Union[str, torch.device] = "cpu",
        valid_token_mask: Optional[torch.Tensor] = None,
        input_ids=None,
    ):
        autocast_ctx = (
            nullcontext()
            if self.model_context.amp
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

    def _init_lm_components(self) -> None:
        """Lazily locate and cache ``_post_block_modules`` and ``_lm_head`` for LFQ loss.

        ``_post_block_modules`` is an ordered list of modules that must be applied
        between the last transformer-block output and the lm_head projection.
        Most architectures have a single final norm; OPT additionally has an
        optional ``project_out`` projection.

        Supported without manual configuration:
            LLaMA / Qwen / Gemma / Mistral / InternLM / Phi-3 — ``model.model.norm``
            OPT — ``model.model.decoder.{final_layer_norm, project_out}`` (both optional)
            GPT-2 / Falcon / Bloom — ``model.transformer.ln_f``
            GPT-NeoX / Pythia — ``model.gpt_neox.final_layer_norm``
            Phi / Phi-2 — ``model.model.final_layernorm``
            MPT — ``model.transformer.norm_f``
            ChatGLM — ``model.transformer.encoder.final_layernorm``
            RWKV — ``model.rwkv.ln_out``

        Raises ``AttributeError`` if no lm_head equivalent can be found.
        """
        if hasattr(self, "_lm_head"):
            return

        model = self.model

        # ── lm_head ──────────────────────────────────────────────────────────
        for name in ("lm_head", "embed_out", "output", "head"):
            if hasattr(model, name):
                self._lm_head = getattr(model, name)
                break
        else:
            raise AttributeError(
                f"Cannot locate lm_head in {type(model).__name__}. " "Checked: lm_head, embed_out, output, head."
            )

        # ── post-block processing (ordered list applied before lm_head) ───────
        # OPT: decoder has both an optional final_layer_norm *and* an optional
        # project_out that maps ffn_dim → word_embed_proj_dim.  Detect by probing
        # for the characteristic project_out attribute (may be None).
        try:
            decoder = model.model.decoder
            _ = decoder.project_out  # raises AttributeError if not an OPT decoder
            self._post_block_modules = [m for m in (decoder.final_layer_norm, decoder.project_out) if m is not None]
            return
        except AttributeError:
            pass

        # All other architectures: single optional final norm.
        norm_getters = [
            lambda: model.model.norm,  # LLaMA / Qwen / Gemma / Mistral / InternLM / Phi-3
            lambda: model.transformer.ln_f,  # GPT-2 / Falcon / Bloom
            lambda: model.gpt_neox.final_layer_norm,  # GPT-NeoX / Pythia
            lambda: model.model.final_layernorm,  # Phi / Phi-2
            lambda: model.transformer.norm_f,  # MPT
            lambda: model.transformer.encoder.final_layernorm,  # ChatGLM
            lambda: model.rwkv.ln_out,  # RWKV
        ]
        self._post_block_modules = []
        for getter in norm_getters:
            try:
                norm = getter()
                if norm is not None:
                    self._post_block_modules = [norm]
                    break
            except AttributeError:
                continue

    # Keywords that identify non-text (visual / audio / multimodal) blocks.
    # LFQ loss is only meaningful for pure language-model decoder blocks.
    _NON_TEXT_BLOCK_KEYWORDS = frozenset(
        {
            "vis",
            "vision",
            "visual",
            "image",
            "img",
            "audio",
            "video",
            "patch",
            "pixel",
            "clip",
            "vit",
            "perceiver",
            "resampler",
            "connector",
            "projector",
        }
    )

    def _is_text_decoder_block(self, block_name: str) -> bool:
        """Return ``True`` if *block_name* refers to a text-decoder block.

        Blocks whose names contain any of the non-text keywords (vision, audio,
        image, …) are considered multimodal and excluded from LFQ loss.
        """
        name_lower = block_name.lower()
        return not any(kw in name_lower for kw in self._NON_TEXT_BLOCK_KEYWORDS)

    def lfq_loss(self, hidden_state: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute LM cross-entropy loss from the last block's hidden states.

        Applies every post-block module (final norm, optional projection, …) in
        order, then runs lm_head and computes next-token prediction loss.
        Positions marked with ``-100`` in *input_ids* are excluded from the loss.

        Args:
            hidden_state: Last block output, shape ``[batch, seq_len, hidden]``.
            input_ids:    Token-ID labels with ``-100`` for ignored positions,
                          shape ``[batch, seq_len]``.

        Returns:
            Scalar cross-entropy loss tensor.
        """
        self._init_lm_components()
        device = hidden_state.device

        for module in self._post_block_modules:
            module.to(device)
            hidden_state = module(hidden_state)

        self._lm_head.to(device)
        logits = self._lm_head(hidden_state)

        if hasattr(self.model, "loss_function"):
            loss = self.model.loss_function(
                logits=logits,
                labels=input_ids.to(device),
                vocab_size=self.model.config.vocab_size,
            )
        else:
            import torch.nn.functional as F

            # Standard causal-LM shift: predict token t+1 from hidden state t.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous().to(device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return loss

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
                empty dict if no trainable parameters were found.
        """
        device = device_manager.device
        loss_device = getattr(self, "_loss_device", device)
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

        from auto_round import envs as _envs
        from auto_round.algorithms.quantization.sign_round.data_parallel import expect_pool_local

        replica_group = None
        mirror_optimizers = []
        mirror_schedules = []
        params_per_replica = []
        _dp_placed = False  # True once pools are distributed across the engaged plan

        import time as _ptime

        from auto_round.algorithms.quantization.sign_round.data_parallel import pre_wrap_shard_candidate

        _defer_wrap_search = pre_wrap_shard_candidate()

        _tp = {}
        _phase_t0 = _ptime.perf_counter()
        quantized_layer_names, unquantized_layer_names = self.wrapper_block(
            block,
            self.enable_minmax_tuning,
            self.enable_norm_bias_tuning,
            enable_torch_compile=self.compress_context.enable_torch_compile,
            device=device,
            defer_init_search=_defer_wrap_search,
        )
        _tp["wrap"] = _ptime.perf_counter() - _phase_t0
        _phase_t0 = _ptime.perf_counter()  # prepare: engagement + params + optimizer

        # shared engagement resolver (also used by the composer's collection
        # pass). Resolve fresh here: this call runs POST-wrap, so the mirror
        # pricing sees the wrapper's fp32 value params and the free-VRAM
        # snapshot is current for THIS block (the composer's pre-wrap
        # resolution is best-effort for collection sharding only).
        from auto_round.algorithms.quantization.sign_round.data_parallel import resolve_tune_ddp_plan_

        self._resolved_ddp_plan = None  # drop the composer's pre-wrap plan
        _plan = resolve_tune_ddp_plan_(self, block, active_inputs, fp_outputs, device)
        if _plan.enabled and not isinstance(fp_outputs, list):
            # the composer may have resolved before reference outputs existed;
            # non-list outputs (diffusion-style) cannot be pool-distributed
            logger.info("[tune-ddp] declining: reference outputs are not a list")
            _plan = type(_plan)(1, _plan.devices[:1], _plan.shard_size)
            self._resolved_ddp_plan = _plan
        _dp_eligible = _plan.enabled
        if _dp_eligible:
            # All-float pinned blocks (a 'bits':16 'data_type':'float' layer_config
            # pin) carry no tuning parameters: the serial path's empty-params
            # guard no-ops them, so the parallel lane would only pay mirror
            # setup + pool distribution for nothing (and the serial early-return
            # never tears the group down). Decline before mirror setup.
            from auto_round.algorithms.quantization.sign_round.data_parallel import block_has_tuning_entries

            if not block_has_tuning_entries(block):
                logger.info("[tune-ddp] declining: no tuning parameters in this block (all-float pinned); serial path")
                _dp_eligible = False
        replica_group = None
        if _dp_eligible:
            from auto_round.algorithms.quantization.sign_round.data_parallel import (
                ReplicaGroup,
                distribute_pool,
                gather_block_for_mirroring_,
            )

            # the source block must sit whole on the home device before
            # mirroring (data-driven multi-GPU may have sharded its leaves)
            gather_block_for_mirroring_(block, _plan.devices[0])
            # distributed calibration pool: shard-local tune reads; each
            # device owns a contiguous 1/world slice of the samples
            distribute_pool(active_inputs, _plan.devices)
            distribute_pool(fp_outputs, _plan.devices)
            _dp_placed = True
            import time as _ptime

            _t0 = _ptime.perf_counter()
            replica_group = ReplicaGroup(block, _plan)
            _ddp_perf = {
                "build": _ptime.perf_counter() - _t0,
                "warm": 0.0,
                "fwd": [],
                "bwd": [],
                "exch": [],
                "step": [],
            }
            for note in _plan.notes:
                logger.info("[tune-ddp] %s", note)

            # mirrors-first: the deferred wrap-time searches run round-robin on
            # the replicas now that the mirrors exist (before any forward --
            # the warm-up and the tune loop both need init_scale present)
            from auto_round.algorithms.quantization.sign_round.data_parallel import (
                run_deferred_wrap_searches,
            )

            run_deferred_wrap_searches(block, replica_group)
        elif _defer_wrap_search:
            # the resolver declined the parallel lane: fill the deferred
            # searches serially on the home device (never enter tuning with
            # init_scale unset)
            from auto_round.algorithms.quantization.sign_round.data_parallel import (
                run_deferred_wrap_searches,
            )

            run_deferred_wrap_searches(block, None)

        def _collect_tuning_params(mod):
            """Collect (round, minmax) params + per-lr groups from a wrapped block."""
            r_params, m_params = [], []
            r_groups: dict[float, list] = {}
            m_groups: dict[float, list] = {}
            for _n, m in mod.named_modules():
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
                        if "min" in key or "max" in key:
                            m_params.append(m.params[key])
                            m_groups.setdefault(float(layer_minmax_lr), []).append(m.params[key])
                        else:
                            r_params.append(m.params[key])
                            r_groups.setdefault(float(layer_lr), []).append(m.params[key])
            return r_params, m_params, r_groups, m_groups

        round_params, minmax_params, round_lr_groups, minmax_lr_groups = _collect_tuning_params(block)

        lr = torch.tensor(self.lr)
        minmax_lr = torch.tensor(self.minmax_lr)

        extra_kwargs = {} if self.momentum is None else {"momentum": self.momentum}

        if len(round_params) + len(minmax_params) <= 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block"
            )
            logger.info(dump_info)
            unwrapper_block(block, {})
            return {}

        # Build optimizer param groups with a per-layer lr for the rounding
        # parameters (and min-max parameters when enabled).
        params = [{"params": ps, "lr": torch.tensor(group_lr)} for group_lr, ps in round_lr_groups.items()]
        if self.enable_minmax_tuning:
            params += [{"params": ps, "lr": torch.tensor(group_lr)} for group_lr, ps in minmax_lr_groups.items()]

        optimizer = self.optimizer(
            params,
            lr=lr,
            weight_decay=0,
            **extra_kwargs,
        )

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)

        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        num_elm = 1
        mse_reduction = "mean"
        if self.gradient_accumulate_steps != 1:
            mse_reduction = "sum"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)
        scaler = self._get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        best_params = {}
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

        block, sync_gradients = setup_ddp_if_needed_(self, block, device_manager.device_list)
        index_sampler = IndexSampler(nsamples, global_batch_size)
        block_fwd = self.block_forward

        # When low_gpu_mem_usage is enabled, active_inputs / fp_outputs are intentionally
        # kept on CPU to limit GPU memory.  However block_fwd normally routes pred_output
        # through CPU (cache_device="cpu") and the very next line moves it back to
        # loss_device — a wasteful GPU→CPU→GPU roundtrip on every batch × iteration.
        # pred_output is a transient single-batch tensor consumed immediately for the
        # loss and then freed, so keeping it on the compute device costs no persistent
        # extra memory.  Pass it as a per-call override so self.cache_device is unchanged.
        _fwd_cache_device = (
            device
            if getattr(self.compress_context, "low_gpu_mem_usage", False) and not str(device).startswith("cpu")
            else None
        )

        # ── Optional single-process data parallelism (AR_TUNE_DDP_WORLD) ─────
        # (eligibility, plan, mirrors and pool distribution ran before
        # parameter collection; mirror optimizers + warm-up remain here)
        _dp_samplers = None
        if replica_group is not None and _dp_placed and global_batch_size % replica_group.world == 0:
            _dp_samplers = shard_samplers(nsamples, replica_group.world, global_batch_size // replica_group.world)
        if _dp_eligible and replica_group is not None:
            # mirror-side optimizers replicate the home group structure
            for mirror in replica_group.mirrors:
                r_ps, m_ps, r_gr, m_gr = _collect_tuning_params(mirror)
                m_params = [{"params": ps, "lr": torch.tensor(g_lr)} for g_lr, ps in r_gr.items()]
                if self.enable_minmax_tuning:
                    m_params += [{"params": ps, "lr": torch.tensor(g_lr)} for g_lr, ps in m_gr.items()]
                m_opt = self.optimizer(m_params, lr=lr, weight_decay=0, **extra_kwargs)
                m_sched = (
                    torch.optim.lr_scheduler.LinearLR(m_opt, start_factor=1.0, end_factor=0.0, total_iters=self.iters)
                    if self.lr_scheduler is None
                    else copy.deepcopy(self.lr_scheduler)
                )
                mirror_optimizers.append(m_opt)
                mirror_schedules.append(m_sched)
            params_per_replica = [
                r_ps + m_ps if self.enable_minmax_tuning else r_ps
                for r_ps, m_ps, _r, _m in (_collect_tuning_params(rep) for rep in replica_group.replicas)
            ]
            # Warm up every replica SERIALLY in the main thread: torch.compile
            # materializes its per-device kernels lazily at first call, and
            # compiling from several worker threads at once races in dynamo.
            # The warm-up also validates each mirror end-to-end; a failure
            # ABORTS the run (DDP was explicitly requested -- continuing
            # serially would silently invalidate the configuration and any
            # measurement against it). Grads are discarded.
            try:
                _t0 = _ptime.perf_counter()
                _pool_shard = max(1, nsamples // replica_group.world)
                for r, rep in enumerate(replica_group.replicas):
                    # warm only one batch's worth (shard_size samples) at the
                    # head of this replica's pool shard -- warming the whole
                    # shard would materialize its full activations on the mirror
                    _warm = list(range(r * _pool_shard, r * _pool_shard + _plan.shard_size))
                    _dev_r = next(rep.parameters()).device
                    with torch.cuda.device(_dev_r):
                        expect_pool_local([fp_outputs[j] for j in _warm], _dev_r, "warmup-ref")
                        _ref_w = torch.cat([fp_outputs[j].to(_dev_r) for j in _warm], dim=0)
                        _pred_w = block_fwd.forward(rep, active_inputs, input_others, _warm, _dev_r)
                        _loss_w = self._get_loss(_pred_w, _ref_w, _warm, mse_loss, _dev_r, valid_token_mask)
                        _loss_w.backward()
                _ddp_perf["warm"] = _ptime.perf_counter() - _t0
                for _opt in [optimizer] + mirror_optimizers:
                    _opt.zero_grad()
            except Exception as _warm_err:  # noqa: BLE001 - re-raised with context
                try:  # fail-visible: tensor-device census + hook inventory per replica
                    for _ri, _rep in enumerate(replica_group.replicas):
                        _census: dict = {}
                        for _on, _pp in _rep.named_parameters():
                            _census[str(_pp.device)] = _census.get(str(_pp.device), 0) + 1
                        # hooks survive deepcopy/replicate and are invisible to
                        # parameter walks; a hook with a baked-in device (e.g. an
                        # AlignDevicesHook pointing at the primary) moves inputs or
                        # reloads weights onto the WRONG GPU mid-forward
                        _hooks: dict = {}
                        for _mn, _mod in _rep.named_modules():
                            for _fn in list(getattr(_mod, "_forward_pre_hooks", {}).values()) + list(
                                getattr(_mod, "_forward_hooks", {}).values()
                            ):
                                _t = type(_fn).__name__
                                _hooks[_t] = _hooks.get(_t, 0) + 1
                        logger.error(
                            "[tune-ddp] warm-up failed; replica[%d] param device census: %s; hooks: %s",
                            _ri,
                            _census,
                            _hooks or "none",
                        )
                        if _ri == 0:  # fingerprint once: kwargs types/shapes + pool geometry
                            try:

                                def _fp(v):
                                    if torch.is_tensor(v):
                                        return tuple(v.shape)
                                    if isinstance(v, tuple):
                                        return "(" + ", ".join(_fp(t) for t in v) + ")"
                                    if isinstance(v, (list,)):
                                        return f"list[{len(v)}]"
                                    return type(v).__name__

                                _pool = len(active_inputs) if isinstance(active_inputs, list) else "dict"
                                logger.error(
                                    "[tune-ddp] kwargs fingerprint (pool n=%s, batch=%s): %s",
                                    _pool,
                                    getattr(self.calibration_context, "batch_size", "?"),
                                    {k: _fp(v) for k, v in input_others.items()},
                                )
                            except Exception:  # pragma: no cover - diagnostics must not mask
                                pass
                except Exception:  # pragma: no cover - diagnostics must not mask
                    pass
                replica_group.teardown()
                raise RuntimeError(
                    "parallel tuning was requested but the replica warm-up failed -- "
                    "refusing to continue on the serial path (that would silently "
                    "invalidate the requested configuration and any measurement "
                    "against it). Fix the underlying failure or run with "
                    "--parallel_quantization off."
                ) from _warm_err

        _tp["prepare"] = _ptime.perf_counter() - _phase_t0
        _phase_t0 = _ptime.perf_counter()  # loop
        tuning_cache = None
        # Only opt-in diffusion tuning can enter the CUDA staging path (the
        # data-parallel lane never coexists with it: engagement requires a
        # multi-device plan, the cache requires a single device).
        cache_budget = getattr(self.model_context, "diffusion_tuning_cache_size", 0)
        use_tuning_cache = (
            getattr(self.model_context, "is_diffusion", False)
            and (cache_budget == "auto" or cache_budget > 0)
            and self.compress_context.low_gpu_mem_usage
            and str(device).startswith("cuda")
            and len(device_manager.device_list) == 1
            and (loss_device is None or torch.device(loss_device) == torch.device(device))
        )

        try:
            for i in range(self.iters):
                # Auto observes a complete forward/backward/optimizer iteration
                # on the legacy path before allocating any extra GPU buffers.
                if use_tuning_cache and i == (1 if cache_budget == "auto" else 0):
                    from auto_round.compressors.diffusion.tuning_cache import DiffusionTuningCache

                    tuning_cache = DiffusionTuningCache.create(
                        block,
                        block_fwd,
                        active_inputs,
                        input_others,
                        fp_outputs,
                        index_sampler,
                        self.iters - i,
                        cache_budget,
                        device,
                    )
                if self.enable_alg_ext and self.scheme.data_type.endswith("dq"):
                    for n, m in block.named_modules():
                        m.cur_iter = i
                total_loss = 0

                if replica_group is not None:
                    _world = replica_group.world
                    if _dp_samplers is not None:
                        _shards = [s_.next_batch() for s_ in _dp_samplers]
                        global_indices = [j for sh in _shards for j in sh]
                    else:
                        global_indices = index_sampler.next_batch()
                        _shard = len(global_indices) // _world
                        _shards = [global_indices[r * _shard : (r + 1) * _shard] for r in range(_world)]
                    if valid_token_mask is not None:
                        # same global normalization as the serial path (reporting only)
                        num_elm = self._get_non_zero_cnt(valid_token_mask, global_indices)
                    _losses = [None] * _world

                    _fwd_walls = [0.0] * _world
                    _bwd_walls = [0.0] * _world

                    def _dp_replica_step(r, shard):
                        rep = replica_group.replicas[r]
                        dev_r = next(rep.parameters()).device
                        with torch.cuda.device(dev_r):
                            expect_pool_local([fp_outputs[j] for j in shard], dev_r, "ddp-ref")
                            ref_r = torch.cat([fp_outputs[j].to(dev_r) for j in shard], dim=0)
                            # always device-local: parking a mirror's output on the
                            # primary GPU would both mismatch the loss and cost a
                            # cross-device copy every iteration
                            _t0 = _ptime.perf_counter()
                            pred_r = block_fwd.forward(rep, active_inputs, input_others, shard, dev_r)
                            # the masked loss divides by the shard's ELEMENT count
                            # (reduction="mean"), which is identical across the
                            # equal shards -- so the exchange's mean-of-shard-means
                            # reproduces the serial masked loss exactly
                            loss_r = self._get_loss(pred_r, ref_r, shard, mse_loss, dev_r, valid_token_mask)
                            _fwd_walls[r] = _ptime.perf_counter() - _t0
                            _losses[r] = loss_r.detach()
                            _t0 = _ptime.perf_counter()
                            loss_r.backward()
                            # NB: backward() returns after ENQUEUE; the device
                            # completion is forced by the loss .item() sum and the
                            # exchange's grad reads, so a tail of bwd GPU time
                            # surfaces in the exch wall below
                            _bwd_walls[r] = _ptime.perf_counter() - _t0

                    replica_group.run_threaded([lambda r=r: _dp_replica_step(r, _shards[r]) for r in range(_world)])
                    _ddp_perf["fwd"].append(max(_fwd_walls))
                    _ddp_perf["bwd"].append(max(_bwd_walls))
                    # sync_grads: cross-replica gradient exchange. sign_exchange:
                    # the update consumes only sign(mean-grad) and weight_decay is
                    # 0, so exchanging int8 signs after the reduce is
                    # bitwise-identical across replicas -- but a momentum buffer
                    # would mix magnitudes back in, so gate on it.
                    _t0 = _ptime.perf_counter()
                    replica_group.sync_grads(
                        params_per_replica,
                        sign_exchange=self.momentum is None or float(self.momentum) == 0.0,
                    )
                    _ddp_perf["exch"].append(_ptime.perf_counter() - _t0)
                    # report the global-batch mean (mean of equal-size shard
                    # means == the serial global mean), normalized by the
                    # valid-element count exactly like the serial path so
                    # best-iter selection and dynamic_max_gap behave identically
                    _ne = 1 if num_elm <= 0 else num_elm
                    total_loss = sum(l.item() for l in _losses if l is not None) / _world / _ne

                else:
                    global_indices = index_sampler.next_batch()
                    if valid_token_mask:
                        num_elm = self._get_non_zero_cnt(valid_token_mask, global_indices)
                    for batch_start in range(0, len(global_indices), batch_size):
                        indices = global_indices[batch_start : batch_start + batch_size]
                        staged = tuning_cache.get(indices) if tuning_cache is not None else None
                        if staged is None:
                            ref_output = _cat_device_safe([fp_outputs[i] for i in indices], dim=0).to(loss_device)
                            pred_output = block_fwd.forward(
                                block, active_inputs, input_others, indices, _fwd_cache_device
                            )
                        else:
                            ref_output = staged[2]
                            pred_output = tuning_cache.forward(block, staged, _fwd_cache_device)
                        if loss_device is not None:
                            pred_output = pred_output.to(loss_device)
                        if (
                            block_ctx.block_index == block_ctx.block_cnt - 1
                            and self.enable_lfq
                            and input_ids is not None
                            and self._is_text_decoder_block(block_ctx.block_name)
                        ):
                            loss = self.lfq_loss(pred_output, torch.cat([input_ids[i] for i in indices], dim=0))
                        else:
                            loss = self._get_loss(pred_output, ref_output, indices, mse_loss, device, valid_token_mask)
                        num_elm = 1 if num_elm <= 0 else num_elm
                        total_loss += loss.item() / num_elm

                        if mid_iter_mem_check:
                            # clear memory to avoid OOM due to memory fragmentation
                            clear_memory_if_reached_threshold(threshold=0.5, device_list=device_manager.device_list)

                        self._scale_loss_and_backward(scaler, loss)

                        if mid_iter_mem_check:
                            # clear memory to avoid OOM due to memory fragmentation
                            clear_memory_if_reached_threshold(threshold=0.8, device_list=device_manager.device_list)

                if i == 0:
                    init_loss = total_loss
                current_lr = optimizer.param_groups[0]["lr"]
                logger.debug("iter %d loss: %.3e lr: %s", i, total_loss, current_lr)

                if total_loss < best_loss:
                    best_loss = total_loss
                    if not self.not_use_best_mse:
                        best_params = (
                            tuning_cache.collect_best_params()
                            if tuning_cache is not None and tuning_cache.best is not None
                            else collect_best_params(block, self.compress_context.cache_device)
                        )
                        last_best_iter = i
                if self.not_use_best_mse and i == self.iters - 1:
                    best_params = (
                        tuning_cache.collect_best_params()
                        if tuning_cache is not None and tuning_cache.best is not None
                        else collect_best_params(block, self.compress_context.cache_device)
                    )

                if not self.not_use_best_mse:
                    if 0 < self.dynamic_max_gap <= i - last_best_iter:
                        break
                if replica_group is not None:
                    # world-sized callable list (home first) so the persistent
                    # pool serves the step too instead of falling back to
                    # per-iteration thread spawn
                    def _mirror_step(opt, sched):
                        opt.step()
                        opt.zero_grad()
                        sched.step()

                    _t0 = _ptime.perf_counter()
                    replica_group.run_threaded(
                        [lambda: self._step(scaler, optimizer, lr_schedule)]
                        + [
                            lambda oo=opt, ss=sch: _mirror_step(oo, ss)
                            for opt, sch in zip(mirror_optimizers, mirror_schedules)
                        ]
                    )
                    _ddp_perf["step"].append(_ptime.perf_counter() - _t0)
                else:
                    sync_gradients()
                    self._step(scaler, optimizer, lr_schedule)

        finally:
            if tuning_cache is not None:
                tuning_cache.close()
            _tp["loop"] = _ptime.perf_counter() - _phase_t0
            _phase_t0 = _ptime.perf_counter()  # tail
            if replica_group is not None:
                # engagement defines both; serial never enters this branch
                _t0 = _ptime.perf_counter()
                replica_group.teardown()
                _ddp_perf["teardown"] = _ptime.perf_counter() - _t0
                _phase_t0 = _ptime.perf_counter()  # tail resumes after teardown

        if replica_group is not None and getattr(_envs, "AR_PERF_COUNTERS", False):
            _fwd, _exch, _step = _ddp_perf["fwd"], _ddp_perf["exch"], _ddp_perf["step"]

            def _ms(x):
                return f"{1000 * x:.0f}ms"

            _bwd = _ddp_perf["bwd"]
            logger.info(
                "[perf] tune-ddp block (%s): mirrors=%s warmup=%s "
                "fwd=%s/%s bwd=%s/%s exch=%s/%s step=%s/%s teardown=%s (mean/max per iter)",
                type(block).__name__,
                _ms(_ddp_perf["build"]),
                _ms(_ddp_perf["warm"]),
                _ms(sum(_fwd) / max(len(_fwd), 1)),
                _ms(max(_fwd)) if _fwd else "n/a",
                _ms(sum(_bwd) / max(len(_bwd), 1)),
                _ms(max(_bwd)) if _bwd else "n/a",
                _ms(sum(_exch) / max(len(_exch), 1)),
                _ms(max(_exch)) if _exch else "n/a",
                _ms(sum(_step) / max(len(_step), 1)),
                _ms(max(_step)) if _step else "n/a",
                _ms(_ddp_perf.get("teardown", 0.0)),
            )

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        if self.iters > 0:
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
        with torch.no_grad():
            unwrapper_block(block, best_params)

        if self.config.is_act_nv_fp:
            # enable moe experts act_max automatic generation for WrapperWALayer
            set_amax_for_all_moe_layers(block, attr_name="orig_layer.act_max")

        logger.infoclean(dump_info)
        _tp["tail"] = _ptime.perf_counter() - _phase_t0
        if getattr(_envs, "AR_PERF_COUNTERS", False):
            logger.info(_tune_phase_line(_tp, self.iters))
        return best_params

    def quantize_layer_outside_block(
        self,
        layer: "torch.nn.Module",
        fp_inputs: Optional[list[torch.Tensor]] = None,
        q_inputs: Optional[list[torch.Tensor]] = None,
        disable_opt_rtn: Optional[bool] = None,
        input_ids: Optional[list[torch.Tensor]] = None,
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

        wrapper_linear = WrapperLinear(
            layer,
            enable_minmax_tuning=self.enable_minmax_tuning,
            enable_torch_compile=self.compress_context.enable_torch_compile,
            device=device,
        ).to(device)
        round_params = []
        minmax_params = []
        for key in wrapper_linear.params.keys():
            if "min" in key or "max" in key:
                minmax_params.append(wrapper_linear.params[key])
            else:
                round_params.append(wrapper_linear.value)
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

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)
        nsamples = len(fp_inputs)
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        best_params = None
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
                with torch.no_grad():
                    current_output = layer(org_input)
                autocast_ctx = (
                    nullcontext()
                    if not self.model_context.amp
                    else autocast(device_type=str(device).split(":")[0], dtype=self.model_context.amp_dtype)
                )
                if valid_token_mask:
                    tmp_valid_mask = [valid_token_mask[i] for i in indices]
                    tmp_valid_mask = torch.cat(tmp_valid_mask, dim=0).to(device)
                    tmp_valid_mask.unsqueeze_(-1)

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
            if i == 0:
                init_loss = total_loss
            current_lr = optimizer.param_groups[0]["lr"]
            logger.debug("iter %d loss: %.3e lr: %s", i, total_loss, current_lr)

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    best_params = collect_best_params(wrapper_linear, self.compress_context.cache_device)
                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_params = collect_best_params(wrapper_linear, self.compress_context.cache_device)

            if not self.not_use_best_mse:
                if 0 < self.dynamic_max_gap <= i - last_best_iter:
                    break
            self._step(scaler, optimizer, lr_schedule)

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
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
        return None

    def _scale_loss_and_backward(self, scaler: Any, loss: torch.Tensor) -> torch.Tensor:
        """Scales the loss and performs backward pass.

        Args:
        scaler: The scaler to be used.
        loss: The loss to be scaled.

        Returns:
        The scaled loss.
        """
        scale_loss = loss * 1000
        scale_loss.backward()
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
        optimizer.zero_grad()
        lr_schedule.step()
