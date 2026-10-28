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
import time as _ptime
from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
from torch import autocast

from auto_round.algorithms.block_runner import BlockForwardRunner
from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.compressors.utils import (
    IndexSampler,
    collect_best_params,
    snapshot_best_params,
)
from auto_round.logger import logger
from auto_round.schemes import is_nvfp4_e5m3_gs
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
    set_amax_for_all_moe_layers,
)
from auto_round.utils.device import clear_memory_if_reached_threshold
from auto_round.utils.device_manager import device_manager
from auto_round.utils.distributed import setup_ddp_if_needed_
from auto_round.wrapper import (
    WrapperLinear,
    store_warm_start_params_and_unwrap,
    unwrapper_block,
    unwrapper_layer,
    wrapper_block,
)

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
        self.lfq_loss_type = getattr(config, "lfq_loss", "ce")
        self.lfq_top_k = getattr(config, "lfq_top_k", 32)
        self.lfq_temperature = getattr(config, "lfq_temperature", 1.0)
        self.lfq_num_blocks = getattr(config, "lfq_num_blocks", 1)
        self.lfq_mse_weight = getattr(config, "lfq_mse_weight", 0.0)
        self.lfq_qad = getattr(config, "lfq_qad", False)
        self.lfq_block_stride = getattr(config, "lfq_block_stride", 1)
        self.lfq_mse_stride = getattr(config, "lfq_mse_stride", 1)
        self.enable_qad = getattr(config, "enable_qad", False)

        self.optimizer = self._get_optimizer(optimizer=config.optimizer)
        self.wrapper_block = wrapper_block
        # Warm-start chaining flag (e.g. SignRound -> QAD): when True, quantize_block
        # keeps the original FP weight and stashes the tuned params on each layer
        # instead of baking them into a quantized weight. Set by the orchestrator
        # for the block-wise warm-up stage that precedes a whole-model quantizer.
        self.warm_start_collect = False
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

    def _maybe_prefetch_teacher(self, indices) -> None:
        """Hook for subclasses (e.g. QAD) to prefetch per-batch teacher data.

        No-op in the base implementation.
        """
        return None

    def _lfq_is_kd_block(self, block_ctx) -> bool:
        """Return ``True`` if the KD (LFQ logit-lens) loss applies to this block.

        * QAD off (default): the last ``lfq_num_blocks`` text-decoder blocks.
        * QAD on: every ``lfq_block_stride`` blocks, counting from the last
          block backward (deep-supervision checkpoints).

        Requires LFQ enabled, token IDs available (checked by the caller) and a
        pure text-decoder block.
        """
        if not self.enable_lfq or not self._is_text_decoder_block(block_ctx.block_name):
            return False
        # QAD: every fused group (chunk) is tuned against the final-output LM loss.
        if self.enable_qad:
            return True
        depth_from_last = block_ctx.block_cnt - 1 - block_ctx.block_index
        if self.lfq_qad:
            return depth_from_last % self.lfq_block_stride == 0
        return block_ctx.block_index >= block_ctx.block_cnt - self.lfq_num_blocks

    def _lfq_is_mse_block(self, block_ctx) -> bool:
        """Return ``True`` if the MSE reconstruction loss applies to this block.

        Only meaningful in QAD mode, where the MSE loss is applied every
        ``lfq_mse_stride`` blocks (from the last block backward). Outside QAD
        mode every block uses MSE (handled directly in the tuning loop).
        """
        depth_from_last = block_ctx.block_cnt - 1 - block_ctx.block_index
        return depth_from_last % self.lfq_mse_stride == 0

    def _get_lfq_downstream_blocks(self, block_ctx, device) -> list:
        """Return the (FP, not-yet-quantized) blocks that follow the current one.

        Only used in the default (non-QAD) LFQ mode with ``lfq_num_blocks`` > 1:
        a non-final KD block's output must go through the remaining decoder
        blocks before the final norm / lm_head, otherwise the loss would be a
        logit-lens on an intermediate hidden state. The returned blocks are
        moved to *device*; the caller is responsible for moving them back.

        Returns an empty list when the current block is the last one, when the
        block order is unknown, or when a downstream block is not materialized
        (e.g. offloaded / meta), in which case it falls back to the logit-lens.
        """
        if self.enable_qad or self.lfq_qad:
            return []
        all_names = getattr(block_ctx, "all_block_names", None)
        cur_names = getattr(block_ctx, "block_names", None) or []
        if not all_names or not cur_names or cur_names[-1] not in all_names:
            if block_ctx.block_index < block_ctx.block_cnt - 1:
                logger.warning_once(
                    "LFQ: block order unavailable, non-final KD blocks fall back to the logit-lens loss "
                    "(block output fed directly to the final norm/lm_head)."
                )
            return []
        downstream_names = all_names[all_names.index(cur_names[-1]) + 1 :]
        if not downstream_names:
            return []

        from auto_round.utils import convert_module_to_hp_if_necessary, get_module

        model = getattr(block_ctx, "model", None) or self.model
        blocks = []
        for name in downstream_names:
            m = get_module(model, name)
            if m is None or any(p.device.type == "meta" or p.numel() == 0 for p in m.parameters()):
                logger.warning_once(
                    "LFQ: downstream block %s is not materialized (offloaded/meta); non-final KD blocks fall "
                    "back to the logit-lens loss. Disable low_cpu_mem_usage/disk streaming to propagate "
                    "through the remaining blocks." % name
                )
                return []
            blocks.append(m)
        for m in blocks:
            convert_module_to_hp_if_necessary(m, self.model_context.amp_dtype, device)
            m.to(device)
        logger.info(
            f"LFQ: propagating {block_ctx.block_name} output through {len(blocks)} downstream FP block(s) "
            "before lm_head"
        )
        return blocks

    def _forward_lfq_downstream(
        self,
        hidden_state: torch.Tensor,
        downstream_blocks: list,
        inputs,
        input_others: dict,
        indices,
        device,
    ) -> torch.Tensor:
        """Forward *hidden_state* through *downstream_blocks* (frozen FP blocks).

        Gradients flow through the activations back to the current block; the
        downstream block parameters themselves are frozen (``requires_grad=False``).
        """
        if not downstream_blocks:
            return hidden_state
        from auto_round.utils import to_device

        block_fwd = self.block_forward
        out_device = hidden_state.device
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        _, batch_others = block_fwd.select_batch(inputs, input_others, indices)
        batch_others = to_device(batch_others, device)
        h = hidden_state.to(device)
        for m in downstream_blocks:
            # block_forward mutates the kwargs dict (pops positional_inputs, injects the
            # hidden-state kwarg), so hand each block a fresh shallow copy.
            raw = block_fwd._forward_one_batch(m, h, dict(batch_others))
            h = block_fwd._normalize_output(raw, m)
        return h.to(out_device)

    def _lfq_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project a block hidden state to vocabulary logits.

        Applies every post-block module (final norm, optional projection, …) in
        order, then runs lm_head.

        Args:
            hidden_state: Block output, shape ``[batch, seq_len, hidden]``.

        Returns:
            Logits tensor of shape ``[batch, seq_len, vocab_size]``.
        """
        self._init_lm_components()
        device = hidden_state.device

        for module in self._post_block_modules:
            module.to(device)
            hidden_state = module(hidden_state)

        self._lm_head.to(device)
        return self._lm_head(hidden_state)

    def lfq_loss(
        self,
        hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
        teacher_hidden_state: torch.Tensor | None = None,
        sample_indices: list | None = None,
    ) -> torch.Tensor:
        """Compute the LFQ loss from the last block's hidden states.

        Two loss types are supported, selected by ``self.lfq_loss_type``:

        * ``"ce"`` (default): next-token cross-entropy against ground-truth
          labels. Positions marked with ``-100`` in *input_ids* are excluded.
        * ``"dtk"``: Decoupled Tempered Top-K forward-KL knowledge distillation
          (``KL(p_T || p_S)``). The FP block output (*teacher_hidden_state*) is
          the teacher and the quantized block output (*hidden_state*) is the
          student. Requires *teacher_hidden_state*; falls back to ``"ce"`` when
          it is ``None``.
        * ``"rkl"``: same decoupled Top-K objective but using the reverse KL
          ``KL(p_S || p_T)`` (mode-seeking). Also requires *teacher_hidden_state*.
        * ``"tk"``: only aligns the Top-K-normalized conditional distributions
          (tempered conditional KL within the teacher Top-K, dropping the binary
          mass term). Also requires *teacher_hidden_state*.

        Args:
            hidden_state: Last (quantized/student) block output,
                shape ``[batch, seq_len, hidden]``.
            input_ids:    Token-ID labels with ``-100`` for ignored positions,
                          shape ``[batch, seq_len]``.
            teacher_hidden_state: FP (teacher) block output used by the ``"dtk"``,
                ``"rkl"`` and ``"tk"`` losses, same shape as *hidden_state*.
            sample_indices: Calibration-sample indices for this batch. Unused by
                the base implementation; QAD subclasses use it to load the
                offline teacher Top-K cache from disk.

        Returns:
            Scalar loss tensor.
        """
        device = hidden_state.device
        logits = self._lfq_logits(hidden_state)

        if self.lfq_loss_type in ("dtk", "rkl", "tk") and teacher_hidden_state is not None:
            with torch.no_grad():
                teacher_logits = self._lfq_logits(teacher_hidden_state.to(device))
            return self._decoupled_topk_kd_loss(
                logits,
                teacher_logits,
                input_ids.to(device),
                reverse=(self.lfq_loss_type == "rkl"),
                conditional_only=(self.lfq_loss_type == "tk"),
            )

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

    def _decoupled_topk_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        reverse: bool = False,
        conditional_only: bool = False,
    ) -> torch.Tensor:
        """Decoupled Tempered Top-K knowledge-distillation loss (per-token mean).

        Decomposes the KL divergence between the teacher and student next-token
        distributions via the chain rule into:

        * ``L_B``: a binary (Bernoulli) term matching the total probability mass
          the teacher/student assign to the teacher's Top-K set, and
        * ``L_T``: the temperature-scaled conditional KL *within* the Top-K set,
          weighted by the expectation distribution's Top-K mass.

        This "decoupled, tempered Top-K" objective avoids support mismatch while
        preserving the benefits of temperature-scaled distillation.

        The Top-K set is always the teacher's. ``reverse`` selects the KL
        direction:

        * ``reverse=False`` (forward KL): the expectation is over the teacher,
          i.e. ``KL(p_T || p_S)`` — mode-covering, the standard distillation
          objective.
        * ``reverse=True`` (reverse KL): the expectation is over the student,
          i.e. ``KL(p_S || p_T)`` — mode-seeking.

        When ``conditional_only`` is ``True`` the binary mass term ``L_B`` and the
        Top-K mass weighting are dropped, so the loss only aligns the two
        distributions *renormalized within the Top-K set* (the tempered
        conditional KL alone).

        Gradients flow only through the student; the teacher is detached.

        Args:
            student_logits: Student logits, shape ``[batch, seq_len, vocab]``.
            teacher_logits: Teacher logits, shape ``[batch, seq_len, vocab]``.
            labels:         Token-ID labels with ``-100`` for ignored positions,
                            shape ``[batch, seq_len]``. Only used to mask out
                            padding / ignored next-token positions.
            reverse:        When ``True`` compute the reverse KL ``KL(p_S || p_T)``
                            instead of the forward KL ``KL(p_T || p_S)``.
            conditional_only: When ``True`` only align the Top-K-normalized
                            conditional distributions (drop ``L_B`` and the mass
                            weighting).

        Returns:
            Scalar loss tensor (mean over valid next-token positions).
        """
        import torch.nn.functional as F

        top_k = int(self.lfq_top_k)
        tau = float(self.lfq_temperature)
        eps = 1e-6

        # Causal-LM shift: position t predicts token t+1.
        s = student_logits[..., :-1, :].contiguous()
        t = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab = s.size(-1)
        s = s.reshape(-1, vocab).float()
        t = t.reshape(-1, vocab).float()
        mask = (shift_labels.reshape(-1) != -100)

        if mask.sum() == 0:
            # Keep the graph connected so backward() is a no-op but valid.
            return student_logits.float().sum() * 0.0

        s = s[mask]
        t = t[mask]
        top_k = min(top_k, vocab)

        log_pS = F.log_softmax(s, dim=-1)
        log_pT = F.log_softmax(t, dim=-1)

        # Teacher Top-K set per token (used for both forward and reverse KL).
        topk_idx = t.topk(top_k, dim=-1).indices  # [N, k]
        log_pS_topk = log_pS.gather(-1, topk_idx)  # [N, k]
        log_pT_topk = log_pT.gather(-1, topk_idx)  # [N, k]

        # Total mass on the Top-K set (in log space for stability).
        log_PT_T = torch.logsumexp(log_pT_topk, dim=-1)  # [N]
        log_PS_T = torch.logsumexp(log_pS_topk, dim=-1)  # [N]

        # Select the expectation (E) and the "other" (O) distribution per KL
        # direction. Forward KL(p_T || p_S): E=teacher, O=student.
        # Reverse KL(p_S || p_T): E=student, O=teacher.
        if reverse:
            log_PE_T, log_PO_T = log_PS_T, log_PT_T
            log_pE_topk, log_pO_topk = log_pS_topk, log_pT_topk
        else:
            log_PE_T, log_PO_T = log_PT_T, log_PS_T
            log_pE_topk, log_pO_topk = log_pT_topk, log_pS_topk

        PE_T = log_PE_T.exp().clamp(min=eps, max=1.0 - eps)

        # Conditional distributions within the Top-K: log p(x|T) = log p(x) - log P(T).
        log_cE = log_pE_topk - log_PE_T.unsqueeze(-1)  # [N, k]
        log_cO = log_pO_topk - log_PO_T.unsqueeze(-1)  # [N, k]

        # Apply temperature tau: p^(tau)(x) = p(x)^(1/tau) / Z, computed in log space.
        log_cE_tau = log_cE / tau
        log_cE_tau = log_cE_tau - torch.logsumexp(log_cE_tau, dim=-1, keepdim=True)
        log_cO_tau = log_cO / tau
        log_cO_tau = log_cO_tau - torch.logsumexp(log_cO_tau, dim=-1, keepdim=True)

        # Tempered KL: D^(tau)_KL = tau^2 * KL( pE^(tau) || pO^(tau) ).
        kl_tau = (log_cE_tau.exp() * (log_cE_tau - log_cO_tau)).sum(dim=-1)  # [N]

        if conditional_only:
            # Only align the Top-K-normalized conditional distributions; drop the
            # binary mass term L_B and the Top-K mass weighting.
            return ((tau * tau) * kl_tau).mean()

        # L_B: KL( Bern(P_E(T)) || Bern(P_O(T)) ).
        log_one_minus_PE = torch.log1p(-PE_T)
        log_one_minus_PO = torch.log1p(-log_PO_T.exp().clamp(min=eps, max=1.0 - eps))
        l_b = PE_T * (log_PE_T - log_PO_T) + (1.0 - PE_T) * (log_one_minus_PE - log_one_minus_PO)

        l_t = PE_T * (tau * tau) * kl_tau

        return (l_b + l_t).mean()

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
                    if "min" in key or "max" in key:
                        minmax_params.append(m.params[key])
                        minmax_lr_groups.setdefault(float(layer_minmax_lr), []).append(m.params[key])
                    else:
                        round_params.append(m.params[key])
                        round_lr_groups.setdefault(float(layer_lr), []).append(m.params[key])

        lr = torch.tensor(self.lr)
        minmax_lr = torch.tensor(self.minmax_lr)

        extra_kwargs = {} if self.momentum is None else {"momentum": self.momentum}

        if len(round_params) + len(minmax_params) <= 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block"
            )
            logger.info(dump_info)
            if self.warm_start_collect:
                store_warm_start_params_and_unwrap(block, {})
            else:
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

        tuning_cache = None
        # Only opt-in diffusion tuning can enter the CUDA staging path.
        cache_budget = getattr(self.model_context, "diffusion_tuning_cache_size", 0)
        use_tuning_cache = (
            getattr(self.model_context, "is_diffusion", False)
            and (cache_budget == "auto" or cache_budget > 0)
            and self.compress_context.low_gpu_mem_usage
            and str(device).startswith("cuda")
            and len(device_manager.device_list) == 1
            and (loss_device is None or torch.device(loss_device) == torch.device(device))
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
        _entry_dev = str(getattr(block_fwd, "device", device)) if block_fwd is not None else str(device)
        if isinstance(active_inputs, list):
            active_inputs = self.pull_tuning_pool(
                active_inputs,
                _entry_dev,
                block,
                batch_size,
                "tune] block input activations",
                charge_activation=True,
            )
        # the diffusion tuning cache requires its pools on the host (it
        # stages batches to the GPU itself via pinned slots); a bulk pull
        # onto the loss device would put the cached reference outputs on
        # cuda and the cache would decline every batch signature
        if fp_outputs and loss_device is not None and not use_tuning_cache:
            fp_outputs = self.pull_tuning_pool(
                fp_outputs,
                loss_device,
                block,
                batch_size,
                "tune] fp reference outputs",
                charge_activation=False,
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
                _perf.start("sampler")
                global_indices = index_sampler.next_batch()
                # Hook: let subclasses (e.g. QAD) prefetch this iteration's teacher
                # data from disk in the background while the forward/loss runs.
                self._maybe_prefetch_teacher(global_indices)
                if valid_token_mask:
                    num_elm = self._get_non_zero_cnt(valid_token_mask, global_indices)
                _perf.stop("sampler")

                # Per-block loss schedule (constant across batches/iters):
                #   lfq_kd_block  -> apply the KD logit-lens loss on this block
                #   lfq_mse_block -> add the MSE reconstruction loss on this block
                # QAD: every fused group is tuned against the final-output LM loss only
                # (no MSE); with the whole network fused this is the true final logits.
                # Non-QAD: KD on the last `lfq_num_blocks`, MSE elsewhere (and on KD
                # blocks only when `lfq_mse_weight` > 0). lfq_qad: KD every
                # `lfq_block_stride` blocks, MSE every `lfq_mse_stride` blocks, with an
                # MSE fallback so no block is ever left untrained.
                lfq_kd_block = input_ids is not None and self._lfq_is_kd_block(block_ctx)
                if self.enable_qad:
                    lfq_mse_block = False
                elif self.lfq_qad:
                    lfq_mse_block = (not lfq_kd_block) or self._lfq_is_mse_block(block_ctx)
                else:
                    lfq_mse_block = (not lfq_kd_block) or self.lfq_mse_weight > 0

                for batch_start in range(0, len(global_indices), batch_size):
                    indices = global_indices[batch_start : batch_start + batch_size]
                    _perf.start("s_fwd")
                    staged = tuning_cache.get(indices) if tuning_cache is not None else None
                    if staged is None:
                        # fp_outputs may be sharded across park devices
                        # (--calibration_data_device): gather onto one device
                        # before cat. Uniform-device selections are returned
                        # untouched (byte-identical to the single-device path);
                        # tensors are immutable, so concurrent gather is safe.
                        _sel = [fp_outputs[i] for i in indices]
                        _gather = BlockForwardRunner._gather_same_device(
                            _sel, str(loss_device) if loss_device is not None else str(_sel[0].device)
                        )
                        ref_output = torch.cat(_gather, dim=0).to(loss_device)
                        pred_output = block_fwd.forward(block, active_inputs, input_others, indices, _fwd_cache_device)
                    else:
                        ref_output = staged[2]
                        pred_output = tuning_cache.forward(block, staged, _fwd_cache_device)
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

                if i == 0:
                    init_loss = total_loss
                current_lr = optimizer.param_groups[0]["lr"]
                logger.debug("iter %d loss: %.3e lr: %s", i, total_loss, current_lr)

                _perf.start("snap")
                if total_loss < best_loss:
                    best_loss = total_loss
                    if not self.not_use_best_mse:
                        best_params = (
                            tuning_cache.collect_best_params()
                            if tuning_cache is not None and tuning_cache.best is not None
                            else snapshot_best_params(block, self.compress_context.cache_device)
                        )
                        last_best_iter = i
                if self.not_use_best_mse and i == self.iters - 1:
                    best_params = (
                        tuning_cache.collect_best_params()
                        if tuning_cache is not None and tuning_cache.best is not None
                        else snapshot_best_params(block, self.compress_context.cache_device)
                    )

                _perf.stop("snap")
                _perf.start("step")
                if not self.not_use_best_mse:
                    if 0 < self.dynamic_max_gap <= i - last_best_iter:
                        break
                sync_gradients()
                self._step(scaler, optimizer, lr_schedule)
                _perf.stop("step")

        finally:
            if tuning_cache is not None:
                tuning_cache.close()
            for m in lfq_downstream_blocks:
                mv_module_from_gpu(m)
            del lfq_teacher_final

        _perf.stop("loop")
        _perf.start("tail")

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
            if self.warm_start_collect:
                # Warm-up stage of a chain (e.g. SignRound -> QAD): keep the FP
                # weight and stash tuned params for the next pass to warm-start from.
                store_warm_start_params_and_unwrap(block, best_params)
            else:
                unwrapper_block(block, best_params)

        _perf.stop("tail")
        _log_tune_phases(_perf, self.iters)

        if self.config.is_act_nv_fp or is_nvfp4_e5m3_gs(self.config.act_data_type or ""):
            # enable moe experts act_max automatic generation for WrapperWALayer
            set_amax_for_all_moe_layers(block, attr_name="orig_layer.act_max")

        logger.infoclean(dump_info)
        return best_params

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
            enable_neuqi=getattr(self.config, "enable_neuqi", False),
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
        optimizer.zero_grad()
        lr_schedule.step()
