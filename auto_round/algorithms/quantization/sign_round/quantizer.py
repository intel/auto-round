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
import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from torch import autocast

from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.compressors.utils import (
    BestParamsSlot,
    IndexSampler,
    collect_best_params,
    snapshot_best_params,
)
from auto_round.logger import logger
from auto_round.utils import (
    htcore,
    is_hpex_available,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
)
from auto_round.utils.device import clear_memory_if_reached_threshold, log_cuda_memory_census
from auto_round.utils.device_manager import device_manager
from auto_round.utils.distributed import setup_ddp_if_needed_
from auto_round.wrapper import WrapperLinear, unwrapper_block, unwrapper_layer, wrapper_block

if TYPE_CHECKING:
    from auto_round.algorithms.composer import BlockContext


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
        # inter-iteration consolidation probes live pressure per device (a
        # ratio check, us-cheap): low-usage tunes never trip it, tight tunes
        # (huge out-of-block layers) consolidate fragmentation
        # every iteration instead of accumulating holes until a mid-backward OOM

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

        quantized_layer_names, unquantized_layer_names = self.wrapper_block(
            block,
            self.enable_minmax_tuning,
            self.enable_norm_bias_tuning,
            enable_torch_compile=self.compress_context.enable_torch_compile,
            device=device,
            weight_qdq_builder=self.build_weight_qdq,
        )

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
                total_loss = 0
                global_indices = index_sampler.next_batch()
                if valid_token_mask:
                    num_elm = self._get_non_zero_cnt(valid_token_mask, global_indices)

                for batch_start in range(0, len(global_indices), batch_size):
                    indices = global_indices[batch_start : batch_start + batch_size]
                    staged = tuning_cache.get(indices) if tuning_cache is not None else None

                    def _compute_loss():
                        if staged is None:
                            ref_output_l = torch.cat([fp_outputs[i] for i in indices], dim=0).to(loss_device)
                            pred_output_l = block_fwd.forward(
                                block, active_inputs, input_others, indices, _fwd_cache_device
                            )
                        else:
                            ref_output_l = staged[2]
                            pred_output_l = tuning_cache.forward(block, staged, _fwd_cache_device)
                        if loss_device is not None:
                            pred_output_l = pred_output_l.to(loss_device)
                        if (
                            block_ctx.block_index == block_ctx.block_cnt - 1
                            and self.enable_lfq
                            and input_ids is not None
                            and self._is_text_decoder_block(block_ctx.block_name)
                        ):
                            return self.lfq_loss(pred_output_l, torch.cat([input_ids[i] for i in indices], dim=0))
                        return self._get_loss(pred_output_l, ref_output_l, indices, mse_loss, device, valid_token_mask)

                    loss = _compute_loss()
                    # clear memory to avoid OOM due to memory fragmentation
                    clear_memory_if_reached_threshold(threshold=0.5, device_list=device_manager.device_list)
                    self._scale_loss_and_backward(scaler, loss)
                    num_elm = 1 if num_elm <= 0 else num_elm
                    total_loss += loss.item() / num_elm

                    # clear memory to avoid OOM due to memory fragmentation
                    clear_memory_if_reached_threshold(threshold=0.8, device_list=device_manager.device_list)

                if i == 0:
                    init_loss = total_loss
                current_lr = optimizer.param_groups[0]["lr"]
                logger.debug("iter %d loss: %.3e lr: %s", i, total_loss, current_lr)

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
                        best_params = (
                            tuning_cache.collect_best_params()
                            if tuning_cache is not None and tuning_cache.best is not None
                            else snapshot_best_params(block, self.compress_context.cache_device)
                        )
                        last_best_iter = i
                if self.not_use_best_mse and i == self.iters - 1:
                    best_params = None
                    best_params = (
                        tuning_cache.collect_best_params()
                        if tuning_cache is not None and tuning_cache.best is not None
                        else snapshot_best_params(block, self.compress_context.cache_device)
                    )

                if not self.not_use_best_mse:
                    if 0 < self.dynamic_max_gap <= i - last_best_iter:
                        break
                sync_gradients()
                self._step(scaler, optimizer, lr_schedule)

        finally:
            if tuning_cache is not None:
                tuning_cache.close()

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
        log_cuda_memory_census(f"outside-block tune start {layer_name}", device, walk=False)
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
            weight_qdq_builder=self.build_weight_qdq,
        ).to(device)
        # header only: the tensor-list walk runs in the lane's failure handler
        log_cuda_memory_census(
            f"outside-block wrapper ready {layer_name} (compile={compile_wrapper})", device, walk=False
        )
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
                log_cuda_memory_census(f"outside-block first backward done {layer_name}", device, walk=False)
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
        # keep the buffers: reallocating a dense gradient as large as the
        # rounding parameter would break the memory budget of huge layers
        # (outside-block lm_head tuning) in every iteration after the first
        optimizer.zero_grad(set_to_none=False)
        lr_schedule.step()
