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

from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.compressors.utils import (
    IndexSampler,
    collect_best_params,
)
from auto_round.logger import logger
from auto_round.utils import (
    get_module,
    htcore,
    is_hpex_available,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
    set_module,
)
from auto_round.utils.device import clear_memory_if_reached_threshold
from auto_round.utils.device_manager import device_manager
from auto_round.utils.distributed import setup_ddp_if_needed_
from auto_round.wrapper import WrapperLinear, unwrapper_block, unwrapper_layer, wrapper_block

if TYPE_CHECKING:
    from auto_round.algorithms.pipeline import BlockContext


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

        self.optimizer = self._get_optimizer(optimizer=config.optimizer)
        self.wrapper_block = wrapper_block

    def dispatch_block(self, block, input_ids, input_others):
        """Multi-GPU aware block dispatch for SignRound tuning.

        Stores card_0_in_high_risk and loss_device on self for use in quantize_block.
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
                self._calibration_state.batch_size,
                device_manager.device,
            )
            if len(device_manager.device_list) > 1:
                from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                for _n, _mod in block.named_modules():
                    if len(list(_mod.children())) != 0 or not hasattr(_mod, "tuning_device"):
                        continue
                    add_hook_to_module(_mod, AlignDevicesHook(_mod.tuning_device, io_same_device=True), True)
        else:
            block = block.to(device_manager.device)
            card_0_in_high_risk, loss_device = False, device_manager.device

        self._card_0_in_high_risk = card_0_in_high_risk
        self._loss_device = loss_device
        return block, card_0_in_high_risk, loss_device

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

    def quantize_block(
        self,
        block,
        fp_inputs,
        input_others,
        fp_outputs,
        q_inputs,
        block_ctx,
        valid_token_mask=None,
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
            valid_token_mask: Per-sample boolean/int masks of shape
                ``[1, seq_len]`` indicating valid (non-padding) token positions.
                ``1`` means valid, ``0`` means padding. ``None`` if no masking
                is needed (e.g. standard string datasets without padding).
                When provided, the loss is computed only over valid token positions.
            **kwargs: Reserved for forward-compatibility with future parameters.

        Returns:
            dict: Best quantization parameters found during optimization, or an
                empty dict if no trainable parameters were found.
        """
        device = device_manager.device
        loss_device = getattr(self, "_loss_device", device)
        card_0_in_high_risk = getattr(self, "_card_0_in_high_risk", False)
        mid_iter_mem_check = self.compress_context.low_gpu_mem_usage and card_0_in_high_risk

        # Use quantized inputs if available and enabled
        active_inputs = q_inputs if (q_inputs is not None and self.enable_quanted_input) else fp_inputs
        nsamples = (
            len(active_inputs)
            if isinstance(active_inputs, list)
            else self.compressor.block_forward._count_samples(active_inputs)
        )

        quantized_layer_names, unquantized_layer_names = self.wrapper_block(
            block,
            self.enable_minmax_tuning,
            self.enable_norm_bias_tuning,
            enable_torch_compile=self.compress_context.enable_torch_compile,
            device=device,
        )

        round_params = []
        minmax_params = []
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                for key in m.params.keys():
                    if "min" in key or "max" in key:
                        minmax_params.append(m.params[key])
                    else:
                        round_params.append(m.params[key])

        lr = torch.tensor(self.lr)
        minmax_lr = torch.tensor(self.minmax_lr)

        extra_kwargs = {} if self.momentum is None else {"momentum": self.momentum}

        if self.enable_minmax_tuning:
            params = [
                {"params": round_params},
                {"params": minmax_params, "lr": minmax_lr},
            ]
        else:
            params = round_params

        optimizer = self.optimizer(
            params,
            lr=lr,
            weight_decay=0,
            **extra_kwargs,
        )

        if len(round_params) + len(minmax_params) <= 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block"
            )
            logger.info(dump_info)
            unwrapper_block(block, {})
            return {}

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
        batch_size = self._calibration_state.batch_size  # TODO delete wenhuach
        global_batch_size = batch_size * self.gradient_accumulate_steps
        global_batch_size = min(nsamples, global_batch_size)
        # We assume the block input and output shape is same
        if self.gradient_accumulate_steps != 1 and not valid_token_mask:
            whole_indices = torch.arange(global_batch_size)
            if isinstance(active_inputs, list):  # dict for diffusion, tricky setting, not sure whether it's correct
                num_elm = sum(active_inputs[i.item()].numel() for i in whole_indices)

        block, sync_gradients = setup_ddp_if_needed_(self, block, device_manager.device_list)
        index_sampler = IndexSampler(nsamples, global_batch_size)
        block_fwd = self.block_forward
        for i in range(self.iters):
            if self.enable_alg_ext and self.scheme.data_type.endswith("dq"):
                for n, m in block.named_modules():
                    m.cur_iter = i
            total_loss = 0
            global_indices = index_sampler.next_batch()
            if valid_token_mask:
                num_elm = self._get_non_zero_cnt(valid_token_mask, global_indices)

            for batch_start in range(0, len(global_indices), batch_size):
                indices = global_indices[batch_start : batch_start + batch_size]
                ref_output = torch.cat([fp_outputs[i] for i in indices], dim=0).to(loss_device)
                pred_output = block_fwd.forward(block, active_inputs, input_others, indices)
                if loss_device is not None:
                    pred_output = pred_output.to(loss_device)
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

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    best_params = collect_best_params(block, self.compress_context.cache_device)
                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_params = collect_best_params(block, self.compress_context.cache_device)

            if not self.not_use_best_mse:
                if 0 < self.dynamic_max_gap <= i - last_best_iter:
                    break
            sync_gradients()
            self._step(scaler, optimizer, lr_schedule)

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        if self.iters > 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block, loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
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

    def quantize_layer_outside_block(
        self,
        layer_name: str,
        fp_input: Optional[list[torch.Tensor]] = None,
        q_input: Optional[list[torch.Tensor]] = None,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        valid_token_mask: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ):
        """Quantize a single layer that lives outside a transformer block.

        When ``fp_input`` is provided the layer is tuned with the sign-gradient
        descent optimizer (same loss loop as block-level quantization).  When
        ``fp_input`` is ``None`` the method falls back to zero-shot RTN.

        Args:
            layer_name: Fully-qualified module name of the layer to quantize
                (e.g. ``"model.lm_head"``).
            fp_input: Per-sample FP activations fed into this layer, used as
                calibration inputs during optimization. ``None`` triggers RTN
                fallback.
            q_input: Per-sample quantized activations from the previous stage,
                used instead of ``fp_input`` during the forward pass when
                cascaded quantized-input is enabled. ``None`` means use
                ``fp_input`` for both reference and tuning forward.
            device: Target device string for running the optimization
                (e.g. ``"cuda:0"``). Defaults to ``"cpu"``.
            dtype: Optional dtype to cast the layer to before quantization.
                ``None`` keeps the existing dtype.
            valid_token_mask: Per-sample boolean/int masks of shape
                ``[1, seq_len]`` indicating valid (non-padding) token positions.
                ``1`` means valid, ``0`` means padding. ``None`` if no masking
                is needed. Forwarded to the loss computation to weight out
                padding tokens.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`quantize_layer_via_rtn` (e.g. ``disable_opt_rtn``).
        """
        if fp_input is None:
            logger.info(f"using rtn to quantize {layer_name}")
            if dtype is not None:
                layer = get_module(self.model, layer_name)
                set_module(self.model, layer_name, layer.to(dtype))
            self.quantize_layer_via_rtn(
                layer_name,
                disable_opt_rtn=kwargs.get("disable_opt_rtn", getattr(self.config, "disable_opt_rtn", True)),
            )
            return

        logger.info(f"quantizing layer {layer_name}")
        layer = get_module(self.model, layer_name)
        if dtype is not None:
            layer = layer.to(dtype)
        if hasattr(layer, "tuning_device"):
            device = layer.tuning_device

        layer = layer.to(device)
        for i in range(len(fp_input)):
            fp_input[i] = fp_input[i].to(layer.weight.dtype)
            if q_input is not None:
                q_input[i] = q_input[i].to(layer.weight.dtype)

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
        nsamples = len(fp_input)
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        best_params = None
        scaler = self._get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        batch_size = self._calibration_state.batch_size
        gradient_accumulate_steps = batch_size  # Force to low gpu

        total_loss = 0
        num_elm = 1
        mse_reduction = "mean"
        if gradient_accumulate_steps != 1:
            mse_reduction = "sum"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)
        batch_size = 1  # Force to low gpu
        global_batch_size = batch_size * gradient_accumulate_steps
        global_batch_size = min(nsamples, global_batch_size)
        if gradient_accumulate_steps != 1 and not self.attention_mask:
            whole_indices = torch.arange(global_batch_size).tolist()
            if q_input is not None:
                num_elm = self._count_layer_input_elements(q_input, whole_indices)
            else:
                num_elm = self._count_layer_input_elements(fp_input, whole_indices)

        index_sampler = IndexSampler(nsamples, global_batch_size)

        for i in range(self.iters):
            total_loss = 0
            global_indices = index_sampler.next_batch()
            if valid_token_mask:
                num_elm = self._get_non_zero_cnt(valid_token_mask, global_indices)

            for batch_start in range(0, len(global_indices), batch_size):
                indices = global_indices[batch_start : batch_start + batch_size]
                if q_input is not None:
                    current_input = [q_input[i] for i in indices]
                    current_input = torch.cat(current_input, dim=0).to(device)
                    org_input = [fp_input[i] for i in indices]
                    org_input = torch.cat(org_input, dim=0).to(device)
                else:
                    current_input = [fp_input[i] for i in indices]
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
                    tmp_attention_mask = [valid_token_mask[i] for i in indices]
                    tmp_attention_mask = torch.cat(tmp_attention_mask, dim=0).to(device)
                    tmp_attention_mask.unsqueeze_(-1)

                    with autocast_ctx:
                        output_q = wrapper_linear(current_input)  # pylint: disable=not-callable
                        loss = mse_loss(  # pylint: disable=not-callable
                            (output_q * tmp_attention_mask).to(torch.float32),
                            (current_output * tmp_attention_mask).to(torch.float32),
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
        dump_info = f"quantized {layer_name},  loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
        logger.info(dump_info)

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
