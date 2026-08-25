# Copyright (c) 2025 Intel Corporation
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
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from functools import wraps
from typing import Iterable, Optional, Union

import torch
from accelerate import dispatch_model
from tqdm import tqdm

from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.auto_scheme.register import register_scheme_methods
from auto_round.auto_scheme.utils import (
    _describe_layer_config,
    _fill_inactive_expert_scores,
    _log_batch_avg_loss,
    _log_scheme_loss_matrix,
    _log_score_summary_by_block_and_nonblock,
    _scheme_short_name,
    apply_quant_scheme,
    build_expert_groups,
    compute_layer_bits,
    merge_lists_unionfind,
    parse_shared_layers,
    remove_quant_scheme,
)
from auto_round.calib_dataset import get_dataloader
from auto_round.data_type.gguf import (
    quant_tensor_gguf_asym_dq,
    quant_tensor_gguf_sym_dq,
    search_gguf_scale_min_asym,
    search_gguf_scale_min_sym,
)
from auto_round.data_type.utils import get_quant_func, reshape_pad_tensor_by_group_size, revert_tensor_by_pad
from auto_round.logger import logger
from auto_round.modeling.fused_moe.replace_modules import safe_to_cpu_
from auto_round.schemes import QuantizationScheme, preset_name_to_scheme
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    clear_memory,
    dispatch_model_by_all_available_devices,
    flatten_list,
    get_block_names,
    get_lm_head_name,
    get_major_device,
    get_module,
    is_mllm_model,
    llm_load_model,
    load_model,
    mllm_load_model,
    parse_available_devices,
    set_avg_auto_device_map,
    set_module,
    set_non_auto_device_map,
    to_device,
    to_dtype,
)
from auto_round.utils.device import MemoryMonitor, memory_monitor
from auto_round.utils.device_manager import get_current_device_manager
from auto_round.utils.model import is_moe_model as _is_moe_model
from auto_round.utils.offload import OffloadManager
from auto_round.wrapper import WrapperLinear

__all__ = ["gen_layer_config"]


class _ScoreLinear(torch.autograd.Function):
    """Linear for scoring passes whose weight lives in the wrapper's CPU cache.

    Saves only the input activation: the dequantized weight is transferred
    from the cache again wherever needed, so the block's autograd graph holds
    no weight-shaped tensors. The weight score is accumulated from the
    explicit gradient instead of a tensor hook.
    """

    @staticmethod
    def forward(ctx, x, wrapper, bias):  # pylint: disable=arguments-differ
        ctx.wrapper = wrapper
        ctx.save_for_backward(x)
        weight = wrapper._score_weight_for_device(x.device)
        return torch.nn.functional.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):  # pylint: disable=arguments-differ
        (x,) = ctx.saved_tensors
        wrapper = ctx.wrapper
        weight = wrapper._score_weight_for_device(x.device)
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.matmul(grad_out, weight)
        if wrapper.grad_mode and wrapper.need_weight_grad:
            grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1])
            x_2d = x.reshape(-1, x.shape[-1])
            grad_w = torch.mm(grad_out_2d.t(), x_2d)
            weight_ref = wrapper.orig_layer.weight
            if weight_ref.device.type == "meta":
                weight_ref = wrapper.orig_layer.get_weight().to(x.device)
            w_diff = weight_ref.to(x.device) - weight
            wrapper.weight_score += torch.abs(grad_w * w_diff).sum().item()
            wrapper.mix_score = wrapper.weight_score + wrapper.act_score
        return grad_x, None, None


class AutoSchemeWrapperLinear(WrapperLinear):

    def __init__(
        self,
        orig_layer,
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        device="cpu",
        enable_round_tuning=True,
        need_weight_grad=False,
        enable_torch_compile=True,
        **kwargs,
    ):
        """Wrap ``orig_layer`` to accumulate a ``mix_score`` (weight + activation loss) during
        forward/backward, used by Delta Loss to rank candidate quantization schemes.
        """
        super().__init__(
            orig_layer,
            enable_minmax_tuning,
            enable_norm_bias_tuning,
            device,
            enable_round_tuning,
            enable_torch_compile=enable_torch_compile,
            **kwargs,
        )
        self.act_score = 0.0
        self.avg_act_score = 0.0
        self.act_cnt = 0.0
        self.weight_score = 0.0
        self.mix_score = 0.0
        self.super_qdq_func = super()._qdq_weight
        self.act_qdq_func = super()._qdq_act
        self.max_act_value = 0
        self.need_weight_grad = need_weight_grad
        self.grad_mode = False
        if self.need_weight_grad:
            self.orig_layer.weight.requires_grad = True
        # scoring cache: the dequantized weight, computed once per wrapper
        # visit and stored on CPU; bounded to the block being replayed so
        # parallel workers' caches stay within host RAM
        self._score_qdq_cpu = None
        # use the recompute-weight scoring forward (see _ScoreLinear)
        self._custom_score_forward = True

    def _qdq_act(self, x, act_min_scale=1.0, act_max_scale=1.0, act_max=None):
        """Quant-dequant the activation and, in ``grad_mode``, register a backward hook that
        accumulates ``act_score`` from ``|grad * (x - qdq_x)|``.
        """
        if hasattr(self.orig_layer, "act_bits") and self.orig_layer.act_bits > 8:
            return x, 1.0, None

        qdq_x, scale, zp = self.act_qdq_func(x, act_min_scale, act_max_scale, act_max)
        if self.grad_mode:
            with torch.no_grad():
                max_act_value = torch.abs(x).max()
                self.max_act_value = max_act_value
                if max_act_value != 0:
                    self.act_cnt += 1
                x_diff = (x - qdq_x).to("cpu")

            def save_grad(grad):
                """Backward hook: accumulate activation score from grad * (x - qdq_x)."""
                if max_act_value == 0:
                    if torch.abs(grad).max() != 0:
                        raise ValueError
                """
                this ut will cause NAN issue sometimes, need to investigate
                    @multi_card
                    def test_multi_card(self):
                     model_name = "/models/Qwen3-8B"
                """
                if torch.isnan(grad).any() or torch.isnan(x_diff).any():
                    self.act_cnt -= 1
                    return None

                self.act_score += torch.abs((grad * x_diff.to(grad.device))).sum().item()
                self.mix_score = self.weight_score + self.act_score
                return None

            if qdq_x.requires_grad:
                qdq_x.register_hook(save_grad)
        return qdq_x, scale, zp

    def _ensure_score_cache(self):
        """Build the per-visit CPU cache of the dequantized weight (scoring layers only)."""
        if self._score_qdq_cpu is not None:
            return
        device = self.device
        with torch.no_grad():
            qdq_w, _, _ = super(AutoSchemeWrapperLinear, self)._qdq_weight(
                torch.tensor(0, device=device), torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)
            )
        self._score_qdq_cpu = qdq_w.detach().to("cpu")

    def _score_weight_for_device(self, device):
        """Return the cached dequantized weight on ``device`` (transfers from the CPU cache)."""
        self._ensure_score_cache()
        return self._score_qdq_cpu.to(device)

    def forward(self, x):
        """In the scoring replay, run the linear through ``_ScoreLinear`` so the block's
        autograd graph saves activations only; every other phase uses the base forward."""
        if getattr(self, "_custom_score_forward", False) and self.grad_mode and torch.is_grad_enabled():
            x = x.to(self.device)
            for hook in self.orig_layer._forward_pre_hooks.values():
                result = hook(self.orig_layer, (x,))
                if result is not None:
                    x = result[0] if isinstance(result, tuple) else result
            bias = self.orig_layer.bias
            if bias is not None and bias.device.type == "meta":
                bias = self.orig_layer.get_bias().to(self.device)
            output = _ScoreLinear.apply(x, self, bias)
            for hook in self.orig_layer._forward_hooks.values():
                hook_result = hook(self.orig_layer, (x,), output)
                if hook_result is not None:
                    output = hook_result
            return output.to(self.output_device)
        return super().forward(x)

    def _qdq_weight(self, value, min_scale, max_scale):
        """Quant-dequant the weight once per scoring pass, cache the result, and, in
        ``grad_mode``, route the score hook through a scalar anchor.

        The cached quantization is recomputed lazily on the first call, so layers
        whose forward never runs (e.g. unrouted MoE experts) never pay this cost.
        In ``grad_mode`` the anchor makes the returned tensor a graph node whose
        gradient is exactly d(loss)/d(qdq_w): the score hook fires without building
        the quant chain into the graph and without materializing weight-shaped
        gradient buffers, which on expert-heavy blocks dominated replay VRAM.
        """
        device = self.device
        if self.orig_layer.bits > 8:
            return super()._qdq_weight(
                torch.tensor(0, device=device), torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)
            )

        scoring = self.grad_mode and self.need_weight_grad
        if not scoring:
            # one-shot layers (capture pass, non-grad phases) never reuse the
            # result, so do not grow a CPU-side cache for them
            with torch.no_grad():
                return super()._qdq_weight(
                    torch.tensor(0, device=device), torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)
                )

        self._ensure_score_cache()
        qdq_w = self._score_qdq_cpu.to(device)
        if scoring:
            anchor = torch.zeros((), dtype=qdq_w.dtype, device=qdq_w.device, requires_grad=True)
            qdq_w = qdq_w + 0.0 * anchor

            def save_grad(grad):
                """Backward hook: accumulate weight score from grad * (weight - qdq_w)."""
                weight = self.orig_layer.weight
                if weight.device.type == "meta":
                    weight = self.orig_layer.get_weight().to(grad.device)
                w_diff = weight.to(grad.device) - self._score_qdq_cpu.to(grad.device)
                self.weight_score += torch.abs(grad.to(w_diff.device) * w_diff).sum().item()
                self.mix_score = self.weight_score + self.act_score
                return None

            qdq_w.register_hook(save_grad)
        return qdq_w, 1.0, None


class AutoSchemeWrapperLinearIMatrix(WrapperLinear):
    """GGUF-K wrapper that scores a layer using an imatrix-aware quant search (RTN, iters=0)."""

    def __init__(
        self,
        orig_layer,
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        device="cpu",
        enable_round_tuning=True,
        need_weight_grad=False,
        enable_torch_compile=True,
        **kwargs,
    ):
        """Wrap ``orig_layer`` and eagerly run the imatrix-aware quant search to build ``qdq_w``."""
        super().__init__(
            orig_layer,
            enable_minmax_tuning,
            enable_norm_bias_tuning,
            device,
            enable_round_tuning,
            enable_torch_compile=enable_torch_compile,
            **kwargs,
        )
        self.act_score = 0.0
        self.avg_act_score = 0.0
        self.act_cnt = 0.0
        self.weight_score = 0.0
        self.mix_score = 0.0
        self.super_qdq_func = super()._qdq_weight
        self.act_qdq_func = super()._qdq_act
        self.max_act_value = 0
        self.need_weight_grad = need_weight_grad
        self.grad_mode = False
        if self.need_weight_grad:
            self.orig_layer.weight.requires_grad = True
        self.weight_search_quant_func, _ = get_quant_func(
            orig_layer.data_type,
            orig_layer.bits,
            orig_layer.sym,
            disable_opt_rtn=False,
            group_size=orig_layer.group_size,
            iters=0,
        )
        self._custom_score_forward = False
        self.post_init_qdqw(device)

    @torch.no_grad()
    def post_init_qdqw(self, device):
        """Run the imatrix-aware quant search once and cache the result as buffer ``qdq_w``,
        registering a backward hook on it to accumulate ``weight_score``.
        """
        qdq_w, _, _ = self.weight_search_quant_func(
            self.orig_layer.weight.to(device),
            bits=self.orig_layer.bits,
            group_size=self.orig_layer.group_size,
            v=torch.tensor(0, device=device),
            min_scale=torch.tensor(1.0, device=device),
            max_scale=torch.tensor(1.0, device=device),
            scale_dtype=self.orig_layer.scale_dtype,
            data_type=self.data_type,
            q_scale_thresh=self.q_scale_thresh,
            imatrix=self.orig_layer.imatrix.to(device) if hasattr(self.orig_layer, "imatrix") else None,
            global_scale=getattr(self, "weight_global_scale", None),
        )

        self.register_buffer("qdq_w", qdq_w.detach().clone().to(self.orig_layer.weight.device))

        def save_grad(grad):
            """Backward hook: accumulate weight score from grad * (weight - qdq_w)."""
            w_diff = self.orig_layer.weight - self.qdq_w.to(self.orig_layer.weight.device)
            self.weight_score += torch.abs((grad.to(torch.float32) * w_diff.to(grad.device))).sum().item()
            self.mix_score = self.weight_score + self.act_score
            return None

        self.qdq_w.requires_grad_(True)
        self.orig_layer.weight.requires_grad_(False)

        self.qdq_w.register_hook(save_grad)

    def _qdq_act(self, x, act_min_scale=1.0, act_max_scale=1.0, act_max=None):
        """Quant-dequant the activation and, in ``grad_mode``, register a backward hook that
        accumulates ``act_score`` from ``|grad * (x - qdq_x)|``.
        """
        if hasattr(self.orig_layer, "act_bits") and self.orig_layer.act_bits > 8:
            return x, 1.0, None

        qdq_x, scale, zp = self.act_qdq_func(x, act_min_scale, act_max_scale, act_max)
        if self.grad_mode:
            with torch.no_grad():
                max_act_value = torch.abs(x).max()
                self.max_act_value = max_act_value
                if max_act_value != 0:
                    self.act_cnt += 1
                x_diff = (x - qdq_x).to("cpu")

            def save_grad(grad):
                """Backward hook: accumulate activation score from grad * (x - qdq_x)."""
                if max_act_value == 0:
                    if torch.abs(grad).max() != 0:
                        raise ValueError
                """
                this ut will cause NAN issue sometimes, need to investigate
                    @multi_card
                    def test_multi_card(self):
                     model_name = "/models/Qwen3-8B"
                """
                if torch.isnan(grad).any() or torch.isnan(x_diff).any():
                    self.act_cnt -= 1
                    return None

                self.act_score += torch.abs((grad * x_diff.to(grad.device))).sum().item()
                self.mix_score = self.weight_score + self.act_score
                return None

            if qdq_x.requires_grad:
                qdq_x.register_hook(save_grad)
        return qdq_x, scale, zp

    def _qdq_weight(self, value, min_scale, max_scale):
        """Return the cached ``qdq_w`` computed eagerly in ``__init__`` (via ``post_init_qdqw``)."""
        return self.qdq_w, 1.0, None


class AutoSchemeWrapperLinearForGGUFK(AutoSchemeWrapperLinear):
    """GGUF-K wrapper (no imatrix): scores a layer using the plain GGUF K-quant search."""

    def __init__(
        self,
        orig_layer,
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        device="cpu",
        enable_round_tuning=True,
        need_weight_grad=False,
        **kwargs,
    ):
        """Wrap ``orig_layer`` and eagerly run the GGUF K-quant search to build ``qdq_w``."""
        super().__init__(
            orig_layer,
            enable_minmax_tuning,
            enable_norm_bias_tuning,
            device,
            enable_round_tuning,
            need_weight_grad,
            **kwargs,
        )
        self._custom_score_forward = False
        self.post_init_qdqw(device)

    @torch.no_grad()
    def post_init_qdqw(self, device):
        """Run the GGUF K-quant search once and cache the result as buffer ``qdq_w``,
        registering a backward hook on it to accumulate ``weight_score``.
        """
        qdq_w, scale, zp = self.super_qdq_func(
            torch.tensor(0).to(device), torch.tensor(1.0).to(device), torch.tensor(1.0).to(device)
        )
        self.register_buffer("qdq_w", qdq_w.detach().clone().to(self.orig_layer.weight.device))

        def save_grad(grad):
            """Backward hook: accumulate weight score from grad * (weight - qdq_w)."""
            w_diff = self.orig_layer.weight - self.qdq_w.to(self.orig_layer.weight.device)
            # TODO strange, grad could be in CPU
            self.weight_score += torch.abs((grad.to(w_diff.device).to(torch.float32) * w_diff)).sum().item()
            self.mix_score = self.weight_score + self.act_score
            return None

        self.qdq_w.requires_grad_(True)
        self.orig_layer.weight.requires_grad_(False)
        self.qdq_w.register_hook(save_grad)

    def _qdq_weight(self, value, min_scale, max_scale):
        """Return the cached ``qdq_w`` computed eagerly in ``__init__`` (via ``post_init_qdqw``)."""
        return self.qdq_w, 1.0, None


class AutoSchemeWrapperLinearForGGUFKImatrix(AutoSchemeWrapperLinear):
    """GGUF-K wrapper (with imatrix): scores a layer using the imatrix-weighted GGUF K-quant
    search (``_init_scale``).
    """

    def __init__(
        self,
        orig_layer,
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        device="cpu",
        enable_round_tuning=True,
        need_weight_grad=False,
        enable_torch_compile=True,
        **kwargs,
    ):
        """Wrap ``orig_layer`` and eagerly run the imatrix-weighted GGUF K-quant search to
        build ``qdq_w``.
        """
        super().__init__(
            orig_layer,
            enable_minmax_tuning,
            enable_norm_bias_tuning,
            device,
            enable_round_tuning,
            need_weight_grad,
            enable_torch_compile=enable_torch_compile,
            **kwargs,
        )
        self._custom_score_forward = False
        self.post_init_qdqw(device)

    @torch.no_grad()
    def post_init_qdqw(self, device):  # Could not place in qdq_w, otherwise vram is much higher
        """Run the imatrix-weighted GGUF K-quant search once and cache the result as buffer
        ``qdq_w``, registering a backward hook on it to accumulate ``weight_score``.
        """
        qdq_w = self._init_scale(device).detach()
        self.register_buffer("qdq_w", qdq_w.detach().clone().to(self.orig_layer.weight.device))

        def save_grad(grad):
            """Backward hook: accumulate weight score from grad * (weight - qdq_w)."""
            w_diff = self.orig_layer.weight - self.qdq_w.to(self.orig_layer.weight.device)
            self.weight_score += torch.abs((grad.to(torch.float32) * w_diff.to(grad.device))).sum().item()
            self.mix_score = self.weight_score + self.act_score
            return None

        self.qdq_w.requires_grad_(True)
        self.orig_layer.weight.requires_grad_(False)

        self.qdq_w.register_hook(save_grad)

    @torch.no_grad()
    def _init_scale(self, device):
        """Compute the imatrix-weighted GGUF K-quant quant-dequant weight for ``bits`` in
        [2,3,4,5,6], returned in the original weight dtype.
        """
        tensor = self.orig_layer.weight.data.to(device)
        bits = self.orig_layer.bits
        scale_dtype = self.orig_layer.scale_dtype
        imatrix = self.orig_layer.imatrix.to(tensor.device)
        orig_dtype = tensor.dtype
        if self.orig_layer.bits in [2, 4, 5]:
            group_size = 16 if bits == 2 else 32
            tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
            scale, wmin, d_scale, d_wmin = search_gguf_scale_min_asym(tensor, bits, scale_dtype, imatrix)
            tensor = revert_tensor_by_pad(tensor, orig_shape=orig_shape, pad_len=pad_len)

            qdq_w, _, _ = quant_tensor_gguf_asym_dq(
                tensor=tensor,
                bits=bits,
                scale_dtype=scale_dtype,
                imatrix=imatrix,
                scale=scale,
                wmin=wmin,
                d_scale=d_scale,
                d_wmin=d_wmin,
            )
        elif bits in [3, 6]:
            qdq_w, _, _ = quant_tensor_gguf_sym_dq(
                tensor=tensor,
                bits=bits,
                scale_dtype=scale_dtype,
                imatrix=imatrix,
                split_num=1,
            )
        else:
            raise ValueError("bits must be in [2,3,4,5,6]")
        return qdq_w.to(orig_dtype)

    def _qdq_weight(self, value, min_scale, max_scale):
        """Return the cached ``qdq_w`` computed eagerly in ``__init__`` (via ``post_init_qdqw``)."""
        return self.qdq_w, 1.0, None


def register_imatrix_hook(model):
    """Registers hooks to accumulate activation squared norms into `imatrix`."""

    def get_imatrix_hook(module, input, output):
        """Forward hook: accumulate the per-channel squared-activation sum into ``module.imatrix``."""
        input = input[0] if isinstance(input, (tuple, list)) else input
        flattened = input.reshape(-1, input.shape[-1]).to(torch.float32)
        squared = torch.sum(torch.pow(flattened, 2), dim=0).to(torch.float32)

        if not hasattr(module, "imatrix"):
            module.imatrix = squared.to("cpu")
        else:
            module.imatrix += squared.to(module.imatrix.device).to("cpu")

    hook_handles = []
    for name, module in model.named_modules():
        if isinstance(module, SUPPORTED_LAYER_TYPES):
            hook = module.register_forward_hook(get_imatrix_hook)
            hook_handles.append(hook)
    return hook_handles


@torch.no_grad()
def cal_imatrix(model, dataloader, major_device, low_gpu_mem_usage):
    """Accumulate an activation-based imatrix on every supported layer by running the
    calibration ``dataloader`` through ``model`` once (dispatches to the low-GPU-memory or
    full-forward variant based on ``low_gpu_mem_usage``).
    """
    if low_gpu_mem_usage:
        cal_imatrix_low_gpu(model, dataloader, major_device)
    else:
        hooks = register_imatrix_hook(model)
        model = model.to(model.device)
        for data in dataloader:
            model.forward(**to_device(data, model.device))
        for hook in hooks:
            hook.remove()


def cal_imatrix_low_gpu(model, dataloader, major_device):
    """Low-GPU-memory variant of ``cal_imatrix``: moves each block to ``major_device`` only
    for the duration of its own forward pass (via pre/post forward hooks), then back to CPU.
    """
    imatrix_hooks = register_imatrix_hook(model)
    block_names = get_block_names(model, quant_vision=True)
    block_names = flatten_list(block_names)

    def move_to_gpu_hook(module, inputs):
        """Pre-forward hook: move this block (and its inputs) to ``major_device``."""
        module.to(major_device)
        to_device(inputs, major_device)

    def move_to_cpu(module, inputs, outputs):
        """Forward hook: move this block back to CPU once its forward pass is done."""
        module.to("cpu")

    def move_to_cpu_clear_memory(module, inputs, outputs):
        """Forward hook: move this block back to CPU and free the device memory it used."""
        module.to("cpu")
        clear_memory(device_list=major_device)

    all_move_device_hooks = []
    i = 0
    for block_name in block_names:
        i += 1
        block_module = get_module(model, block_name)
        hook_move_gpu = block_module.register_forward_pre_hook(move_to_gpu_hook)

        hook_move_cpu = block_module.register_forward_hook(move_to_cpu)

        all_move_device_hooks.append(hook_move_gpu)
        all_move_device_hooks.append(hook_move_cpu)

    for data in dataloader:
        model.forward(**to_device(data, model.device))

    for hook in imatrix_hooks:
        hook.remove()
    for hook in all_move_device_hooks:
        hook.remove()
    clear_memory(device_list=major_device)


class MyCustomError(Exception):
    """Raised from ``backward_pre_hook`` to deliberately interrupt ``loss.backward()`` at the
    last block, so gradients can be replayed manually block-by-block in ``model_forward_low_gpu``.
    """

    def __init__(self, message):
        """Create the interrupt signal with the given ``message``."""
        super().__init__(message)


def prepare_model_low_gpu(model, block_inputs: dict = None, pbar=None, major_device="cpu", disk_index=None):
    """Wrap every block's forward so that, for one calibration batch, it (1) moves itself to
    ``major_device`` on demand, (2) records its own inputs into ``block_inputs`` (on CPU) so
    they can be replayed later, and (3) moves itself back to CPU once done.

    Called once per calibration batch before ``model_forward_low_gpu`` runs the actual
    forward+backward -- the recorded ``block_inputs`` are what let the backward pass be
    replayed manually, one block at a time, without keeping every block resident on GPU.

    When ``disk_index`` is set (streaming mode -- the model is a meta-device skeleton,
    see ``gen_layer_config``/``disk_stream_util.py``), each block's real weights are
    materialized from the checkpoint right before its own forward and released back to
    meta right after, instead of assuming the block already has real CPU-resident weights
    to shuffle to GPU and back.
    """
    block_inputs.clear()
    for n, m in model.named_modules():
        if hasattr(m, "grad_mode"):
            m.grad_mode = False

    block_names = get_block_names(model)[0]

    def wrap_forward(module, module_name):
        """Build a replacement ``forward`` for ``module`` that captures its inputs/outputs
        (see ``prepare_model_low_gpu`` docstring) while moving it to/from ``major_device``.
        """
        original_forward = module.forward

        @wraps(original_forward)
        def new_forward(*args, **kwargs):
            """Move the block to device, run its original forward, cache its (CPU) inputs
            for later replay, then move the block back to CPU.
            """
            if disk_index is not None:
                from auto_round.utils.disk_stream_util import materialize_module

                materialize_module(module, module_name, disk_index, device=major_device)
            move_module_to_tuning_device(module, major_device=major_device)
            # The block now sits on major_device; its incoming tensors may have
            # been emitted on another device (e.g. CPU-resident embeddings, or
            # CPU-recorded replay inputs), so align them before the forward --
            # a same-device .to() is a no-op reference return.
            args = tuple(a.to(major_device) if isinstance(a, torch.Tensor) else a for a in args)
            kwargs = {k: v.to(major_device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

            # Call the original forward
            with torch.no_grad():
                result = original_forward(*args, **kwargs)

            # Save input information and ensure tensors are on CPU
            input_info = {
                "args": [arg.detach().clone().to("cpu") if isinstance(arg, torch.Tensor) else arg for arg in args],
                "kwargs": {
                    k: v.detach().clone().to("cpu") if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
                },
            }
            block_inputs[module_name] = input_info

            if disk_index is not None:
                from auto_round.utils.disk_stream_util import free_module

                free_module(module)
            else:
                module.to("cpu")
            memory_monitor.update(device_list=major_device)
            # clear_memory(device_list=major_device) #slow
            # memory_monitor.log_summary()

            # Enable gradients for the output of the last block
            if module.tmp_name == block_names[-1]:
                if isinstance(result, torch.Tensor):
                    if result.is_floating_point():
                        result = result.requires_grad_(True)
                elif isinstance(result, tuple):
                    result = tuple(
                        r.requires_grad_(True) if isinstance(r, torch.Tensor) and r.is_floating_point() else r
                        for r in result
                    )

            if pbar is not None:
                pbar.update(1)
            return result

        return new_forward

    # Assign a temporary name to each module
    for n, m in model.named_modules():
        m.tmp_name = n

    # Wrap the forward method of each block
    for block_name in block_names:
        module = get_module(model, block_name)
        module.forward = wrap_forward(module, block_name)


def _prepare_mllm_inputs(data, model):
    """Normalize one batch from a (possibly mllm) dataloader into a form the
    model can be called on.

    Crucially this casts ``images`` / ``pixel_values`` / ``pixel_values_videos``
    to ``model.dtype`` — without it, vision tensors arrive as float32 while
    the vision tower is bf16/fp16, and several HF VLM implementations silently
    bypass the vision branch on dtype mismatch (which manifests downstream as
    "vision grad = 0").

    Returns ``(prepared, kind)`` where ``kind`` is ``"tensor" | "seq" | "dict"``
    so the caller knows whether to use ``model(x)`` / ``model(*x)`` /
    ``model(**x)``.
    """
    _img_keys = ("images", "image", "pixel_values", "pixel_values_videos", "pixel_values_images", "image_pixel_values")

    if isinstance(data, torch.Tensor):
        return data.to(model.device), "tensor"

    if isinstance(data, (tuple, list)):
        return to_device(data, model.device), "seq"

    # Plain dict (the common path: HF VLM ``data_collator`` outputs).
    new = {}
    for key, value in data.items():
        t = to_device(value, model.device)
        if key in _img_keys:
            t = to_dtype(t, model.dtype)
        new[key] = t
    return new, "dict"


def model_forward(model, data, **forward_kwargs):
    """Single entry point for "run a (possibly multimodal) batch through the
    model". Used by both AutoScheme paths so that ``pixel_values`` / ``images``
    are cast to ``model.dtype`` (otherwise VLMs silently skip the vision tower
    → vision grad = 0)."""
    prepared, kind = _prepare_mllm_inputs(data, model)
    if kind == "tensor":
        return model(prepared, **forward_kwargs), prepared
    if kind == "seq":
        return model(*prepared, **forward_kwargs), prepared
    return model(**prepared, **forward_kwargs), prepared


def _clear_wrapper_score_caches(block_module):
    """Drop the per-wrapper scoring caches of every layer in ``block_module`."

    Wrappers persist for a whole scoring pass while their caches are only valid
    while the owning layer's weights are materialized; without this the CPU-side
    caches accumulate across blocks of the model.
    """
    for module in block_module.modules():
        if getattr(module, "_score_qdq_cpu", None) is not None:
            module._score_qdq_cpu = None


def _replay_retain_graph(block_module) -> bool:
    """Whether this block's backward must keep its autograd graph alive.

    MX-family data types re-run quantization code inside backward hooks that
    expect the block graph to still exist.  Every other data type frees the
    graph per block: the reverse replay holds one block's saved activations at
    a time, and retaining graphs across blocks accumulates them for the whole
    pass, which is what pushes streamed scoring workers OOM mid-pass.
    """
    for _, module in block_module.named_modules():
        data_type = getattr(module, "data_type", None)
        if isinstance(data_type, str) and data_type.startswith("mx"):
            return True
    return False


def _prepare_replay_input(block_input_args, block_input_kwargs, block_name):
    """Find the floating hidden-state tensor whose gradient feeds the preceding block."""
    candidates = []
    if "hidden_states" in block_input_kwargs:
        candidates.append(block_input_kwargs["hidden_states"])
    candidates.extend(block_input_args)
    candidates.extend(value for key, value in block_input_kwargs.items() if key != "hidden_states")
    for value in candidates:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            value.requires_grad_(True)
            return value
    raise RuntimeError(f"No floating replay input found for block {block_name}")


def _vram_inventory_text(top_k: int = 12) -> str:
    """Return the live-CUDA-tensor census as text (see ``_vram_inventory``).

    Grouped by (shape, dtype) signature, largest first -- block weight shapes
    are recognizable, which is what makes an at-failure census actionable.
    """
    import gc as _gc
    from collections import defaultdict

    if not torch.cuda.is_available():
        return "cuda unavailable; no census"
    groups = defaultdict(lambda: [0, 0])
    for obj in _gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                key = (tuple(obj.shape), str(obj.dtype))
                groups[key][0] += 1
                groups[key][1] += obj.element_size() * obj.numel()
        except Exception:  # noqa: BLE001
            continue
    ranked = sorted(groups.items(), key=lambda kv: -kv[1][1])[:top_k]
    total = sum(v[1] for v in groups.values())
    alloc = torch.cuda.memory_allocated() / 2**30
    lines = [f"live tensors {total / 2**30:.2f} GiB (allocator {alloc:.2f} GiB)"]
    for (shape, dtype), (cnt, nb) in ranked:
        lines.append(f"  {nb / 2**30:6.2f} GiB x{cnt:<4} {dtype} {shape}")
    # name the retaining objects for the largest group so the census points at
    # the retainer, not only the retained shape
    for rank, ((shape, dtype), _) in enumerate(ranked[:2]):
        target = None
        objs = _gc.get_objects()
        for obj in objs:
            try:
                if torch.is_tensor(obj) and obj.is_cuda and tuple(obj.shape) == shape:
                    target = obj
                    break
            except Exception:  # noqa: BLE001
                continue
        # The enumeration list itself references every object it yielded; drop
        # it before asking for referrers or it shadows the real retainers.
        del objs
        if target is not None:
            trace = _tensor_referrer_snippet(target)
            if trace:
                lines.append(f"  group#{rank} referrers {shape}:\n{trace}")
    return "\n".join(lines)


def _tensor_referrer_snippet(tensor, max_depth: int = 3, max_entries: int = 4) -> str:
    """Describe what holds ``tensor`` alive, as a compact referrer chain.

    Walks ``gc.get_referrers`` breadth-first, reporting container types (list /
    dict sizes, module class names) so a retention census names the retaining
    object rather than only the retained shape.
    """
    import gc as _gc

    import torch.nn as _nn

    def _describe(obj, refers_to=None):
        if isinstance(obj, (list, tuple)):
            return f"{type(obj).__name__}[{len(obj)}]"
        if isinstance(obj, dict):
            keys = list(obj.keys())[:max_entries]
            shown = [k if isinstance(k, str) else type(k).__name__ for k in keys]
            desc = "dict{" + ", ".join(map(str, shown)) + "}"
            if refers_to is not None:
                # name the holding attribute deterministically: identity-search
                # the walked object among the dict's values, so the snippet does
                # not depend on key order or gc traversal order across platforms
                for key, value in obj.items():
                    if value is refers_to and isinstance(key, str):
                        return f"{desc}[holding {key!r}]"
            return desc
        if isinstance(obj, _nn.Module):
            desc = f"module:{type(obj).__name__}"
            if refers_to is not None:
                for store_name in ("_parameters", "_buffers", "_modules"):
                    store = getattr(obj, store_name, None)
                    if isinstance(store, dict):
                        for attr_name, value in store.items():
                            if value is refers_to:
                                return f"{desc}.{store_name}[{attr_name!r}]"
                for attr_name, value in obj.__dict__.items():
                    if value is refers_to:
                        return f"{desc}.{attr_name}"
            return desc
        return type(obj).__name__

    seen = {id(tensor)}
    frontier = [tensor]
    lines = []
    for _depth in range(max_depth):
        next_frontier = []
        for obj in frontier:
            try:
                referrers = _gc.get_referrers(obj)
            except Exception:  # noqa: BLE001
                continue
            for ref in referrers:
                if id(ref) in seen:
                    continue
                seen.add(id(ref))
                desc = _describe(ref)
                if desc in ("frame", "builtin_function_or_method", "function"):
                    continue
                if isinstance(ref, (list, tuple)) and len(ref) > 10000:
                    # giant containers at this depth are bookkeeping artifacts
                    # (object registries, gc internals), not model state
                    continue
                lines.append(f"  depth{_depth + 1} <- {_describe(ref, refers_to=obj)}")
                if isinstance(ref, (list, tuple, dict)):
                    next_frontier.append(ref)
            if len(lines) >= max_entries * 3:
                break
        frontier = next_frontier
        if not frontier or len(lines) >= max_entries * 3:
            break
    return "\n".join(lines[: max_entries * 3]) if lines else "no external referrers found"


def _annotate_worker_oom(worker_index, exc):
    """Attach a live-tensor census and the failing op to a CUDA OOM in a worker."""
    import traceback as _tb

    try:
        census = _vram_inventory_text(top_k=20)
    except Exception:  # noqa: BLE001  the census must never mask the OOM
        census = "census unavailable"
    tb_lines = _tb.format_exc().splitlines()
    tb_head = "\n".join(tb_lines[:4])
    tb_tail = "\n".join(tb_lines[-8:])
    return RuntimeError(
        f"_score_scheme_worker[{worker_index}] CUDA OOM during scoring. "
        f"Live-tensor census at failure:\n{census}\n\n"
        f"Traceback head:\n{tb_head}\n...\nTraceback tail:\n{tb_tail}"
        f"Original error: {exc}"
    )


def _vram_inventory(tag: str, top_k: int = 12):
    """Env-gated (AR_SCHEME_MEM_INVENTORY=1) per-device live-tensor census:
    resident CUDA tensors grouped by (shape, dtype) signature, largest first --
    used to spot unreleased module weights or retained graphs during streamed
    scoring (block shapes are recognizable: e.g. [5120, 17408] = mlp.down)."""
    import gc as _gc
    import os as _os
    from collections import defaultdict

    if _os.getenv("AR_SCHEME_MEM_INVENTORY", "0") != "1":
        return
    groups = defaultdict(lambda: [0, 0])
    for obj in _gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                key = (tuple(obj.shape), str(obj.dtype))
                groups[key][0] += 1
                groups[key][1] += obj.element_size() * obj.numel()
        except Exception:  # noqa: BLE001
            continue
    ranked = sorted(groups.items(), key=lambda kv: -kv[1][1])[:top_k]
    total = sum(v[1] for v in groups.values())
    alloc = torch.cuda.memory_allocated() / 2**30
    print(f"[mem-inv] {tag}: live tensors {total / 2**30:.2f} GiB " f"(allocator {alloc:.2f} GiB)", flush=True)
    for (shape, dtype), (cnt, nb) in ranked:
        print(f"[mem-inv]   {nb / 2**30:6.2f} GiB x{cnt:<4} {dtype} {shape}", flush=True)


def model_forward_low_gpu(
    model,
    dataloader,
    major_device="cuda",
    pbar=None,
    scheme_tag=None,
    disk_index=None,
    skip_batches=0,
    batch_checkpoint=None,
):
    """Run one full scoring pass (all calibration batches) in low-GPU-memory mode.

    For each batch: capture per-block inputs via ``prepare_model_low_gpu``, run a forward
    pass whose backward is deliberately interrupted at the last block (``backward_pre_hook``
    raising ``MyCustomError``), then manually replay the backward pass block-by-block
    (moving each block to ``major_device`` only for its own recompute + backward, then back
    to CPU) so only one block's weights need to be resident on GPU at a time.

    When ``disk_index`` is set (streaming mode -- the model is a meta-device skeleton),
    each block's real weights are materialized from the checkpoint right before use and
    released back to meta right after, both here (the manual reverse-order backward
    replay) and in ``prepare_model_low_gpu`` (the initial forward capture pass).
    """
    block_inputs = {}
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None

    block_names = get_block_names(model)[0]
    for name in block_names:
        module = get_module(model, name)
        module.orig_forward = module.forward

    captured_grad = None

    def backward_pre_hook(module, grad_input):
        """Hook executed before backward propagation."""
        nonlocal captured_grad
        captured_grad = grad_input
        get_current_device_manager().synchronize()
        raise MyCustomError("Interrupt backward pass")

    for batch_idx, data in enumerate(dataloader, start=1):
        if batch_idx <= skip_batches:
            # resumed run: these batches' contributions are already in the
            # restored accumulators; advance the loader and the progress bar only
            if pbar is not None:
                pbar.update(len(block_names) * 2)
            continue
        captured_grad = None
        interrupted = False
        last_block_backward_hook = None
        try:
            prepare_model_low_gpu(model, block_inputs, major_device=major_device, pbar=pbar, disk_index=disk_index)

            # lm_head sits outside every decoder block, so it never gets `grad_mode=True`
            # in the manual block-by-block backward below. Scope the fix narrowly to
            # just lm_head (rather than every non-block module) to avoid enabling grad
            # tracking / scoring hooks on unrelated out-of-block layers, which would
            # add extra autograd-graph memory for no benefit. The backward flow is:
            #   loss → lm_head (hook fires here) → norm → last_block (hook raises error)
            head_name = get_lm_head_name(model)
            if head_name is not None:
                # Once lm_head has been wrapped for scoring, `get_lm_head_name` resolves
                # to the inner original Linear (e.g. "lm_head.orig_layer") rather than
                # the wrapper itself ("lm_head") -- strip the suffix to reach the wrapper.
                head_name = head_name.removesuffix(".orig_layer")
                head_module = get_module(model, head_name)
                if hasattr(head_module, "grad_mode"):
                    head_module.grad_mode = True

            last_block = get_module(model, block_names[-1])
            last_block_backward_hook = last_block.register_full_backward_pre_hook(backward_pre_hook)

            data = to_device(data, model.device)
            # VLM datasets often already include ``labels``; LLM ones don't. Strip
            # any pre-existing ``labels`` from kwargs so we don't pass it twice.
            labels = data["labels"] if isinstance(data, dict) and "labels" in data else data["input_ids"]
            if isinstance(data, dict):
                data_for_forward = {k: v for k, v in data.items() if k != "labels"}
            else:
                data_for_forward = data
            # Route through the unified mllm forward so ``pixel_values`` /
            # ``images`` get cast to ``model.dtype`` (otherwise the vision tower
            # is silently bypassed on dtype mismatch and vision grad stays 0).
            output, _prepared = model_forward(model, data_for_forward, labels=labels, use_cache=False)
            clear_memory(device_list=major_device)
            memory_monitor.log_summary()

            try:
                output.loss.to(torch.float32).backward()
            except MyCustomError:
                interrupted = True
            if not interrupted or captured_grad is None:
                raise RuntimeError("AutoScheme failed to capture the last block gradient for replay")
            current_grad = captured_grad
        finally:
            if last_block_backward_hook is not None:
                last_block_backward_hook.remove()
            for name in block_names:
                module = get_module(model, name)
                module.forward = module.orig_forward

        del output, data

        # Manually compute gradients block by block
        for block_name in reversed(block_names):
            # Retrieve stored inputs for the block
            block_input_info = block_inputs.get(block_name, {})

            block_input_args = to_device(block_input_info.get("args", []), major_device)
            block_input_kwargs = to_device(block_input_info.get("kwargs", {}), major_device)
            replay_input = _prepare_replay_input(block_input_args, block_input_kwargs, block_name)

            # Move the block module to GPU
            block_module = get_module(model, block_name)
            for n, m in block_module.named_modules():
                if hasattr(m, "grad_mode"):
                    m.grad_mode = True
            materialized = False
            try:
                if disk_index is not None:
                    from auto_round.utils.disk_stream_util import materialize_module

                    materialize_module(block_module, block_name, disk_index, device=major_device)
                    materialized = True
                move_module_to_tuning_device(block_module, major_device=major_device)

                block_module.eval()
                block_output = block_module(*block_input_args, **block_input_kwargs)

                if isinstance(block_output, tuple):
                    main_output = block_output[0]
                    if isinstance(main_output, torch.Tensor) and main_output.is_floating_point():
                        main_output = main_output.requires_grad_(True)
                elif isinstance(block_output, torch.Tensor) and block_output.is_floating_point():
                    main_output = block_output.requires_grad_(True)
                else:
                    main_output = block_output

                torch.autograd.backward(
                    tensors=main_output,
                    grad_tensors=current_grad,
                    retain_graph=_replay_retain_graph(block_module),
                )

                if replay_input.grad is None:
                    logger.warning(f"No gradient found for input of {block_name}, stopping backward replay")
                    break
                current_grad = replay_input.grad.detach().clone()
            finally:
                for parameter in block_module.parameters():
                    parameter.grad = None
                _clear_wrapper_score_caches(block_module)
                if disk_index is not None and materialized:
                    from auto_round.utils.disk_stream_util import free_module

                    free_module(block_module)
                elif disk_index is None:
                    block_module.to("cpu")

            # clear_memory(device_list=major_device) # this one is very slow and seems does not affect max ram usage
            memory_monitor.update()

            if pbar is not None:
                pbar.update(1)

        _vram_inventory(f"scheme={scheme_tag} batch={batch_idx}/{total_batches} post-replay")

        if batch_checkpoint is not None:
            batch_checkpoint(batch_idx, total_batches or -1)

        _log_batch_avg_loss(
            model,
            batch_idx,
            pbar=pbar,
            block_names=block_names,
            total_batches=total_batches,
            scheme_tag=scheme_tag,
        )


def get_score_for_scheme(
    model,
    tokenizer,
    quant_layer_names,
    fixed_layer_scheme,
    dataset,
    ignore_scale_zp_bits=False,
    nsamples=16,
    seqlen=256,
    skip_batches=0,
    batch_checkpoint=None,
    pbar=None,
    shared_layers=None,
    need_weight_grad=False,
    enable_torch_compile=True,
    low_gpu_mem_usage=True,
    major_device="cpu",
    batch_size=1,
    offload_context: Optional[OffloadManager] = None,
    processor=None,
    is_vlm: bool = False,
    force_mllm: bool = False,
    model_name: Optional[str] = None,
    scheme_tag: Optional[str] = None,
    disk_index=None,
):
    """Wrap every quantizable layer in ``quant_layer_names`` with a scoring wrapper, run
    forward(+backward, unless RTN-only) calibration over ``nsamples`` examples from
    ``dataset``/``dataloader``, then unwrap and return each layer's ``[bits, loss]``.
    """
    scores_dict = {}  # Key=name,Val=[quant_total_bits, loss]
    # Include the visual block(s) when scoring VLMs with ``--quant_nontext_module``
    # (``force_mllm=True``) so vision-tower layer losses match a block below instead
    # of silently falling through to "non_block" in the logging/inactive-expert-fill
    # helpers. Mirrors the same ``quant_vision=force_mllm`` pattern used in
    # ``_gen_layer_config``.
    block_names = get_block_names(model, quant_vision=force_mllm)[0]
    for n, m in model.named_modules():
        if type(m) in SUPPORTED_LAYER_TYPES:
            m.weight.requires_grad = False
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = False

    has_imatrix = False
    for name in quant_layer_names:
        if name in fixed_layer_scheme.keys():
            continue
        m = get_module(model, name)
        if hasattr(m, "imatrix") and m.imatrix is not None:
            has_imatrix = True
            break

    for name in quant_layer_names:
        if offload_context is not None:
            offload_context.ensure_loaded(model, name)
        if name in fixed_layer_scheme.keys():
            continue
        m = get_module(model, name)
        if m is None:
            raise RuntimeError(f"AutoScheme scoring layer {name!r} is missing after model preprocessing")
        if not check_to_quantized(m):
            layer_bits, _ = compute_layer_bits(m, ignore_scale_zp_bits)
            scores_dict[name] = [layer_bits, 0.0]
            continue
        if m.act_bits > 8 and m.super_bits is not None:
            m.scale_dtype = torch.float32  # TODO set this via API
        elif m.act_bits > 8:
            m.scale_dtype = torch.float16
        else:
            m.scale_dtype = torch.bfloat16

        WrapperLayer = AutoSchemeWrapperLinear
        # if has_imatrix: # no better result
        #     WrapperLayer = AutoSchemeWrapperLinearIMatrix
        if hasattr(m, "super_group_size") and m.super_group_size is not None:
            if has_imatrix:
                WrapperLayer = AutoSchemeWrapperLinearForGGUFKImatrix
            else:
                WrapperLayer = AutoSchemeWrapperLinearForGGUFK

        with torch.no_grad():
            if low_gpu_mem_usage:
                device = m.tuning_device if hasattr(m, "tuning_device") else major_device
                # Any non-CPU device (cuda/xpu/hpu/...) is consolidated to the major device.
                if str(device).split(":")[0] not in ("cpu", "meta", "disk"):
                    device = major_device
            else:
                device = m.weight.device
            # Replacement materialization may create fresh expert Linear modules
            # without the metadata assigned on the pre-materialized tree.
            m.tuning_device = device

            new_m = WrapperLayer(
                m,
                device=device,
                enable_minmax_tuning=False,
                enable_norm_bias_tuning=False,
                enable_round_tuning=False,
                need_weight_grad=need_weight_grad,
                enable_torch_compile=enable_torch_compile,
            )
            set_module(model, name, new_m)
    if offload_context is not None:
        offload_context.flush_loaded(model)

    # ---- Memory: only wrapper.orig_layer.weight needs ``requires_grad`` ---- #
    # AutoScheme scoring uses ``iters=0`` (RTN), so we never UPDATE any
    # parameter. The only parameters that need to participate in autograd
    # are the wrappers' ``orig_layer.weight`` (so qdq_w in the backward
    # graph can trace back to them and the weight-grad hook fires).
    # All other parameters (norms, non-wrapped linears, vision-tower
    # layers when ``--quant_nontext_module`` is off, …) just waste a full
    # ``.grad`` buffer (~one model-worth of VRAM) during ``loss.backward()``.
    # This is the biggest single VRAM win for the non-low_gpu_mem_usage
    # path used to score VLMs.
    wrapper_weight_ids = set()
    for _, _m in model.named_modules():
        if hasattr(_m, "orig_layer") and hasattr(_m.orig_layer, "weight") and _m.orig_layer.weight is not None:
            wrapper_weight_ids.add(id(_m.orig_layer.weight))
    _trimmed = 0
    for _p in model.parameters():
        if id(_p) not in wrapper_weight_ids and _p.requires_grad:
            _p.requires_grad_(False)
            _trimmed += 1
    # if _trimmed:
    #     logger.info(
    #         "AutoScheme: disabled requires_grad on %d non-wrapper parameters "
    #         "(only wrapper.orig_layer.weight needs grad for scoring; saves "
    #         "~one model-worth of grad buffer during backward).",
    #         _trimmed,
    #     )

    # When scoring vision-tower layers, keep the autograd chain alive end-to-end:
    #   (1) every wrapper's orig weight must require grad — otherwise the STE
    #       output of W-only-low-bit wrappers has no grad path and act-score
    #       hooks never fire.
    #   (2) every vision sub-tree leaf param must require grad — so the very
    #       first vision op (patch_embed / first conv) enters autograd; its
    #       input ``pixel_values`` is a plain tensor with no grad.
    if force_mllm:
        _re_enabled_w = 0
        for _, _m in model.named_modules():
            if (
                hasattr(_m, "orig_layer")
                and hasattr(_m.orig_layer, "weight")
                and _m.orig_layer.weight is not None
                and not _m.orig_layer.weight.requires_grad
            ):
                _m.orig_layer.weight.requires_grad_(True)
                _re_enabled_w += 1

        _vision_markers = ("vision", "visual", "image_encoder", "img_encoder", "patch_embed")
        _re_enabled_v = 0
        _seen = set()
        for _mod_name, _mod in model.named_modules():
            if not any(mk in _mod_name.lower() for mk in _vision_markers):
                continue
            for _p in _mod.parameters(recurse=False):
                if id(_p) in _seen:
                    continue
                _seen.add(id(_p))
                if not _p.requires_grad:
                    _p.requires_grad_(True)
                    _re_enabled_v += 1

        logger.info(
            "AutoScheme(force_mllm): kept requires_grad on %d wrapper weights, " "%d vision-side params.",
            _re_enabled_w,
            _re_enabled_v,
        )

    def _build_calib_dataloader():
        """Pick the calibration dataloader.

        Since AutoScheme only scores the language tower (``get_block_names``
        already skips the vision/audio sub-trees on VLMs), a pure-text
        calibration dataset is sufficient and far cheaper for VLMs too — most
        VLMs accept a text-only forward and simply skip the vision encoder.
        We therefore use ``get_dataloader`` (text-only) by default and only
        fall back to the multimodal ``get_mllm_dataloader`` if a VLM truly
        rejects text-only inputs (caller can detect that in the calling loop).
        """
        return get_dataloader(tokenizer, seqlen, dataset_name=dataset, seed=42, bs=batch_size, nsamples=nsamples)

    def _build_mllm_calib_dataloader():
        """Build the multimodal calibration dataloader (image + text).

        Returns ``None`` if we can't build one (no processor / template /
        dataset issue) so the caller can surface a clearer error.
        """
        if processor is None:
            return None
        import os as _os

        from auto_round.compressors.mllm.dataset import MLLM_DATASET, get_mllm_dataloader

        template = None
        if hasattr(model, "config") and hasattr(model.config, "model_type"):
            template = model.config.model_type

        # Decide the effective dataset.
        # ``get_mllm_dataloader`` only treats ``dataset`` as multimodal when it
        # is either a local file OR a key registered in ``MLLM_DATASET``.
        # Otherwise it silently falls back to ``get_dataloader`` (text-only),
        # which produces batches with NO ``pixel_values`` -> the vision tower
        # is never invoked -> every vision score / grad collapses to 0. We
        # explicitly catch that case here and override to a known-good
        # multimodal dataset so the user doesn't end up with silent garbage.
        ds = dataset
        _is_real_mllm = isinstance(ds, str) and (_os.path.isfile(ds) or ds in MLLM_DATASET.keys())
        if not _is_real_mllm:
            _fallback = "liuhaotian/llava_conv_58k"
            logger.warning_once(
                "AutoScheme(force_mllm): dataset=%r is text-only, " "overriding to %r.",
                ds,
                _fallback,
            )
            ds = _fallback

        try:
            loader, _, _, _ = get_mllm_dataloader(
                template=template,
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                dataset=ds,
                seqlen=seqlen,
                bs=batch_size,
                nsamples=nsamples,
                # If, for any reason, get_mllm_dataloader still falls back to
                # text-only, force it to hard-error rather than silently
                # producing image-less batches.
                quant_nontext_module=True,
            )
            return loader
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to build mllm calibration dataloader: {exc}")
            return None

    if low_gpu_mem_usage:
        if force_mllm:
            mllm_loader = _build_mllm_calib_dataloader()
            if mllm_loader is None:
                raise RuntimeError(
                    "AutoScheme(force_mllm): cannot build mllm dataloader. "
                    "Provide a `processor` and a multimodal `dataset`."
                )
            model_forward_low_gpu(
                model,
                mllm_loader,
                major_device=major_device,
                pbar=pbar,
                scheme_tag=scheme_tag,
                disk_index=disk_index,
                skip_batches=skip_batches,
                batch_checkpoint=batch_checkpoint,
            )
        else:
            try:
                dataloader = _build_calib_dataloader()
                model_forward_low_gpu(
                    model,
                    dataloader,
                    major_device=major_device,
                    pbar=pbar,
                    scheme_tag=scheme_tag,
                    disk_index=disk_index,
                    skip_batches=skip_batches,
                    batch_checkpoint=batch_checkpoint,
                )
            except Exception as exc:  # noqa: BLE001
                if not is_vlm:
                    raise
                logger.warning(
                    f"Text-only calibration failed on VLM ({exc}); "
                    f"falling back to multimodal calibration dataloader."
                )
                mllm_loader = _build_mllm_calib_dataloader()
                batch_size = 1
                if mllm_loader is None:
                    raise
                model_forward_low_gpu(
                    model,
                    mllm_loader,
                    major_device=major_device,
                    pbar=pbar,
                    scheme_tag=scheme_tag,
                    disk_index=disk_index,
                )
    else:
        for n, m in model.named_modules():
            if hasattr(m, "grad_mode"):
                m.grad_mode = True
            # if hasattr(m, "post_init_qdqw"):
            #     m.post_init_qdqw()

        def _run_forward_loop(loader):
            """Run the full (non-low-GPU) forward+backward calibration loop over ``loader``,
            accumulating ``mix_score`` on every wrapped layer and periodically logging progress
            via ``_log_batch_avg_loss``.
            """
            total_batches = len(loader) if hasattr(loader, "__len__") else None
            _checked_pixel = False
            _pixel_keys = (
                "pixel_values",
                "pixel_values_videos",
                "pixel_values_images",
                "image_pixel_values",
                "images",
                "image",
            )
            for batch_idx, data in enumerate(loader, start=1):
                # Pull labels out of the batch (VLM datasets often carry them;
                # LLM ones don't) before mllm_model_forward casts dtypes.
                _src = data if isinstance(data, dict) else None
                labels = (
                    _src["labels"]
                    if _src is not None and "labels" in _src
                    else (_src["input_ids"] if _src is not None and "input_ids" in _src else None)
                )
                if _src is not None and "labels" in _src:
                    data_for_forward = {k: v for k, v in _src.items() if k != "labels"}
                else:
                    data_for_forward = data

                # Unified mllm-aware forward (casts pixel_values/images to
                # model.dtype, handles dict-with-text/str/tuple paths the same
                # way the multimodal compressor calibration does).
                output, _prepared = model_forward(model, data_for_forward, labels=labels, use_cache=False)
                output.loss.backward()

                # One-shot sanity check: when scoring vision layers, the batch
                # MUST carry image data, otherwise the vision tower is bypassed
                # by the model and every vision score is silently 0.
                if not _checked_pixel and force_mllm:
                    _checked_pixel = True
                    _has_pixel = isinstance(_prepared, dict) and any(k in _prepared for k in _pixel_keys)
                    if not _has_pixel:
                        _keys = list(_prepared.keys()) if isinstance(_prepared, dict) else type(_prepared).__name__
                        raise RuntimeError(
                            f"AutoScheme(force_mllm) batch has no pixel_values "
                            f"(keys: {_keys}). Vision scores would all be 0. "
                            f"Use a real multimodal dataset (e.g. "
                            f"liuhaotian/llava_conv_58k) and pass a processor."
                        )

                for _, m in model.named_parameters():  # zero grads to keep VRAM low
                    m.grad = None
                if pbar is not None:
                    pbar.update(1)
                _log_batch_avg_loss(
                    model,
                    batch_idx,
                    pbar=pbar,
                    block_names=block_names,
                    total_batches=total_batches,
                    scheme_tag=scheme_tag,
                )

        if force_mllm:
            mllm_loader = _build_mllm_calib_dataloader()
            if mllm_loader is None:
                raise RuntimeError(
                    "AutoScheme(force_mllm): cannot build mllm dataloader. "
                    "Provide a `processor` and a multimodal `dataset`."
                )
            _run_forward_loop(mllm_loader)
        else:
            try:
                _run_forward_loop(_build_calib_dataloader())
            except Exception as exc:  # noqa: BLE001
                if not is_vlm:
                    raise
                logger.warning(
                    f"Text-only calibration failed on VLM ({exc}); "
                    f"falling back to multimodal calibration dataloader."
                )
                mllm_loader = _build_mllm_calib_dataloader()
                if mllm_loader is None:
                    raise
                _run_forward_loop(mllm_loader)

        for n, m in model.named_parameters():
            m.grad = None

    for n, m in model.named_modules():
        if hasattr(m, "mix_score"):
            if m.orig_layer.act_bits <= 8:
                if m.act_cnt == 0:
                    logger.warning_once(
                        "layer{n} max abs activation is 0, please use more data to improve the accuracy"
                    )
            layer_bits, _ = compute_layer_bits(m.orig_layer, ignore_scale_zp_bits=ignore_scale_zp_bits)
            scores_dict[n] = [layer_bits, m.mix_score]
    _fill_inactive_expert_scores(scores_dict, block_names)
    _log_score_summary_by_block_and_nonblock(
        scores_dict,
        block_names,
        model=model,
        scheme_tag=scheme_tag,
        summary_stage="final",
    )

    for n, m in model.named_modules():
        if hasattr(m, "orig_layer"):
            # Explicitly break reference cycles to ensure GC can free the wrapper.
            # Hook closures capture `self` (wrapper), creating cycles:
            #   wrapper → qdq_w → _backward_hooks → closure → wrapper
            # PyTorch's C-level tensor storage prevents Python's cyclic GC from
            # collecting these without explicitly breaking the cycle first.
            if hasattr(m, "qdq_w") and m.qdq_w is not None:
                if hasattr(m.qdq_w, "_backward_hooks") and m.qdq_w._backward_hooks:
                    m.qdq_w._backward_hooks.clear()
                # Use detach_() rather than requires_grad_(False) because
                # block_module.to("cpu") may have turned qdq_w into a non-leaf
                # (ToCopyBackward grad_fn from .to()), and requires_grad_ only
                # works on leaf tensors.
                m.qdq_w.detach_()
                m.qdq_w = None
            if hasattr(m, "x_diff"):
                m.x_diff = None
            if hasattr(m, "super_qdq_func"):
                m.super_qdq_func = None
            if hasattr(m, "act_qdq_func"):
                m.act_qdq_func = None
            set_module(model, n, m.orig_layer)

    gc.collect()
    return scores_dict


def choose_bits_per_layer_with_path(layers: dict, P: int, max_states: int = None):
    """
    Args:
        layers: A dict mapping each layer name to a list of candidate options.
                Each option is a tuple of (scheme, bits_cost, loss_cost, layer_names).
        P: Upper bound on the total parameter (bit) budget.
        max_states: Maximum number of DP states to retain after each layer
                    (beam width). Limits memory usage for models with many
                    layers and incommensurate layer sizes.

    Returns:
        (min_loss, best_path), where best_path is a list of
        (layer_names, scheme) for each layer, or (None, None) if no feasible
        solution exists.
    """
    # dp: total_params -> (accumulated_loss, path_node)
    # Each path node points to its parent. Tuple/list concatenation still copies
    # the entire path on every transition, which becomes quadratic for large
    # models; linked nodes keep each transition O(1) and are expanded only once.
    dp: dict[int, tuple[float, tuple]] = {0: (0.0, ())}
    for layer_name, opts in layers.items():
        new_dp: dict[int, tuple[float, tuple]] = {}
        for cur_params, (cur_loss, cur_path) in dp.items():
            for opt in opts:
                scheme, bits_cost, loss_cost, layer_names = opt
                np_total = cur_params + bits_cost
                if np_total > P:
                    continue

                new_loss = cur_loss + loss_cost
                new_path = (cur_path, layer_names, scheme)

                # Keep the path with smaller loss for the same parameter budget
                if np_total not in new_dp or new_loss < new_dp[np_total][0]:
                    new_dp[np_total] = (new_loss, new_path)

        if not new_dp:
            return None, None
        # Pareto pruning: remove dominated (params, loss) states
        items = sorted(new_dp.items(), key=lambda x: x[0])  # (params, (loss, path))
        pruned: dict[int, tuple[float, tuple]] = {}
        best_loss_so_far = float("inf")
        for params_val, (loss_val, path_val) in items:
            if loss_val < best_loss_so_far:
                pruned[params_val] = (loss_val, path_val)
                best_loss_so_far = loss_val

        # Beam width limit: if too many states survive Pareto pruning,
        # uniformly subsample to bound memory usage. For models with many
        # layers whose sizes are incommensurate, the number of distinct
        # cumulative-bit sums can grow to millions, each storing a full
        # path copy — easily exceeding 70 GB of RAM.
        if max_states is not None and len(pruned) > max_states:
            if max_states <= 1:
                best_k = min(pruned.keys(), key=lambda k: pruned[k][0])
                pruned = {best_k: pruned[best_k]}
            else:
                sorted_keys = sorted(pruned.keys())
                n = len(sorted_keys)
                # Uniformly pick max_states indices (always include first and last)
                step = (n - 1) / (max_states - 1)
                selected: dict[int, tuple[float, tuple]] = {}
                for i in range(max_states):
                    idx = int(round(i * step))
                    if idx >= n:
                        idx = n - 1
                    k = sorted_keys[idx]
                    selected[k] = pruned[k]
                pruned = selected

        dp = pruned

    # Select the solution with the minimum loss
    best_params = min(dp.keys(), key=lambda k: dp[k][0])
    best_loss, best_path = dp[best_params]
    path = []
    while best_path:
        best_path, layer_names, scheme = best_path
        path.append((layer_names, scheme))
    path.reverse()
    return best_loss, path


def move_module_to_tuning_device(module, major_device="cpu"):
    """Move every submodule of ``module`` to its own tuning device: wrapper submodules go to
    ``orig_layer.tuning_device``/``tuning_device`` (set per-layer earlier), leaf modules with
    no such attribute fall back to ``major_device``, and any directly-held parameters/buffers
    (not just the standard ``.to()`` targets) are relocated along with their ``.grad``.
    """

    def _normalize(dev):
        """Coerce ``dev`` (str or ``torch.device``) into a ``torch.device``."""
        return dev if isinstance(dev, torch.device) else torch.device(dev)

    def _move_own_tensors(m, device):
        """Move ``m``'s directly-owned (non-recursive) parameters/buffers (and their
        ``.grad``) to ``device``.
        """
        # Cover non-leaf modules that directly hold nn.Parameter / buffers
        # (e.g. Mamba/GDN linear_attn with A_log & dt_bias). Also relocate
        # p.grad together with p.data — otherwise the next backward's grad
        # accumulation hits a cuda/cpu device mismatch.
        target = _normalize(device)
        for p in m.parameters(recurse=False):
            if p.device != target:
                p.data = p.data.to(target)
            if p.grad is not None and p.grad.device != target:
                p.grad.data = p.grad.data.to(target)
        for b_name, b in list(m.named_buffers(recurse=False)):
            if b is None:
                continue
            if b.device != target:
                m._buffers[b_name] = b.to(target)

    for n, m in module.named_modules():
        if hasattr(m, "orig_layer"):
            target = getattr(m.orig_layer, "tuning_device", getattr(m, "tuning_device", major_device))
            m.to(target)
            _move_own_tensors(m, target)
        elif hasattr(m, "tuning_device"):
            target = m.tuning_device
            m.to(target)
            _move_own_tensors(m, target)
        elif len(list(m.children())) == 0:
            m.to(major_device)
            _move_own_tensors(m, major_device)
        else:
            _move_own_tensors(m, major_device)


def _get_scheme_bits(scheme):
    """Extract the weight bits from a scheme (str or dict)."""
    if isinstance(scheme, str):
        scheme = asdict(preset_name_to_scheme(scheme))
    elif isinstance(scheme, QuantizationScheme):
        scheme = asdict(scheme)
    return scheme.get("bits", 16)


def _get_next_scheme_bits(schemes, indices, floor_bits):
    """Return the smallest candidate bit width strictly above ``floor_bits``."""
    higher_bits = {
        _get_scheme_bits(schemes[index]) for index in indices if _get_scheme_bits(schemes[index]) > floor_bits
    }
    return min(higher_bits, default=None)


# Delta loss does not handle lm-head well, it is prone to assign low bit to lm-head which is not optimal
def _apply_head_trick(head_name, schemes, sorted_indices, target_bits, target_params_cnt, total_scores):

    # ------------------------------------------------------------------ #
    # lm_head option restriction for DP                                   #
    # lm_head is critical — its quantization error goes directly into     #
    # logits with no subsequent LayerNorm dampening. Instead of removing  #
    # it from DP, we bias its candidate options toward higher precision   #
    # or lower loss, then relax the restriction if it cannot fit budget.  #
    #                                                                      #
    # Rules (only if user hasn't already fixed it):                        #
    #   1. No option has bits >= 6      → prefer lowest-loss available    #
    #   2. Exactly one option bits >= 6 → prefer that high-bit option     #
    #   3. Multiple options bits >= 6:                                      #
    #      - target_bits > 6  → restrict to only the highest-bit option   #
    #      - target_bits <= 6 → keep all >=6 options, let DP decide       #
    #   Any restriction above is relaxed if it makes the budget infeasible.#
    # ------------------------------------------------------------------ #

    high_bit_indices = [i for i in range(len(schemes)) if _get_scheme_bits(schemes[i]) >= 6]

    if len(high_bit_indices) == 0:
        # Rule 1: no option >= 6 bit → keep the lowest-loss scheme if budget allows.
        allowed_indices = {sorted_indices[0]} if sorted_indices else None
    elif len(high_bit_indices) == 1:
        # Rule 2: exactly one >= 6 bit option → restrict to it
        allowed_indices = set(high_bit_indices)
    else:
        # Rule 3: multiple >= 6 bit options
        if target_bits > 6:
            # Restrict to only the highest-bit option
            highest_idx = max(high_bit_indices, key=lambda i: _get_scheme_bits(schemes[i]))
            allowed_indices = {highest_idx}
        else:
            # Keep all >= 6 bit options, let DP decide among them
            allowed_indices = set(high_bit_indices)

    # Feasibility check: ensure the restricted lm_head options + min bits
    # for all other layers don't exceed the budget. If infeasible, relax
    # by adding options from sorted_indices (lowest loss first) until
    # a feasible combination exists.
    if allowed_indices is not None:
        # Compute budget remaining after fixed layers

        _remaining_budget = target_params_cnt

        # Compute min bits for non-lm_head DP layers
        _min_other_bits = 0
        for key, opts in total_scores.items():
            if key != head_name:
                _min_other_bits += min(opt[1] for opt in opts)

        # Compute min bits for lm_head under allowed_indices
        _min_head_bits = 0

        if head_name in total_scores:
            head_opts = [opt for opt in total_scores[head_name] if opt[0] in allowed_indices]
            if head_opts:
                _min_head_bits += min(opt[1] for opt in head_opts)
            else:
                _min_head_bits += min(opt[1] for opt in total_scores[head_name])

        # If infeasible, relax by adding cheaper options from sorted_indices
        if _min_head_bits + _min_other_bits > _remaining_budget:
            for fallback_idx in sorted_indices:
                if fallback_idx in allowed_indices:
                    continue
                allowed_indices.add(fallback_idx)
                # Recompute min head bits with expanded options
                _min_head_bits = 0

                if head_name in total_scores:
                    head_opts = [opt for opt in total_scores[head_name] if opt[0] in allowed_indices]
                    if head_opts:
                        _min_head_bits += min(opt[1] for opt in head_opts)
                    else:
                        _min_head_bits += min(opt[1] for opt in total_scores[head_name])
                if _min_head_bits + _min_other_bits <= _remaining_budget:
                    break

    # Filter lm_head's entries in total_scores to only allowed options
    if allowed_indices is not None:
        if head_name in total_scores:
            filtered = [opt for opt in total_scores[head_name] if opt[0] in allowed_indices]
            if filtered:
                total_scores[head_name] = filtered


# ---------------------------------------------------------------------------
# AutoScheme scoring cache helpers
# ---------------------------------------------------------------------------


def _scheme_repr(s):
    """Normalize a scheme to a stable representation independent of preset aliases."""
    if isinstance(s, str):
        try:
            s = preset_name_to_scheme(s)
        except KeyError:
            return s.upper()
    if isinstance(s, QuantizationScheme):
        s = asdict(s)
    if isinstance(s, dict):
        return {key: value for key, value in sorted(s.items()) if value is not None}
    return str(s)


def _stable_model_id(model_name):
    """Return a portable model identifier for local paths and Hub model IDs."""
    if not isinstance(model_name, str):
        return model_name
    normalized = model_name.rstrip("/\\")
    return os.path.basename(normalized) or normalized


def _autoscheme_cache_config(
    model_name,
    dataset,
    nsamples,
    seqlen,
    batch_size,
    quant_layer_names,
    fixed_layer_scheme,
    scheme,
    force_mllm,
    low_gpu_mem_usage,
    need_weight_grad=False,
):
    """Build the portable, implementation-independent identity of a scoring run."""
    return {
        "model_id": _stable_model_id(model_name),
        "dataset": dataset,
        "nsamples": nsamples,
        "seqlen": seqlen,
        "batch_size": batch_size,
        "quant_layer_names": sorted(quant_layer_names),
        "fixed_layer_scheme": {key: _scheme_repr(value) for key, value in sorted(fixed_layer_scheme.items())},
        "scheme": _scheme_repr(scheme),
        "force_mllm": force_mllm,
        "low_gpu_mem_usage": low_gpu_mem_usage,
        "need_weight_grad": need_weight_grad,
    }


def _autoscheme_cache_key(
    model_name,
    dataset,
    nsamples,
    seqlen,
    batch_size,
    quant_layer_names,
    fixed_layer_scheme,
    scheme,
    force_mllm,
    low_gpu_mem_usage,
    need_weight_grad=False,
):
    """Return a 16-char hex digest that uniquely identifies a **single-scheme** scoring run.

    The key covers every parameter that directly affects per-layer loss values
    **except** ``avg_bits`` / ``target_bits`` (only drive the DP step).
    Unlike the old version, this key is generated **per-scheme**, not per-run,
    so caching is granular: adding/removing schemes doesn't invalidate cached
    scores for unchanged schemes.
    """
    key_data = _autoscheme_cache_config(
        model_name,
        dataset,
        nsamples,
        seqlen,
        batch_size,
        quant_layer_names,
        fixed_layer_scheme,
        scheme,
        force_mllm,
        low_gpu_mem_usage,
        need_weight_grad,
    )
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _autoscheme_cache_path(cache_key, scheme_index):
    """Return the full path to the JSON cache file for a **single scheme**.

    Each scheme gets its own cache file under ``AR_AUTO_SCHEME_CACHE`` or the
    default ``~/.cache/auto_round`` directory
    to enable granular reuse: adding/removing schemes or changing non-scoring
    parameters (e.g., target_bits) doesn't invalidate caches for unmodified schemes.

    Args:
        cache_key: Per-scheme cache key (includes model, dataset, scheme, etc.)
        scheme_index: Index of the scheme (for human readability in filenames)
    """
    from auto_round import envs as _envs

    cache_dir = os.path.expanduser(_envs.AR_AUTO_SCHEME_CACHE or "~/.cache/auto_round")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"scheme_{scheme_index:02d}_{cache_key}.json")


def _extract_score_accumulators(model):
    """Snapshot the additive per-layer scoring accumulators for mid-scheme resume."""
    state = {}
    for n, m in model.named_modules():
        if hasattr(m, "mix_score"):
            state[n] = [float(m.act_score), float(m.weight_score), float(m.act_cnt)]
    return state


def _inject_score_accumulators(model, state):
    """Restore scoring accumulators snapshotted by ``_extract_score_accumulators``."""
    for n, m in model.named_modules():
        if n in state and hasattr(m, "mix_score"):
            m.act_score, m.weight_score, m.act_cnt = state[n]


def _partial_scores_path(cache_path):
    return cache_path + ".partial"


def _save_partial_scores(cache_path, batch_idx, total_batches, state):
    """Atomically persist a batch-granularity scoring checkpoint (JSON)."""
    payload = {
        "version": 1,
        "batches_done": batch_idx,
        "total_batches": total_batches,
        "scores_state": state,
    }
    tmp = _partial_scores_path(cache_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, _partial_scores_path(cache_path))


def _load_partial_scores(cache_path, expected_total_batches):
    """Load a batch checkpoint if present and still applicable."""
    path = _partial_scores_path(cache_path)
    if cache_path is None or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("total_batches") != expected_total_batches:
            return None
        if not 0 < data.get("batches_done", 0) < expected_total_batches:
            return None
        return data
    except Exception:  # noqa: BLE001
        return None


def _save_autoscheme_scores(
    cache_path,
    cache_key,
    scheme_index,
    scheme_dict,
    layer_scores,
    total_loss_for_scheme,
    total_params,
    cache_config,
):
    """Persist scoring results for **a single scheme** to *cache_path* as JSON.

    Each scheme's scores are stored independently so that adding/removing schemes
    or changing unrelated parameters (e.g., target_bits) doesn't invalidate cached
    scores for unchanged schemes.

        Schema version 1 (portable per-scheme, per-op cache)::

        {
                    "version": 1,
          "score_granularity": "per_op",
          "cache_key": "<hex>",
                    "cache_config": { ... scoring inputs ... },
          "scheme_index": 0,
          "scheme": { ... scheme dict ... },
          "created_at": "<ISO datetime>",
          "layer_scores": { layer_key: [bits, loss], ... },
          "total_loss_for_scheme": 1.234,
          "total_params": 12345
        }
    """
    if cache_path is None or cache_key is None:
        return
    # Persist only per-layer independent scores. Grouping (e.g. shared_layers
    # or MoE expert groups) is intentionally NOT stored here — callers should
    # re-apply grouping when loading a cache so the on-disk format stays
    # per-op and backward/forward compatible.
    data = {
        "version": 1,
        "score_granularity": "per_op",
        "cache_key": cache_key,
        "cache_config": cache_config,
        "scheme_index": scheme_index,
        "scheme": scheme_dict,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "layer_scores": layer_scores,
        "total_loss_for_scheme": total_loss_for_scheme,
        "total_params": total_params,
    }
    try:
        with open(cache_path, "w", encoding="utf-8") as _f:
            json.dump(data, _f, indent=2, default=str)
        logger.info("AutoScheme: per-scheme cache saved → %s", cache_path)
    except Exception as _exc:  # noqa: BLE001
        logger.warning("AutoScheme: failed to save per-scheme cache: %s", _exc)


def _load_autoscheme_scores(cache_path):
    """Load and validate a **single-scheme** per-op scoring cache file (version 1).

    Returns the parsed dict with keys ``layer_scores``, ``total_loss_for_scheme``,
    and ``total_params`` on success, or ``None`` if the file is missing, malformed,
    or fails the version sanity check.
    """
    _required = ("cache_config", "layer_scores", "total_loss_for_scheme", "total_params")
    try:
        with open(cache_path, encoding="utf-8") as _f:
            data = json.load(_f)
        if data.get("version") != 1 or data.get("score_granularity") != "per_op":
            logger.warning(
                "AutoScheme: per-scheme cache schema mismatch "
                "(expected version=1, score_granularity=per_op; got version=%s, score_granularity=%s)",
                data.get("version"),
                data.get("score_granularity"),
            )
            return None
        for _k in _required:
            if _k not in data:
                logger.warning("AutoScheme: per-scheme cache missing required field %s", _k)
                return None
        return data
    except Exception as _exc:  # noqa: BLE001
        logger.warning("AutoScheme: failed to read per-scheme cache %s: %s", cache_path, _exc)
        return None


def _is_per_op_cache_compatible(cached_data, quant_layer_names, fixed_layer_scheme):
    """Return whether a cache contains exactly one score for every non-fixed quant layer."""
    expected_layers = set(quant_layer_names) - set(fixed_layer_scheme)
    return set(cached_data["layer_scores"]) == expected_layers


def _find_compatible_autoscheme_cache(
    expected_path,
    cache_config,
    quant_layer_names,
    fixed_layer_scheme,
    total_params,
):
    """Find a compatible cache even when a downloaded JSON has a different filename."""
    candidates = [expected_path]
    cache_dir = os.path.dirname(expected_path)
    try:
        candidates.extend(
            os.path.join(cache_dir, filename)
            for filename in sorted(os.listdir(cache_dir))
            if filename.endswith(".json") and os.path.join(cache_dir, filename) != expected_path
        )
    except OSError:
        pass

    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        cached_data = _load_autoscheme_scores(candidate)
        if cached_data is None or not _is_per_op_cache_compatible(cached_data, quant_layer_names, fixed_layer_scheme):
            continue
        if cached_data["cache_config"] != cache_config:
            continue
        if cached_data.get("total_params") != total_params:
            continue
        cached_data["_cache_path"] = candidate
        if candidate != expected_path:
            logger.info("AutoScheme: using compatible downloaded cache %s", candidate)
        return cached_data
    return None


def _refresh_cached_layer_bits(
    model,
    quant_layer_names,
    fixed_layer_scheme,
    scheme,
    cached_layer_scores,
    ignore_scale_zp_bits,
):
    """Recompute bit costs for the current accounting mode while preserving cached losses."""
    apply_quant_scheme(
        model,
        quant_layer_names=quant_layer_names,
        fixed_layer_scheme=fixed_layer_scheme,
        scheme=scheme,
    )
    refreshed_scores = {}
    for name, (_, loss) in cached_layer_scores.items():
        bits, _ = compute_layer_bits(get_module(model, name), ignore_scale_zp_bits)
        refreshed_scores[name] = [bits, loss]
    return refreshed_scores


def _parallel_scoring_must_raise(parallel_error: Exception) -> bool:
    """Decide whether a parallel-scoring failure aborts instead of falling back to serial.

    Set AR_AUTO_SCHEME_NO_SERIAL_FALLBACK=1 to fail fast: the serial fallback is
    ~workers-count times slower and, on models where it cannot run at all, only
    burns hours before crashing.  Schemes lost with their worker always raise
    regardless of the env, since their scores are unrecoverable.
    """
    from auto_round import envs as _envs

    if "was lost with its worker" in str(parallel_error):
        return True
    return bool(_envs.AR_AUTO_SCHEME_NO_SERIAL_FALLBACK)


def _assign_scheme_worker_devices(worker_count, available_devices):
    """Assign workers round-robin within the devices selected by the caller."""
    if not available_devices:
        raise ValueError("available_devices must contain at least one device")
    return [available_devices[worker_index % len(available_devices)] for worker_index in range(worker_count)]


class _ProgressQueueProxy:
    """Forward worker progress events to the parent process that owns tqdm."""

    def __init__(self, progress_queue):
        self.progress_queue = progress_queue

    def update(self, steps=1):
        self.progress_queue.put(("update", steps))

    def write(self, message):
        self.progress_queue.put(("write", message))


def _drain_progress_queue(progress_queue, pbar):
    """Apply all currently queued worker progress events to the parent tqdm instance."""
    import queue

    while True:
        try:
            event, payload = progress_queue.get_nowait()
        except queue.Empty:
            break
        if event == "update":
            pbar.update(payload)
        elif event == "write":
            pbar.write(payload)


def _get_worker_memory_report(worker_device):
    """Return this worker's peak RAM and VRAM for its assigned logical device."""
    memory_monitor.update(device_list=worker_device)
    device_key = str(worker_device).split(":")[-1]
    return {
        "device": device_key,
        "peak_ram": memory_monitor.peak_ram,
        "peak_vram": memory_monitor.peak_vram.get(device_key, 0.0),
    }


def _merge_worker_memory_reports(monitor, reports):
    """Merge child-process RAM/VRAM peaks into the parent monitor."""
    # Worker reports are process-local. Sum them to estimate the concurrent
    # worker peak, then add the parent RSS after workers have exited. The
    # parent's sampled peak may already contain the full live process tree,
    # so retain whichever aggregate is larger rather than adding both.
    worker_peak_ram = sum(report.get("peak_ram", 0.0) for report in reports)
    parent_ram = monitor._process_tree_rss() if hasattr(monitor, "_process_tree_rss") else monitor.peak_ram
    monitor.peak_ram = max(monitor.peak_ram, parent_ram + worker_peak_ram)

    worker_peaks = {}
    for report in reports:
        device = str(report["device"])
        worker_peaks[device] = worker_peaks.get(device, 0.0) + report["peak_vram"]
    for device, peak_vram in worker_peaks.items():
        monitor.peak_vram[device] = max(monitor.peak_vram.get(device, 0.0), peak_vram)


def _get_scheme_worker_count(num_schemes, num_gpus):
    """Use one worker per scoring scheme; workers may share a visible GPU."""
    if num_gpus < 1:
        raise ValueError("AutoScheme multiprocessing requires at least one GPU")
    return num_schemes


def _weights_span_multiple_gpus(devices) -> bool:
    """Whether a device collection places weights on more than one GPU."""
    cuda_devices = {str(device) for device in devices if str(device).startswith("cuda")}
    return len(cuda_devices) > 1


def _serial_scoring_device_safe(model, visible_cuda_devices=None) -> bool:
    """Whether in-process (serial) full-model scoring can run on ``model``.

    Serial scoring runs the whole-model forward through per-layer wrappers.
    Two placement sources can spread work over several GPUs: live parameter
    devices (a materialized model), and ``hf_device_map`` - which is what a
    disk-stream (meta-skeleton) model consults when its blocks materialize
    block-by-block during scoring. Either spanning more than one GPU makes
    the first cross-device layer raise a tensor-device error mid-forward.

    Additionally, a disk-stream (meta-skeleton) model scored serially with
    more than one visible CUDA device is not considered safe: block
    materialization and wrapper bookkeeping are only validated with a
    single visible GPU, and workers pin one GPU each. Such models are
    routed through disk-stream workers instead.
    """
    if _weights_span_multiple_gpus({parameter.device for parameter in model.parameters()}):
        return False
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        if _weights_span_multiple_gpus(set(hf_device_map.values())):
            return False
    if visible_cuda_devices is not None and len(visible_cuda_devices) > 1:
        if getattr(model, "_disk_stream_index", None) is not None:
            return False
    return True


def _can_parallel_scheme_scoring(
    parallel_enabled,
    model_id,
    num_gpus,
    uncached_count,
    need_imatrix,
    disk_stream_model,
    is_vlm,
    low_gpu_mem_usage=True,
    force_mllm=False,
    min_uncached=2,
):
    """Return whether candidate schemes can be scored in separate workers.

    ``is_vlm`` is accepted for call-site compatibility; text-only scoring of a
    VLM's language tower streams blocks just like a text model. Only vision
    scoring (``force_mllm``) requires a full-model backward and is excluded
    from the block-wise materialize/free path used by disk streaming.
    ``min_uncached`` is the parallel-worthiness floor (2 by default); the call
    site lowers it to 1 when serial scoring cannot run on the loaded model and
    a disk-stream worker can."""
    return (
        parallel_enabled
        and low_gpu_mem_usage
        and model_id is not None
        and num_gpus >= 1
        and uncached_count >= min_uncached
        and not need_imatrix
        and (not disk_stream_model or not force_mllm)
    )


def _load_scheme_worker_model(model_name, use_model_replacements, low_cpu_mem_usage):
    """Load an isolated worker model without an extra full-size CPU initialization copy."""
    return load_model(
        model_name,
        device="cpu",
        use_auto_mapping=False,
        use_model_replacements=use_model_replacements,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )


def _load_disk_stream_scheme_worker_model(model_name, use_model_replacements=False):
    """Build an isolated meta model and checkpoint index for a scoring worker."""
    from auto_round.utils.disk_stream_util import build_meta_model

    model, tokenizer, disk_index = build_meta_model(model_name)
    # The regular load pipeline applies custom replacements, among them the
    # structural MoE unfusing that turns fused expert containers into per-expert
    # Linear modules; the meta-skeleton build skips that pipeline.  Unfuse here
    # so per-expert quant layers resolve -- weights stay on meta and are filled
    # per-block from the checkpoint index during scoring.
    from auto_round.modeling.fused_moe.replace_modules import _handle_moe_modules

    unfused = _handle_moe_modules(model)
    logger.info("disk-stream scoring worker: structural MoE unfuse produced %d unfused experts modules", len(unfused))
    if use_model_replacements:
        from auto_round.special_model_handler import _handle_special_model, update_module

        model = update_module(model, formats=None, cleanup_original=False)
        model = _handle_special_model(model)
    return model, tokenizer, disk_index


def _prefer_disk_stream_scheme_worker(model_id, is_vlm, low_gpu_mem_usage):
    """Prefer block-wise disk streaming whenever the worker scoring path supports it.

    Multimodal archs are covered: build_meta_model resolves the class from
    config.architectures and the workers materialize non-block params (vision
    tower included) via materialize_non_block_params."""
    return model_id is not None and low_gpu_mem_usage


def _score_scheme_worker(args):
    """Score one scheme and return its index, scores, and worker VRAM peak."""
    import traceback as _tb

    (
        index,
        scheme,
        model_name,
        is_vlm,
        quant_layer_names,
        fixed_layer_scheme,
        dataset,
        nsamples,
        seqlen,
        batch_size,
        need_weight_grad,
        enable_torch_compile,
        low_cpu_mem_usage,
        low_gpu_mem_usage,
        ignore_scale_zp_bits,
        force_mllm,
        use_model_replacements,
        worker_device,
        total_schemes,
        progress_queue,
        disk_stream_model,
        worker_cache_path,
    ) = args

    from auto_round.auto_scheme.utils import _scheme_short_name as _short_name
    from auto_round.auto_scheme.utils import apply_quant_scheme as _apply_quant_scheme
    from auto_round.auto_scheme.utils import compute_layer_bits as _compute_layer_bits
    from auto_round.utils import get_block_names as _get_block_names
    from auto_round.utils import get_module as _get_module

    disk_index = None
    if disk_stream_model:
        try:
            model, tokenizer, disk_index = _load_disk_stream_scheme_worker_model(
                model_name, use_model_replacements=use_model_replacements
            )
            processor = None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_score_scheme_worker[%d]: disk-stream load failed for %r (%s); falling back to regular loading.",
                index,
                model_name,
                exc,
            )
            disk_index = None

    if disk_index is None:
        try:
            model, tokenizer, processor, _, _, is_vlm, _ = _load_scheme_worker_model(
                model_name,
                use_model_replacements,
                low_cpu_mem_usage,
            )
        except Exception as exc:
            raise RuntimeError(
                f"_score_scheme_worker[{index}]: failed to load model {model_name!r}\n{_tb.format_exc()}"
            ) from exc

    safe_to_cpu_(model)
    block_names = _get_block_names(model, quant_vision=force_mllm)[0]
    if disk_index is not None:
        from auto_round.utils.disk_stream_util import materialize_non_block_params

        materialize_non_block_params(model, block_names, disk_index, device="cpu")
    for block_name in block_names:
        block = _get_module(model, block_name)
        block.in_block = True
        for _, module in block.named_modules():
            module.in_block = True
    for _, module in model.named_modules():
        if len(list(module.children())) == 0:
            if not hasattr(module, "in_block"):
                module.in_block = False
            if not module.in_block and low_gpu_mem_usage:
                module.to(worker_device)

    from auto_round.modeling.fused_moe.replace_modules import materialize_model_

    if disk_index is None:
        materialize_model_(model)
    # MoE materialization can replace fused expert modules with newly-created
    # Linear layers. Assign tuning metadata only after the final module tree exists.
    for layer_name in quant_layer_names:
        layer = _get_module(model, layer_name)
        if layer is None:
            parent_name = layer_name.rsplit(".", 1)[0]
            parent = _get_module(model, parent_name)
            expert_containers = sum(1 for _, m in model.named_modules() if m.__class__.__name__ == "_ExpertContainer")
            raise RuntimeError(
                f"_score_scheme_worker[{index}]: layer {layer_name!r} is missing after model preprocessing "
                f"(parent {parent_name!r} resolves to {type(parent).__name__ if parent is not None else None}; "
                f"expert containers in model: {expert_containers}; disk-stream build: {disk_index is not None})"
            )
        if not hasattr(layer, "tuning_device"):
            layer.tuning_device = worker_device
        layer.tmp_name = layer_name

    _apply_quant_scheme(
        model,
        quant_layer_names=quant_layer_names,
        fixed_layer_scheme=fixed_layer_scheme,
        scheme=scheme,
    )

    # mid-scheme resume: reload additive accumulators and skip completed batches
    skip_batches = 0
    if worker_cache_path is not None:
        expected_total = max(1, math.ceil(nsamples / max(1, batch_size)))
        partial = _load_partial_scores(worker_cache_path, expected_total)
        if partial is not None:
            _inject_score_accumulators(model, partial["scores_state"])
            skip_batches = partial["batches_done"]
            logger.info(
                "_score_scheme_worker[%d]: resuming from batch checkpoint %d/%d",
                index,
                skip_batches,
                expected_total,
            )

        def batch_checkpoint(batch_idx, total_batches):
            _save_partial_scores(worker_cache_path, batch_idx, total_batches, _extract_score_accumulators(model))

    else:
        batch_checkpoint = None

    is_bf16 = isinstance(scheme, str) and scheme.upper() == "BF16"
    if isinstance(scheme, dict):
        is_bf16 = scheme.get("bits", 16) >= 16 and scheme.get("act_bits", 16) >= 16
    if is_bf16:
        scores = {}
        for layer_name in quant_layer_names:
            if layer_name in fixed_layer_scheme:
                continue
            layer_bits, _ = _compute_layer_bits(_get_module(model, layer_name), ignore_scale_zp_bits)
            scores[layer_name] = [layer_bits, 0.0]
        return index, scores, _get_worker_memory_report(worker_device)

    from auto_round.auto_scheme.delta_loss import get_score_for_scheme

    try:
        scores = get_score_for_scheme(
            model,
            tokenizer,
            quant_layer_names,
            fixed_layer_scheme,
            dataset,
            ignore_scale_zp_bits=ignore_scale_zp_bits,
            pbar=_ProgressQueueProxy(progress_queue),
            nsamples=nsamples,
            seqlen=seqlen,
            skip_batches=skip_batches,
            batch_checkpoint=batch_checkpoint,
            need_weight_grad=need_weight_grad,
            enable_torch_compile=enable_torch_compile,
            low_gpu_mem_usage=low_gpu_mem_usage,
            major_device=worker_device,
            batch_size=batch_size,
            offload_context=None,
            processor=processor,
            is_vlm=is_vlm,
            force_mllm=force_mllm,
            model_name=model_name,
            scheme_tag=f"{index + 1}/{total_schemes} {_short_name(scheme)}",
            disk_index=disk_index,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        raise _annotate_worker_oom(index, exc) from exc
    return index, scores, _get_worker_memory_report(worker_device)


def _gen_layer_config(
    auto_scheme: AutoScheme,
    model: Union[str, torch.nn.Module],
    quant_layer_names: Iterable[str],
    fixed_layer_scheme: dict[str, dict],
    min_avg_bit_scheme,
    dataset: str = "pile-10k",
    tokenizer=None,
    device_map=None,
    enable_torch_compile=True,
    model_name=None,
    major_device="cpu",
    device_list=None,
    processor=None,
    is_vlm: bool = False,
    disk_index=None,
):
    """Score every candidate scheme in ``auto_scheme.options`` against ``quant_layer_names``
    and return per-layer per-scheme losses used by the caller to pick a final bit-width
    assignment (via the DP knapsack in ``choose_bits_per_layer_with_path``).

    For each scheme: wraps every quantizable layer with a scoring wrapper, runs
    forward+backward calibration to accumulate ``mix_score`` (weight + activation loss), then
    unwraps and records the result before moving to the next scheme.
    """
    # Initialize memory tracking for AutoScheme
    memory_monitor = MemoryMonitor()
    # memory_monitor.reset()
    memory_monitor.update_cpu()

    # Create offload context for CPU RAM optimization
    # Note: low_cpu_mem_usage only works when low_gpu_mem_usage is also enabled,
    # because it requires layer-by-layer processing
    #
    # When disk_index is set, gen_layer_config already built the model as a
    # meta-device skeleton and materialize_module/free_module (called directly
    # around each block's use, see get_score_for_scheme/model_forward_low_gpu/
    # prepare_model_low_gpu above) are the actual streaming mechanism --
    # OffloadManager's hook-based approach doesn't apply to a model that never
    # had real CPU-resident weights to begin with.
    offload_context = None
    if disk_index is None and auto_scheme.low_cpu_mem_usage and auto_scheme.low_gpu_mem_usage:
        _model_dir = model_name
        if _model_dir is None and hasattr(model, "config"):
            _model_dir = getattr(model.config, "_name_or_path", None)
        offload_mode = "clean"
        offload_kwargs = {"model_dir": _model_dir}
        # Rotation mutates weights in memory before AutoScheme starts. Clean-mode
        # reloads from the original checkpoint and would silently discard those
        # transformed weights during scoring and final restore.
        if getattr(model, "_rotation_config", None):
            offload_mode = "offload"
            offload_kwargs = {"offload_dir_prefix": "autoscheme", "retain_saved_entries": True}
        offload_context = OffloadManager(enabled=True, mode=offload_mode, cache_numel=True, **offload_kwargs)

    target_bits = auto_scheme.avg_bits
    # HF gates gradient checkpointing on ``self.training`` — it's a no-op in eval mode.
    # In the non-low_gpu path we run a full forward+backward through the whole model,
    # so we want checkpointing to actually kick in (train mode). In the low_gpu path
    # we drive the blocks manually and don't want dropout / training-only side effects,
    # so keep eval mode.
    if auto_scheme.low_gpu_mem_usage:
        model.eval()
    else:
        # To trigger gradient checkpoint, but it will enable dropout, batchnorm, which is not good for accuracy
        model.train()

    # Filter out embedding layers from the scoring set (they aren't linear
    # quantization targets in any of our schemes).
    embedding_layers_names = []
    for name in quant_layer_names:
        module = get_module(model, name)
        if isinstance(module, torch.nn.Embedding):
            embedding_layers_names.append(name)
    quant_layer_names = list(set(quant_layer_names) - set(embedding_layers_names))

    is_moe_model = _is_moe_model(model)

    # Decide whether AutoScheme has to score vision-tower layers (typically
    # because the user passed ``--quant_nontext_module``). Used below to
    # clamp batch_size to 1 (image sizes vary) and to pick the multimodal
    # dataloader. The actual switch from low_gpu to full forward+backward
    # is done upstream in ``gen_layer_config``.
    vision_markers = ("vision", "visual", "image", "img")
    force_mllm = is_vlm and any(any(marker in n.lower() for marker in vision_markers) for n in quant_layer_names)

    # When scoring vision-tower layers (``force_mllm``, typically because the
    # caller passed ``--quant_nontext_module``), include the visual block(s)
    # so they get ``in_block=True``, count towards ``block_num``/progress, and
    # participate in offload hooks below just like language blocks.
    block_name = get_block_names(model, quant_vision=force_mllm)[0]
    for name in block_name:
        module = get_module(model, name)
        module.in_block = True
        for n, m in module.named_modules():
            m.in_block = True

    for n, m in model.named_modules():
        if len(list(m.children())) == 0:
            if not hasattr(m, "in_block"):
                m.in_block = False
            if not m.in_block and auto_scheme.low_gpu_mem_usage:
                m.to(major_device)

    total_scores = {}
    schemes = auto_scheme.options

    def check_bf16_scheme(scheme):
        """Return True if ``scheme`` is effectively BF16/no-op (bits >= 16 and act_bits >= 16),
        in which case scoring can skip the expensive wrap/forward/backward cycle entirely.
        """
        if isinstance(scheme, str) and scheme.upper() == "BF16":
            return True
        if isinstance(scheme, QuantizationScheme):
            scheme = asdict(scheme)
        if isinstance(scheme, dict):
            return scheme.get("bits", 16) >= 16 and scheme.get("act_bits", 16) >= 16
        return False

    from auto_round import envs as _envs

    _env_nsamples = _envs.AR_AUTO_SCHEME_NSAMPLES
    # Priority for nsamples: env > API > default
    if _env_nsamples is not None:
        nsamples = _env_nsamples
    elif auto_scheme.nsamples is not None:
        nsamples = auto_scheme.nsamples
    else:
        nsamples = 16

    # seqlen: env > API explicit setting > MoE-aware default
    _env_seqlen = _envs.AR_AUTO_SCHEME_SEQLEN
    if _env_seqlen is not None:
        seqlen = _env_seqlen
    elif auto_scheme.seqlen is not None:
        seqlen = auto_scheme.seqlen
    else:
        seqlen = 128 if is_moe_model else 256

    # 2-bit options benefit from more/longer calibration data. Warn when 2-bit
    # (non-GGUF) schemes are present but nsamples/seqlen are below the recommended settings.
    def _scheme_has_2bit(scheme):
        if isinstance(scheme, str):
            try:
                scheme = asdict(preset_name_to_scheme(scheme))
            except Exception:
                return False
        if isinstance(scheme, QuantizationScheme):
            scheme = asdict(scheme)
        if isinstance(scheme, dict):
            # GGUF (super_bits set) uses its own (double) quantization and does
            # not follow this nsamples/seqlen recommendation, so skip it.
            if scheme.get("super_bits") is not None:
                return False
            return scheme.get("bits", 16) == 2
        return False

    if any(_scheme_has_2bit(s) for s in schemes) and (nsamples < 128 or seqlen < 1024):
        logger.warning(
            "AutoScheme: 2-bit scheme(s) detected. For better results, consider nsamples>=128 and "
            "seqlen>=1024 (current: nsamples=%d, seqlen=%d). "
            "Override via env vars AR_AUTO_SCHEME_NSAMPLES / AR_AUTO_SCHEME_SEQLEN.",
            nsamples,
            seqlen,
        )

    if auto_scheme.batch_size is not None:
        batch_size = auto_scheme.batch_size
    else:
        _env_batch_size = _envs.AR_AUTO_SCHEME_BATCH_SIZE
        if _env_batch_size is not None:
            batch_size = _env_batch_size
        else:
            if auto_scheme.low_gpu_mem_usage:
                batch_size = 8
            else:
                batch_size = 1

    # ------------------------------------------------------------------ #
    # Multimodal calibration: ``batch_size`` must be 1 because image      #
    # sizes differ across samples (the multimodal collator can't stack    #
    # them otherwise).                                                    #
    # ------------------------------------------------------------------ #
    if force_mllm:
        if batch_size != 1:
            logger.info("AutoScheme(force_mllm): clamping batch_size %d -> 1.", batch_size)
            batch_size = 1

    pbar_cnt = 0
    need_weight_grad = False
    need_imatrix = False  # only trigger it for gguf q-k quant
    effective_scheme_num = 0
    block_num = len(block_name)
    for index, scheme in enumerate(schemes):
        if check_bf16_scheme(scheme):
            continue
        effective_scheme_num += 1
        if isinstance(scheme, str):
            scheme = asdict(preset_name_to_scheme(scheme))
        elif isinstance(scheme, QuantizationScheme):
            scheme = asdict(scheme)
        bits = scheme.get("bits", 16)
        act_bits = scheme.get("act_bits", 16)
        if scheme.get("super_group_size"):
            need_imatrix = True
        # Weight scores must accumulate for every weight-quantized scheme, not
        # only A16 ones: MX options (W4A4/W8A8) quantify weight error solely via
        # the weight score, and skipping it zeroes every layer score, so the
        # bit allocation collapses to the cheapest option.
        if bits <= 8:
            need_weight_grad = True
        if not auto_scheme.low_gpu_mem_usage:
            pbar_cnt += nsamples
        if auto_scheme.low_gpu_mem_usage:
            pbar_cnt += len(block_name) * 2 * ((nsamples + batch_size - 1) // batch_size)  # forward backward

    # Formula-style step log for paper/debug readability.
    # In low_gpu mode, one calibration mini-batch uses block-wise forward+backward replay.
    # so base_step_per_scheme = block_num * 2.
    base_total_steps = effective_scheme_num * block_num * 2
    logger.info(f"AutoScheme steps(total)={base_total_steps}")
    logger.info(
        "AutoScheme steps variables: "
        f"scheme_num={effective_scheme_num}, block_num={block_num}, "
        f"nsamples={nsamples}, batch_size={batch_size}"
    )
    logger.info(
        "AutoScheme: nsamples/batch_size can be overridden via env vars "
        "AR_AUTO_SCHEME_NSAMPLES / AR_AUTO_SCHEME_BATCH_SIZE "
        "(e.g. `export AR_AUTO_SCHEME_NSAMPLES=1` for a quick run); "
        "see docs/environments.md for details."
    )
    if auto_scheme.low_gpu_mem_usage:
        n_batches = (nsamples + batch_size - 1) // batch_size
        logger.info(
            "AutoScheme steps expanded(low_gpu): "
            "total_steps = scheme_num * block_num * 2(forward+backward) * n_batches = "
            f"{effective_scheme_num} * {block_num} * 2 * {n_batches} = {pbar_cnt}"
        )
    else:
        logger.info(
            "AutoScheme steps expanded(full_backward): "
            f"total_steps = scheme_num * nsamples = {effective_scheme_num} * {nsamples} = {pbar_cnt}"
        )
    shared_layers = parse_shared_layers(model, auto_scheme.shared_layers)

    # Auto-group MoE expert layers so DP treats all experts in one block as a unit.
    if is_moe_model:
        expert_groups = build_expert_groups(model, quant_layer_names, fixed_layer_scheme)
        if expert_groups:
            shared_layers = merge_lists_unionfind(shared_layers + expert_groups)

    # Pre-compute per-key weight numel (for loss/elem display).  Mirrors the
    # shared_layers grouping used in the scoring loop so keys match total_scores.
    _dp_names = set(quant_layer_names) - set(fixed_layer_scheme.keys())
    _shared_seen: set[str] = set()
    layer_numel: dict[str, int] = {}
    for _share_layer in shared_layers:
        _nl = [n for n in _share_layer if n in _dp_names]
        if not _nl:
            continue
        _total = 0
        for _n in _nl:
            _m = get_module(model, _n)
            _np = _m.weight.numel() if hasattr(_m, "weight") and _m.weight is not None else 0
            if _np == 0 and hasattr(_m, "_cached_weight_numel"):
                _np = _m._cached_weight_numel
            _total += _np
            _shared_seen.add(_n)
        layer_numel[_nl[0]] = _total
    for _n in _dp_names:
        if _n in _shared_seen:
            continue
        _m = get_module(model, _n)
        _np = _m.weight.numel() if hasattr(_m, "weight") and _m.weight is not None else 0
        if _np == 0 and hasattr(_m, "_cached_weight_numel"):
            _np = _m._cached_weight_numel
        layer_numel[_n] = _np

    options_scores = []
    pbar = None

    # ---- Scoring cache (per-scheme) -------------------------------------------------- #
    # Each scheme gets its own cache file so that adding/removing schemes or changing
    # unrelated parameters (e.g., target_bits) doesn't invalidate cached scores for
    # unchanged schemes.
    _model_id_for_cache = model_name or getattr(getattr(model, "config", None), "_name_or_path", None)

    # In per-scheme caching, each scheme is checked independently for cache hits.
    # We always prepare the common resources (imatrix, pbar, etc.) since they're
    # reusable across schemes and their setup cost is negligible compared to scoring.
    if True:
        if need_imatrix:
            dataloader = get_dataloader(
                tokenizer,
                seqlen=max(seqlen * 2, 2048),
                dataset_name=dataset,
                seed=42,
                bs=batch_size,
                nsamples=min(nsamples, 128),
            )
            logger.info("start to compute imatrix in AutoScheme")
            cal_imatrix(model, dataloader, major_device, low_gpu_mem_usage=auto_scheme.low_gpu_mem_usage)
            memory_monitor.update()
            memory_monitor.log_summary()
            logger.info("finish calculating imatrix")

        # Register hooks and clear all block weights before the scheme loop.
        # Hooks will transparently reload weights on demand during forward passes.
        if offload_context is not None:
            offload_context.add_offload_hooks(model, block_name)

        pbar = tqdm(total=pbar_cnt, desc="Generating AutoScheme")
        scored_layer_names = set(quant_layer_names + embedding_layers_names)
        cache_total_params = sum(
            (
                module.weight.numel()
                if hasattr(module, "weight") and module.weight is not None
                else getattr(module, "_cached_weight_numel", 0)
            )
            for name, module in model.named_modules()
            if name in scored_layer_names
        )
        scheme_cache_configs = []

        def _group_per_op_scores(index, per_op_scores):
            """Apply the current shared-layer grouping without mutating cached per-op scores."""
            grouped_scores = {}
            remaining = {name: list(score) for name, score in per_op_scores.items()}
            for share_layer in shared_layers:
                param_bits = 0
                tmp_loss = 0
                name_list = []
                for name in share_layer:
                    if name in remaining:
                        bits, loss = remaining.pop(name)
                        param_bits += bits
                        tmp_loss += loss
                        name_list.append(name)
                if name_list:
                    grouped_scores[name_list[0]] = [index, param_bits, tmp_loss, name_list]
            for name, (bits, loss) in remaining.items():
                grouped_scores[name] = [index, bits, loss, [name]]
            return grouped_scores

        def _record_scheme_scores(index, per_op_scores):
            grouped_scores = _group_per_op_scores(index, per_op_scores)
            total_loss = sum(item[2] for item in grouped_scores.values())
            for key, item in grouped_scores.items():
                total_scores.setdefault(key, []).append(item)
            options_scores.append(total_loss)
            return total_loss

        def _save_per_op_scores(index, scheme, cache_key, cache_path, per_op_scores):
            if cache_key is None or cache_path is None:
                return
            try:
                os.remove(_partial_scores_path(cache_path))
            except OSError:
                pass
            if isinstance(scheme, str):
                scheme_dict = asdict(preset_name_to_scheme(scheme))
            elif isinstance(scheme, QuantizationScheme):
                scheme_dict = asdict(scheme)
            else:
                scheme_dict = scheme if isinstance(scheme, dict) else {"preset": str(scheme)}
            _save_autoscheme_scores(
                cache_path=cache_path,
                cache_key=cache_key,
                scheme_index=index,
                scheme_dict=scheme_dict,
                layer_scores={name: list(score) for name, score in per_op_scores.items()},
                total_loss_for_scheme=sum(score[1] for score in per_op_scores.values()),
                total_params=cache_total_params,
                cache_config=scheme_cache_configs[index],
            )

        scheme_cache_meta = []
        for index, scheme in enumerate(schemes):
            if check_bf16_scheme(scheme) or _model_id_for_cache is None:
                scheme_cache_meta.append((None, None, None))
                scheme_cache_configs.append(None)
                if check_bf16_scheme(scheme):
                    logger.info(
                        "AutoScheme: scheme %d/%d (%s) is a BF16 baseline; skipping scoring and cache lookup.",
                        index + 1,
                        len(schemes),
                        _scheme_short_name(scheme),
                    )
                continue
            cache_config = _autoscheme_cache_config(
                model_name=_model_id_for_cache,
                dataset=dataset,
                nsamples=nsamples,
                seqlen=seqlen,
                batch_size=batch_size,
                quant_layer_names=quant_layer_names,
                fixed_layer_scheme=fixed_layer_scheme,
                scheme=scheme,
                force_mllm=force_mllm,
                low_gpu_mem_usage=auto_scheme.low_gpu_mem_usage,
                need_weight_grad=need_weight_grad,
            )
            cache_key = hashlib.sha256(json.dumps(cache_config, sort_keys=True, default=str).encode()).hexdigest()[:16]
            cache_path = _autoscheme_cache_path(cache_key, index)
            cached_data = _find_compatible_autoscheme_cache(
                cache_path,
                cache_config,
                quant_layer_names,
                fixed_layer_scheme,
                cache_total_params,
            )
            scheme_cache_meta.append((cache_key, cache_path, cached_data))
            scheme_cache_configs.append(cache_config)

        uncached_indices = [
            index
            for index, (_, _, cached) in enumerate(scheme_cache_meta)
            if not check_bf16_scheme(schemes[index]) and cached is None
        ]
        worker_device_pool = [device for device in device_list if str(device).startswith("cuda:")]
        num_gpus = len(worker_device_pool)
        parallel_enabled = _envs.AR_ENABLE_AUTO_SCHEME_PARALLEL
        worker_disk_stream_model = _prefer_disk_stream_scheme_worker(
            _model_id_for_cache, is_vlm, auto_scheme.low_gpu_mem_usage
        )
        # Serial scoring runs the full-model forward, which cannot cross GPUs on
        # a manually sharded model. When that is the loaded configuration and a
        # disk-stream worker is possible, lower the parallel floor to 1 so even
        # a single uncached scheme is scored through a worker instead of the
        # broken in-process pass.
        serial_device_safe = _serial_scoring_device_safe(model, visible_cuda_devices=worker_device_pool)
        min_uncached = 2
        if not serial_device_safe and worker_disk_stream_model:
            min_uncached = 1
            logger.info(
                "AutoScheme: serial in-process scoring is unsafe (multi-GPU weights, "
                "hf_device_map, or a streamed model with several visible GPUs); "
                "routing even a single uncached scheme through a disk-stream scoring worker."
            )
        elif not serial_device_safe and uncached_indices:
            raise RuntimeError(
                "AutoScheme serial scoring cannot run on a model whose weights span "
                "multiple GPUs without accelerate dispatch (the full-model forward has "
                "no cross-device activation transfer). Score via disk-stream workers: "
                "point AR_DISK_STREAM_MODEL at a local checkpoint directory and keep "
                "AR_ENABLE_AUTO_SCHEME_PARALLEL enabled, or load the model on a single "
                "GPU / with device_map='auto'."
            )
        if __debug__:
            from collections import Counter

            _param_devs = Counter(str(p_.device) for p_ in model.parameters())
            _hf_map = getattr(model, "hf_device_map", None)
            _hf_devs = sorted({str(v_) for v_ in _hf_map.values()}) if _hf_map else None
            logger.info(
                "AutoScheme serial-safety census: serial_device_safe=%s "
                "param_devices=%s hf_device_map_devices=%s disk_stream_index=%s meta_params=%d "
                "cuda_visible=%s",
                serial_device_safe,
                dict(_param_devs),
                _hf_devs,
                getattr(model, "_disk_stream_index", None) is not None,
                sum(1 for p_ in model.parameters() if p_.device.type == "meta"),
                [str(d_) for d_ in worker_device_pool],
            )
        # Vision scoring requires a full-model backward and therefore cannot use
        # the block-wise materialize/free path used by disk streaming.
        can_parallel = _can_parallel_scheme_scoring(
            parallel_enabled,
            _model_id_for_cache,
            num_gpus,
            len(uncached_indices),
            need_imatrix,
            worker_disk_stream_model,
            is_vlm,
            low_gpu_mem_usage=auto_scheme.low_gpu_mem_usage,
            force_mllm=force_mllm,
            min_uncached=min_uncached,
        )
        logger.info(
            "AutoScheme scoring mode: parallel_configured=%s, parallel_enabled=%s, disk_stream_enabled=%s",
            parallel_enabled,
            can_parallel,
            worker_disk_stream_model and can_parallel,
        )
        if not parallel_enabled and len(uncached_indices) >= 2:
            logger.info(
                "AutoScheme: parallel scoring was disabled by AR_ENABLE_AUTO_SCHEME_PARALLEL=0; "
                "scoring %d uncached non-BF16 schemes serially.",
                len(uncached_indices),
            )

        from auto_round.modeling.fused_moe.replace_modules import ReplacementModuleBase

        use_model_replacements = any(isinstance(module, ReplacementModuleBase) for module in model.modules())

        parallel_done = False
        if can_parallel:
            try:
                import torch.multiprocessing as multiprocessing

                def _serialize_scheme(scheme):
                    return asdict(scheme) if isinstance(scheme, QuantizationScheme) else scheme

                num_workers = _get_scheme_worker_count(len(uncached_indices), num_gpus)
                worker_devices = _assign_scheme_worker_devices(len(uncached_indices), worker_device_pool)
                logger.info(
                    "AutoScheme: starting %d parallel scoring workers for %d uncached non-BF16 schemes "
                    "(devices=%s; workers may share devices)",
                    num_workers,
                    len(uncached_indices),
                    worker_device_pool,
                )
                logger.info(
                    "AutoScheme: if parallel scoring runs out of RAM/VRAM or automatic serial fallback cannot "
                    "recover, set AR_ENABLE_AUTO_SCHEME_PARALLEL=0 and rerun."
                )
                # free the parent's GPU before spawning workers: non-block params
                # (embed/lm_head/vision) parked on a device by earlier setup are
                # dead weight during parallel scoring -- the parent never runs a
                # forward, and each worker owns its private copy. A shared-GPU
                # worker then gets the card's full budget.
                _moved = 0
                _parked = []
                for _p in model.parameters():
                    if _p.device.type == "cuda":
                        _parked.append((_p, _p.device))
                        _p.data = _p.data.to("cpu")
                        _moved += 1
                for _b in model.buffers():
                    if _b.device.type == "cuda":
                        _parked.append((_b, _b.device))
                        _b.data = _b.data.to("cpu")

                def _restore_parked_tensors():
                    """Move tensors parked before spawning back to their devices.

                    The serial fallback and the post-parallel phases run
                    full-model forwards; leaving non-block params on CPU while
                    ``model.device`` still reports the GPU crashes the first
                    non-block module on a device mismatch (e.g. final norm)."""
                    for _tensor, _device in _parked:
                        if _tensor.device.type == "cpu" and _device.type != "cpu":
                            _tensor.data = _tensor.data.to(_device)
                    _parked.clear()

                if _moved:
                    logger.info("AutoScheme: parked %d parent-side GPU tensors on CPU before spawning workers", _moved)
                    clear_memory()

                _vram_inventory("parent pre-spawn")

                spawn_context = multiprocessing.get_context("spawn")
                with spawn_context.Manager() as manager:
                    progress_queue = manager.Queue()
                    worker_args = [
                        (
                            index,
                            _serialize_scheme(schemes[index]),
                            _model_id_for_cache,
                            is_vlm,
                            list(quant_layer_names),
                            dict(fixed_layer_scheme),
                            dataset,
                            nsamples,
                            seqlen,
                            batch_size,
                            need_weight_grad,
                            enable_torch_compile,
                            auto_scheme.low_cpu_mem_usage,
                            auto_scheme.low_gpu_mem_usage,
                            auto_scheme.ignore_scale_zp_bits,
                            force_mllm,
                            use_model_replacements,
                            worker_devices[slot],
                            len(schemes),
                            progress_queue,
                            worker_disk_stream_model,
                            scheme_cache_meta[index][1],  # cache_path for batch checkpoints
                        )
                        for slot, index in enumerate(uncached_indices)
                    ]
                    with spawn_context.Pool(processes=num_workers) as pool:
                        # incremental consumption: persist each scheme's scores to
                        # the per-scheme cache as its worker returns, so a partial
                        # failure (or serial-fallback crash) never discards
                        # completed schemes -- the rerun picks them up from cache
                        result_iter = pool.imap_unordered(_score_scheme_worker, worker_args)
                        worker_results = []
                        failed_workers = 0
                        while True:
                            try:
                                worker_results.append(result_iter.next(timeout=0.5))
                            except multiprocessing.TimeoutError:
                                _drain_progress_queue(progress_queue, pbar)
                                memory_monitor.update_cpu()
                                continue
                            except StopIteration:
                                break
                            except Exception as worker_error:  # noqa: BLE001
                                # one worker's failure must not abort its siblings:
                                # keep consuming their results (each already-saved
                                # scheme survives in the per-scheme cache)
                                failed_workers += 1
                                logger.warning(
                                    "AutoScheme: a scoring worker failed (%s); collecting remaining workers' results.",
                                    worker_error,
                                )
                                _drain_progress_queue(progress_queue, pbar)
                                continue
                            index, scores, _report = worker_results[-1]
                            cache_key, cache_path, _ = scheme_cache_meta[index]
                            _save_per_op_scores(index, schemes[index], cache_key, cache_path, scores)
                        _drain_progress_queue(progress_queue, pbar)
                        if not worker_results:
                            raise RuntimeError("all parallel scoring workers failed") from None

                    parallel_results = {index: scores for index, scores, _ in worker_results}
                    _merge_worker_memory_reports(memory_monitor, [report for _, _, report in worker_results])
                    logger.info(
                        "AutoScheme parallel aggregate memory (parent + %d workers): %s",
                        len(worker_results),
                        memory_monitor.get_summary(),
                    )

                for index, scheme in enumerate(schemes):
                    cache_key, cache_path, cached_data = scheme_cache_meta[index]
                    if check_bf16_scheme(scheme):
                        apply_quant_scheme(
                            model,
                            quant_layer_names=quant_layer_names,
                            fixed_layer_scheme=fixed_layer_scheme,
                            scheme=scheme,
                        )
                        per_op_scores = {}
                        for name in quant_layer_names:
                            if name in fixed_layer_scheme:
                                continue
                            bits, _ = compute_layer_bits(get_module(model, name), auto_scheme.ignore_scale_zp_bits)
                            per_op_scores[name] = [bits, 0.0]
                    elif cached_data is not None:
                        loaded_cache_path = cached_data.get("_cache_path", cache_path)
                        logger.info(
                            "AutoScheme: loading per-scheme cache for scheme %d from %s."
                            " Delete this file to disable reuse and rescore.",
                            index,
                            loaded_cache_path,
                        )
                        per_op_scores = _refresh_cached_layer_bits(
                            model,
                            quant_layer_names,
                            fixed_layer_scheme,
                            scheme,
                            cached_data["layer_scores"],
                            auto_scheme.ignore_scale_zp_bits,
                        )
                        if not check_bf16_scheme(scheme):
                            pbar.update(pbar_cnt // effective_scheme_num if effective_scheme_num > 0 else 1)
                    else:
                        if index in parallel_results:
                            per_op_scores = parallel_results[index]
                            _save_per_op_scores(index, scheme, cache_key, cache_path, per_op_scores)
                        else:
                            # worker died for this scheme; siblings' results are saved.
                            # Rerunning resumes/scores only the missing schemes.
                            raise RuntimeError(
                                f"AutoScheme: scheme {index} ({_scheme_short_name(scheme)}) was lost with its "
                                "worker; completed schemes are persisted in the per-scheme cache -- rerun to "
                                "score only the failed schemes."
                            )
                    total_loss = _record_scheme_scores(index, per_op_scores)
                    logger.info(
                        "AutoScheme transition: scheme %d/%d scoring finished (total_loss=%.6f)",
                        index + 1,
                        len(schemes),
                        total_loss,
                    )
                parallel_done = True
                logger.info("AutoScheme: parallel scoring completed.")
                _restore_parked_tensors()
                post_scoring_started = time.perf_counter()
            except Exception as parallel_error:  # noqa: BLE001
                _restore_parked_tensors()
                if _parallel_scoring_must_raise(parallel_error):
                    logger.error(
                        "AutoScheme: keeping the parallel scoring failure as a hard error "
                        "(AR_AUTO_SCHEME_NO_SERIAL_FALLBACK is set, or a scheme was lost with its worker); "
                        "completed schemes and batches are persisted in the per-scheme cache -- "
                        "rerun to score only the failed parts, e.g. with a smaller "
                        "AR_AUTO_SCHEME_BATCH_SIZE / AR_AUTO_SCHEME_NSAMPLES."
                    )
                    raise
                logger.warning(
                    "AutoScheme: parallel scoring failed, falling back to serial: %s. "
                    "If fallback cannot recover (for example after RAM/VRAM exhaustion), rerun with "
                    "AR_ENABLE_AUTO_SCHEME_PARALLEL=0.",
                    parallel_error,
                )
                total_scores.clear()
                options_scores.clear()
                pbar.reset(total=pbar_cnt)

        if not parallel_done:
            if uncached_indices and disk_index is None:
                # Skipped in streaming mode: materialize_model_ only acts on
                # ReplacementModuleBase (fused-MoE) instances -- a no-op for
                # dense models like ours -- but it also warns once per
                # still-meta parameter/buffer, which would flood the log with
                # one warning per decoder-block tensor (intentionally still
                # meta, to be streamed on demand later).
                from auto_round.modeling.fused_moe.replace_modules import materialize_model_

                materialize_model_(model)
            for index, scheme in enumerate(schemes):
                scheme_tag = f"{index + 1}/{len(schemes)} {_scheme_short_name(scheme)}"
                logger.info(f"AutoScheme transition: switch to scheme {index + 1}/{len(schemes)} ({scheme})")
                cache_key, cache_path, cached_data = scheme_cache_meta[index]

                if cached_data is not None:
                    loaded_cache_path = cached_data.get("_cache_path", cache_path)
                    logger.info(
                        "AutoScheme: loading per-scheme cache for scheme %d from %s."
                        " Delete this file to disable reuse and rescore.",
                        index,
                        loaded_cache_path,
                    )
                    per_op_scores = _refresh_cached_layer_bits(
                        model,
                        quant_layer_names,
                        fixed_layer_scheme,
                        scheme,
                        cached_data["layer_scores"],
                        auto_scheme.ignore_scale_zp_bits,
                    )
                    if not check_bf16_scheme(scheme):
                        pbar.update(pbar_cnt // effective_scheme_num if effective_scheme_num > 0 else 1)
                else:
                    apply_quant_scheme(
                        model,
                        quant_layer_names=quant_layer_names,
                        fixed_layer_scheme=fixed_layer_scheme,
                        scheme=scheme,
                    )
                    if check_bf16_scheme(scheme):
                        per_op_scores = {}
                        for name in quant_layer_names:
                            if name in fixed_layer_scheme:
                                continue
                            bits, _ = compute_layer_bits(get_module(model, name), auto_scheme.ignore_scale_zp_bits)
                            per_op_scores[name] = [bits, 0.0]
                    else:
                        per_op_scores = get_score_for_scheme(
                            model,
                            tokenizer,
                            quant_layer_names,
                            fixed_layer_scheme,
                            dataset,
                            ignore_scale_zp_bits=auto_scheme.ignore_scale_zp_bits,
                            pbar=pbar,
                            nsamples=nsamples,
                            seqlen=seqlen,
                            need_weight_grad=need_weight_grad,
                            enable_torch_compile=enable_torch_compile,
                            low_gpu_mem_usage=auto_scheme.low_gpu_mem_usage,
                            major_device=major_device,
                            batch_size=batch_size,
                            offload_context=offload_context,
                            processor=processor,
                            is_vlm=is_vlm,
                            force_mllm=force_mllm,
                            model_name=model_name,
                            scheme_tag=scheme_tag,
                            disk_index=disk_index,
                        )
                    memory_monitor.update()
                    memory_monitor.log_summary()
                    if not check_bf16_scheme(scheme):
                        _save_per_op_scores(index, scheme, cache_key, cache_path, per_op_scores)

                total_loss = _record_scheme_scores(index, per_op_scores)
                logger.info(
                    "AutoScheme transition: scheme %d/%d scoring finished (total_loss=%.6f)",
                    index + 1,
                    len(schemes),
                    total_loss,
                )
                clear_memory(device_list=device_list)

        # Serial scoring leaves the main model configured with the final scheme and
        # applies fixed_layer_scheme as a side effect. Parallel workers and cache hits
        # do not touch the main model, so restore that state before bit-budget math.
        if schemes:
            apply_quant_scheme(
                model,
                quant_layer_names=quant_layer_names,
                fixed_layer_scheme=fixed_layer_scheme,
                scheme=schemes[-1],
            )

        # Remove hooks and restore original weights from disk for final bit-budget computations
        if offload_context is not None:
            offload_context.remove_offload_hooks(model, block_name)

        total_params = 0
        for n, m in model.named_modules():
            if n in quant_layer_names + embedding_layers_names:
                n_param = m.weight.numel()
                if n_param == 0 and hasattr(m, "_cached_weight_numel"):
                    n_param = m._cached_weight_numel
                total_params += n_param

        if parallel_done:
            logger.info(
                "AutoScheme post-scoring: model restore and parameter accounting took %.2fs",
                time.perf_counter() - post_scoring_started,
            )

    target_params_cnt = int(total_params * target_bits)
    sorted_indices = sorted(range(len(options_scores)), key=lambda i: options_scores[i])
    # Layers that are not fixed in fixed_layer_scheme. Note that
    # `embedding_layers_names` was carved out of `quant_layer_names` above, so
    # every entry is a quantization target; checking `quant_layer_names` again
    # here would always fail and silently leave embeddings outside the budget.
    not_fixed_embedding_layers_names = [name for name in embedding_layers_names if name not in fixed_layer_scheme]

    # Determine if model has shared lm_head (tie_word_embeddings)
    has_tied_lm_head = getattr(getattr(model, "config", None), "tie_word_embeddings", False)

    def _to_scheme_dict(scheme):
        """Normalize a scheme (str/QuantizationScheme/dict) to a plain dict."""
        if isinstance(scheme, str):
            return asdict(preset_name_to_scheme(scheme))
        elif isinstance(scheme, QuantizationScheme):
            return asdict(scheme)
        return scheme

    def _compute_embedding_bits(scheme_dict):
        """Compute total bits consumed by non-fixed embedding layers under scheme_dict."""
        total = 0
        for emb_name in not_fixed_embedding_layers_names:
            emb_layer = get_module(model, emb_name)
            n_param = emb_layer.weight.numel()
            if n_param == 0 and hasattr(emb_layer, "_cached_weight_numel"):
                n_param = emb_layer._cached_weight_numel
            # With ignore_scale_zp_bits, bits_cost = n_param * bits
            emb_bits = scheme_dict.get("bits", 16)
            total += n_param * emb_bits
        return total

    # Compute minimum bits needed for DP layers (non-fixed, non-embedding)
    min_dp_bits = 0
    for layer_name, opts in total_scores.items():
        min_dp_bits += min(opt[1] for opt in opts)

    def _fits_budget(scheme_dict):
        """Check if applying scheme_dict to embeddings leaves enough budget for DP layers.

        Called after user-fixed layers have already been subtracted from
        ``target_params_cnt``, so only the embedding cost is deducted here.
        """
        emb_bits = _compute_embedding_bits(scheme_dict)
        remaining = target_params_cnt - emb_bits
        return remaining >= min_dp_bits

    def _select_embedding_scheme_index():
        """Select the best scheme index for embedding layers based on model type and target_bits.

        For models with shared lm_head (tie_word_embeddings=True):
          - target_bits > 6: use the lowest-loss option (same as before)
          - target_bits <= 6: use the lowest-loss option among those with bits <= 6
        For models without shared lm_head:
          - use the lowest-loss option among those with bits >= ceil(target_bits)

        In all cases, the selected scheme must not exceed the remaining bit budget
        (i.e., embedding bits + min DP bits <= target_params_cnt, where user-fixed
        layers have already been subtracted from target_params_cnt).
        """

        if has_tied_lm_head:
            if target_bits > 6:
                candidates = list(sorted_indices)
            else:
                # Prefer options with bits <= 6, sorted by loss
                candidates = [idx for idx in sorted_indices if _get_scheme_bits(schemes[idx]) <= 6]
                if not candidates:
                    candidates = list(sorted_indices)
        else:
            # Not shared lm_head: prefer the nearest available bit width at or
            # above floor(target_bits), then choose the lowest-loss option at
            # that width.
            floor_bits = math.floor(target_bits)
            candidates = [idx for idx in sorted_indices if _get_scheme_bits(schemes[idx]) == floor_bits]
            if not candidates:
                embedding_bits = _get_next_scheme_bits(schemes, sorted_indices, floor_bits)
                if embedding_bits is not None:
                    candidates = [idx for idx in sorted_indices if _get_scheme_bits(schemes[idx]) == embedding_bits]
            candidates.extend(sorted_indices)  # to make sure if the above candidate exceed the budget

        # Among candidates (ordered by loss), pick the first that fits the budget
        for idx in candidates:
            scheme_dict = _to_scheme_dict(schemes[idx])
            if _fits_budget(scheme_dict):
                return idx

        # Fallback: try ALL options sorted by bits ascending (cheapest first)
        all_by_bits = sorted(range(len(schemes)), key=lambda i: _get_scheme_bits(schemes[i]))
        for idx in all_by_bits:
            scheme_dict = _to_scheme_dict(schemes[idx])
            if _fits_budget(scheme_dict):
                return idx

        # Last resort: use the cheapest option regardless
        return all_by_bits[0] if all_by_bits else 0

    # Minus fixed_layer
    for name, layer_scheme in fixed_layer_scheme.items():
        m = get_module(model, name)
        # apply_quant_scheme only covers quant_layer_names; embedding layers were
        # carved out of it, so a user-fixed embedding still has no scheme attrs here
        # and compute_layer_bits would price it at 16 bits, wrecking the budget.
        for key, item in _to_scheme_dict(layer_scheme).items():
            setattr(m, key, item)
        layer_bits, _ = compute_layer_bits(m, auto_scheme.ignore_scale_zp_bits)
        target_params_cnt -= layer_bits

    # As only a small amount of calibration data is used and embedding layers are inherently sparse,
    # we cannot obtain a reliable score.
    if not_fixed_embedding_layers_names:
        selected_index = _select_embedding_scheme_index()
        tmp_scheme = _to_scheme_dict(schemes[selected_index])

        for embedding_layer_name in not_fixed_embedding_layers_names:
            fixed_layer_scheme[embedding_layer_name] = tmp_scheme
            embedding_layer = get_module(model, embedding_layer_name)
            for key, item in tmp_scheme.items():
                setattr(embedding_layer, key, item)
            layer_bits, _ = compute_layer_bits(embedding_layer, auto_scheme.ignore_scale_zp_bits)
            target_params_cnt -= layer_bits

    head_name = get_lm_head_name(model)
    if head_name is not None and (head_name not in fixed_layer_scheme and head_name in quant_layer_names):
        _apply_head_trick(head_name, schemes, sorted_indices, target_bits, target_params_cnt, total_scores)

    if target_params_cnt <= 0:
        raise ValueError("Avg bits is too small")

    cleanup_started = time.perf_counter()
    remove_quant_scheme(model)  # Must place after minus fixed_layer
    memory_monitor.update()
    memory_monitor.log_summary()
    logger.info(
        "AutoScheme post-scoring: scheme cleanup and memory accounting took %.2fs",
        time.perf_counter() - cleanup_started,
    )

    dp_started = time.perf_counter()
    best_loss, best_path = choose_bits_per_layer_with_path(total_scores, target_params_cnt)
    logger.info(
        "AutoScheme post-scoring: DP selection took %.2fs (layers=%d)",
        time.perf_counter() - dp_started,
        len(total_scores),
    )

    if best_path is None:
        raise ValueError("Avg bits is too small")

    # print(best_loss, best_path)  # TODO better log
    layer_config = copy.deepcopy(fixed_layer_scheme)
    options = list(copy.deepcopy(auto_scheme.options))
    # Replace scheme preset names with actual QuantizationScheme objects
    for index in range(len(options)):
        if isinstance(options[index], str):
            options[index] = preset_name_to_scheme(options[index])
    for item in best_path:
        layer_names = item[0]
        layer_scheme = options[item[1]]
        for layer_name in layer_names:
            layer_config[layer_name] = asdict(layer_scheme)
    reporting_started = time.perf_counter()
    _log_scheme_loss_matrix(total_scores, options, block_name, model=model, layer_numel=layer_numel)
    _describe_layer_config(layer_config, total_scores, options, block_name, model=model)
    logger.info("AutoScheme post-scoring: result reporting took %.2fs", time.perf_counter() - reporting_started)
    if model_name is not None:
        model = None
        del model
    else:
        safe_to_cpu_(model)
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            import accelerate

            accelerate.hooks.remove_hook_from_submodules(model)
            delattr(model, "hf_device_map")
        for n, m in model.named_modules():
            if hasattr(m, "scale_dtype"):  # TODO refine code
                delattr(m, "scale_dtype")
            if hasattr(m, "imatrix"):
                delattr(m, "imatrix")
            if hasattr(m, "tuning_device"):
                delattr(m, "tuning_device")
        for n, m in model.named_parameters():
            if hasattr(m, "grad"):
                m.grad = None
    clear_memory(device_list=device_list)

    # # Log AutoScheme memory usage
    # memory_monitor.update_cpu()
    low_cpu_str = "enabled" if auto_scheme.low_cpu_mem_usage else "disabled"
    memory_monitor.log_summary(f"AutoScheme complete (low_cpu_mem_usage={low_cpu_str})")

    if pbar is not None:
        pbar.close()
    return layer_config


# Supports model with gradient clearing between iterations
@register_scheme_methods(("default", "DeltaLoss"))
def gen_layer_config(
    auto_scheme: AutoScheme,
    model: Union[str, torch.nn.Module],
    quant_layer_names: Iterable[str],
    fixed_layer_scheme: dict[str, dict],
    dataset: str = "pile-10k",
    tokenizer=None,
    device_map=None,
    enable_torch_compile=True,
    low_gpu_mem_usage=True,
    min_avg_bit_scheme=None,
    processor=None,
    **kwargs,
):
    """Public AutoScheme entry.

    This wrapper performs model loading/dispatch and environment preparation,
    then delegates to `_gen_layer_config` for staged scoring + DP selection.
    """
    model_name = None
    is_vlm = False
    disk_index = None
    if isinstance(model, str):
        model_name = model
        is_vlm = is_mllm_model(model_name)
        if not is_vlm and auto_scheme.low_cpu_mem_usage and low_gpu_mem_usage:
            # Disk-streamed load (meta-device skeleton + on-demand per-block
            # materialize/free, see disk_stream_util.py) instead of
            # load_model()'s full-checkpoint CPU RAM load -- infeasible for a
            # checkpoint bigger than available RAM. Falls back to load_model()
            # for anything build_meta_model doesn't cover.
            try:
                from auto_round.utils.disk_stream_util import build_meta_model, materialize_non_block_params

                model, tokenizer, disk_index = build_meta_model(model_name)
                block_prefixes = flatten_list(get_block_names(model, quant_vision=is_vlm))
                materialize_non_block_params(model, block_prefixes, disk_index, device="cpu")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"AutoScheme streaming load failed ({exc}); falling back to "
                    f"load_model() (needs the whole checkpoint resident in RAM)."
                )
                disk_index = None
                model, tokenizer, processor, _, _, is_vlm, _ = load_model(
                    model_name, device="cpu", use_auto_mapping=False
                )
        else:
            model, tokenizer, processor, _, _, is_vlm, _ = load_model(model_name, device="cpu", use_auto_mapping=False)
    else:
        # Object passed in: still try to detect VLM so we can pick the right dataloader later.
        try:
            _, _, _, _, _, is_vlm, _ = load_model(model)
        except Exception:  # noqa: BLE001
            is_vlm = False
        # By the time AutoRound's compressor calls into AutoScheme, ModelContext
        # has already turned a string model into a real object -- meaning the
        # `isinstance(model, str)` branch above never actually runs in the
        # standard `AutoRound(model=path, ...)` API flow. When ModelContext
        # itself built the object as a meta skeleton (AR_DISK_STREAM_MODEL=1),
        # it stashes the SafetensorsIndex on the model so we can pick it up
        # here instead of re-detecting streaming mode from scratch (or,
        # worse, silently treating a meta model as if it were fully real).
        disk_index = getattr(model, "_disk_stream_index", None)

    # ---- Vision-tower scoring requires a full backward ---- #
    # ``model_forward_low_gpu`` only walks the language tower (it uses
    # ``get_block_names(model)[0]``, which excludes vision blocks by default)
    # and interrupts ``loss.backward()`` at the LAST language block via
    # ``backward_pre_hook`` -> ``MyCustomError``. As a result, gradient never
    # propagates into the vision tower and any AutoScheme score for vision
    # layers comes out as 0. If the caller asked us to score vision layers
    # (typically because ``--quant_nontext_module`` was passed), force a
    # full forward+backward instead.
    vision_markers = ("vision", "visual", "image", "img")
    force_mllm_for_vision = is_vlm and any(
        any(marker in n.lower() for marker in vision_markers) for n in quant_layer_names
    )
    if force_mllm_for_vision and low_gpu_mem_usage:
        logger.warning("AutoScheme: scoring vision layers requires full backward; " "disabling low_gpu_mem_usage.")
        low_gpu_mem_usage = False
        try:
            auto_scheme.low_gpu_mem_usage = False
        except Exception:  # noqa: BLE001
            pass
    # Get major device
    major_device = get_major_device(device_map)
    if not low_gpu_mem_usage:
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            model = dispatch_model(model, device_map=model.hf_device_map)
        else:
            model = dispatch_model_by_all_available_devices(model, device_map)
    else:
        safe_to_cpu_(model)
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            import accelerate

            accelerate.hooks.remove_hook_from_submodules(model)
        if (isinstance(device_map, str) and "," in device_map) or device_map == "auto":
            set_avg_auto_device_map(model, device_map)
        else:
            set_non_auto_device_map(model, device_map)

        for n in quant_layer_names:
            m = get_module(model, n)
            if not hasattr(m, "tuning_device"):
                m.tuning_device = major_device

    device_list = parse_available_devices(device_map)

    # Enable gradient checkpointing if supported.
    #
    # IMPORTANT: we must use ``use_reentrant=False``. The reentrant
    # implementation requires the inputs that enter the checkpointed region to
    # have ``requires_grad=True`` — otherwise its backward sees "no grad-
    # requiring input" and returns ``None`` for the input gradient, which kills
    # the autograd chain *before* the checkpoint boundary. In AutoScheme we
    # aggressively turn off ``requires_grad`` on every non-wrapper parameter
    # (token embeddings, norms, vision-tower non-linear layers, patch embeds,
    # …), so ``inputs_embeds`` entering the first text decoder block often does
    # NOT require grad. With reentrant=True that means gradient never flows
    # back into the vision tower → vision wrapper hooks see grad=0.
    # ``use_reentrant=False`` (saved-tensor-hooks impl) does not have this
    # restriction.
    def _enable_gc(mod):
        """Enable gradient checkpointing on ``mod`` with ``use_reentrant=False`` if supported
        (see rationale above); no-op if the module doesn't support checkpointing.
        """
        if not getattr(mod, "supports_gradient_checkpointing", False):
            return
        try:
            mod.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            # Older transformers without the kwargs argument.
            mod.gradient_checkpointing_enable()

    _enable_gc(model)

    for name in quant_layer_names:
        m = get_module(model, name)
        m.tmp_name = name

    try:
        res = _gen_layer_config(
            auto_scheme,
            model,
            quant_layer_names,
            fixed_layer_scheme,
            dataset=dataset,
            tokenizer=tokenizer,
            model_name=model_name,
            enable_torch_compile=enable_torch_compile,
            device_map=device_map,
            major_device=major_device,
            device_list=device_list,
            min_avg_bit_scheme=min_avg_bit_scheme,
            processor=processor,
            is_vlm=is_vlm,
            disk_index=disk_index,
        )
    except torch.OutOfMemoryError:
        logger.warning(
            "Fallback to CPU for automatic scheme generation."
            " Using multiple devices is strongly recommended (e.g., --device_map 0,1,2,3)."
        )
        safe_to_cpu_(model)
        for n, m in model.named_modules():
            if hasattr(m, "orig_layer"):
                set_module(model, n, m.orig_layer)
        clear_memory(device_list=device_list)
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            import accelerate

            accelerate.hooks.remove_hook_from_submodules(model)
            delattr(model, "hf_device_map")
        res = _gen_layer_config(
            auto_scheme,
            model,
            quant_layer_names,
            fixed_layer_scheme,
            dataset=dataset,
            tokenizer=tokenizer,
            model_name=model_name,
            enable_torch_compile=enable_torch_compile,
            device_map=device_map,
            major_device=major_device,
            device_list=device_list,
            min_avg_bit_scheme=min_avg_bit_scheme,
            processor=processor,
            is_vlm=is_vlm,
            disk_index=disk_index,
        )

    return res
