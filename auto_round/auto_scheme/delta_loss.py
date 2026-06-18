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
from dataclasses import asdict
from functools import wraps
from typing import Iterable, Optional, Union

import torch
from accelerate import dispatch_model
from tqdm import tqdm

from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.auto_scheme.register import register_scheme_methods
from auto_round.auto_scheme.utils import (
    apply_quant_scheme,
    compute_layer_bits,
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
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad
from auto_round.logger import logger
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
from auto_round.utils.offload import OffloadManager
from auto_round.wrapper import WrapperLinear

__all__ = ["gen_layer_config"]


class AutoSchemeWrapperLinear(WrapperLinear):

    def __init__(
        self,
        orig_layer,
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        device="cpu",
        enable_round_tuning=True,
        need_weight_grad=False,
        enable_torch_compile=False,
        **kwargs,
    ):
        super().__init__(
            orig_layer,
            enable_minmax_tuning,
            enable_norm_bias_tuning,
            device,
            enable_round_tuning,
            enable_torch_compile=enable_torch_compile,
            **kwargs,
        )
        self.total_act_score = 0.0
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

    def _qdq_act(self, x, act_min_scale=1.0, act_max_scale=1.0, act_max=None):
        if hasattr(self.orig_layer, "act_bits") and self.orig_layer.act_bits > 8:
            return x, 1.0, None

        qdq_x, scale, zp = self.act_qdq_func(x, act_min_scale, act_max_scale, act_max)
        if self.grad_mode:
            with torch.no_grad():
                self.max_act_value = torch.abs(x).max()
                if torch.abs(x).max() != 0:
                    self.act_cnt += 1
                x_diff = x - qdq_x
                self.x_diff = x_diff.to("cpu")

            def save_grad(grad):
                if self.max_act_value == 0:
                    if torch.abs(grad).max() != 0:
                        raise ValueError
                """
                this ut will cause NAN issue sometimes, need to investigate
                    @multi_card
                    def test_multi_card(self):
                     model_name = "/models/Qwen3-8B"
                """
                if torch.isnan(grad).any() or torch.isnan(self.x_diff).any():
                    self.act_cnt -= 1
                    return None

                self.total_act_score += torch.abs((grad * self.x_diff.to(grad.device))).sum().item()
                self.act_score = 0.0 if self.act_cnt <= 0 else self.total_act_score / self.act_cnt
                self.mix_score = self.weight_score + self.act_score
                self.x_diff = None
                return None

            qdq_x.register_hook(save_grad)
        return qdq_x, scale, zp

    def _qdq_weight(self, value, min_scale, max_scale):
        device = self.device
        if self.orig_layer.bits > 8 or not self.need_weight_grad:
            qdq_w, scale, zp = super()._qdq_weight(
                torch.tensor(0, device=device), torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)
            )

            return qdq_w, 1.0, None

        qdq_w, scale, zp = super()._qdq_weight(
            torch.tensor(0, device=device), torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)
        )

        if self.grad_mode:

            def save_grad(grad):
                qdq_w, scale, zp = self.super_qdq_func(
                    torch.tensor(0, device=device), torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)
                )
                w_diff = self.orig_layer.weight - qdq_w.to(self.orig_layer.weight.device)
                self.weight_score += torch.abs((grad.to(w_diff.device) * w_diff)).sum().item()
                act_score = 0.0 if self.act_cnt <= 0 else self.total_act_score / self.act_cnt
                self.mix_score = self.weight_score + act_score
                return None

            qdq_w.register_hook(save_grad)
        return qdq_w, 1.0, None


class AutoSchemeWrapperLinearForGGUFK(AutoSchemeWrapperLinear):

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
        super().__init__(
            orig_layer,
            enable_minmax_tuning,
            enable_norm_bias_tuning,
            device,
            enable_round_tuning,
            need_weight_grad,
            **kwargs,
        )
        with torch.no_grad():
            qdq_w, scale, zp = self.super_qdq_func(
                torch.tensor(0).to(device), torch.tensor(1.0).to(device), torch.tensor(1.0).to(device)
            )
        self.register_buffer("qdq_w", qdq_w.detach().clone().to(self.orig_layer.weight.device))

        def save_grad(grad):
            w_diff = self.orig_layer.weight - self.qdq_w.to(self.orig_layer.weight.device)
            # TODO strange, grad could be in CPU
            self.weight_score += torch.abs((grad.to(w_diff.device).to(torch.float32) * w_diff)).sum().item()
            act_score = 0.0 if self.act_cnt <= 0 else self.total_act_score / self.act_cnt
            self.mix_score = self.weight_score + act_score
            return None

        self.qdq_w.requires_grad_(True)
        self.orig_layer.weight.requires_grad_(False)
        self.qdq_w.register_hook(save_grad)

    def _qdq_weight(self, value, min_scale, max_scale):
        return self.qdq_w, 1.0, None


class AutoSchemeWrapperLinearForGGUFKImatrix(AutoSchemeWrapperLinear):

    def __init__(
        self,
        orig_layer,
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        device="cpu",
        enable_round_tuning=True,
        need_weight_grad=False,
        enable_torch_compile=False,
        **kwargs,
    ):
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
        self.post_init_qdqw(device)

    @torch.no_grad()
    def post_init_qdqw(self, device):  # Could not place in qdq_w, otherwise vram is much higher
        qdq_w = self._init_scale(device).detach()
        self.register_buffer("qdq_w", qdq_w.detach().clone().to(self.orig_layer.weight.device))

        def save_grad(grad):
            w_diff = self.orig_layer.weight - self.qdq_w.to(self.orig_layer.weight.device)
            self.weight_score += torch.abs((grad.to(torch.float32) * w_diff.to(grad.device))).sum().item()
            act_score = 0.0 if self.act_cnt <= 0 else self.total_act_score / self.act_cnt
            self.mix_score = self.weight_score + act_score
            return None

        self.qdq_w.requires_grad_(True)
        self.orig_layer.weight.requires_grad_(False)

        self.qdq_w.register_hook(save_grad)

    @torch.no_grad()
    def _init_scale(self, device):
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
            group_size = 16
            tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
            scale, d_scale = search_gguf_scale_min_sym(tensor, bits, imatrix, scale_dtype, split_num=1)
            tensor = revert_tensor_by_pad(tensor, orig_shape=orig_shape, pad_len=pad_len)
            qdq_w, _, _ = quant_tensor_gguf_sym_dq(
                tensor=tensor, bits=bits, scale_dtype=scale_dtype, imatrix=imatrix, scale=scale, d_scale=d_scale
            )
        else:
            raise ValueError("bits must be in [2,3,4,5,6]")
        return qdq_w.to(orig_dtype)

    def _qdq_weight(self, value, min_scale, max_scale):
        return self.qdq_w, 1.0, None


def register_imatrix_hook(model):
    """Registers hooks to accumulate activation squared norms into `imatrix`."""

    def get_imatrix_hook(module, input, output):
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
    imatrix_hooks = register_imatrix_hook(model)
    block_names = get_block_names(model, quant_vision=True)
    block_names = flatten_list(block_names)

    def move_to_gpu_hook(module, inputs):
        module.to(major_device)
        to_device(inputs, major_device)

    def move_to_cpu(module, inputs, outputs):
        module.to("cpu")

    def move_to_cpu_clear_memory(module, inputs, outputs):
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

    def __init__(self, message):
        super().__init__(message)


last_grad_input = None


def prepare_model_low_gpu(model, block_inputs: dict = None, pbar=None, major_device="cpu"):
    block_inputs.clear()
    for n, m in model.named_modules():
        if hasattr(m, "grad_mode"):
            m.grad_mode = False

    block_names = get_block_names(model)[0]

    def wrap_forward(module, module_name):
        original_forward = module.forward

        @wraps(original_forward)
        def new_forward(*args, **kwargs):
            move_module_to_tuning_device(module, major_device=major_device)
            # for n,m in module.named_modules():
            #     if hasattr(m, "post_init_qdqw"):
            #         m.post_init_qdqw()

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

            module.to("cpu")
            # clear_memory(device_list=major_device) #slow
            memory_monitor.log_summary()

            # Enable gradients for the output of the last block
            if module.tmp_name == block_names[-1]:
                if isinstance(result, torch.Tensor):
                    result = result.requires_grad_(True)
                elif isinstance(result, tuple):
                    result = tuple(r.requires_grad_(True) if isinstance(r, torch.Tensor) else r for r in result)

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


def model_forward_low_gpu(model, dataloader, major_device="cuda", pbar=None):
    block_inputs = {}

    block_names = get_block_names(model)[0]
    for name in block_names:
        module = get_module(model, name)
        module.orig_forward = module.forward

    def backward_pre_hook(module, grad_input):
        """Hook executed before backward propagation."""
        global last_grad_input
        last_grad_input = grad_input
        get_current_device_manager().synchronize()
        raise MyCustomError("Interrupt backward pass")

    for data in dataloader:
        prepare_model_low_gpu(model, block_inputs, major_device=major_device, pbar=pbar)

        # Register backward hook on the last block
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
            # Backward pass (will be interrupted by the hook)
            output.loss.to(torch.float32).backward()
        except MyCustomError:
            pass

        current_grad = last_grad_input
        del output, data

        # Manually compute gradients block by block
        last_block_backward_hook.remove()

        for name in block_names:
            module = get_module(model, name)
            module.forward = module.orig_forward
        index = 0
        for block_name in reversed(block_names):
            index += 1
            # Retrieve stored inputs for the block
            block_input_info = block_inputs.get(block_name, {})

            block_input_args = to_device(block_input_info.get("args", []), major_device)
            block_input_kwargs = to_device(block_input_info.get("kwargs", {}), major_device)
            block_input_args[0].requires_grad_(True)

            # Move the block module to GPU
            block_module = get_module(model, block_name)
            for n, m in block_module.named_modules():
                if hasattr(m, "grad_mode"):
                    m.grad_mode = True
            move_module_to_tuning_device(block_module, major_device=major_device)

            # Set the block to eval mode while enabling gradient computation
            block_module.eval()

            # Recompute the block output
            block_output = block_module(*block_input_args, **block_input_kwargs)

            # Ensure the output requires gradients
            if isinstance(block_output, tuple):
                # For tuple outputs, we usually care about the first element (hidden states)
                main_output = block_output[0]
                main_output = main_output.requires_grad_(True)
            else:
                main_output = block_output.requires_grad_(True)

            # Backward pass for the current block
            torch.autograd.backward(
                tensors=main_output,
                # inputs=block_input_args,
                grad_tensors=current_grad,
                retain_graph=True,  # False may lead to zero gradients for some cases (e.g., MXFP4)
            )

            # Extract gradients w.r.t. the block input
            if block_input_args and isinstance(block_input_args[0], torch.Tensor):
                current_grad = block_input_args[0].grad.detach().clone()
            else:
                logger.warning(f"No suitable input gradient found for {block_name}")
                break

            del block_output, main_output, block_input_args, block_input_kwargs
            block_module.to("cpu")
            # if index%4==0:
            #     clear_memory(device_list=major_device) # this one is very slow and seems does not affect max ram usage
            memory_monitor.update()
            memory_monitor.log_summary()
            pbar.update(1)


def get_score_for_scheme(
    model,
    tokenizer,
    quant_layer_names,
    fixed_layer_scheme,
    dataset,
    ignore_scale_zp_bits=False,
    nsamples=16,
    seqlen=256,
    pbar=None,
    shared_layers=None,
    need_weight_grad=False,
    enable_torch_compile=False,
    low_gpu_mem_usage=True,
    major_device="cpu",
    batch_size=1,
    disable_opt_rtn=True,
    offload_context: Optional[OffloadManager] = None,
    processor=None,
    is_vlm: bool = False,
    force_mllm: bool = False,
    model_name: Optional[str] = None,
):
    scores_dict = {}  # Key=name,Val=[quant_total_bits, loss]
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
                m.tuning_device = m.weight.device

            new_m = WrapperLayer(
                m,
                device=device,
                enable_minmax_tuning=False,
                enable_norm_bias_tuning=False,
                enable_round_tuning=False,
                need_weight_grad=need_weight_grad,
                enable_torch_compile=enable_torch_compile,
                disable_opt_rtn=disable_opt_rtn,
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
            model_forward_low_gpu(model, mllm_loader, major_device=major_device, pbar=pbar)
        else:
            # try:
            dataloader = _build_calib_dataloader()
            model_forward_low_gpu(model, dataloader, major_device=major_device, pbar=pbar)
            # except Exception as exc:  # noqa: BLE001
            #     if not is_vlm:
            #         raise
            #     logger.warning(
            #         f"Text-only calibration failed on VLM ({exc}); "
            #         f"falling back to multimodal calibration dataloader."
            #     )
            #     mllm_loader = _build_mllm_calib_dataloader()
            #     batch_size = 1
            #     if mllm_loader is None:
            #         raise
            #     model_forward_low_gpu(model, mllm_loader, major_device=major_device, pbar=pbar)
    else:
        for n, m in model.named_modules():
            if hasattr(m, "grad_mode"):
                m.grad_mode = True
            # if hasattr(m, "post_init_qdqw"):
            #     m.post_init_qdqw()

        def _run_forward_loop(loader):
            _checked_pixel = False
            _pixel_keys = (
                "pixel_values",
                "pixel_values_videos",
                "pixel_values_images",
                "image_pixel_values",
                "images",
                "image",
            )
            for data in loader:
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
                # way AutoRoundMLLM.calib does).
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

    scores_dict = {}
    for n, m in model.named_modules():
        if hasattr(m, "mix_score"):
            if m.orig_layer.act_bits <= 8:
                if m.act_cnt == 0:
                    logger.warning_once(
                        "layer{n} max abs activation is 0, please use more data to improve the accuracy"
                    )
            layer_bits, _ = compute_layer_bits(m.orig_layer, ignore_scale_zp_bits=ignore_scale_zp_bits)
            scores_dict[n] = [layer_bits, m.mix_score]
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
    # dp: total_params -> (accumulated_loss, chosen_path)
    # The path explicitly stores the selected options.
    dp: dict[int, tuple[float, list]] = {0: (0.0, [])}
    for layer_name, opts in layers.items():
        new_dp: dict[int, tuple[float, list]] = {}
        for cur_params, (cur_loss, cur_path) in dp.items():
            for opt in opts:
                scheme, bits_cost, loss_cost, layer_names = opt
                np_total = cur_params + bits_cost
                if np_total > P:
                    continue

                new_loss = cur_loss + loss_cost
                new_path = cur_path + [(layer_names, scheme)]

                # Keep the path with smaller loss for the same parameter budget
                if np_total not in new_dp or new_loss < new_dp[np_total][0]:
                    new_dp[np_total] = (new_loss, new_path)

        if not new_dp:
            return None, None
        # Pareto pruning: remove dominated (params, loss) states
        items = sorted(new_dp.items(), key=lambda x: x[0])  # (params, (loss, path))
        pruned: dict[int, tuple[float, list]] = {}
        best_loss_so_far = float("inf")
        for params_val, (loss_val, path_val) in items:
            if loss_val < best_loss_so_far:
                pruned[params_val] = (loss_val, path_val)
                best_loss_so_far = loss_val

        # Beam width limit: if too many states survive Pareto pruning,
        # uniformly subsample to bound memory usage.  For models with many
        # layers whose sizes are incommensurate, the number of distinct
        # cumulative-bit sums can grow to millions, each storing a full
        # path copy — easily exceeding 70 GB of RAM.
        if max_states is not None and len(pruned) > max_states:
            sorted_keys = sorted(pruned.keys())
            n = len(sorted_keys)
            # Uniformly pick max_states indices (always include first and last)
            step = (n - 1) / (max_states - 1)
            selected: dict[int, tuple[float, list]] = {}
            for i in range(max_states):
                idx = int(round(i * step))
                k = sorted_keys[idx]
                selected[k] = pruned[k]
            pruned = selected

        dp = pruned

    # Select the solution with the minimum loss
    best_params = min(dp.keys(), key=lambda k: dp[k][0])
    best_loss, best_path = dp[best_params]
    return best_loss, best_path


def move_module_to_tuning_device(module, major_device="cpu"):

    def _normalize(dev):
        return dev if isinstance(dev, torch.device) else torch.device(dev)

    def _move_own_tensors(m, device):
        # Cover non-leaf modules that directly hold nn.Parameter / buffers
        # (e.g. Mamba/GDN linear_attn with A_log & dt_bias). Also relocate
        # p.grad together with p.data — otherwise the next backward's grad
        # accumulation hits a cuda/cpu device mismatch.
        target = _normalize(device)
        for p in m.parameters(recurse=False):
            if p.device != target:  # TODO have check
                p.data = p.data.to(target)
            if p.grad is not None and p.grad.device != target:
                p.grad.data = p.grad.data.to(target)
        for b_name, b in list(m.named_buffers(recurse=False)):
            if b is None:
                continue
            if b.device != target:
                m._buffers[b_name] = b.to(target)

    for n, m in module.named_modules():
        # if "imatrix" in m.__class__.__name__.lower():
        #     target = m.orig_layer.tuning_device
        #     if hasattr(m, "qdq_w"):
        #         m.qdq_w.to(target)
        #         if m.orig_layer.bias is not None:
        #             m.orig_layer.bias.to(major_device)
        #     else:
        #         m.to(major_device)
        #     _move_own_tensors(m, target)

        if hasattr(m, "orig_layer"):
            target = m.orig_layer.tuning_device
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


# Delta loss does not handle lm-head well, it is prone to assign low bit to lm-head which is not optimal
def _apply_head_trick(head_name, schemes, sorted_indices, target_bits, target_params_cnt, total_scores):

    # ------------------------------------------------------------------ #
    # lm_head option restriction for DP                                   #
    # lm_head is critical — its quantization error goes directly into     #
    # logits with no subsequent LayerNorm dampening. Instead of removing  #
    # it from DP, we restrict its candidate options so the DP can only    #
    # assign it >= 6 bit schemes, then let DP globally optimize.          #
    #                                                                      #
    # Rules (only if user hasn't already fixed it):                        #
    #   1. No option has bits >= 6      → keep all options (DP picks best)#
    #   2. Exactly one option bits >= 6 → restrict to only that option    #
    #   3. Multiple options bits >= 6:                                      #
    #      - target_bits > 6  → restrict to only the highest-bit option   #
    #      - target_bits <= 6 → keep all >=6 options, let DP decide       #
    # ------------------------------------------------------------------ #

    high_bit_indices = [i for i in range(len(schemes)) if _get_scheme_bits(schemes[i]) >= 6]

    if len(high_bit_indices) == 0:
        # Rule 1: no option >= 6 bit → restrict to the lowest-loss scheme
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


def _gen_layer_config(
    auto_scheme: AutoScheme,
    model: Union[str, torch.nn.Module],
    quant_layer_names: Iterable[str],
    fixed_layer_scheme: dict[str, dict],
    min_avg_bit_scheme,
    dataset: str = "pile-10k",
    tokenizer=None,
    device_map=None,
    enable_torch_compile=False,
    disable_opt_rtn=True,
    model_name=None,
    major_device="cpu",
    device_list=None,
    processor=None,
    is_vlm: bool = False,
):
    # Initialize memory tracking for AutoScheme
    memory_monitor = MemoryMonitor()
    # memory_monitor.reset()
    memory_monitor.update_cpu()

    # Create offload context for CPU RAM optimization
    # Note: low_cpu_mem_usage only works when low_gpu_mem_usage is also enabled,
    # because it requires layer-by-layer processing
    offload_context = None
    if auto_scheme.low_cpu_mem_usage and auto_scheme.low_gpu_mem_usage:
        _model_dir = model_name
        if _model_dir is None and hasattr(model, "config"):
            _model_dir = getattr(model.config, "_name_or_path", None)
        offload_mode = "clean"
        offload_kwargs = {"model_dir": _model_dir}
        # Rotation mutates weights in memory before AutoScheme starts. Clean-mode
        # reloads from the original checkpoint and would silently discard those
        # transformed weights during scoring and final restore.
        if getattr(model, "rotation_config", None):
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

    # Decide whether AutoScheme has to score vision-tower layers (typically
    # because the user passed ``--quant_nontext_module``). Used below to
    # clamp batch_size to 1 (image sizes vary) and to pick the multimodal
    # dataloader. The actual switch from low_gpu to full forward+backward
    # is done upstream in ``gen_layer_config``.
    vision_markers = ("vision", "visual", "image", "img")
    force_mllm = is_vlm and any(any(marker in n.lower() for marker in vision_markers) for n in quant_layer_names)

    block_name = get_block_names(model)[0]  # TODO need change to support vlm
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
        if isinstance(scheme, str) and scheme.upper() == "BF16":
            return True
        if isinstance(scheme, QuantizationScheme):
            scheme = asdict(scheme)
            if scheme["bits"] >= 16 and scheme["act_bits"] >= 16:
                return True
        return False

    if auto_scheme.nsamples is not None:
        nsamples = auto_scheme.nsamples
    else:
        is_moe_model = False
        if hasattr(model, "config"):
            for key in model.config.to_dict().keys():
                if "moe" in key or "expert" in key:
                    is_moe_model = True
                    break
        if is_moe_model:
            logger.info(
                "The model appears to be an MoE  model. "
                "Using more samples to help generate a better auto-scheme recipe."
            )
            nsamples = 64
        else:
            nsamples = 16
    seqlen = auto_scheme.seqlen if auto_scheme.seqlen is not None else 256

    if auto_scheme.batch_size is not None:
        batch_size = auto_scheme.batch_size
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
    for index, scheme in enumerate(schemes):
        if check_bf16_scheme(scheme):
            continue
        if isinstance(scheme, str):
            scheme = asdict(preset_name_to_scheme(scheme))
        elif isinstance(scheme, QuantizationScheme):
            scheme = asdict(scheme)
        bits = scheme.get("bits", 16)
        act_bits = scheme.get("act_bits", 16)
        if bits <= 8 < act_bits:
            need_weight_grad = True
        if not auto_scheme.low_gpu_mem_usage:
            pbar_cnt += nsamples
        if auto_scheme.low_gpu_mem_usage:
            pbar_cnt += len(block_name) * 2 * ((nsamples + batch_size - 1) // batch_size)  # forward backward
    shared_layers = parse_shared_layers(model, auto_scheme.shared_layers)

    options_scores = []
    # if auto_scheme.low_gpu_mem_usage and not disable_opt_rtn:
    #     logger.warning("low_gpu_mem_usage is enabled, disable_opt_rtn will be set to True automatically")
    #     disable_opt_rtn = True
    disable_opt_rtn = False
    if not disable_opt_rtn:
        need_imatrix = False
        for scheme in schemes:
            if isinstance(scheme, str):
                scheme = asdict(preset_name_to_scheme(scheme))
            elif isinstance(scheme, QuantizationScheme):
                scheme = asdict(scheme)

            need_imatrix = scheme["super_bits"] is not None
            if need_imatrix:
                break
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
    for index, scheme in enumerate(schemes):
        apply_quant_scheme(
            model, quant_layer_names=quant_layer_names, fixed_layer_scheme=fixed_layer_scheme, scheme=scheme
        )
        scores = {}  # name: bits, loss
        if check_bf16_scheme(scheme):
            for n in quant_layer_names:
                if n in fixed_layer_scheme.keys():
                    continue
                m = get_module(model, n)
                bits, _ = compute_layer_bits(m, auto_scheme.ignore_scale_zp_bits)
                scores[n] = [bits, 0.0]
        else:
            scores = get_score_for_scheme(
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
                disable_opt_rtn=auto_scheme.disable_opt_rtn,
                offload_context=offload_context,
                processor=processor,
                is_vlm=is_vlm,
                force_mllm=force_mllm,
                model_name=model_name,
            )
        # Track peak RAM after each scheme scoring
        memory_monitor.update()
        memory_monitor.log_summary()

        new_scores = {}
        for share_layer in shared_layers:
            param_bits = 0
            tmp_loss = 0
            name_list = []
            for name in share_layer:
                if name in scores.keys():
                    param_bits += scores[name][0]
                    tmp_loss += scores[name][1]
                    name_list.append(name)
                    scores.pop(name)
            new_scores[name_list[0]] = [index, param_bits, tmp_loss, name_list]
        for name, item in scores.items():
            new_scores[name] = [index, item[0], item[1], [name]]
        options_total_loss = 0.0
        for key, item in new_scores.items():
            options_total_loss += item[2]
            if key in total_scores:
                total_scores[key].append(item)
            else:
                total_scores[key] = [item]
        options_scores.append(options_total_loss)
        clear_memory(device_list=device_list)

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

    target_params_cnt = int(total_params * target_bits)
    sorted_indices = sorted(range(len(options_scores)), key=lambda i: options_scores[i])
    # Layers that are not fixed in fixed_layer_scheme
    not_fixed_embedding_layers_names = [
        name for name in embedding_layers_names if (name not in fixed_layer_scheme and name in quant_layer_names)
    ]

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

    # Compute bits already consumed by user-fixed layers (excluding embeddings we'll set)
    already_fixed_bits = 0
    for name in fixed_layer_scheme.keys():
        m = get_module(model, name)
        layer_bits, _ = compute_layer_bits(m, auto_scheme.ignore_scale_zp_bits)
        already_fixed_bits += layer_bits

    # Compute minimum bits needed for DP layers (non-fixed, non-embedding)
    min_dp_bits = 0
    for layer_name, opts in total_scores.items():
        min_dp_bits += min(opt[1] for opt in opts)

    def _fits_budget(scheme_dict):
        """Check if applying scheme_dict to embeddings leaves enough budget for DP layers."""
        emb_bits = _compute_embedding_bits(scheme_dict)
        remaining = target_params_cnt - already_fixed_bits - emb_bits
        return remaining >= min_dp_bits

    def _select_embedding_scheme_index():
        """Select the best scheme index for embedding layers based on model type and target_bits.

        For models with shared lm_head (tie_word_embeddings=True):
          - target_bits > 6: use the lowest-loss option (same as before)
          - target_bits <= 6: use the lowest-loss option among those with bits <= 6
        For models without shared lm_head:
          - use the lowest-loss option among those with bits >= ceil(target_bits)

        In all cases, the selected scheme must not exceed the total bit budget
        (i.e., embedding bits + fixed bits + min DP bits <= target_params_cnt).
        """
        import math

        if has_tied_lm_head:
            if target_bits > 6:
                candidates = list(sorted_indices)
            else:
                # Prefer options with bits <= 6, sorted by loss
                candidates = [idx for idx in sorted_indices if _get_scheme_bits(schemes[idx]) <= 6]
                if not candidates:
                    candidates = list(sorted_indices)
        else:
            # Not shared lm_head: prefer options with bits < floor(target_bits)
            floor_bits = math.floor(target_bits)
            candidates = [idx for idx in sorted_indices if _get_scheme_bits(schemes[idx]) == floor_bits]
            if not candidates:
                # find the first bits that greater than floor bits
                embedding_bits = [bits for idx in sorted_indices if _get_scheme_bits(schemes[idx]) > floor_bits]
                if len(embedding_bits) > 0:
                    sorted(embedding_bits)
                    embedding_bits = embedding_bits[0]
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
    for name in fixed_layer_scheme.keys():  # The Scheme should have been applied
        m = get_module(model, name)
        layer_bits, _ = compute_layer_bits(m, auto_scheme.ignore_scale_zp_bits)
        target_params_cnt -= layer_bits

    head_name = get_lm_head_name(model)
    if head_name is not None and (head_name not in fixed_layer_scheme and head_name in quant_layer_names):
        _apply_head_trick(head_name, schemes, sorted_indices, target_bits, target_params_cnt, total_scores)
    # As a little fo calibration data is used and embedding is a sparse op, we could not get reliable score
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

    if target_params_cnt <= 0:
        raise ValueError("Avg bits is too small")

    remove_quant_scheme(model)  # Must place after minus fixed_layer
    memory_monitor.update()
    memory_monitor.log_summary()

    best_loss, best_path = choose_bits_per_layer_with_path(total_scores, target_params_cnt)

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
    if model_name is not None:
        model = None
        del model
    else:
        model = model.to("cpu")  # TODO this requires large ram
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
    global last_grad_input
    last_grad_input = None
    clear_memory(device_list=device_list)

    # # Log AutoScheme memory usage
    # memory_monitor.update_cpu()
    low_cpu_str = "enabled" if auto_scheme.low_cpu_mem_usage else "disabled"
    memory_monitor.log_summary(f"AutoScheme complete (low_cpu_mem_usage={low_cpu_str})")

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
    enable_torch_compile=False,
    disable_opt_rtn=True,
    low_gpu_mem_usage=True,
    min_avg_bit_scheme=None,
    processor=None,
    **kwargs,
):
    model_name = None
    is_vlm = False
    if isinstance(model, str):
        model_name = model
        # Detect VLM (Qwen-VL / Qwen3-VL / LLaVA / etc.) and load via the MLLM
        # path so we get a usable ``processor`` for the multimodal calibration
        # dataloader. The block walker auto-detects VLMs too and skips the
        # vision tower when scoring.
        is_vlm = is_mllm_model(model_name)
        if is_vlm:
            model, processor, tokenizer, _ = mllm_load_model(
                model_name,
                device="cpu",
                use_auto_mapping=False,
            )
        else:
            # Load model on CPU only; do not apply automatic device map or tuning-aware placement at load time.
            model, tokenizer, _ = llm_load_model(model_name, device_map="cpu")
    else:
        # Object passed in: still try to detect VLM so we can pick the right dataloader later.
        try:
            is_vlm = is_mllm_model(model)
        except Exception:  # noqa: BLE001
            is_vlm = False

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
        model.to("cpu")
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
            disable_opt_rtn=disable_opt_rtn,
            device_map=device_map,
            major_device=major_device,
            device_list=device_list,
            min_avg_bit_scheme=min_avg_bit_scheme,
            processor=processor,
            is_vlm=is_vlm,
        )
    except torch.OutOfMemoryError:
        logger.warning(
            "Fallback to CPU for automatic scheme generation."
            " Using multiple devices is strongly recommended (e.g., --device_map 0,1,2,3)."
        )
        model.to("cpu")
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
            disable_opt_rtn=disable_opt_rtn,
            device_map=device_map,
            major_device=major_device,
            device_list=device_list,
            min_avg_bit_scheme=min_avg_bit_scheme,
            processor=processor,
            is_vlm=is_vlm,
        )

    return res
