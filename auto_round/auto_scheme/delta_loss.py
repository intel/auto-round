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
import os
import shutil
import tempfile

from safetensors.torch import load_file, save_file
from dataclasses import asdict
from functools import wraps
from typing import Any, Iterable, Optional, Union

import torch
from accelerate import dispatch_model
from tqdm import tqdm

from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.auto_scheme.register import register_scheme_methods
from auto_round.auto_scheme.utils import (
    apply_quant_scheme,
    compute_avg_bits_for_scheme,
    compute_layer_bits,
    dispatch_model_by_all_available_devices,
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
    get_block_names,
    get_major_device,
    get_module,
    llm_load_model,
    parse_available_devices,
    set_avg_auto_device_map,
    set_module,
    set_non_auto_device_map,
    to_device,
)
from auto_round.utils.device import MemoryMonitor
from auto_round.wrapper import WrapperLinear

__all__ = ["gen_layer_config"]


def _group_layers_by_block(quant_layer_names, block_names):
    """Group quantization layer names by their containing block."""
    groups = {bn: [] for bn in block_names}
    non_block = []
    for name in quant_layer_names:
        matched = False
        for bn in block_names:
            if name.startswith(bn + "."):
                groups[bn].append(name)
                matched = True
                break
        if not matched:
            non_block.append(name)
    return groups, non_block


# ============================================================================
# CPU RAM Offload Management for AutoScheme
# ============================================================================


class AutoSchemeOffloadContext:
    """Manages disk offload state for AutoScheme to reduce CPU RAM usage.

    Maintains two separate on-disk stores:
      - *original* weights (unwrapped, saved once at the start)
      - *wrapped* weights (saved per-scheme iteration for forward/backward)

    This allows block weights to stay on disk between scheme iterations,
    keeping only one block in RAM at a time.
    """

    def __init__(self, low_cpu_mem_usage: bool = False):
        self.low_cpu_mem_usage = low_cpu_mem_usage
        # Wrapped-state storage (changes every scheme iteration)
        self._offload_tempdir: Optional[str] = None
        self._offloaded_blocks: dict[str, dict] = {}
        # Original-state storage (saved once, read many)
        self._original_dir: Optional[str] = None
        self._original_blocks: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    def init_offload_dir(self) -> Optional[str]:
        """Initialize the temporary directory for wrapped-state offload."""
        if not self.low_cpu_mem_usage:
            return None
        if self._offload_tempdir is None:
            self._offload_tempdir = tempfile.mkdtemp(prefix="autoscheme_offload_")
            logger.info(f"AutoScheme CPU offload directory: {self._offload_tempdir}")
        return self._offload_tempdir

    def _init_original_dir(self) -> str:
        """Initialize the temporary directory for original-weight storage."""
        if self._original_dir is None:
            self._original_dir = tempfile.mkdtemp(prefix="autoscheme_original_")
        return self._original_dir

    # ------------------------------------------------------------------
    # Original (unwrapped) weight management — saved once at start
    # ------------------------------------------------------------------
    def save_original_block_weights(self, block_name: str, block: torch.nn.Module) -> None:
        """Save original (unwrapped) block weights to disk. Skips if already saved."""
        if not self.low_cpu_mem_usage:
            return
        if block_name in self._original_blocks:
            return
        orig_dir = self._init_original_dir()
        safe_name = block_name.replace(".", "_")
        save_path = os.path.join(orig_dir, f"{safe_name}.safetensors")
        try:
            state_dict = {k: v.cpu().contiguous() for k, v in block.state_dict().items()}
            save_file(state_dict, save_path)
            self._original_blocks[block_name] = {"save_path": save_path}
            del state_dict
        except Exception as e:
            logger.warning(f"Failed to save original block {block_name}: {e}")

    def _load_state_into_block(self, save_path: str, block: torch.nn.Module) -> None:
        """Low-level helper: load a safetensors file into *block*."""
        state_dict = load_file(save_path, device="cpu")
        for name, param in state_dict.items():
            parts = name.split(".")
            target = block
            try:
                for part in parts[:-1]:
                    target = getattr(target, part)
            except AttributeError:
                continue  # key belongs to a different module tree (e.g. wrapper vs orig)
            param_name = parts[-1]
            if hasattr(target, param_name):
                old_param = getattr(target, param_name)
                if isinstance(old_param, torch.nn.Parameter):
                    setattr(target, param_name,
                            torch.nn.Parameter(param, requires_grad=old_param.requires_grad))
                else:
                    setattr(target, param_name, param)
        del state_dict

    def load_original_block_weights(self, block_name: str, block: torch.nn.Module) -> None:
        """Load original (unwrapped) weights from disk into *block*."""
        if not self.low_cpu_mem_usage:
            return
        metadata = self._original_blocks.get(block_name)
        if not metadata:
            return
        save_path = metadata["save_path"]
        if not os.path.exists(save_path):
            logger.warning(f"Original weights not found: {save_path}")
            return
        try:
            self._load_state_into_block(save_path, block)
        except Exception as e:
            logger.warning(f"Failed to load original block {block_name}: {e}")

    def save_and_clear_all_original_blocks(
        self, model: torch.nn.Module, block_names: list[str]
    ) -> None:
        """Save all original block weights to disk and clear them from RAM."""
        if not self.low_cpu_mem_usage:
            return
        logger.info("AutoScheme: saving original block weights to disk...")
        for block_name in block_names:
            block = get_module(model, block_name)
            if block is not None:
                self.save_original_block_weights(block_name, block)
                for submodule in block.modules():
                    _clear_module_weights(submodule)
        gc.collect()
        clear_memory()
        logger.info("AutoScheme: original weights saved and cleared")

    def load_all_original_blocks(
        self, model: torch.nn.Module, block_names: list[str]
    ) -> None:
        """Load all original block weights back into RAM."""
        if not self.low_cpu_mem_usage:
            return
        for block_name in block_names:
            block = get_module(model, block_name)
            if block is not None:
                self.load_original_block_weights(block_name, block)

    # ------------------------------------------------------------------
    # Wrapped-state management — re-saved each scheme iteration
    # ------------------------------------------------------------------
    def offload_block_weights(self, block_name: str, block: torch.nn.Module) -> None:
        """Offload a block's (possibly wrapped) weights to disk."""
        if not self.low_cpu_mem_usage:
            return
        offload_dir = self.init_offload_dir()
        if offload_dir is None:
            return

        safe_name = block_name.replace(".", "_")
        save_path = os.path.join(offload_dir, f"{safe_name}.safetensors")

        already_saved = block_name in self._offloaded_blocks

        try:
            if not already_saved:
                state_dict = {k: v.cpu().contiguous() for k, v in block.state_dict().items()}
                save_file(state_dict, save_path)
                self._offloaded_blocks[block_name] = {"save_path": save_path}
                del state_dict

            for submodule in block.modules():
                _clear_module_weights(submodule)
        except Exception as e:
            logger.warning(f"Failed to offload block {block_name}: {e}")

    def load_block_weights(self, block_name: str, block: torch.nn.Module) -> None:
        """Load wrapped block weights from disk back into memory."""
        if not self.low_cpu_mem_usage:
            return
        metadata = self._offloaded_blocks.get(block_name)
        if not metadata:
            return
        save_path = metadata.get("save_path")
        if not save_path or not os.path.exists(save_path):
            logger.warning(f"Cannot load block weights: file {save_path} does not exist")
            return
        try:
            self._load_state_into_block(save_path, block)
        except Exception as e:
            logger.warning(f"Failed to load block weights from {save_path}: {e}")

    def offload_all_blocks(self, model: torch.nn.Module, block_names: list[str]) -> None:
        """Offload all block weights to disk."""
        if not self.low_cpu_mem_usage:
            return
        logger.info("AutoScheme: offloading all block weights to disk for RAM optimization...")
        for block_name in block_names:
            block = get_module(model, block_name)
            if block is not None:
                self.offload_block_weights(block_name, block)
        gc.collect()
        clear_memory()
        logger.info("AutoScheme: block weights offload complete")

    def reset_scheme_state(self) -> None:
        """Clear wrapped-state tracking so the next scheme iteration re-saves."""
        self._offloaded_blocks = {}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        """Clean up all temporary directories."""
        for d in (self._offload_tempdir, self._original_dir):
            if d and os.path.isdir(d):
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    logger.warning(f"Failed to cleanup dir {d}: {e}")
        self._offload_tempdir = None
        self._offloaded_blocks = {}
        self._original_dir = None
        self._original_blocks = {}


def _clear_module_weights(module: torch.nn.Module) -> None:
    """Clear module's weight and bias to free CPU RAM.

    Note: Skips WrapperLayer modules since their weight/bias are properties
    that delegate to orig_layer. Clearing the actual orig_layer is sufficient.
    Caches weight.numel() as ``_cached_weight_numel`` before clearing so that
    ``compute_layer_bits`` can still compute correct results with empty tensors.
    """
    if module is None:
        return
    # Skip WrapperLayer - its weight is a property, assigning to it would create
    # an instance attribute shadowing the property. We clear orig_layer directly instead.
    if hasattr(module, "orig_layer"):
        return
    with torch.no_grad():
        if hasattr(module, "weight") and module.weight is not None:
            # Cache numel / shape before replacing with empty tensor
            if module.weight.numel() > 0:
                module._cached_weight_numel = module.weight.numel()
                module._cached_weight_shape = tuple(module.weight.shape)
            if isinstance(module.weight, torch.nn.Parameter):
                module.weight = torch.nn.Parameter(
                    torch.empty(0, dtype=module.weight.dtype, device="cpu"),
                    requires_grad=module.weight.requires_grad
                )
            else:
                module.weight = torch.empty(0, dtype=module.weight.dtype, device="cpu")
        if hasattr(module, "bias") and module.bias is not None:
            if isinstance(module.bias, torch.nn.Parameter):
                module.bias = torch.nn.Parameter(
                    torch.empty(0, dtype=module.bias.dtype, device="cpu"),
                    requires_grad=module.bias.requires_grad
                )
            else:
                module.bias = torch.empty(0, dtype=module.bias.dtype, device="cpu")


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
        # self.device = device
        self.max_act_value = 0
        self.need_weight_grad = need_weight_grad
        self.grad_mode = False
        if self.need_weight_grad:
            self.orig_layer.weight.requires_grad = True

    def _qdq_act(self, x, act_max_scale, act_max=None):
        if hasattr(self.orig_layer, "act_bits") and self.orig_layer.act_bits > 8:
            return x, 1.0, None

        qdq_x, scale, zp = self.act_qdq_func(x, act_max_scale, act_max)
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

        if self.need_weight_grad:

            def save_grad(grad):
                w_diff = self.orig_layer.weight - self.qdq_w.to(self.orig_layer.weight.device)
                # TODO strange, grad could be in CPU
                self.weight_score += torch.abs((grad.to(w_diff.device) * w_diff)).sum().item()  # TODO add 2nd order
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
        with torch.no_grad():
            qdq_w = self._init_scale().detach()
            self.register_buffer("qdq_w", qdq_w.detach().clone().to(self.orig_layer.weight.device))
        if self.need_weight_grad:

            def save_grad(grad):
                w_diff = self.orig_layer.weight - self.qdq_w.to(self.orig_layer.weight.device)
                self.weight_score += torch.abs((grad * w_diff.to(grad.device))).sum().item()
                act_score = 0.0 if self.act_cnt <= 0 else self.total_act_score / self.act_cnt
                self.mix_score = self.weight_score + act_score
                return None

            self.qdq_w.requires_grad_(True)
            self.orig_layer.weight.requires_grad_(False)

            self.qdq_w.register_hook(save_grad)

    @torch.no_grad()
    def _init_scale(self):
        tensor = self.orig_layer.weight.data
        bits = self.orig_layer.bits
        scale_dtype = self.orig_layer.scale_dtype
        imatrix = self.orig_layer.imatrix
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


@torch.no_grad()
def cal_imatrix(model, dataloader):
    def register_act_hook(model):
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

    hooks = register_act_hook(model)
    for data in dataloader:
        model.forward(**to_device(data, model.device))
    for hook in hooks:
        hook.remove()


class MyCustomError(Exception):

    def __init__(self, message):
        super().__init__(message)


last_grad_input = None


def prepare_model_low_gpu(
    model,
    block_inputs: dict = None,
    pbar=None,
    major_device="cpu",
    offload_context: Optional[AutoSchemeOffloadContext] = None,
):
    block_inputs.clear()
    for n, m in model.named_modules():
        if hasattr(m, "grad_mode"):
            m.grad_mode = False

    block_names = get_block_names(model)[0]

    def wrap_forward(module, module_name):
        original_forward = module.forward

        @wraps(original_forward)
        def new_forward(*args, **kwargs):
            # Load block weights from disk if using low_cpu_mem_usage
            if offload_context is not None:
                offload_context.load_block_weights(module_name, module)

            move_module_to_tuning_device(module, major_device=major_device)

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
            # torch.cuda.empty_cache()

            # Offload block weights back to disk if using low_cpu_mem_usage
            if offload_context is not None:
                offload_context.offload_block_weights(module_name, module)

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


def model_forward_low_gpu(
    model,
    dataloader,
    major_device="cuda",
    pbar=None,
    offload_context: Optional[AutoSchemeOffloadContext] = None,
):
    block_inputs = {}

    block_names = get_block_names(model)[0]
    for name in block_names:
        module = get_module(model, name)
        module.orig_forward = module.forward

    def backward_pre_hook(module, grad_input):
        """Hook executed before backward propagation."""
        global last_grad_input
        last_grad_input = grad_input
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        raise MyCustomError("Interrupt backward pass")

    for data in dataloader:
        prepare_model_low_gpu(
            model, block_inputs, major_device=major_device, pbar=pbar, offload_context=offload_context
        )

        # Register backward hook on the last block
        last_block = get_module(model, block_names[-1])
        last_block_backward_hook = last_block.register_full_backward_pre_hook(backward_pre_hook)

        data = to_device(data, model.device)
        output = model.forward(**data, labels=data["input_ids"], use_cache=False)

        try:
            # Backward pass (will be interrupted by the hook)
            output.loss.backward()
        except MyCustomError:
            pass

        current_grad = last_grad_input

        # Manually compute gradients block by block
        last_block_backward_hook.remove()

        for name in block_names:
            module = get_module(model, name)
            module.forward = module.orig_forward

        for block_name in reversed(block_names):

            # Retrieve stored inputs for the block
            block_input_info = block_inputs.get(block_name, {})

            block_input_args = to_device(block_input_info.get("args", []), major_device)
            block_input_kwargs = to_device(block_input_info.get("kwargs", {}), major_device)
            block_input_args[0].requires_grad_(True)

            # Move the block module to GPU
            block_module = get_module(model, block_name)

            # Load block weights from disk if offloaded (for backward pass)
            if offload_context is not None:
                offload_context.load_block_weights(block_name, block_module)

            for n, m in block_module.named_modules():
                if hasattr(m, "grad_mode"):
                    m.grad_mode = True
                if hasattr(m, "orig_layer"):
                    m.to(m.orig_layer.tuning_device)
                elif hasattr(m, "tuning_device"):
                    m.to(m.tuning_device)
                elif len(list(m.children())) == 0:
                    m.to(major_device)

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
                print(f"Warning: No suitable input gradient found for {block_name}")
                break

            block_module.to("cpu")
            # clear_memory()

            # Offload block weights to disk if low_cpu_mem_usage is enabled (after backward)
            if offload_context is not None:
                offload_context.offload_block_weights(block_name, block_module)

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
    low_cpu_mem_usage=False,
    major_device="cpu",
    batch_size=1,
    disable_opt_rtn=True,
    offload_context: Optional[AutoSchemeOffloadContext] = None,
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

    def wrap_layer(name: str) -> None:
        if name in fixed_layer_scheme.keys():
            return
        m = get_module(model, name)
        if not check_to_quantized(m):
            layer_bits, _ = compute_layer_bits(m, ignore_scale_zp_bits)
            scores_dict[name] = [layer_bits, 0.0]
            return
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
                if "cuda" in device or "xpu" in device:
                    device = major_device
            else:
                device = m.weight.device
                m.tuning_device = m.weight.device

            new_m = WrapperLayer(
                m,
                device=device,
                enable_minmax_tuning=False,  # TODO this should be change
                enable_norm_bias_tuning=False,
                enable_round_tuning=False,
                need_weight_grad=need_weight_grad,
                enable_torch_compile=enable_torch_compile,
                disable_opt_rtn=disable_opt_rtn,
            )
            set_module(model, name, new_m)

    # ------------------------------------------------------------------
    # Wrapping + forward/backward
    # ------------------------------------------------------------------
    if low_gpu_mem_usage:
        dataloader = get_dataloader(tokenizer, seqlen, dataset_name=dataset, seed=42, bs=batch_size, nsamples=nsamples)

        if offload_context is not None and low_cpu_mem_usage:
            # Block-by-block wrapping: load original weights -> wrap -> offload wrapped state
            block_names = get_block_names(model)[0]
            layer_groups, non_block_layers = _group_layers_by_block(quant_layer_names, block_names)
            offload_context.reset_scheme_state()

            for block_name in block_names:
                block = get_module(model, block_name)
                offload_context.load_original_block_weights(block_name, block)
                for name in layer_groups.get(block_name, []):
                    wrap_layer(name)
                offload_context.offload_block_weights(block_name, block)

            # Wrap layers that live outside of blocks (e.g. lm_head)
            for name in non_block_layers:
                wrap_layer(name)

            gc.collect()
            clear_memory()
        else:
            for name in quant_layer_names:
                wrap_layer(name)

        model_forward_low_gpu(
            model, dataloader, major_device=major_device, pbar=pbar, offload_context=offload_context
        )

        # NOTE: do NOT load all blocks back — scores are read block-by-block below
    else:
        for name in quant_layer_names:
            wrap_layer(name)

        dataloader = get_dataloader(tokenizer, seqlen, dataset_name=dataset, seed=42, bs=batch_size, nsamples=nsamples)
        for data in dataloader:
            data = to_device(data, model.device)
            output = model.forward(**data, labels=data["input_ids"], use_cache=False)
            output.loss.backward()
            for n, m in model.named_parameters():  # This should be kept to reduce VRAM footprint
                m.grad = None
            if pbar is not None:
                pbar.update(1)

        for n, m in model.named_parameters():
            m.grad = None

    # ------------------------------------------------------------------
    # Score reading + unwrapping
    # ------------------------------------------------------------------
    if offload_context is not None and low_cpu_mem_usage:
        # Block-by-block: load wrapped state -> read scores -> unwrap -> clear
        scores_dict = {}
        for block_name in block_names:
            block = get_module(model, block_name)
            offload_context.load_block_weights(block_name, block)

            # Read scores from wrapper attributes in this block
            for n, m in block.named_modules():
                full_name = f"{block_name}.{n}" if n else block_name
                if hasattr(m, "mix_score"):
                    if m.orig_layer.act_bits <= 8:
                        if m.act_cnt == 0:
                            logger.warning_once(
                                f"layer {full_name} max abs activation is 0, "
                                "please use more data to improve the accuracy"
                            )
                    layer_bits, _ = compute_layer_bits(
                        m.orig_layer, ignore_scale_zp_bits=ignore_scale_zp_bits
                    )
                    scores_dict[full_name] = [layer_bits, m.mix_score]

            # Unwrap layers in this block
            unwrap_pairs = []
            for n, m in block.named_modules():
                full_name = f"{block_name}.{n}" if n else block_name
                if hasattr(m, "orig_layer"):
                    unwrap_pairs.append((full_name, m.orig_layer))
            for full_name, orig_layer in unwrap_pairs:
                set_module(model, full_name, orig_layer)

            # Clear weights so this block no longer occupies RAM
            block = get_module(model, block_name)
            for submodule in block.modules():
                _clear_module_weights(submodule)

        # Handle non-block layers
        for n, m in model.named_modules():
            if hasattr(m, "mix_score") and n not in scores_dict:
                if m.orig_layer.act_bits <= 8 and m.act_cnt == 0:
                    logger.warning_once(
                        f"layer {n} max abs activation is 0, "
                        "please use more data to improve the accuracy"
                    )
                layer_bits, _ = compute_layer_bits(
                    m.orig_layer, ignore_scale_zp_bits=ignore_scale_zp_bits
                )
                scores_dict[n] = [layer_bits, m.mix_score]
        for n, m in model.named_modules():
            if hasattr(m, "orig_layer"):
                set_module(model, n, m.orig_layer)

        gc.collect()
        clear_memory()
    else:
        scores_dict = {}
        for n, m in model.named_modules():
            if hasattr(m, "mix_score"):
                if m.orig_layer.act_bits <= 8:
                    if m.act_cnt == 0:
                        logger.warning_once(
                            f"layer {n} max abs activation is 0, "
                            "please use more data to improve the accuracy"
                        )
                layer_bits, _ = compute_layer_bits(m.orig_layer, ignore_scale_zp_bits=ignore_scale_zp_bits)
                scores_dict[n] = [layer_bits, m.mix_score]
        for n, m in model.named_modules():
            if hasattr(m, "orig_layer"):
                set_module(model, n, m.orig_layer)
    return scores_dict


def choose_bits_per_layer_with_path(layers: dict, P: int):
    """
    Args:
        layers: A dict mapping each layer name to a list of candidate options.
                Each option is a tuple of (scheme, bits_cost, loss_cost, layer_names).
        P: Upper bound on the total parameter (bit) budget.

    Returns:
        (min_loss, best_path), where best_path is a list of
        (layer_names, scheme) for each layer, or (None, None) if no feasible
        solution exists.
    """
    # dp: total_params -> accumulated_loss
    # Use backtracking pointers instead of storing full paths to avoid
    # O(layers) list copies per state transition.
    dp: dict[int, float] = {0: 0.0}

    layer_list = list(layers.items())
    total_layers = len(layer_list)
    logger.info(f"Starting DP for {total_layers} layers, budget P={P}")

    # history[layer_idx][params] = (prev_params, opt_idx) for path reconstruction
    history: list[dict[int, tuple[int, int]]] = []

    pbar = tqdm(range(total_layers), desc="DP bit allocation", leave=True)
    for idx in pbar:
        layer_name, opts = layer_list[idx]
        pbar.set_postfix(dp_states=len(dp))

        new_dp: dict[int, float] = {}
        new_bt: dict[int, tuple[int, int]] = {}

        for cur_params, cur_loss in dp.items():
            for opt_idx, opt in enumerate(opts):
                scheme, bits_cost, loss_cost, layer_names = opt
                np_total = cur_params + bits_cost
                if np_total > P:
                    continue

                new_loss = cur_loss + loss_cost

                # Keep the option with smaller loss for the same parameter budget
                if np_total not in new_dp or new_loss < new_dp[np_total]:
                    new_dp[np_total] = new_loss
                    new_bt[np_total] = (cur_params, opt_idx)

        if not new_dp:
            return None, None

        # Pareto pruning: remove dominated (params, loss) states
        items = sorted(new_dp.items(), key=lambda x: x[0])  # sort by params
        pruned_dp: dict[int, float] = {}
        pruned_bt: dict[int, tuple[int, int]] = {}
        best_loss_so_far = float("inf")
        for params_val, loss_val in items:
            if loss_val < best_loss_so_far:
                pruned_dp[params_val] = loss_val
                pruned_bt[params_val] = new_bt[params_val]
                best_loss_so_far = loss_val

        dp = pruned_dp
        history.append(pruned_bt)

    # Select the solution with the minimum loss
    best_params = min(dp.keys(), key=lambda k: dp[k])
    best_loss = dp[best_params]

    # Backtrack to reconstruct the chosen path
    path = []
    cur_params = best_params
    for layer_idx in range(total_layers - 1, -1, -1):
        prev_params, opt_idx = history[layer_idx][cur_params]
        scheme, bits_cost, loss_cost, layer_names = layer_list[layer_idx][1][opt_idx]
        path.append((layer_names, scheme))
        cur_params = prev_params
    path.reverse()

    return best_loss, path


def move_module_to_tuning_device(module, major_device="cpu"):
    for n, m in module.named_modules():
        if hasattr(m, "orig_layer"):
            m.to(m.orig_layer.tuning_device)
        elif hasattr(m, "tuning_device"):
            m.to(m.tuning_device)
        elif len(list(m.children())) == 0:
            m.to(major_device)


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
):
    # Initialize memory tracking for AutoScheme
    memory_monitor = MemoryMonitor()
    memory_monitor.reset()
    memory_monitor.update_cpu()

    # Create offload context for CPU RAM optimization
    # Note: low_cpu_mem_usage only works when low_gpu_mem_usage is also enabled,
    # because disk offloading requires layer-by-layer processing
    offload_context = None
    if auto_scheme.low_cpu_mem_usage and auto_scheme.low_gpu_mem_usage:
        offload_context = AutoSchemeOffloadContext(low_cpu_mem_usage=True)

    target_bits = auto_scheme.avg_bits
    model.eval()

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

    embedding_layers_names = []
    for name in quant_layer_names:
        module = get_module(model, name)
        if isinstance(module, torch.nn.Embedding):
            embedding_layers_names.append(name)
    quant_layer_names = list(set(quant_layer_names) - set(embedding_layers_names))

    options_scores = []
    if auto_scheme.low_gpu_mem_usage and not disable_opt_rtn:
        logger.warning("low_gpu_mem_usage is enabled, disable_opt_rtn will be set to True automatically")
        disable_opt_rtn = True
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
        if need_imatrix:  # TODO change to block way in low_gpu_mem_usage
            dataloader = get_dataloader(tokenizer, seqlen=256, dataset_name=dataset, seed=42, bs=8, nsamples=16)
            logger.info("start to compute imatrix for GGUF-K quantization in AutoScheme")
            cal_imatrix(model, dataloader)
            logger.info("finish calculating imatrix")

    # Offload all original block weights to disk before the scheme loop
    # so that only one block needs to be in RAM at a time during scoring.
    if offload_context is not None:
        offload_context.save_and_clear_all_original_blocks(model, block_name)

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
                low_cpu_mem_usage=auto_scheme.low_cpu_mem_usage,
                major_device=major_device,
                batch_size=batch_size,
                disable_opt_rtn=auto_scheme.disable_opt_rtn,
                offload_context=offload_context,
            )
        # Track peak RAM after each scheme scoring
        memory_monitor.update_cpu()

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

    # Restore original weights from disk for final bit-budget computations
    if offload_context is not None:
        offload_context.load_all_original_blocks(model, block_name)

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
    not_fixed_embedding_layers_names = [name for name in embedding_layers_names if name not in fixed_layer_scheme]
    for sorted_index in sorted_indices:
        tmp_scheme = schemes[sorted_index]
        if isinstance(tmp_scheme, str):
            tmp_scheme = asdict(preset_name_to_scheme(tmp_scheme))

        for embedding_layer_name in not_fixed_embedding_layers_names:

            fixed_layer_scheme[embedding_layer_name] = tmp_scheme
            embedding_layer = get_module(model, embedding_layer_name)
            for key, item in tmp_scheme.items():
                setattr(embedding_layer, key, item)
        current_avg_bits, _ = compute_avg_bits_for_scheme(
            model,
            quant_layer_names + embedding_layers_names,
            fixed_layer_scheme,
            min_avg_bit_scheme,
            ignore_scale_zp_bits=auto_scheme.ignore_scale_zp_bits,
            clean_scheme=False,
        )

        if current_avg_bits <= target_bits:  # compute_avg_bit remove setting
            for embedding_layer_name in not_fixed_embedding_layers_names:

                fixed_layer_scheme[embedding_layer_name] = tmp_scheme
                embedding_layer = get_module(model, embedding_layer_name)
                for key, item in tmp_scheme.items():
                    setattr(embedding_layer, key, item)
            break

    # Minus fixed_layer
    for name in fixed_layer_scheme.keys():  # The Scheme should have been applied
        m = get_module(model, name)
        layer_bits, _ = compute_layer_bits(m, auto_scheme.ignore_scale_zp_bits)
        target_params_cnt -= layer_bits
    if target_params_cnt <= 0:
        raise ValueError("Avg bits is too small")

    remove_quant_scheme(model)  # Must place after minus fixed_layer

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

    # Cleanup offload context
    if offload_context is not None:
        offload_context.cleanup()

    clear_memory(device_list=device_list)

    # Log AutoScheme memory usage
    memory_monitor.update_cpu()
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
    **kwargs,
):
    model_name = None
    if isinstance(model, str):
        model_name = model
        # Load model on CPU only; do not apply automatic device map or tuning-aware placement at load time.
        model, tokenizer, _ = llm_load_model(model_name, device_map="cpu")
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

    # Enable gradient checkpointing if supported
    if model.supports_gradient_checkpointing:
        model.gradient_checkpointing_enable()

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
        )

    return res
