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

"""Unified block-level disk offloading for CPU RAM optimization.

This module provides ``BlockOffloadManager``, the single point of truth for
saving / loading / clearing model block weights to / from disk.

It also provides ``load_block_from_model_files`` for reloading block weights
directly from the original HuggingFace model checkpoint (safetensors or
pytorch bin files) — eliminating the need to write an extra copy to a temp
directory when the original weights have not been modified.
"""

import gc
import json
import os
import shutil
import tempfile
from collections import defaultdict
from typing import Any, Optional

import torch

from auto_round.logger import logger
from auto_round.utils.model import get_module

__all__ = ["AutoSchemeOffloadContext", "BlockOffloadManager", "group_layers_by_block", "load_block_from_model_files"]


# =====================================================================
# Generic helpers
# =====================================================================


def group_layers_by_block(quant_layer_names, block_names):
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


# =====================================================================
# Low-level helpers
# =====================================================================


def _load_state_dict_into_block(state_dict: dict, block: torch.nn.Module) -> None:
    """Assign every key in *state_dict* to the corresponding sub-module of *block*.

    This handles the case where the target parameter has already been replaced
    by an empty tensor (after ``clear_block``), and where modules have been
    replaced by wrapper objects (e.g. ``WrapperLinear``) that store the original
    layer in an ``orig_layer`` attribute.
    """
    for name, param in state_dict.items():
        parts = name.split(".")
        target = block
        try:
            for part in parts[:-1]:
                target = getattr(target, part)
        except AttributeError:
            continue  # key belongs to a different module tree
        # If target is a wrapper, redirect to the underlying module
        if hasattr(target, "orig_layer"):
            target = target.orig_layer
        param_name = parts[-1]
        if hasattr(target, param_name):
            old_param = getattr(target, param_name)
            if isinstance(old_param, torch.nn.Parameter):
                # Cast to the module's existing dtype/device to avoid mismatches
                param = param.to(dtype=old_param.dtype, device=old_param.device)
                setattr(target, param_name, torch.nn.Parameter(param, requires_grad=old_param.requires_grad))
            else:
                setattr(target, param_name, param)
    del state_dict


def _clear_submodule_weights(module: torch.nn.Module, cache_numel: bool = False) -> None:
    """Replace a single (leaf-ish) module's weight / bias with empty tensors.

    Args:
        module: The module to clear.
        cache_numel: If *True*, store ``_cached_weight_numel`` and
            ``_cached_weight_shape`` before clearing so that downstream code
            (e.g. ``compute_layer_bits``) can still obtain the original size.
    """
    if module is None:
        return
    # Skip WrapperLayer – its weight is a property delegating to orig_layer.
    if hasattr(module, "orig_layer"):
        return

    with torch.no_grad():
        if hasattr(module, "weight") and module.weight is not None:
            if cache_numel and module.weight.numel() > 0:
                module._cached_weight_numel = module.weight.numel()
                module._cached_weight_shape = tuple(module.weight.shape)
            if isinstance(module.weight, torch.nn.Parameter):
                module.weight = torch.nn.Parameter(
                    torch.empty(0, dtype=module.weight.dtype, device="cpu"),
                    requires_grad=module.weight.requires_grad,
                )
            else:
                module.weight = torch.empty(0, dtype=module.weight.dtype, device="cpu")
        if hasattr(module, "bias") and module.bias is not None:
            if isinstance(module.bias, torch.nn.Parameter):
                module.bias = torch.nn.Parameter(
                    torch.empty(0, dtype=module.bias.dtype, device="cpu"),
                    requires_grad=module.bias.requires_grad,
                )
            else:
                module.bias = torch.empty(0, dtype=module.bias.dtype, device="cpu")


# =====================================================================
# Reload from original model checkpoint
# =====================================================================

def _resolve_model_dir(model_dir: str) -> str:
    """Resolve a model name/path to a local directory containing weight files.

    Handles both local paths and HuggingFace hub cache paths.
    """
    if os.path.isdir(model_dir):
        return model_dir
    # Try HuggingFace hub cache resolution
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_dir, local_files_only=True)
    except Exception:
        return model_dir


def _build_weight_map(model_dir: str) -> dict[str, str]:
    """Build a mapping  ``{tensor_name: shard_filename}`` from the model directory.

    Supports both sharded models (with ``model.safetensors.index.json``) and
    single-file models (``model.safetensors``).
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index_data = json.load(f)
        return index_data["weight_map"]

    single_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single_path):
        from safetensors import safe_open
        with safe_open(single_path, framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}

    # Fallback: pytorch binary format
    bin_index_path = os.path.join(model_dir, "pytorch_model.bin.index.json")
    if os.path.exists(bin_index_path):
        with open(bin_index_path) as f:
            index_data = json.load(f)
        return index_data["weight_map"]

    single_bin = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(single_bin):
        state_dict = torch.load(single_bin, map_location="cpu")
        return {k: "pytorch_model.bin" for k in state_dict.keys()}

    raise FileNotFoundError(
        f"Could not find model weight files in {model_dir}. "
        "Expected model.safetensors or pytorch_model.bin (with optional index.json)."
    )


def load_block_from_model_files(
    model_dir: str,
    block_name: str,
    block: torch.nn.Module,
) -> None:
    """Reload a block's weights directly from the original model checkpoint files.

    This function selectively loads only the tensors belonging to *block_name*
    from the safetensors (or pytorch bin) files in *model_dir*, without loading
    the entire model into memory.

    Args:
        model_dir: Path to the model directory (local or HuggingFace hub name).
        block_name: The block name prefix (e.g. ``"model.layers.0"``).
        block: The ``nn.Module`` to load weights into.
    """
    model_dir = _resolve_model_dir(model_dir)
    weight_map = _build_weight_map(model_dir)

    # Find all tensors belonging to this block
    prefix = block_name + "."
    matching = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
    if not matching:
        logger.warning(f"No tensors found for block {block_name} in {model_dir}")
        return

    # Group by shard file to minimize file opens
    shard_to_tensors: dict[str, list[str]] = defaultdict(list)
    for tensor_name, shard_file in matching.items():
        shard_to_tensors[shard_file].append(tensor_name)

    # Load tensors and build a state_dict with block-relative keys
    state_dict = {}
    for shard_file, tensor_names in shard_to_tensors.items():
        shard_path = os.path.join(model_dir, shard_file)
        if shard_file.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for name in tensor_names:
                    # Convert absolute name to block-relative name
                    relative_name = name[len(prefix):]
                    state_dict[relative_name] = f.get_tensor(name)
        else:
            # pytorch bin format
            full_state = torch.load(shard_path, map_location="cpu")
            for name in tensor_names:
                relative_name = name[len(prefix):]
                if name in full_state:
                    state_dict[relative_name] = full_state[name]
            del full_state

    _load_state_dict_into_block(state_dict, block)


# =====================================================================
# AutoSchemeOffloadContext
# =====================================================================


class AutoSchemeOffloadContext:
    """Manages CPU RAM reduction for AutoScheme by releasing and reloading block weights.

    Instead of duplicating weights to a temporary directory, this context simply
    **clears** block weights from memory and **reloads** them on demand from the
    original model checkpoint files (safetensors / pytorch bin).  This avoids
    writing any extra data to disk.
    """

    def __init__(self, low_cpu_mem_usage: bool = False, model_dir: str | None = None):
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self._model_dir = model_dir
        self._cleared_blocks: set[str] = set()
        self._cache_numel = True

        # Hook state
        self._hook_handles: list = []
        self._model_ref: torch.nn.Module | None = None
        self._block_names: list[str] = []
        self._last_loaded_block: str | None = None

        # Wrapping-loop state (used by ensure_block_for_layer / flush_loaded_block)
        self._wrapping_current_block: str | None = None

    @property
    def model_dir(self) -> str | None:
        return self._model_dir

    @model_dir.setter
    def model_dir(self, value: str | None) -> None:
        self._model_dir = value

    # ------------------------------------------------------------------
    # Hook-based attach / detach
    # ------------------------------------------------------------------
    def attach(self, model: torch.nn.Module, block_names: list[str]) -> None:
        """Register pre-forward hooks on every block and clear all block weights.

        After this call, every ``block.forward()`` will transparently reload
        weights from the model checkpoint before executing.
        """
        if not self.low_cpu_mem_usage:
            return
        self._model_ref = model
        self._block_names = list(block_names)
        self._register_hooks(model, block_names)
        self._clear_all_blocks(model, block_names)

    def detach(self, model: torch.nn.Module, block_names: list[str]) -> None:
        """Remove hooks and reload all block weights from the checkpoint."""
        self._remove_hooks()
        self._load_all_blocks(model, block_names)
        self._model_ref = None

    def _register_hooks(self, model: torch.nn.Module, block_names: list[str]) -> None:
        from functools import partial

        for block_name in block_names:
            block = get_module(model, block_name)
            if block is None:
                continue
            handle = block.register_forward_pre_hook(
                partial(self._pre_forward_hook, block_name=block_name)
            )
            self._hook_handles.append(handle)

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._last_loaded_block = None

    # ------------------------------------------------------------------
    # Pre-forward hook (the core of transparent offloading)
    # ------------------------------------------------------------------
    def _pre_forward_hook(self, module: torch.nn.Module, args, *, block_name: str) -> None:
        """Reload block weights if cleared; lazily clear the previous block."""
        if not self._needs_loading(module):
            return
        # Clear previous block before loading the new one
        if self._last_loaded_block is not None and self._last_loaded_block != block_name:
            prev_block = get_module(self._model_ref, self._last_loaded_block)
            if prev_block is not None:
                self._clear_block(prev_block)
        load_block_from_model_files(self._model_dir, block_name, module)
        self._last_loaded_block = block_name

    @staticmethod
    def _needs_loading(block: torch.nn.Module) -> bool:
        """Return *True* if any sub-module's weight has been cleared (numel == 0)."""
        for submodule in block.modules():
            if hasattr(submodule, "orig_layer"):
                submodule = submodule.orig_layer
            if hasattr(submodule, "weight") and submodule.weight is not None:
                if submodule.weight.numel() == 0:
                    return True
        return False

    # ------------------------------------------------------------------
    # Wrapping-loop helpers
    # ------------------------------------------------------------------
    def ensure_block_for_layer(self, model: torch.nn.Module, layer_name: str) -> None:
        """Ensure the block containing *layer_name* has weights loaded.

        Call this before accessing a layer's weight (e.g. to create a
        ``WrapperLinear``).  When the loop moves to a layer in a different
        block, the previous block is automatically cleared.  After the loop,
        call :meth:`flush_loaded_block` to release the last block.
        """
        if not self.low_cpu_mem_usage or not self._block_names:
            return
        target_block = None
        for bn in self._block_names:
            if layer_name.startswith(bn + "."):
                target_block = bn
                break
        if target_block is None:
            return  # non-block layer (e.g. lm_head)
        if target_block == self._wrapping_current_block:
            return  # already loaded
        # Clear previous block
        if self._wrapping_current_block is not None:
            prev = get_module(model, self._wrapping_current_block)
            if prev is not None:
                self._clear_block(prev)
        # Load new block
        block = get_module(model, target_block)
        if block is not None:
            load_block_from_model_files(self._model_dir, target_block, block)
        self._wrapping_current_block = target_block

    def flush_loaded_block(self, model: torch.nn.Module) -> None:
        """Clear the last block loaded by :meth:`ensure_block_for_layer`."""
        if self._wrapping_current_block is not None:
            block = get_module(model, self._wrapping_current_block)
            if block is not None:
                self._clear_block(block)
            self._wrapping_current_block = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _clear_block(self, block: torch.nn.Module) -> None:
        """Replace all weight/bias tensors in *block* with empty tensors."""
        for submodule in block.modules():
            _clear_submodule_weights(submodule, cache_numel=self._cache_numel)

    def _clear_all_blocks(self, model: torch.nn.Module, block_names: list[str]) -> None:
        """Clear all block weights and release memory."""
        logger.info("AutoScheme: clearing block weights to free RAM...")
        for block_name in block_names:
            block = get_module(model, block_name)
            if block is not None:
                self._clear_block(block)
                self._cleared_blocks.add(block_name)
        gc.collect()
        from auto_round.utils import clear_memory
        clear_memory()
        logger.info("AutoScheme: block weights cleared")

    def _load_all_blocks(self, model: torch.nn.Module, block_names: list[str]) -> None:
        """Reload all blocks from the model checkpoint files."""
        if self._model_dir is None:
            return
        for block_name in block_names:
            block = get_module(model, block_name)
            if block is not None:
                load_block_from_model_files(self._model_dir, block_name, block)
        self._cleared_blocks.clear()

    def reset_scheme_state(self) -> None:
        """Reset tracking for a new scheme iteration."""
        self._cleared_blocks.clear()
        self._last_loaded_block = None

    # --- Cleanup ---
    def cleanup(self) -> None:
        """Remove hooks and reset tracking state."""
        self._remove_hooks()
        self._cleared_blocks.clear()


# =====================================================================
# BlockOffloadManager
# =====================================================================


class BlockOffloadManager:
    """Manages disk offloading of model block ``state_dict``s to reduce CPU RAM.

    Each instance owns **one** temporary directory and an internal tracking dict
    that maps ``block_name → {"save_path": ...}``.

    For the auto-scheme dual-store pattern (original weights + wrapped weights),
    simply create **two** manager instances.

    Parameters
    ----------
    enabled : bool
        When *False* every public method is a no-op.
    prefix : str
        Prefix for the temporary directory name.
    cache_numel : bool
        If *True*, ``clear_block`` caches ``weight.numel()`` on each sub-module
        before replacing the tensor, so callers like ``compute_layer_bits`` can
        still determine the original size.
    """

    def __init__(self, enabled: bool = False, prefix: str = "autoround_offload", cache_numel: bool = False):
        self.enabled = enabled
        self._prefix = prefix
        self._cache_numel = cache_numel
        self._tempdir: Optional[str] = None
        self._blocks: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------
    def _ensure_dir(self) -> str:
        """Lazily create and return the temp directory."""
        if self._tempdir is None:
            self._tempdir = tempfile.mkdtemp(prefix=f"{self._prefix}_")
            logger.info(f"BlockOffloadManager ({self._prefix}): tempdir = {self._tempdir}")
        return self._tempdir

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def save_block(self, name: str, block: torch.nn.Module, *, skip_if_saved: bool = False) -> None:
        """Save the full ``state_dict`` of *block* to disk.

        Args:
            name: Logical block name (e.g. ``"model.layers.0"``).
            block: The ``nn.Module`` whose state to persist.
            skip_if_saved: If *True* and *name* was already saved, do nothing.
        """
        if not self.enabled:
            return
        if skip_if_saved and name in self._blocks:
            return
        tmpdir = self._ensure_dir()
        safe_name = name.replace(".", "_")
        save_path = os.path.join(tmpdir, f"{safe_name}.pt")
        try:
            state_dict = {k: v.cpu().contiguous() for k, v in block.state_dict().items()}
            torch.save(state_dict, save_path)
            self._blocks[name] = {"save_path": save_path}
            del state_dict
        except Exception as e:
            logger.warning(f"BlockOffloadManager: failed to save block {name}: {e}")

    def load_block(self, name: str, block: torch.nn.Module) -> None:
        """Load a previously-saved ``state_dict`` from disk into *block*."""
        if not self.enabled:
            return
        metadata = self._blocks.get(name)
        if not metadata:
            return
        save_path = metadata["save_path"]
        if not os.path.exists(save_path):
            logger.warning(f"BlockOffloadManager: file not found {save_path}")
            return
        try:
            state_dict = torch.load(save_path, map_location="cpu")
            _load_state_dict_into_block(state_dict, block)
        except Exception as e:
            logger.warning(f"BlockOffloadManager: failed to load block {name}: {e}")

    def clear_block(self, block: torch.nn.Module) -> None:
        """Replace all weight / bias tensors in *block*'s sub-modules with empty tensors."""
        for submodule in block.modules():
            _clear_submodule_weights(submodule, cache_numel=self._cache_numel)

    def save_and_clear(self, name: str, block: torch.nn.Module, *, skip_if_saved: bool = False) -> None:
        """Convenience: ``save_block`` then ``clear_block``."""
        self.save_block(name, block, skip_if_saved=skip_if_saved)
        self.clear_block(block)

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------
    def save_all(self, model: torch.nn.Module, block_names: list[str], *, clear: bool = True) -> None:
        """Save (and optionally clear) every block in *block_names*."""
        if not self.enabled:
            return
        for block_name in block_names:
            block = get_module(model, block_name)
            if block is not None:
                if clear:
                    self.save_and_clear(block_name, block, skip_if_saved=True)
                else:
                    self.save_block(block_name, block, skip_if_saved=True)
        gc.collect()

    def load_all(self, model: torch.nn.Module, block_names: list[str]) -> None:
        """Load every block in *block_names* from disk."""
        if not self.enabled:
            return
        for block_name in block_names:
            block = get_module(model, block_name)
            if block is not None:
                self.load_block(block_name, block)

    def restore_all(self, model: torch.nn.Module) -> None:
        """Load **all** tracked blocks back into *model*."""
        if not self._blocks:
            return
        for block_name in list(self._blocks.keys()):
            block = get_module(model, block_name)
            if block is not None:
                self.load_block(block_name, block)

    def discard_and_resave(self, name: str, block: torch.nn.Module) -> None:
        """Remove the old save file, re-save the (possibly quantized) block, and clear it.

        Useful when the block has been quantized in-place and the original
        checkpoint is no longer needed.
        """
        if not self.enabled:
            return
        # Remove old file
        old_meta = self._blocks.pop(name, None)
        if old_meta:
            old_path = old_meta.get("save_path")
            if old_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except Exception as e:
                    logger.warning(f"BlockOffloadManager: could not remove {old_path}: {e}")
        # Re-save
        self.save_and_clear(name, block)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear tracking dict so the next round of ``save_block`` re-saves.

        The temp directory itself is **not** removed.
        """
        self._blocks = {}

    def has(self, name: str) -> bool:
        """Return *True* if *name* has been saved (and not yet discarded)."""
        return name in self._blocks

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_tensor_size_gb(tensor: Any) -> float:
        """Estimate the size of a tensor (or nested tensors) in GB."""
        if tensor is None:
            return 0.0
        if isinstance(tensor, torch.Tensor):
            return tensor.numel() * tensor.element_size() / (1024**3)
        if isinstance(tensor, list):
            return sum(BlockOffloadManager.estimate_tensor_size_gb(t) for t in tensor)
        if isinstance(tensor, dict):
            return sum(BlockOffloadManager.estimate_tensor_size_gb(v) for v in tensor.values())
        return 0.0

    @staticmethod
    def estimate_inputs_size_gb(all_inputs: dict) -> float:
        """Estimate the total size of calibration inputs in GB."""
        total = 0.0
        for _, inputs in all_inputs.items():
            total += BlockOffloadManager.estimate_tensor_size_gb(inputs)
        return total

    @staticmethod
    def estimate_model_size_gb(model: torch.nn.Module) -> float:
        """Estimate the model weights size in GB."""
        total = 0.0
        for param in model.parameters():
            if param.numel() > 0:
                total += param.numel() * param.element_size() / (1024**3)
        return total

    @staticmethod
    def estimate_block_size_gb(block: torch.nn.Module) -> float:
        """Estimate a block's parameter size in GB."""
        total = 0.0
        for param in block.parameters():
            if param.numel() > 0:
                total += param.numel() * param.element_size() / (1024**3)
        return total

    # ------------------------------------------------------------------
    # Stream offload
    # ------------------------------------------------------------------
    def stream_offload_all_blocks(
        self, model: torch.nn.Module, all_blocks: list[list[str]], device_list=None
    ) -> None:
        """Offload all block weights to disk upfront to minimize peak CPU RAM.

        Each block's full ``state_dict`` is saved and then cleared from memory.
        Blocks already saved are skipped.
        """
        if not self.enabled:
            return
        logger.info("stream offloading block weights to disk...")
        total_offloaded_gb = 0.0
        for block_names in all_blocks:
            for block_name in block_names:
                if self.has(block_name):
                    continue
                block = get_module(model, block_name)
                if block is None:
                    continue
                total_offloaded_gb += self.estimate_block_size_gb(block)
                self.save_and_clear(block_name, block)
        from auto_round.utils import clear_memory

        clear_memory(device_list=device_list)
        logger.info(f"stream offload done, offloaded {total_offloaded_gb:.2f} GB of block weights")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        """Remove the temp directory and clear all tracking state."""
        if self._tempdir and os.path.isdir(self._tempdir):
            try:
                shutil.rmtree(self._tempdir)
            except Exception as e:
                logger.warning(f"BlockOffloadManager: cleanup failed for {self._tempdir}: {e}")
        self._tempdir = None
        self._blocks = {}
