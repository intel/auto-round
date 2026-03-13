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

"""Offloading module for CPU RAM optimization.

This module provides :class:`OffloadManager`, a single class that can
**offload** any named module (block, layer, or sub-module) to free CPU RAM
and **reload** it on demand.  Two operating modes are supported:

* ``"offload"`` -- saves the module's ``state_dict`` to a temporary directory
  before clearing. Best when the original weights may have been modified
  (e.g. after quantization) and need to be restored later.
* ``"clean"``   -- simply clears weights without saving; reloads from the
  original HuggingFace model checkpoint files.  Zero extra disk writes.

Usage patterns
--------------

**Compressor (offload mode)**::

    offloader = OffloadManager(mode="offload")
    offloader.offload(model, "model.layers.0")              # save + clear one module
    offloader.offload(model, block_names, clear_memory=True) # save + clear many + gc
    offloader.reload(model, "model.layers.0")               # load back + auto-clean temp file

**AutoScheme (clean mode with hooks)**::

    offloader = OffloadManager(mode="clean",
                         model_dir="/path/to/model")
    offloader.add_offload_hooks(model, block_names)       # clear all + register hooks
    # ... forward passes transparently reload on demand ...
    offloader.remove_offload_hooks(model, block_names)    # remove hooks + restore all
"""

import gc
import json
import os
import shutil
import tempfile
from collections import defaultdict
from functools import partial
from typing import Any, Optional, Union

import torch

from auto_round.logger import logger
from auto_round.utils.model import get_module

__all__ = ["OffloadManager"]

# =====================================================================
# Low-level helpers
# =====================================================================


def _load_state_dict_into_module(state_dict: dict, module: torch.nn.Module) -> None:
    """Assign every key in *state_dict* to the corresponding sub-module.

    Handles cleared parameters (empty tensors) and wrapper objects that store
    the original layer in an ``orig_layer`` attribute.
    """
    for name, param in state_dict.items():
        parts = name.split(".")
        target = module
        try:
            for part in parts[:-1]:
                target = getattr(target, part)
        except AttributeError:
            continue
        if hasattr(target, "orig_layer"):
            target = target.orig_layer
        param_name = parts[-1]
        if hasattr(target, param_name):
            old_param = getattr(target, param_name)
            if isinstance(old_param, torch.nn.Parameter):
                param = param.to(dtype=old_param.dtype, device=old_param.device)
                setattr(target, param_name, torch.nn.Parameter(param, requires_grad=old_param.requires_grad))
            else:
                setattr(target, param_name, param)
    del state_dict


def _clear_module_weights(module: torch.nn.Module, cache_numel: bool = False) -> None:
    """Replace a single module's weight/bias with empty tensors.

    Args:
        module: The leaf module to clear.
        cache_numel: If *True*, store ``_cached_weight_numel`` and
            ``_cached_weight_shape`` before clearing.
    """
    if module is None:
        return
    if hasattr(module, "orig_layer"):
        return

    with torch.no_grad():
        for name, param in list(module.named_parameters(recurse=False)):
            if param is None or param.numel() == 0:
                continue
            if cache_numel and name == "weight":
                module._cached_weight_numel = param.numel()
                module._cached_weight_shape = tuple(param.shape)
            setattr(
                module,
                name,
                torch.nn.Parameter(torch.empty(0, dtype=param.dtype, device="cpu"), requires_grad=param.requires_grad),
            )
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf is None or buf.numel() == 0:
                continue
            module.register_buffer(name, torch.empty(0, dtype=buf.dtype, device="cpu"))


# =====================================================================
# Checkpoint file helpers
# =====================================================================


def _resolve_model_dir(model_dir: str) -> str:
    """Resolve a model name/path to a local directory containing weight files."""
    if os.path.isdir(model_dir):
        return model_dir
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_dir, local_files_only=True)
    except Exception:
        return model_dir


def _build_weight_map(model_dir: str) -> dict[str, str]:
    """Build ``{tensor_name: shard_filename}`` from the model directory."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)["weight_map"]

    single_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single_path):
        from safetensors import safe_open

        with safe_open(single_path, framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}

    bin_index_path = os.path.join(model_dir, "pytorch_model.bin.index.json")
    if os.path.exists(bin_index_path):
        with open(bin_index_path) as f:
            return json.load(f)["weight_map"]

    single_bin = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(single_bin):
        state_dict = torch.load(single_bin, map_location="cpu")
        return {k: "pytorch_model.bin" for k in state_dict.keys()}

    raise FileNotFoundError(
        f"Could not find model weight files in {model_dir}. "
        "Expected model.safetensors or pytorch_model.bin (with optional index.json)."
    )


def load_block_from_model_files(model_dir: str, block_name: str, block: torch.nn.Module) -> None:
    """Reload a module's weights directly from the original model checkpoint.

    Selectively loads only tensors belonging to *block_name* without loading
    the entire model into memory.

    Args:
        model_dir: Path to the model directory (local or HuggingFace hub name).
        block_name: The module name prefix (e.g. ``"model.layers.0"``).
        block: The ``nn.Module`` to load weights into.
    """
    model_dir = _resolve_model_dir(model_dir)
    weight_map = _build_weight_map(model_dir)

    prefix = block_name + "."
    matching = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
    if not matching:
        logger.warning(f"No tensors found for {block_name} in {model_dir}")
        return

    shard_to_tensors: dict[str, list[str]] = defaultdict(list)
    for tensor_name, shard_file in matching.items():
        shard_to_tensors[shard_file].append(tensor_name)

    state_dict = {}
    for shard_file, tensor_names in shard_to_tensors.items():
        shard_path = os.path.join(model_dir, shard_file)
        if shard_file.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for name in tensor_names:
                    state_dict[name[len(prefix) :]] = f.get_tensor(name)
        else:
            full_state = torch.load(shard_path, map_location="cpu")
            for name in tensor_names:
                if name in full_state:
                    state_dict[name[len(prefix) :]] = full_state[name]
            del full_state

    _load_state_dict_into_module(state_dict, block)


# =====================================================================
# OffloadManager -- the offload manager class
# =====================================================================


class OffloadManager:
    """Module offloader for CPU RAM optimization.

    Provides two core operations:

    * `offload` -- save/clear a named module's weights to free CPU RAM.
    * `reload` -- restore the weights back into the module.

    Works at any granularity (block, layer, individual module) and supports
    two operating modes:

    * ``"offload"`` -- saves ``state_dict`` to a temp directory on disk
      before clearing.  Supports ``overwrite=True`` for re-saving modified
      (e.g. quantized) weights.
    * ``"clean"`` -- simply clears weights; reloads from the original
      HuggingFace model checkpoint files.  Zero extra disk writes but
      cannot persist modifications.

    The manager tracks its own state and supports automatic cleanup via
    context manager (``with`` statement) or explicit `cleanup`.

    Parameters
    ----------
    enabled : bool
        When *False*, every public method is a no-op.
    mode : str
        ``"offload"`` or ``"clean"``.
    model_dir : str, optional
        Path to the model checkpoint directory. Required for ``"clean"``
        mode; optional for ``"offload"``.
    offload_dir_prefix : str
        Prefix for the temp directory name (``"offload"`` mode only).
    cache_numel : bool
        If *True*, cache original ``weight.numel()`` before clearing so
        downstream code can still determine the original size.
    """

    def __init__(
        self,
        enabled: bool = True,
        mode: str = "offload",
        model_dir: Optional[str] = None,
        offload_dir_prefix: str = "ar_offload",
        cache_numel: bool = False,
    ):
        self.enabled = enabled
        self.mode = mode
        self.model_dir = model_dir
        self.cache_numel = cache_numel
        self._prefix = offload_dir_prefix

        # Disk state (offload mode)
        self._tempdir: Optional[str] = None
        self._saved: dict[str, dict] = {}  # name -> {"save_path": str}

        # Hook state (for add_offload_hooks/remove_offload_hooks transparent offloading)
        self._hook_handles: list = []
        self._model_ref: Optional[torch.nn.Module] = None
        self._module_names: list[str] = []
        self._last_loaded: Optional[str] = None

        # Ensure-style state (for wrapping loops)
        self._current_loaded: Optional[str] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        try:
            self.cleanup(_skip_reload=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def offload(
        self,
        model: torch.nn.Module,
        names: Union[str, list[str], list[list[str]]],
        *,
        skip_if_saved: bool = False,
        overwrite: bool = False,
        clear_memory: bool = False,
        device_list=None,
    ) -> float:
        """Offload one or more named modules' weights to free CPU RAM.

        Accepts a single module name, a flat list, or a nested list of names.
        For ``"offload"`` mode: saves state_dict to disk, then clears weights.
        For ``"clean"`` mode: simply clears weights (will reload from
        model files later).

        Args:
            model: The root model containing the module(s).
            names: Module name(s) -- a single string, a list of strings,
                or a nested list (e.g. ``all_blocks``).
            skip_if_saved: If *True* and a name was already saved, skip it.
            overwrite: If *True*, discard any previous save and re-save the
                current (possibly modified) weights before clearing.  Useful
                after quantization when updated weights must be persisted.
                Only meaningful for ``"offload"`` mode.
            clear_memory: If *True*, call ``clear_memory()`` after offloading
                all modules and log the total freed size.
            device_list: Device list passed to ``clear_memory`` (only used
                when *clear_memory* is *True*).

        Returns:
            Total offloaded size in GB (non-zero only when *clear_memory* is True).
        """
        if not self.enabled:
            return 0.0
        if not self._check_disk_space(model, names):
            self.enabled = False
            return 0.0
        if isinstance(names, str):
            self._offload(model, names, skip_if_saved=skip_if_saved, overwrite=overwrite)
            return 0.0
        flat_names = self._flatten_names(names)
        if clear_memory:
            logger.info("offloading module weights...")
        total_gb = 0.0
        for name in flat_names:
            if skip_if_saved and self.has(name):
                continue
            if clear_memory:
                module = get_module(model, name)
                if module is not None:
                    total_gb += self.estimate_module_size_gb(module)
            self._offload(model, name, skip_if_saved=skip_if_saved, overwrite=overwrite)
        if clear_memory:
            from auto_round.utils import clear_memory as _clear_memory

            _clear_memory(device_list=device_list)
            logger.info(f"offload done, freed {total_gb:.2f} GB")
        return total_gb

    def _check_disk_space(self, model: torch.nn.Module, names: Union[str, list[str], list[list[str]]]) -> bool:
        """Check whether there is enough disk space to offload the given modules.

        Args:
            model: The root model containing the module(s).
            names: Module name(s) to check.

        Returns:
            True if sufficient disk space is available, False otherwise.
        """
        if isinstance(names, str):
            flat_names = [names]
        else:
            flat_names = self._flatten_names(names)
        total_bytes = 0
        for name in flat_names:
            module = get_module(model, name)
            if module is not None:
                total_bytes += sum(
                    p.numel() * p.element_size() for p in module.parameters() if p.numel() > 0
                )
        # torch.save adds serialization overhead; use 1.2x safety margin
        required_bytes = int(total_bytes * 1.2)
        tmpdir = self._ensure_dir()
        free_bytes = shutil.disk_usage(tmpdir).free
        if free_bytes < required_bytes:
            total_gb = total_bytes / (1024**3)
            free_gb = free_bytes / (1024**3)
            logger.warning(
                f"Insufficient disk space for offloading: need ~{total_gb:.2f} GB "
                f"but only {free_gb:.2f} GB available at {tmpdir}. Skipping offload."
            )
            return False
        return True

    def _offload(
        self, model: torch.nn.Module, name: str, *, skip_if_saved: bool = False, overwrite: bool = False
    ) -> None:
        """Offload a single named module (internal helper)."""
        self._model_ref = model
        module = get_module(model, name)
        if module is None:
            return
        if self.mode == "offload":
            if overwrite:
                old_meta = self._saved.pop(name, None)
                if old_meta:
                    old_path = old_meta.get("save_path")
                    if old_path and os.path.exists(old_path):
                        try:
                            os.remove(old_path)
                        except Exception as e:
                            logger.warning(f"OffloadManager: could not remove {old_path}: {e}")
            elif skip_if_saved and name in self._saved:
                return
            self._save_to_disk(name, module)
        self._clear(module)

    def reload(self, model: torch.nn.Module, names: Union[str, list[str], None] = None) -> None:
        """Reload previously offloaded module(s).

        For ``"offload"`` mode: loads from the temp directory, then
        removes the temp file.  When all offloaded modules have been
        reloaded the temp directory is automatically deleted.
        For ``"clean"`` mode: loads from original model files.

        Args:
            model: The root model containing the module(s).
            names: A single module name, a list of names, or *None*
                to reload all tracked modules.
        """
        if not self.enabled:
            return
        if names is None:
            if self.mode == "offload":
                names = list(self._saved.keys())
            else:
                names = list(self._module_names)
        if isinstance(names, str):
            self._reload(model, names)
            return
        for name in names:
            self._reload(model, name)

    def _reload(self, model: torch.nn.Module, name: str) -> None:
        """Reload a single named module (internal helper)."""
        module = get_module(model, name)
        if module is None:
            return
        if self.mode == "offload":
            self._load_from_disk(name, module)
            self._remove_saved_entry(name)
        else:
            if self.model_dir is None:
                logger.warning("OffloadManager: model_dir is required for clean mode")
                return
            load_block_from_model_files(self.model_dir, name, module)

    # ------------------------------------------------------------------
    # Hook-based transparent offloading
    # ------------------------------------------------------------------

    def add_offload_hooks(self, model: torch.nn.Module, names: list[str]) -> None:
        """Clear all named modules and register pre-forward hooks for
        transparent reload-on-demand.

        Args:
            model: The root model.
            names: List of module names to manage.
        """
        if not self.enabled:
            return
        self._model_ref = model
        self._module_names = list(names)
        # Register hooks
        for name in names:
            module = get_module(model, name)
            if module is None:
                continue
            handle = module.register_forward_pre_hook(partial(self._pre_forward_hook, name=name))
            self._hook_handles.append(handle)
        # Clear all
        logger.info("clearing module weights to free RAM...")
        for name in names:
            self._offload(model, name, skip_if_saved=True)
        gc.collect()
        from auto_round.utils import clear_memory

        clear_memory()
        logger.info("module weights cleared")

    def remove_offload_hooks(self, model: torch.nn.Module, names: Optional[list[str]] = None) -> None:
        """Remove hooks and reload all managed modules.

        Args:
            model: The root model.
            names: Module names to reload.  If None, uses the names from
                `add_offload_hooks`.
        """
        self._remove_hooks()
        if names is None:
            names = self._module_names
        self.reload(model, names)
        self._model_ref = None

    def _pre_forward_hook(self, module: torch.nn.Module, args, *, name: str) -> None:
        """Pre-forward hook: reload this module if cleared, clear previous."""
        if not self._needs_loading(module):
            return
        # Clear previous to keep only one module loaded at a time
        if self._last_loaded is not None and self._last_loaded != name:
            prev = get_module(self._model_ref, self._last_loaded)
            if prev is not None:
                self._clear(prev)
        self.reload(self._model_ref, name)
        self._last_loaded = name

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._last_loaded = None

    # ------------------------------------------------------------------
    # Ensure-style API (for wrapping loops)
    # ------------------------------------------------------------------

    def ensure_loaded(self, model: torch.nn.Module, layer_name: str) -> None:
        """Ensure the module containing *layer_name* is loaded.

        When the loop moves to a layer in a different module, the previous
        module is automatically cleared.  Call `flush_loaded` after
        the loop to release the last module.

        Args:
            model: The root model.
            layer_name: Full layer name (e.g. ``"model.layers.0.attn.q"``).
        """
        if not self.enabled or not self._module_names:
            return
        target = None
        for mn in self._module_names:
            if layer_name.startswith(mn + "."):
                target = mn
                break
        if target is None:
            return
        if target == self._current_loaded:
            return
        # Clear previous
        if self._current_loaded is not None:
            module = get_module(model, self._current_loaded)
            if module is not None:
                self._clear(module)
        # Load new
        self.reload(model, target)
        self._current_loaded = target

    def flush_loaded(self, model: torch.nn.Module) -> None:
        """Clear the last module loaded by `ensure_loaded`."""
        if self._current_loaded is not None:
            module = get_module(model, self._current_loaded)
            if module is not None:
                self._clear(module)
            self._current_loaded = None

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def has(self, name: str) -> bool:
        """Return *True* if *name* has been offloaded."""
        if self.mode == "offload":
            return name in self._saved
        return False

    def reset(self) -> None:
        """Clear tracking state (temp directory is kept)."""
        self._saved = {}
        self._current_loaded = None
        self._last_loaded = None

    def cleanup(self, _skip_reload: bool = False) -> None:
        """Reload remaining modules and remove temp files.

        Normally this is called automatically -- ``reload`` deletes each
        temp file as it goes and removes the temp directory when done.
        Explicit calls are usually unnecessary; the context-manager exit
        and ``__del__`` use this as a fallback.
        """
        self._remove_hooks()
        # Auto-reload offloaded modules if we still have a model reference
        if not _skip_reload and self._model_ref is not None and self._saved:
            try:
                self.reload(self._model_ref)
            except Exception as e:
                logger.warning(f"OffloadManager: auto-reload during cleanup failed: {e}")
        self._cleanup_tempdir()
        self._saved = {}
        self._current_loaded = None
        self._module_names = []
        self._model_ref = None

    # ------------------------------------------------------------------
    # Static estimation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_tensor_size_gb(tensor: Any) -> float:
        """Estimate size of a tensor (or nested container) in GB."""
        if tensor is None:
            return 0.0
        if isinstance(tensor, torch.Tensor):
            return tensor.numel() * tensor.element_size() / (1024**3)
        if isinstance(tensor, list):
            return sum(OffloadManager.estimate_tensor_size_gb(t) for t in tensor)
        if isinstance(tensor, dict):
            return sum(OffloadManager.estimate_tensor_size_gb(v) for v in tensor.values())
        return 0.0

    @staticmethod
    def estimate_inputs_size_gb(all_inputs: dict) -> float:
        """Estimate total size of calibration inputs in GB."""
        return sum(OffloadManager.estimate_tensor_size_gb(v) for v in all_inputs.values())

    @staticmethod
    def estimate_model_size_gb(model: torch.nn.Module) -> float:
        """Estimate model weights size in GB."""
        return sum(p.numel() * p.element_size() / (1024**3) for p in model.parameters() if p.numel() > 0)

    @staticmethod
    def estimate_module_size_gb(module: torch.nn.Module) -> float:
        """Estimate a module's parameter size in GB."""
        return sum(p.numel() * p.element_size() / (1024**3) for p in module.parameters() if p.numel() > 0)

    # ------------------------------------------------------------------
    # Internal: disk operations (offload mode)
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> str:
        if self._tempdir is None:
            self._tempdir = tempfile.mkdtemp(prefix=f"{self._prefix}_")
            logger.info(f"OffloadManager ({self._prefix}): tempdir = {self._tempdir}")
        return self._tempdir

    def _save_to_disk(self, name: str, module: torch.nn.Module) -> None:
        tmpdir = self._ensure_dir()
        safe_name = name.replace(".", "_")
        save_path = os.path.join(tmpdir, f"{safe_name}.pt")
        try:
            # Skip meta tensors: they contain no real data (e.g. quantized weights
            # already flushed to disk by an immediate-saving shard writer).
            state_dict = {k: v.cpu().contiguous() for k, v in module.state_dict().items() if v.device.type != "meta"}
            torch.save(state_dict, save_path)
            self._saved[name] = {"save_path": save_path}
            del state_dict
        except Exception as e:
            logger.warning(f"OffloadManager: failed to save {name}: {e}")

    def _load_from_disk(self, name: str, module: torch.nn.Module) -> None:
        metadata = self._saved.get(name)
        if not metadata:
            return
        save_path = metadata["save_path"]
        if not os.path.exists(save_path):
            logger.warning(f"OffloadManager: file not found {save_path}")
            return
        try:
            state_dict = torch.load(save_path, map_location="cpu")
            _load_state_dict_into_module(state_dict, module)
        except Exception as e:
            logger.warning(f"OffloadManager: failed to load {name}: {e}")

    def _remove_saved_entry(self, name: str) -> None:
        """Remove a single saved entry and its temp file; clean tempdir if empty."""
        meta = self._saved.pop(name, None)
        if meta:
            path = meta.get("save_path")
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"OffloadManager: could not remove {path}: {e}")
        if not self._saved:
            self._cleanup_tempdir()

    def _cleanup_tempdir(self) -> None:
        """Remove the temp directory if it exists."""
        if self._tempdir and os.path.isdir(self._tempdir):
            try:
                shutil.rmtree(self._tempdir)
            except Exception as e:
                logger.warning(f"OffloadManager: cleanup failed for {self._tempdir}: {e}")
        self._tempdir = None

    # ------------------------------------------------------------------
    # Internal: clearing
    # ------------------------------------------------------------------

    def _clear(self, module: torch.nn.Module) -> None:
        """Clear all weight/bias tensors in *module* and its sub-modules."""
        for submodule in module.modules():
            _clear_module_weights(submodule, cache_numel=self.cache_numel)

    @staticmethod
    def _needs_loading(module: torch.nn.Module) -> bool:
        """Return *True* if any parameter in *module* has been cleared."""
        for submodule in module.modules():
            if hasattr(submodule, "orig_layer"):
                submodule = submodule.orig_layer
            for param in submodule.parameters(recurse=False):
                if param is not None and param.numel() == 0:
                    return True
        return False

    @staticmethod
    def _flatten_names(names: Union[list[str], list[list[str]]]) -> list[str]:
        """Flatten a potentially nested list of names."""
        flat = []
        for item in names:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
