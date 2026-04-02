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

import json
import os
from collections import OrderedDict

import torch

from auto_round.compressors_new.utils import _get_save_folder_name
from auto_round.context.compress import CompressContext
from auto_round.context.model import ModelContext
from auto_round.logger import logger
from auto_round.utils import get_lm_head_name, get_module


class ShardWriter:
    """
    Handles shard-saving of model parameters to disk with memory management.
    """

    _instance = None
    _initialized = False

    model = None
    lm_head_name = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = {}
        return cls._instance

    def __init__(
        self,
        model,
        bits,
        max_shard_size=None,
        safe_serialization=True,
    ):
        if ShardWriter._initialized:
            return
        self.model = model
        self.lm_head_name = get_lm_head_name(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        # Heuristic estimate of model size in GB used to choose a default max_shard_size:
        # - total_params * rounder.bits       -> total number of bits in all parameters
        # - // 8                              -> convert bits to bytes
        # - // 1e9                            -> approx convert bytes to GB (1e9 bytes ~= 1 GB)
        # - final // 10                       -> apply a safety margin so default shards are
        #                                         smaller than the full model; this intentionally
        #                                         underestimates size before clamping below.
        max_split_num = 10
        model_size = int(total_params * bits // 1e9 // 8 + max_split_num - 1) / max_split_num
        model_size = max(1, min(int(model_size), 5))

        # Configuration
        max_shard_size = max_shard_size or f"{model_size}GB"
        self.max_shard_size = self._parse_size(max_shard_size)
        self.safe_serialization = safe_serialization

        # Internal State
        self.use_safetensors = self._check_safetensors()
        self.shard_suffix = "safetensors" if self.use_safetensors else "bin"
        self.current_shard_tensors = OrderedDict()
        self.current_shard_size = 0
        self.shard_meta = []  # List of {tmp_file: str, params: list}
        self.global_weight_map = {}
        self.shard_counter = 0

        # Persistent set of all parameter names already flushed to a shard file.
        # Maintained incrementally in _flush_shard to avoid O(N^2) rebuilds in _add_tensor.
        self._all_saved = set()

        # Stats
        self.total_param_elems = 0
        self.total_param_size_bytes = 0
        self.skipped_meta_tensors = []

        ShardWriter._initialized = True

    @property
    def output_dir(self) -> str:
        """Derive the output directory from the current CompressContext at access time.

        Reading from context rather than caching the path at construction time ensures
        the ShardWriter always uses the final export directory even if
        ``CompressContext.output_dir`` is updated after the ShardWriter was created
        (e.g. by ``_get_export_dir()`` in ``quantize_and_save()``).
        """
        compress_context = CompressContext.get_context()
        formats = compress_context.formats
        base_dir = _get_save_folder_name(formats[0])
        subfolder = getattr(self.model, "_autoround_pipeline_subfolder", None)
        if subfolder:
            base_dir = os.path.join(base_dir, subfolder)
        return os.path.join(base_dir, "")

    @classmethod
    def reset(cls):
        """Reset the singleton state so the next instantiation creates a fresh ShardWriter."""
        cls._initialized = False
        cls._instance = None

    @classmethod
    def get_shard_writer(cls, *args, **kwargs):
        """Return the current singleton instance, or None if not yet initialized.

        Callers that require a valid writer should guard the result with
        ``if self.compress_context.is_immediate_saving`` before use.
        """
        return cls._instance

    def _parse_size(self, size_str: str) -> int:
        if isinstance(size_str, int):
            return size_str
        s = size_str.strip().upper()
        units = {"GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}
        for unit, mult in units.items():
            if s.endswith(unit):
                return int(float(s[: -len(unit)]) * mult)
        return int(s)

    def _check_safetensors(self) -> bool:
        if self.safe_serialization:
            try:
                import safetensors.torch

                return True
            except ImportError:
                logger.warning("safetensors not installed; falling back to torch.save.")
        return False

    def save_module(self, m: torch.nn.Module, name: str = None):
        """Extracts and accumulates tensors from a module."""
        prefix = name if name is not None else getattr(m, "global_name", "model")
        sd = m.state_dict()

        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            param_name = f"{prefix}.{k}"
            self._add_tensor(param_name, v)

    def _add_tensor(self, name: str, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "meta":
            self.skipped_meta_tensors.append(name)
            return

        # Guard against duplicate saving of the same parameter
        if name in self._all_saved or name in self.current_shard_tensors:
            return

        t_size = tensor.nbytes
        self.total_param_elems += tensor.numel()
        self.total_param_size_bytes += t_size
        tensor = tensor.detach().cpu()
        # If single tensor exceeds limit, flush current, save it solo, then continue
        if t_size > self.max_shard_size:
            self._flush_shard()
            self.current_shard_tensors[name] = tensor
            self.current_shard_size = t_size
            self._flush_shard()
        # If adding exceeds limit, flush first
        elif self.current_shard_size + t_size > self.max_shard_size and self.current_shard_size > 0:
            self._flush_shard()
            self.current_shard_tensors[name] = tensor
            self.current_shard_size = t_size
        else:
            self.current_shard_tensors[name] = tensor
            self.current_shard_size += t_size

    def _flush_shard(self):
        if not self.current_shard_tensors:
            return

        self.shard_counter += 1
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        tmp_name = f"model-shard-{self.shard_counter:05d}.{self.shard_suffix}"
        tmp_path = os.path.join(output_dir, tmp_name)

        if self.use_safetensors:
            from safetensors.torch import save_file

            save_file(self.current_shard_tensors, tmp_path)
        else:
            torch.save(self.current_shard_tensors, tmp_path)

        saved_params = list(self.current_shard_tensors.keys())
        self.shard_meta.append({"tmp_file": tmp_name, "params": saved_params, "dir": output_dir})
        self._all_saved.update(saved_params)

        # Offload logic: move modules to meta device once all params are saved
        self._offload_to_meta(saved_params)

        self.current_shard_tensors = OrderedDict()
        self.current_shard_size = 0

    def _offload_to_meta(self, saved_params):
        """Attempts to move fully saved modules to the 'meta' device to free RAM."""
        for param_full_name in saved_params:
            module_path = param_full_name.rsplit(".", 1)[0]

            module = get_module(self.model, module_path)
            # Check if all parameters of this module are now in '_all_saved'
            if (
                module is not None
                and isinstance(module, torch.nn.Module)
                and all(f"{module_path}.{k}" in self._all_saved for k in module.state_dict().keys())
            ):
                module.to("meta")

    def finalize(self):
        """Saves remaining weights, renames files, and writes the index JSON."""
        # 1. Capture remaining weights not yet saved
        full_sd = self.model.state_dict()
        tie_word_embeddings = False
        if hasattr(self.model, "config") and hasattr(self.model.config, "tie_word_embeddings"):
            tie_word_embeddings = self.model.config.tie_word_embeddings

        finalize_skipped_meta_tensors = []
        for pname, tensor in full_sd.items():
            if pname in self._all_saved:
                continue
            if tensor.device.type == "meta":
                continue
            layer_name = ".".join(pname.split(".")[:-1])
            if self.lm_head_name is not None and layer_name == self.lm_head_name and tie_word_embeddings:
                lm_head_module = get_module(self.model, self.lm_head_name)
                lm_head_module.to("meta")  # Must to meta, otherwise model's saver will dump it again
                continue
            self._add_tensor(pname, tensor.detach().to("cpu"))

        self._flush_shard()

        total_skipped = len(self.skipped_meta_tensors) + len(finalize_skipped_meta_tensors)
        if total_skipped > 0:
            examples = (self.skipped_meta_tensors + finalize_skipped_meta_tensors)[:5]

        # 2. Rename temp files to HF standard and map weights
        if self.shard_counter == 0:
            logger.warning("No tensors saved.")
            return

        output_dir = self.output_dir
        for idx, meta in enumerate(self.shard_meta, start=1):
            shard_dir = meta.get("dir", output_dir)
            old_path = os.path.join(shard_dir, meta["tmp_file"])
            new_name = (
                f"model.{self.shard_suffix}"
                if self.shard_counter == 1
                else f"model-{idx:05d}-of-{self.shard_counter:05d}.{self.shard_suffix}"
            )
            new_path = os.path.join(shard_dir, new_name)
            os.rename(old_path, new_path)
            for p in meta["params"]:
                self.global_weight_map[p] = new_name

        # 3. Write Index JSON
        index_ext = "safetensors.index.json" if self.use_safetensors else "bin.index.json"
        index_path = os.path.join(output_dir, f"model.{index_ext}")

        index_data = {
            "metadata": {
                "format": "safetensors" if self.use_safetensors else "pytorch",
                "total_shards": self.shard_counter,
                "total_parameters": int(self.total_param_elems),
                "total_size": int(self.total_param_size_bytes),
            },
            "weight_map": self.global_weight_map,
        }

        if self.shard_counter > 1:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2)

        logger.info(f"model has been saved to {self.output_dir}")

    @torch.no_grad()
    def write(self, m: torch.nn.Module = None, name: str = None, is_finalize: bool = False):
        if m is None and name is None and not is_finalize and not is_finalize:
            raise ValueError("Must specify either name or m")
        if m is None and name is not None:
            m = get_module(self.model, name)
            # Perform the save
        if m is not None:
            self.save_module(m, name)

        if is_finalize:
            self.finalize()
            ShardWriter._initialized = False
            ShardWriter._instance = None
