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

from auto_round.logger import logger
from auto_round.utils import get_lm_head_name, get_module


class ShardSaver:
    """
    Handles shard-saving of model parameters to disk with memory management.
    """

    def __init__(self, rounder):
        self.model = rounder.model
        self.lm_head_name = get_lm_head_name(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        model_size = int(total_params * rounder.bits // 1e9 // 8) // 10
        model_size = max(1, min(int(model_size), 5))

        # Configuration
        self.max_shard_size = self._parse_size(getattr(rounder, "max_shard_size", f"{model_size}GB"))
        self.safe_serialization = getattr(rounder, "safe_serialization", True)

        # Internal State
        self.use_safetensors = self._check_safetensors()
        self.shard_suffix = "safetensors" if self.use_safetensors else "bin"
        self.current_shard_tensors = OrderedDict()
        self.current_shard_size = 0
        self.shard_meta = []  # List of {tmp_file: str, params: list}
        self.global_weight_map = {}
        self.shard_counter = 0

        # Stats
        self.total_param_elems = 0
        self.total_param_size_bytes = 0

        # Directory Setup
        self.output_dir = os.path.join(rounder._get_save_folder_name(rounder.formats[0]), "")
        os.makedirs(self.output_dir, exist_ok=True)

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
        t_size = tensor.numel() * tensor.element_size()
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
        tmp_name = f"model-shard-{self.shard_counter:05d}.{self.shard_suffix}"
        tmp_path = os.path.join(self.output_dir, tmp_name)

        if self.use_safetensors:
            from safetensors.torch import save_file

            save_file(self.current_shard_tensors, tmp_path)
        else:
            torch.save(self.current_shard_tensors, tmp_path)

        saved_params = list(self.current_shard_tensors.keys())
        self.shard_meta.append({"tmp_file": tmp_name, "params": saved_params})

        # Offload logic: move modules to meta device once all params are saved
        self._offload_to_meta(saved_params)

        self.current_shard_tensors = OrderedDict()
        self.current_shard_size = 0

    def _offload_to_meta(self, saved_params):
        """Attempts to move fully saved modules to the 'meta' device to free RAM."""
        # Using a set for faster lookup of all saved parameters
        all_saved = {p for meta in self.shard_meta for p in meta["params"]}

        for param_full_name in saved_params:
            module_path = param_full_name.rsplit(".", 1)[0]
            try:
                module = get_module(self.model, module_path)
                # Check if all parameters of this module are now in 'all_saved'
                if all(f"{module_path}.{k}" in all_saved for k in module.state_dict().keys()):
                    module.to("meta")
            except Exception:
                continue

    def finalize(self):
        """Saves remaining weights, renames files, and writes the index JSON."""
        # 1. Capture remaining weights not yet saved
        full_sd = self.model.state_dict()
        tie_word_embeddings = getattr(getattr(self.model, "config", None), "tie_word_embeddings", True)
        all_saved_names = {p for meta in self.shard_meta for p in meta["params"]}

        for pname, tensor in full_sd.items():
            if pname in all_saved_names:
                continue
            layer_name = ".".join(pname.split(".")[:-1])
            if self.lm_head_name is not None and layer_name == self.lm_head_name and tie_word_embeddings:
                lm_head_module = get_module(self.model, self.lm_head_name)
                lm_head_module.to("meta")  # Must to
            self._add_tensor(pname, tensor.detach())

        self._flush_shard()

        # 2. Rename temp files to HF standard and map weights
        if self.shard_counter == 0:
            logger.warning("No tensors saved.")
            return

        for idx, meta in enumerate(self.shard_meta, start=1):
            old_path = os.path.join(self.output_dir, meta["tmp_file"])
            new_name = (
                f"model.{self.shard_suffix}"
                if self.shard_counter == 1
                else f"model-{idx:05d}-of-{self.shard_counter:05d}.{self.shard_suffix}"
            )

            os.rename(old_path, os.path.join(self.output_dir, new_name))
            for p in meta["params"]:
                self.global_weight_map[p] = new_name

        # 3. Write Index JSON
        index_ext = "safetensors.index.json" if self.use_safetensors else "bin.index.json"
        index_path = os.path.join(self.output_dir, f"model.{index_ext}")

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

        logger.info(f"Saved {self.shard_counter} shards to {self.output_dir}")


# Entry point function to maintain compatibility with your current flow
@torch.no_grad()
def shard_saver(rounder: object, m: torch.nn.Module=None, name: str = None, is_finalize: bool = False):
    if not hasattr(rounder, "_shard_saver"):
        rounder._shard_saver = ShardSaver(rounder)

    # Handle logic for determining if this is actually the last group
    if not hasattr(rounder, "quantized_layer_names_outside_blocks"):
        rounder.quantized_layer_names_outside_blocks = rounder._get_quantized_layer_names_outside_blocks()

    layer_names = rounder.quantized_layer_names_outside_blocks
    if len(layer_names) > 0 and name != layer_names[-1]:
        is_finalize = False

    # Perform the save
    if m is not None:
        rounder._shard_saver.save_module(m, name)

    if is_finalize:
        rounder._shard_saver.finalize()
        # Optional: cleanup the saver object from rounder
        del rounder._shard_saver
