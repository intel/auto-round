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

import torch

from auto_round.logger import logger
from auto_round.utils import copy_python_files_from_model_cache, get_lm_head_name, get_module, set_module

# TODO decouple max_shard_size with dump shard size

@torch.inference_mode()
class ShardWriter:
    """
    HF-style shard writer with immediate flushing, module-level freeing,
    safetensors support, and full cleanup on finalize.
    """

    def __init__(self, rounder):
        self.model = rounder.model

        self.max_shard_bytes = self._parse_size(getattr(rounder, "max_shard_size", "5GB"))

        self.use_safetensors = False
        if getattr(rounder, "safe_serialization", True):
            try:
                from safetensors.torch import save_file  # noqa

                self.use_safetensors = True
            except ImportError:
                logger.warning("fallback to torch.save as safetensors not installed")

        self.shard_suffix = "safetensors" if self.use_safetensors else "bin"

        self.root = os.path.join(rounder._get_save_folder_name(rounder.formats[0]), "")
        os.makedirs(self.root, exist_ok=True)

        # -------- shard state --------
        self.current_shard = {}
        self.current_size = 0
        self.shard_meta = []
        self.shard_counter = 0
        self.global_weight_map = {}

        # # -------- performance bookkeeping --------
        # self.saved_params = set()
        # self.module_param_count = {}
        # self.module_saved_count = {}

        # -------- stats --------
        self.total_param_elems = 0
        self.total_param_size_bytes = 0

        self._closed = False
        self.lm_head_name = get_lm_head_name(self.model)

    # ==================================================
    @staticmethod
    def _parse_size(size_str: str) -> int:
        s = size_str.strip().upper()
        if s.endswith("GB"):
            return int(s[:-2]) * (1024**3)
        if s.endswith("MB"):
            return int(s[:-2]) * (1024**2)
        if s.endswith("KB"):
            return int(s[:-2]) * 1024
        return int(s)

    @torch.inference_mode()
    def _flush(self):
        if not self.current_shard:
            return

        self.shard_counter += 1
        fname = f"model-shard-{self.shard_counter:05d}.{self.shard_suffix}"
        fpath = os.path.join(self.root, fname)
        current_shard_dict = {k: v for sub in self.current_shard.values() for k, v in sub.items()}
        self.shard_meta.append({"tmp_file": fname, "params": current_shard_dict.keys()})
        if self.use_safetensors:
            from safetensors.torch import save_file

            save_file(current_shard_dict, fpath)
        else:
            torch.save(current_shard_dict, fpath)

        layer_names = list(self.current_shard.keys())

        def clear_module_state(module: torch.nn.Module):
            # parameters
            for name, p in list(module._parameters.items()):
                module._parameters[name] = None

            # buffers（如 running_mean）
            for name in list(module._buffers.keys()):
                module._buffers[name] = None

        for layer_name in layer_names:
            module = get_module(self.model, layer_name)
            clear_module_state(module)

            module.to("meta")

        self.current_shard.clear()
        self.current_size = 0

    @torch.inference_mode()
    def add_module(self, module, name=None):
        if self._closed:
            raise RuntimeError("ShardWriter already finalized/closed")

        prefix = name if name is not None else module.tmp_name
        state = module.state_dict()
        layer_state = {}
        for k, v in state.items():
            if k in self.current_shard.keys():
                continue
            if not isinstance(v, torch.Tensor):
                continue
            pname = f"{prefix}.{k}"
            layer_name = ".".join(pname.split(".")[:-1])
            if layer_name not in layer_state:
                layer_state[layer_name] = {}
            layer_state[layer_name][prefix + "." + k] = v
        for k, v in layer_state.items():
            size = sum([t.numel() * t.element_size() for t in v.values()])
            self.total_param_size_bytes += size
            if size > self.max_shard_bytes:
                self._flush()
                self.current_shard[k] = v
                self.current_size = size
                self._flush()
                continue

            if self.current_size + size > self.max_shard_bytes and self.current_size > 0:
                self._flush()

            self.current_shard[k] = v
            self.current_size += size
        #
        # for k, v in state.items():
        #     if not isinstance(v, torch.Tensor):
        #         continue
        #
        #     pname = f"{prefix}.{k}"
        #     elems = v.numel()
        #     size = elems * v.element_size()
        #
        #     self.total_param_elems += elems
        #     self.total_param_size_bytes += size
        #
        #     if size > self.max_shard_bytes:
        #         self._flush()
        #         self.current_shard[pname] = v
        #         self.current_size = size
        #         self._flush()
        #         continue
        #
        #     if self.current_size + size > self.max_shard_bytes and self.current_size > 0:
        #         self._flush()
        #
        #     self.current_shard[pname] = v
        #     self.current_size += size

    # ==================================================
    def finalize(self):
        """
        Finalize sharding:
        - add remaining unsaved parameters
        - flush
        - rename shards
        - write index
        - copy python files
        """

        if self._closed:
            return

        try:
            full_sd = self.model.state_dict()
        except Exception as e:
            logger.warning(f"failed to get full state_dict: {e}")
            full_sd = {}

        tie_word_embeddings = getattr(
            getattr(self.model, "config", None),
            "tie_word_embeddings",
            True,
        )
        layer_state = {}
        for k, v in full_sd.items():
            pname = k
            layer_name = ".".join(pname.split(".")[:-1])
            if layer_name in self.current_shard.keys():
                continue
            if not isinstance(v, torch.Tensor):
                continue

            if layer_name not in layer_state:
                layer_state[layer_name] = {}
            layer_state[layer_name][k] = v

        for layer_name, v in layer_state.items():
            if self.lm_head_name is not None and layer_name == self.lm_head_name and tie_word_embeddings:
                lm_head_module = get_module(self.model, self.lm_head_name)
                lm_head_module.to(
                    "meta"
                )  # Must to meta, otherwise, the save_pretrained will save it and override some checkpoints
                continue
            size = sum([t.numel() * t.element_size() for t in v.values()])
            self.total_param_size_bytes += size

            if size > self.max_shard_bytes:
                self._flush()
                self.current_shard[layer_name] = v
                self.current_size = size
                self._flush()
                continue

            if self.current_size + size > self.max_shard_bytes and self.current_size > 0:
                self._flush()

            self.current_shard[layer_name] = v
            self.current_size += size

        # for pname, tensor in full_sd.items():
        #     if self.lm_head_name is not None and self.lm_head_name in pname and tie_word_embeddings:
        #         continue
        #     if not isinstance(tensor, torch.Tensor):
        #         continue
        #
        #     elems = tensor.numel()
        #     size = elems * tensor.element_size()
        #
        #     self.total_param_elems += elems
        #     self.total_param_size_bytes += size
        #
        #     if size > self.max_shard_bytes:
        #         self._flush()
        #         self.current_shard[pname] = tensor.detach().cpu()
        #         self.current_size = size
        #         self._flush()
        #         continue
        #
        #     if self.current_size + size > self.max_shard_bytes and self.current_size > 0:
        #         self._flush()
        #
        #     self.current_shard[pname] = tensor.detach().cpu()
        #     self.current_size += size

        self._flush()

        # -------- rename shards + index --------
        total = self.shard_counter
        for i, meta in enumerate(self.shard_meta, 1):
            old = meta["tmp_file"]
            if total == 1:
                new = f"model.{self.shard_suffix}"
            else:
                new = f"model-{i:05d}-of-{total:05d}.{self.shard_suffix}"

            os.rename(
                os.path.join(self.root, old),
                os.path.join(self.root, new),
            )
            for p in meta["params"]:
                self.global_weight_map[p] = new

        if total > 1:
            index_name = "model.safetensors.index.json" if self.use_safetensors else "model.bin.index.json"
            with open(os.path.join(self.root, index_name), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metadata": {
                            "format": "safetensors" if self.use_safetensors else "pytorch",
                            "total_shards": total,
                            "total_parameters": int(self.total_param_elems),
                            "total_size": int(self.total_param_size_bytes),
                        },
                        "weight_map": self.global_weight_map,
                    },
                    f,
                    indent=2,
                )

        # -------- copy python files --------
        try:
            copy_python_files_from_model_cache(
                self.model,
                self.root,
            )
        except Exception as e:
            logger.warning("Skipping python file copy: %s", e)

        self._closed = True
        self._close()

    # ==================================================
    def _close(self):
        """
        Explicit cleanup (safe to call multiple times)
        """
        if not self._closed:
            return
        self.current_shard.clear()
        self.shard_meta.clear()
        self.global_weight_map.clear()
