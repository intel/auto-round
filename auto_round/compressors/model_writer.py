import os
from auto_round.logger import logger
from auto_round.utils import copy_python_files_from_model_cache, get_lm_head_name
import torch
from auto_round.utils import get_module
import json


#TODO decouple max_shard_size with dump shard size
class ShardWriter:
    """
    HF-style shard writer with immediate flushing, module-level freeing,
    safetensors support, and full cleanup on finalize.
    """

    def __init__(self, rounder):
        self.model = rounder.model

        self.max_shard_bytes = self._parse_size(
            getattr(rounder, "max_shard_size", "5GB")
        )

        self.use_safetensors = False
        if getattr(rounder, "safe_serialization", True):
            try:
                from safetensors.torch import save_file  # noqa
                self.use_safetensors = True
            except ImportError:
                logger.warning("fallback to torch.save as safetensors not installed")

        self.shard_suffix = "safetensors" if self.use_safetensors else "bin"

        self.root = os.path.join(
            rounder._get_save_folder_name(rounder.formats[0]), ""
        )
        os.makedirs(self.root, exist_ok=True)

        # -------- shard state --------
        self.current_shard = {}
        self.current_size = 0
        self.shard_meta = []
        self.shard_counter = 0
        self.global_weight_map = {}

        # -------- performance bookkeeping --------
        self.saved_params = set()
        self.module_param_count = {}
        self.module_saved_count = {}

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

    # ==================================================
    def _flush(self):
        if not self.current_shard:
            return

        self.shard_counter += 1
        fname = f"model-shard-{self.shard_counter:05d}.{self.shard_suffix}"
        fpath = os.path.join(self.root, fname)

        if self.use_safetensors:
            from safetensors.torch import save_file
            save_file(self.current_shard, fpath)
        else:
            torch.save(self.current_shard, fpath)

        params = list(self.current_shard.keys())
        self.shard_meta.append({"tmp_file": fname, "params": params})

        for p in params:
            self.saved_params.add(p)
            module_name = p.rsplit(".", 1)[0]
            self.module_saved_count[module_name] = (
                self.module_saved_count.get(module_name, 0) + 1
            )

            if (
                self.module_saved_count[module_name]
                == self.module_param_count.get(module_name, 0)
            ):
                try:
                    get_module(self.model, module_name).to("meta")
                except Exception:
                    pass

        self.current_shard.clear()
        self.current_size = 0

    # ==================================================
    def add_module(self, module, name=None):
        import torch

        if self._closed:
            raise RuntimeError("ShardWriter already finalized/closed")

        prefix = name if name is not None else module.tmp_name
        state = module.state_dict() # 这个需要同一个module的一起，这样好释放内存

        if prefix not in self.module_param_count:
            self.module_param_count[prefix] = len(state)
            self.module_saved_count.setdefault(prefix, 0)

        for k, v in state.items():
            if not isinstance(v, torch.Tensor):
                continue

            pname = f"{prefix}.{k}"
            elems = v.numel()
            size = elems * v.element_size()

            self.total_param_elems += elems
            self.total_param_size_bytes += size

            if size > self.max_shard_bytes:
                self._flush()
                self.current_shard[pname] = v
                self.current_size = size
                self._flush()
                continue

            if self.current_size + size > self.max_shard_bytes and self.current_size > 0:
                self._flush()

            self.current_shard[pname] = v
            self.current_size += size

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

        for pname, tensor in full_sd.items():
            if pname in self.saved_params:
                continue
            if self.lm_head_name is not None and self.lm_head_name in pname and tie_word_embeddings:
                continue
            if not isinstance(tensor, torch.Tensor):
                continue

            elems = tensor.numel()
            size = elems * tensor.element_size()

            self.total_param_elems += elems
            self.total_param_size_bytes += size

            if size > self.max_shard_bytes:
                self._flush()
                self.current_shard[pname] = tensor.detach().cpu()
                self.current_size = size
                self._flush()
                continue

            if self.current_size + size > self.max_shard_bytes and self.current_size > 0:
                self._flush()

            self.current_shard[pname] = tensor.detach().cpu()
            self.current_size += size

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
            index_name = (
                "model.safetensors.index.json"
                if self.use_safetensors
                else "model.bin.index.json"
            )
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
        self.saved_params.clear()
        self.module_param_count.clear()
        self.module_saved_count.clear()
        self.shard_meta.clear()
        self.global_weight_map.clear()