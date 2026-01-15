import torch
import os
from auto_round import logger
from auto_round.utils import copy_python_files_from_model_cache


def immediate_saving(rounder: object, m: torch.nn.Module, name: str = None, last_group: bool = False):
    """
    Shard-saves the parameters of a model block (or group of blocks) immediately into disk,
    accumulating tensors into size-limited shards, optionally finalizing all remaining
    model weights when processing the last group.

    Args:
        rounder (object): The object of compressor.
        m (torch.nn.Module): The current block (or composite module) whose parameters will be added to the shard set.
        name (str): Override module name used as prefix for saved parameter keys. If None, falls back to m.global_name.
        last_group (bool): If True, triggers final pass over the entire model to include unsaved weights,
            writes shard index, renames shard files, copies source files, and releases temporary state.
    """
    import json
    from collections import OrderedDict

    from auto_round.utils import clear_memory, get_module

    # User configurable (can be preset on rounder)
    max_shard_size = getattr(rounder, "max_shard_size", "5GB")
    safe_serialization = getattr(rounder, "safe_serialization", True)
    if not hasattr(rounder, "quantized_layer_names_outside_blocks"):
        rounder.quantized_layer_names_outside_blocks = rounder._get_quantized_layer_names_outside_blocks()
    layer_names = rounder.quantized_layer_names_outside_blocks
    if len(layer_names) > 0 and name != layer_names[-1]:
        last_group = False

    def _parse_size(size_str: str) -> int:
        s = size_str.strip().upper()
        if s.endswith("GB"):
            return int(s[:-2]) * (1024**3)
        if s.endswith("MB"):
            return int(s[:-2]) * (1024**2)
        if s.endswith("KB"):
            return int(s[:-2]) * 1024
        return int(s)

    # Init global accumulators (once)
    if not hasattr(rounder, "_shard_init_done"):
        rounder._shard_init_done = True
        rounder._max_shard_bytes = _parse_size(str(max_shard_size))
        rounder._use_safetensors = False
        if safe_serialization:
            try:
                from safetensors.torch import save_file as _sf  # noqa

                rounder._use_safetensors = True
            except ImportError:
                logger.warning("safe_serialization=True but safetensors not installed; fallback to torch.save.")
        rounder._current_shard_tensors = OrderedDict()
        rounder._current_shard_size = 0
        rounder._shard_meta = []  # list of dicts: {file, params}
        rounder._global_weight_map = {}  # param_name -> final shard file (filled after finalize)
        rounder._shard_counter = 0
        rounder._shard_suffix = "safetensors" if rounder._use_safetensors else "bin"
        # new global counters
        rounder._total_param_elems = 0
        rounder._total_param_size_bytes = 0
        # Directory
        rounder._packed_blocks_root = os.path.join(rounder._get_save_folder_name(rounder.formats[0]), "")
        os.makedirs(rounder._packed_blocks_root, exist_ok=True)

    # Collect tensors directly from current (multi)block `m`
    flat_tensors = OrderedDict()
    for k, v in m.state_dict().items():
        global_name = name if name is not None else m.global_name
        if isinstance(v, torch.Tensor):
            flat_tensors[f"{global_name}.{k}"] = v

    # Append tensors into the running shard(s)
    def _flush_current_shard():
        if len(rounder._current_shard_tensors) == 0:
            return
        rounder._shard_counter += 1
        global_name = f"model-shard-{rounder._shard_counter:05d}.{rounder._shard_suffix}"  # temporary name
        tmp_path = os.path.join(rounder._packed_blocks_root, global_name)
        if rounder._use_safetensors:
            from safetensors.torch import save_file

            save_file(rounder._current_shard_tensors, tmp_path)
        else:
            torch.save(rounder._current_shard_tensors, tmp_path)
        params = list(rounder._current_shard_tensors.keys())
        rounder._shard_meta.append({"tmp_file": global_name, "params": params})
        for param in params:
            free_module_name = param.rsplit(".", 1)[0]
            free_module = get_module(rounder.model, free_module_name)

            # free module only when all its parameters have been saved
            free_flag = True
            free_module_state_dict = free_module.state_dict()
            already_saved_name = []
            for _meta in rounder._shard_meta:
                already_saved_name += _meta.get("params", [])
            for free_module_key in free_module_state_dict:
                free_module_key_full_name = f"{free_module_name}.{free_module_key}"
                if free_module_key_full_name not in already_saved_name:
                    free_flag = False
            if free_flag:
                free_module.to("meta")
                del rounder._current_shard_tensors[param]
        rounder._current_shard_tensors = OrderedDict()
        rounder._current_shard_size = 0

    for pname, tensor in flat_tensors.items():
        t_elems = tensor.numel()
        t_size = t_elems * tensor.element_size()
        # accumulate global stats
        rounder._total_param_elems += t_elems
        rounder._total_param_size_bytes += t_size
        if t_size > rounder._max_shard_bytes:
            _flush_current_shard()
            rounder._current_shard_tensors[pname] = tensor
            rounder._current_shard_size = t_size
            _flush_current_shard()
            continue
        if rounder._current_shard_size + t_size > rounder._max_shard_bytes and rounder._current_shard_size > 0:
            _flush_current_shard()
        rounder._current_shard_tensors[pname] = tensor
        rounder._current_shard_size += t_size

    if last_group:

        # 1) Add the remaining (unsaved) model weights into new shard(s),
        # do not overwrite the already saved weights.
        try:
            full_sd = rounder.model.state_dict()
        except Exception as e:
            logger.warning(f"failed to obtain full state_dict for remaining weights: {e}")
            full_sd = {}
        tie_word_embeddings: bool = getattr(getattr(rounder.model, "config", None), "tie_word_embeddings", True)
        for pname, tensor in full_sd.items():
            if "lm_head" in pname and tie_word_embeddings:
                continue
            if not isinstance(tensor, torch.Tensor):
                continue
            # Check whether pname already stored in previous shards via _shard_meta
            already_saved = False
            for _meta in rounder._shard_meta:
                if pname in _meta.get("params", []):
                    already_saved = True
                    break
            if already_saved:
                continue  # skip weights already saved
            # Size accounting
            t_elems = tensor.numel()
            t_size = t_elems * tensor.element_size()

            # Update global stats (these counters may already include earlier packed weights)
            rounder._total_param_elems += t_elems
            rounder._total_param_size_bytes += t_size

            # If this tensor alone exceeds shard size -> dedicated shard
            if t_size > rounder._max_shard_bytes:
                _flush_current_shard()
                rounder._current_shard_tensors[pname] = tensor.detach().cpu()
                rounder._current_shard_size = t_size
                _flush_current_shard()
                continue

            # If adding this tensor would overflow current shard -> flush current first
            if rounder._current_shard_size + t_size > rounder._max_shard_bytes and rounder._current_shard_size > 0:
                _flush_current_shard()

            # Add to current shard
            rounder._current_shard_tensors[pname] = tensor.detach().cpu()
            rounder._current_shard_size += t_size

        # 2) Flush any remaining unsaved leftover tensors
        _flush_current_shard()

        # 3) Finalize: rename temp shard files to HF-style names and build index
        total_shards = rounder._shard_counter
        if total_shards == 0:
            logger.warning("no tensors saved across all blocks")
        else:
            final_names = []
            for idx, meta in enumerate(rounder._shard_meta, start=1):
                old_tmp = meta["tmp_file"]
                old_path = os.path.join(rounder._packed_blocks_root, old_tmp)
                if total_shards == 1:
                    new_name = f"model.{rounder._shard_suffix}"
                else:
                    new_name = f"model-{idx:05d}-of-{total_shards:05d}.{rounder._shard_suffix}"
                new_path = os.path.join(rounder._packed_blocks_root, new_name)
                os.rename(old_path, new_path)
                final_names.append(new_name)
                for p in meta["params"]:
                    rounder._global_weight_map[p] = new_name

            index_fname = "model.safetensors.index.json" if rounder._use_safetensors else "model.bin.index.json"
            index_path = os.path.join(rounder._packed_blocks_root, index_fname)
            index_data = {
                "metadata": {
                    "format": "safetensors" if rounder._use_safetensors else "pytorch",
                    "total_shards": total_shards,
                    "total_parameters": int(getattr(rounder, "_total_param_elems", 0)),
                    "total_size": int(getattr(rounder, "_total_param_size_bytes", 0)),
                },
                "weight_map": rounder._global_weight_map,
            }
            if total_shards > 1:
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(index_data, f, indent=2)
            logger.info(
                f"saved {total_shards} shard(s) (HF-style, including remaining unsaved weights) to "
                f"{rounder._packed_blocks_root} (index: {index_fname})"
            )

            try:
                copy_python_files_from_model_cache(rounder.model, rounder._get_save_folder_name(rounder.formats[0]))
            except Exception as e:
                logger.warning("Skipping source model Python file copy due to error: %s", e)

        # 4) Cleanup attributes to release memory after final shard is written
        try:
            attrs_to_cleanup = [
                "_current_shard_tensors",
                "_current_shard_size",
                "_shard_counter",
                "_max_shard_bytes",
                "_use_safetensors",
                "_shard_suffix",
                "_packed_blocks_root",
                "_total_param_elems",
                "_total_param_size_bytes",
                "_shard_init_done",
                "_shard_meta",
                "_global_weight_map",
            ]
            for _attr in attrs_to_cleanup:
                if hasattr(rounder, _attr):
                    delattr(rounder, _attr)
        except Exception as _cleanup_err:
            logger.warning(f"shard cleanup warning: {_cleanup_err}")
