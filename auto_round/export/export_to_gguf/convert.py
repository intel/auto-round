# Copyright (c) 2024 Intel Corporation
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

# Copyright (c) 2023-2024 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import json
import os
import re
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import torch
from transformers import AutoConfig

from auto_round.export.export_to_gguf.config import ModelType
from auto_round.export.export_to_gguf.gguf_dtype import GGUFDTypeSelector
from auto_round.export.export_to_gguf.hf_checkpoint_restorer import HFCheckpointRestorer, RestoredTensor
from auto_round.export.export_to_gguf.packing import ggml_quant
from auto_round.utils import (
    LazyImport,
    clean_module_parameter,
    clear_memory,
    get_module,
    get_packing_device,
    is_quantized_input_module,
    is_separate_tensor,
    is_transformers_version_greater_or_equal_5_4_0,
    logger,
)

gguf = LazyImport("gguf")

if TYPE_CHECKING:
    from torch import Tensor


def wrapper_model_instance(
    model_instance,
    model,
    layer_config,
    low_cpu_mem_usage=False,
    device=None,
    quant_nontext_module=False,
    is_auto_scheme=False,
):
    if model_instance.model_arch == gguf.MODEL_ARCH.MMPROJ and model_instance.fname_out.is_dir():
        model_instance.fname_out = model_instance.fname_out / "mmproj-model.gguf"
    model_instance.model = model
    model_instance.layer_config = layer_config
    model_instance.is_auto_scheme = is_auto_scheme
    model_instance.low_cpu_mem_usage = _need_low_cpu_mem(low_cpu_mem_usage)
    model_instance._gguf_written_hf_names = set()
    model_instance._gguf_written_checkpoint_names = set()

    model_instance.get_tensors = partial(get_restored_tensors, model_instance)
    model_instance.prepare_tensors = partial(prepare_tensors, model_instance)

    model_instance.device = device
    model_instance.quant_nontext_module = quant_nontext_module

    return model_instance


def _need_low_cpu_mem(low_cpu_mem_usage):
    if not low_cpu_mem_usage:
        return False

    # process = psutil.Process(os.getpid())
    # mem_usage = process.memory_info().rss
    # memory_info = psutil.virtual_memory()
    # if memory_info.available > mem_usage / 3:
    #     return False
    # else:
    #     logger.info("use low cpu memory mode.")
    return True


def get_moe_name(cls, name, new_name):
    type_mapping = {
        "FFN_GATE_EXP": ["gate_proj", "w1", "linear"],
        "FFN_DOWN_EXP": ["down_proj", "w2", "linear_1"],
        "FFN_UP_EXP": ["up_proj", "w3", "linear_v"],
    }
    nums = re.findall(r"\.(\d+)\.", name)
    if len(nums) != 2:
        return name
    name_tmp = name[: -len(".weight")].replace(f".{nums[1]}", "")
    new_name_tmp = new_name[: -len(".weight")]
    if cls.tensor_map.get_type(name_tmp) == cls.tensor_map.get_type(new_name_tmp):
        return name

    tensor_type = cls.tensor_map.get_type(new_name_tmp).name
    experts_name = name_tmp.split(".")[-1]
    for k, v in type_mapping.items():
        if experts_name in v:
            idx = v.index(experts_name)
            name = name.replace(experts_name, type_mapping[tensor_type][idx])
            return name
    return name


def is_mmproj_tensor_name(name):
    from auto_round.utils.common import MM_KEYS

    name = name.lower()
    return any(key in name for key in MM_KEYS)


def _iter_extra_tensors(cls):
    def is_extra_tensor(tensor_name):
        if getattr(cls.model, "_is_quantized_input_module", False) and "scale" in tensor_name.split(".")[-1]:
            return False
        return tensor_name not in getattr(cls.model, "tensor_name_list", [])

    extra_tensor = {}
    if hasattr(cls.model, "name_or_path"):
        from safetensors import safe_open

        from auto_round.export.export_to_gguf.special_handle import get_tensor_from_file
        from auto_round.utils import download_or_get_path

        dir_path = cls.model.name_or_path
        if not os.path.isdir(dir_path):
            dir_path = download_or_get_path(dir_path)
        INDEX_FILE = "model.safetensors.index.json"
        if INDEX_FILE in os.listdir(dir_path):
            with open(os.path.join(dir_path, INDEX_FILE)) as f:
                tensor_index = json.load(f)
            for tensor_name in tensor_index["weight_map"]:
                if is_extra_tensor(tensor_name):
                    extra_tensor[tensor_name] = get_tensor_from_file(dir_path, tensor_name)
        else:
            model_file = os.path.join(dir_path, "model.safetensors")
            if os.path.exists(model_file):
                f = safe_open(model_file, framework="pt")
                for tensor_name in f.keys():
                    if is_extra_tensor(tensor_name):
                        extra_tensor[tensor_name] = get_tensor_from_file(dir_path, tensor_name)

    yield from extra_tensor.items()


def get_restored_tensors(cls) -> Iterator[RestoredTensor]:
    written_hf_names = getattr(cls, "_gguf_written_hf_names", set())
    written_checkpoint_names = getattr(cls, "_gguf_written_checkpoint_names", set())
    cls.model.tensor_name_list = list(written_hf_names | written_checkpoint_names)

    pending_checkpoint_tensors = {}
    for restored in HFCheckpointRestorer(cls.model, completed_hf_names=written_hf_names).iter_tensors():
        tensor = restored.tensor_fn()
        if tensor is None or tensor.numel() == 0:
            pending_checkpoint_tensors[restored.checkpoint_name] = restored
            continue
        for tensor_name in (restored.checkpoint_name, *restored.hf_names):
            if tensor_name not in cls.model.tensor_name_list:
                cls.model.tensor_name_list.append(tensor_name)
        yield RestoredTensor(
            restored.checkpoint_name,
            lambda tensor=tensor: tensor,
            restored.hf_names,
            restored.transform_kind,
        )

    for name, tensor in _iter_extra_tensors(cls):
        pending = pending_checkpoint_tensors.pop(name, None)
        if pending is None:
            yield RestoredTensor(name, lambda tensor=tensor: tensor, (name,), "extra")
        else:
            yield RestoredTensor(name, lambda tensor=tensor: tensor, pending.hf_names, pending.transform_kind)


def _quant_data_with_args(
    data_torch, data_qtype, scale, zp, d_scale=None, wmin=None, d_wmin=None, imatrix=None, device=None
):
    device = data_torch.device if device is None else device
    data_torch = data_torch.to(torch.float32)
    scale = scale.to(torch.float32) if isinstance(scale, torch.Tensor) else scale
    zp = zp.to(torch.float32) if isinstance(zp, torch.Tensor) else zp
    d_scale = d_scale.to(torch.float32) if isinstance(d_scale, torch.Tensor) else d_scale
    d_wmin = d_wmin.to(torch.float32) if isinstance(d_wmin, torch.Tensor) else d_wmin
    wmin = wmin.to(torch.float32) if isinstance(wmin, torch.Tensor) else wmin
    imatrix = imatrix.to(torch.float32) if isinstance(imatrix, torch.Tensor) else imatrix

    data = ggml_quant(
        data_torch,
        data_qtype.name.lower(),
        scale,
        zp,
        wmin=wmin,
        d_scale=d_scale,
        d_wmin=d_wmin,
        imatrix=imatrix,
        device=device,
    )
    return data


def _tensors_equal(a, b):
    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        b = b.to(a.dtype)
    if a.device != b.device:
        b = b.to(a.device)
    return torch.equal(a, b)


def _trace_modify_transform(cls, in_shape, modify_name, new_name, bid, out_shape):
    """Probe ``cls.modify_tensors`` with index tensors to identify the weight transform.

    Runs the transform on row-index and column-index tensors (each twice, with an
    affine pair ``2*x+1`` to reject arithmetic transforms that would fake a valid
    index map). Returns ``(rowmap, colmap)`` mapping every output row/column back
    to its source row/column when the transform is a pure gather that keeps rows
    and columns independent, else None.
    """
    rows, cols = in_shape

    def run(probe):
        outputs = dict(cls.modify_tensors(probe, modify_name, bid))
        out = outputs.get(new_name)
        if out is None or tuple(out.shape) != tuple(out_shape):
            return None
        return out.to(torch.float32)

    try:
        row_probe = torch.arange(rows, dtype=torch.float32).unsqueeze(1).expand(rows, cols).contiguous()
        out_r = run(row_probe)
        out_r2 = run(row_probe * 2 + 1)
        del row_probe
        if out_r is None or out_r2 is None or not torch.equal(out_r2, out_r * 2 + 1):
            return None  # not a pure gather (arithmetic transform)
        if not torch.equal(out_r, out_r[:, :1].expand_as(out_r)):
            return None  # mixes different source rows into one output row
        del out_r2

        col_probe = torch.arange(cols, dtype=torch.float32).unsqueeze(0).expand(rows, cols).contiguous()
        out_c = run(col_probe)
        out_c2 = run(col_probe * 2 + 1)
        del col_probe
        if out_c is None or out_c2 is None or not torch.equal(out_c2, out_c * 2 + 1):
            return None
        if not torch.equal(out_c, out_c[:1, :].expand_as(out_c)):
            return None  # mixes different source columns into one output column

        rowmap = out_r[:, 0].round().to(torch.long)
        colmap = out_c[0].round().to(torch.long)
    except Exception as e:
        logger.debug("transform tracing failed for %s: %s", new_name, e)
        return None
    if rowmap.numel() == 0 or rowmap.min() < 0 or rowmap.max() >= rows:
        return None
    if colmap.numel() == 0 or colmap.min() < 0 or colmap.max() >= cols:
        return None
    return rowmap, colmap


def _remap_traced_attrs(kwargs, rowmap, colmap, in_shape):
    """Apply a traced (rowmap, colmap) gather to per-group quant params.

    Quant params are row-major with one or more values per group of weight
    columns, so row gathers always transfer; column gathers transfer only when
    they move whole param groups. Returns the remapped params, or None when any
    present param cannot be remapped (params must be used all-or-nothing).
    """
    in_rows, in_cols = in_shape
    col_identity = colmap.numel() == in_cols and torch.equal(colmap, torch.arange(in_cols, dtype=colmap.dtype))
    remapped = {}
    for key, t in kwargs.items():
        if t is None or not isinstance(t, torch.Tensor):
            remapped[key] = t
            continue
        if key == "imatrix":
            # imatrix is per input column and unaffected by row gathers
            if col_identity:
                remapped[key] = t
            elif t.dim() == 1 and t.numel() == in_cols:
                remapped[key] = t[colmap.to(t.device)]
            else:
                return None
            continue
        if t.numel() == 0 or t.numel() % in_rows != 0:
            return None
        m = t.reshape(in_rows, -1)
        m = m.index_select(0, rowmap.to(t.device))
        if not col_identity:
            per_row = m.shape[1]
            if per_row == 0 or in_cols % per_row != 0:
                return None
            group_size = in_cols // per_row  # weight columns covered by one param value
            if colmap.numel() % group_size != 0:
                return None
            groups = colmap.reshape(-1, group_size)
            starts = groups[:, 0]
            expected = starts.unsqueeze(1) + torch.arange(group_size, dtype=colmap.dtype)
            if not (torch.equal(groups, expected) and bool((starts % group_size == 0).all())):
                return None  # column moves cross param-group boundaries
            m = m.index_select(1, (starts // group_size).to(t.device))
        remapped[key] = m
    return remapped


def _adapt_quant_attrs_to_transform(cls, module, data_torch, modify_name, new_name, bid, kwargs):
    """Make the module's quant params (stored in the original HF weight layout)
    valid for the tensor produced by ``modify_tensors``.

    Returns the (possibly remapped) params, or None when the transform cannot be
    applied to them — the caller must then drop the params so ``ggml_quant``
    re-searches them on the transformed weights.
    """
    weight = getattr(module, "weight", None)
    if weight is None or not isinstance(weight, torch.Tensor):
        return None
    if _tensors_equal(data_torch, weight):
        return kwargs
    if weight.dim() != 2 or data_torch.dim() != 2:
        return None
    trace = _trace_modify_transform(cls, tuple(weight.shape), modify_name, new_name, bid, tuple(data_torch.shape))
    if trace is None:
        return None
    return _remap_traced_attrs(kwargs, trace[0], trace[1], tuple(weight.shape))


def _verify_packed_tensor(data, data_qtype, ref, new_name):
    """Guard against packing with mismatched quant params: dequantize a sample of
    the packed tensor and compare against the (already fake-quantized) weights.
    When the provided params match the tensor layout the round-trip is near-exact,
    so any sizable deviation means the export would be silently corrupted.

    ``ref`` must be a snapshot of the leading weight rows taken BEFORE ggml_quant,
    whose kernels modify their input buffer in place."""
    try:
        arr = data if isinstance(data, np.ndarray) else np.asarray(data)
        if arr.ndim < 2 or ref.dim() < 2:
            return
        k = min(arr.shape[0], ref.shape[0])
        dequantized = gguf.quants.dequantize(arr[:k], data_qtype)
        ref = ref[:k].detach().to(torch.float32).cpu().numpy()
        if dequantized.shape != ref.shape:
            return
        rel_err = float(np.linalg.norm(dequantized - ref)) / max(float(np.linalg.norm(ref)), 1e-12)
    except Exception as e:
        logger.debug("pack verification skipped for %s: %s", new_name, e)
        return
    if rel_err > 0.1:
        raise ValueError(
            f"{new_name}: packed tensor deviates from the quantized weights by {rel_err:.1%}; "
            "the quantization params do not match the tensor layout, the export would be corrupted"
        )
    if rel_err > 0.02:
        logger.warning(f"{new_name}: packed tensor deviates from the quantized weights by {rel_err:.1%}")


def _expected_scale_columns(ggml_type):
    from auto_round.export.export_to_gguf.config import QK_K

    if ggml_type in {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0"}:
        return 1
    if ggml_type in {"q2_k", "q3_k", "q6_k"}:
        return QK_K // 16
    if ggml_type in {"q4_k", "q5_k"}:
        return QK_K // 32
    return None


def _validate_quant_attr_shapes(data_torch, data_qtype, kwargs, name, new_name):
    from auto_round.export.export_to_gguf.config import GGML_QUANT_SIZES

    ggml_type = data_qtype.name.lower()
    if ggml_type not in GGML_QUANT_SIZES or kwargs.get("scale") is None:
        return

    block_size, _ = GGML_QUANT_SIZES[ggml_type]
    n_blocks = data_torch.numel() // block_size
    scale_cols = _expected_scale_columns(ggml_type)
    if scale_cols is None:
        return

    scale = kwargs["scale"]
    if scale.numel() != n_blocks * scale_cols:
        raise ValueError(
            f"{name} -> {new_name}: {ggml_type} scale shape {tuple(scale.shape)} does not match "
            f"tensor shape {tuple(data_torch.shape)}; expected {n_blocks * scale_cols} scale values"
        )

    for attr in ["d_scale", "d_wmin"]:
        tensor = kwargs.get(attr)
        if tensor is not None and tensor.numel() != n_blocks:
            raise ValueError(
                f"{name} -> {new_name}: {ggml_type} {attr} shape {tuple(tensor.shape)} does not match "
                f"tensor shape {tuple(data_torch.shape)}; expected {n_blocks} values"
            )


def _quant_data(cls, data_torch, data_qtype, name, modify_name, new_name, bid, device=None, use_layer_attrs=True):
    """

    Args:
        data_torch: original data tensor
        data_qtype: quantization type
        name: original tensor name, for getting auto_round config, model.language_model.layers.0.input_linear.weight
        modify_name: modified tensor name, for gguf mapping, model.layers.0.input_linear.weight
        new_name: name after modify_tensors, gguf using this to save tensor, like blk.0.ffn_gate.weight
        bid: block id
        device: device to perform quantization. Defaults to None.

    """
    suffix = ".weight"
    device = data_torch.device if device is None else device

    if name.endswith(suffix):
        layer_name = name[: -len(suffix)]
    else:
        layer_name = name
    module = get_module(cls.model, layer_name)
    kwargs = {"scale": None, "zp": None, "d_scale": None, "d_wmin": None, "wmin": None, "imatrix": None}
    source_qtype = get_qtype_by_layer_config(cls.layer_config, name, data_qtype, explicit_only=True)
    use_layer_attrs = use_layer_attrs and source_qtype == data_qtype
    if use_layer_attrs:
        kwargs = {
            "scale": module.scale if hasattr(module, "scale") else None,
            "zp": module.zp if hasattr(module, "zp") else None,
            "d_scale": module.w_d_scale if hasattr(module, "w_d_scale") else None,
            "d_wmin": module.w_d_wmin if hasattr(module, "w_d_wmin") else None,
            "wmin": module.w_wmin if hasattr(module, "w_wmin") else None,
            "imatrix": module.imatrix if hasattr(module, "imatrix") else None,
        }
        if any(isinstance(v, torch.Tensor) for v in kwargs.values()):
            # modify_tensors may have permuted/split the weights (e.g. llama Q/K
            # interleave, Qwen3.5 V-head reorder); the stored quant params must
            # follow the same transform or they describe the wrong rows/columns.
            adapted = _adapt_quant_attrs_to_transform(cls, module, data_torch, modify_name, new_name, bid, kwargs)
            if adapted is None:
                logger.info(
                    f"{new_name}: weight transform is incompatible with the stored quantization params,"
                    " re-searching them on the transformed weights"
                )
                adapted = dict.fromkeys(kwargs)
            kwargs = {
                k: v.to(torch.float32) if isinstance(v, torch.Tensor) and k != "imatrix" else v
                for k, v in adapted.items()
            }
    data_torch = data_torch.to(torch.float32)
    _validate_quant_attr_shapes(data_torch, data_qtype, kwargs, name, new_name)
    # ggml_quant kernels modify their input buffer in place; snapshot the rows
    # used by the post-pack verification first.
    verify_ref = data_torch[:64].detach().clone() if kwargs.get("scale") is not None and data_torch.dim() >= 2 else None
    data = ggml_quant(data_torch, data_qtype.name.lower(), device=device, **kwargs)
    if verify_ref is not None:
        _verify_packed_tensor(data, data_qtype, verify_ref, new_name)
    # else:
    #     # if data_torch.dtype ==torch.float32:
    #     #     data_qtype = gguf.GGMLQuantizationType.F32
    #     # else:
    #     #     data_qtype = gguf.GGMLQuantizationType.F16
    #     data_qtype = gguf.GGMLQuantizationType.F32  ##FP16 has issues at inference
    #     data = data_torch.to(torch.float32).squeeze().cpu().numpy()
    return data, data_qtype


def get_qtype_by_layer_config(layer_config, name, data_qtype, *, explicit_only=False):
    name = name[: -len(".weight")]
    if name not in layer_config and name.endswith("embed_tokens"):
        embedding_names = [key for key in layer_config if key.endswith("embed_tokens")]
        if len(embedding_names) == 1:
            name = embedding_names[0]
    elif name == "token_embd":
        embedding_names = [key for key in layer_config if key.endswith("embed_tokens")]
        if len(embedding_names) == 1:
            name = embedding_names[0]
    if name not in layer_config:
        return None if explicit_only else data_qtype
    if layer_config[name]["bits"] >= 16:
        return data_qtype
    bits = layer_config[name].get("bits")
    super_bits = layer_config[name].get("super_bits")
    sym = layer_config[name].get("sym")
    if bits == 2:
        return gguf.GGMLQuantizationType.Q2_K
    if bits == 3:
        return gguf.GGMLQuantizationType.Q3_K
    if bits == 4:
        if super_bits is not None:
            return gguf.GGMLQuantizationType.Q4_K
        if super_bits is None and sym:
            return gguf.GGMLQuantizationType.Q4_0
        if super_bits is None and not sym:
            return gguf.GGMLQuantizationType.Q4_1
    if bits == 5:
        if super_bits is not None:
            return gguf.GGMLQuantizationType.Q5_K
        if super_bits is None and sym:
            return gguf.GGMLQuantizationType.Q5_0
        if super_bits is None and not sym:
            return gguf.GGMLQuantizationType.Q5_1
    if bits == 6:
        return gguf.GGMLQuantizationType.Q6_K
    if bits == 8:
        return gguf.GGMLQuantizationType.Q8_0
    raise ValueError(f"Unknown file type: {data_qtype}")

    # data_qtype = re.sub(r"Q\d", f"Q{bits}", cls.ftype.name)
    #
    # if data_qtype == "MOSTLY_Q8_0":
    #     data_qtype = gguf.GGMLQuantizationType.Q8_0
    # elif data_qtype == "MOSTLY_Q4_0":
    #     data_qtype = gguf.GGMLQuantizationType.Q4_0
    # elif data_qtype == "MOSTLY_Q4_1":
    #     data_qtype = gguf.GGMLQuantizationType.Q4_1
    # elif data_qtype == "MOSTLY_Q5_0":
    #     data_qtype = gguf.GGMLQuantizationType.Q5_0
    # elif data_qtype == "MOSTLY_Q5_1":
    #     data_qtype = gguf.GGMLQuantizationType.Q5_1
    # elif data_qtype.startswith("MOSTLY_Q2_K"):
    #     data_qtype = gguf.GGMLQuantizationType.Q2_K
    # elif data_qtype.startswith("MOSTLY_Q3_K"):
    #     data_qtype = gguf.GGMLQuantizationType.Q3_K
    # elif data_qtype.startswith("MOSTLY_Q4_K"):
    #     data_qtype = gguf.GGMLQuantizationType.Q4_K
    # elif data_qtype.startswith("MOSTLY_Q5_K"):
    #     data_qtype = gguf.GGMLQuantizationType.Q5_K
    # elif data_qtype.startswith("MOSTLY_Q6_K"):
    #     data_qtype = gguf.GGMLQuantizationType.Q6_K
    # else:
    #     raise ValueError(f"Unknown file type: {data_qtype}")
    # return data_qtype


def _qtype_name(qtype):
    return qtype.name if hasattr(qtype, "name") else str(qtype)


def _qtype_precision_rank(qtype):
    ranks = {
        gguf.GGMLQuantizationType.Q2_K: 2,
        gguf.GGMLQuantizationType.Q3_K: 3,
        gguf.GGMLQuantizationType.Q4_0: 4,
        gguf.GGMLQuantizationType.Q4_1: 4,
        gguf.GGMLQuantizationType.Q4_K: 4,
        gguf.GGMLQuantizationType.Q5_0: 5,
        gguf.GGMLQuantizationType.Q5_1: 5,
        gguf.GGMLQuantizationType.Q5_K: 5,
        gguf.GGMLQuantizationType.Q6_K: 6,
        gguf.GGMLQuantizationType.Q8_0: 8,
        gguf.GGMLQuantizationType.Q8_1: 8,
        gguf.GGMLQuantizationType.Q8_K: 8,
        gguf.GGMLQuantizationType.F16: 16,
        gguf.GGMLQuantizationType.BF16: 16,
        gguf.GGMLQuantizationType.F32: 32,
    }
    return ranks.get(qtype, 0)


def _should_keep_recipe_qtype(layer_config_qtype, fallback_qtype, allow_recipe_fallback):
    """Keep llama.cpp fixed-format recipe upgrades unless AutoScheme owns dtype selection."""
    if not allow_recipe_fallback or layer_config_qtype is None:
        return False
    return _qtype_precision_rank(fallback_qtype) > _qtype_precision_rank(layer_config_qtype)


def resolve_restored_qtype(
    layer_config,
    hf_names,
    checkpoint_name,
    gguf_name,
    fallback_qtype,
    diagnostics,
    allow_recipe_fallback=False,
):
    source_qtypes = [
        get_qtype_by_layer_config(layer_config, hf_name, fallback_qtype, explicit_only=True) for hf_name in hf_names
    ]
    matched_qtypes = [qtype for qtype in source_qtypes if qtype is not None]

    if len(matched_qtypes) == 1 and len(hf_names) == 1:
        if _should_keep_recipe_qtype(matched_qtypes[0], fallback_qtype, allow_recipe_fallback):
            return None
        return matched_qtypes[0]
    if len(matched_qtypes) > 0 and len(matched_qtypes) == len(source_qtypes):
        unique_qtypes = set(matched_qtypes)
        if len(unique_qtypes) == 1:
            if _should_keep_recipe_qtype(matched_qtypes[0], fallback_qtype, allow_recipe_fallback):
                return None
            return matched_qtypes[0]

    if len(hf_names) <= 1 and len(matched_qtypes) == 0:
        reason = "layer_config_miss"
    elif len(matched_qtypes) == 0:
        reason = "multi_source_layer_config_miss"
    elif len(matched_qtypes) != len(source_qtypes):
        reason = "multi_source_partial_layer_config_miss"
    else:
        reason = "multi_source_dtype_conflict"

    diagnostics.append(
        {
            "checkpoint_name": checkpoint_name,
            "gguf_name": gguf_name,
            "reason": reason,
            "fallback_qtype": _qtype_name(fallback_qtype),
            "hf_sources": [
                {
                    "name": hf_name,
                    "layer_config_qtype": None if qtype is None else _qtype_name(qtype),
                }
                for hf_name, qtype in zip(hf_names, source_qtypes)
            ],
        }
    )
    return None


def _special_name_handle(cls, name):
    # using transformers >= 5.4, model.language_model.embed_tokens.weight changed to
    # model.language_model.model.embed_tokens.weight after saved
    if is_transformers_version_greater_or_equal_5_4_0():
        name = name.replace("language_model.model.", "language_model.")

    # for Qwen2VL
    def remove_prefix(name, key_list):
        for key in key_list:
            if key in name and not name.startswith(key):
                name = ".".join(name.split(".")[1:])
                break
        return name

    if cls.model_arch == gguf.MODEL_ARCH.QWEN2VL:
        name = name.replace("language_model.", "")
        visual_keys = ["thinker", "visual", "audio", "talker", "token2wav"]
        name = remove_prefix(name, visual_keys)
    if cls.__class__.__name__ == "Qwen2VLVisionModel":
        if "visual." in name and not name.startswith("visual."):
            name = ".".join(name.split(".")[1:])

    # # for gemma3
    if cls.model_arch == gguf.MODEL_ARCH.GEMMA3 or cls.__class__.__name__ == "Gemma3VisionModel":
        visual_keys = ["multi_modal_projector", "vision_tower", "multimodal_projector"]
        name = remove_prefix(name, visual_keys)

        # AutoRound exports tensors from a live transformers model, while gguf tensor_mapping
        # expects checkpoint-style names. Normalize known Gemma3 vision/projector variants.
        if name.startswith("vision_tower.") and not name.startswith("vision_tower.vision_model."):
            name = name.replace("vision_tower.", "vision_tower.vision_model.", 1)
        if name.startswith("multi_modal_projector.mm_input_projection_weight"):
            name = name.replace("mm_input_projection_weight", "mm_input_projection.weight", 1)

    # for LlavaForConditionalGeneration
    if cls.model_arch == gguf.MODEL_ARCH.LLAMA:
        name = name.replace("language_model.", "")
        # name = name.replace("model.", "")
    if cls.__class__.__name__ == "LlavaVisionModel":
        visual_keys = ["multi_modal_projector", "vision_tower"]
        name = remove_prefix(name, visual_keys)

    # for InternVisionModel
    if cls.__class__.__name__ == "InternVisionModel":
        visual_keys = ["vision_model"]
        name = remove_prefix(name, visual_keys)

    return name


def _to_restored_tensor(item):
    if isinstance(item, RestoredTensor):
        return item
    name, data_torch = item
    return RestoredTensor(name, lambda data_torch=data_torch: data_torch, (name,), "passthrough")


def filter_restored_tensor(cls, restored):
    filtered = cls.filter_tensors((restored.checkpoint_name, restored.tensor_fn))
    if filtered is None:
        return None
    return RestoredTensor(filtered[0], filtered[1], restored.hf_names, restored.transform_kind)


def _gguf_writer_has_tensor(gguf_writer, tensor_name):
    for tensor_info in gguf_writer.tensors:
        if isinstance(tensor_info, dict):
            if tensor_name in tensor_info:
                return True
            continue
        existing_name = getattr(tensor_info, "name", None)
        if existing_name is None and isinstance(tensor_info, (list, tuple)) and len(tensor_info) > 0:
            existing_name = tensor_info[0]
        if tensor_name == existing_name:
            return True
    return False


def _count_attention_wv_tensors(cls):
    count = 0
    for item in chain(cls.generate_extra_tensors(), cls.get_tensors()):
        restored = _to_restored_tensor(item)
        filtered = filter_restored_tensor(cls, restored)
        if filtered is None:
            continue
        data_torch = filtered.tensor_fn()
        if data_torch is None or data_torch.numel() == 0:
            continue
        modify_name = _special_name_handle(cls, filtered.checkpoint_name)
        try:
            new_name = cls.map_tensor_name(modify_name)
        except Exception:
            continue
        count += any(key in new_name for key in ("attn_v.weight", "attn_qkv.weight", "attn_kv_b.weight"))
    return count or None


def _flush_name_resolution_diagnostics(cls):
    diagnostics = getattr(cls, "_gguf_name_resolution_diagnostics", None)
    if diagnostics is None or len(diagnostics) == 0:
        return
    logger.debug("gguf: recorded %d in-memory name resolution fallback diagnostics", len(diagnostics))


def prepare_tensors(cls):
    device = get_packing_device(cls.device)
    if not hasattr(cls, "_gguf_dtype_selector"):
        has_tied_embeddings = cls.hparams.get("tie_word_embeddings", False)
        cls._gguf_dtype_selector = GGUFDTypeSelector(
            cls.hparams, cls.ftype, cls.model_arch, has_tied_embeddings=has_tied_embeddings
        )
        cls._gguf_dtype_selector.n_attention_wv = _count_attention_wv_tensors(cls)
    if not hasattr(cls, "_gguf_name_resolution_diagnostics"):
        cls._gguf_name_resolution_diagnostics = []

    # Handle empty tensor_map for models with block_count=0 (like MobileNetV5)
    if cls.tensor_map.mapping:
        max_name_len = max(len(s) for _, s in cls.tensor_map.mapping.values()) + len(".weight,")
    else:
        max_name_len = len("vision_encoder.weight,")  # Default reasonable length

    for item in chain(cls.generate_extra_tensors(), cls.get_tensors()):
        restored = _to_restored_tensor(item)
        name = restored.checkpoint_name
        is_mmproj_model = cls.model_arch == gguf.MODEL_ARCH.MMPROJ
        if is_mmproj_model != is_mmproj_tensor_name(name):
            continue
        if any(
            hf_name in getattr(cls.model, "_tied_weights_keys", []) and not is_separate_tensor(cls.model, hf_name)
            for hf_name in restored.hf_names
        ):
            continue
        data_torch = restored.tensor_fn()
        if data_torch is None or data_torch.numel() == 0:
            continue
        # we don't need these
        if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
            continue
        if hasattr(cls, "current_packing_block") and cls.current_packing_block is not None:  # pylint: disable=E1101
            current_packing_block_split = cls.current_packing_block.split(".")  # pylint: disable=E1101
            name_split = name.split(".")
            if (
                len(name_split) < len(current_packing_block_split)
                or name_split[: len(current_packing_block_split)] != current_packing_block_split
            ):
                continue

        filtered = filter_restored_tensor(cls, restored)
        if filtered is None:
            continue
        checkpoint_name = filtered.checkpoint_name
        hf_names = filtered.hf_names or (checkpoint_name,)
        hf_name = hf_names[0]

        old_dtype = data_torch.dtype

        # convert any unsupported data types to float32
        if data_torch.dtype not in (torch.float16, torch.float32):
            data_torch = data_torch.to(torch.float32)

        # use the first number-like part of the tensor name as the block id
        bid = None
        for part in name.split("."):
            if part.isdecimal():
                bid = int(part)
                break

        clean_weight_list = []

        modify_name = _special_name_handle(cls, checkpoint_name)
        restored_outputs_completed = False
        for new_name, data_torch in cls.modify_tensors(data_torch, modify_name, bid):
            restored_outputs_completed = True
            if _gguf_writer_has_tensor(cls.gguf_writer, new_name):
                logger.debug("%s already added to gguf_writer, skip", new_name)
                continue
            # squeeze is necessary for reloading in transformers.
            data = data_torch.squeeze()
            n_dims = len(data.shape)
            data_qtype: gguf.GGMLQuantizationType | bool = cls.tensor_force_quant(
                checkpoint_name, new_name, bid, n_dims
            )

            # Most of the codebase that takes in 1D tensors or norms only handles F32 tensors
            if n_dims <= 1 or new_name.endswith("_norm.weight"):
                data_qtype = gguf.GGMLQuantizationType.F32

            # Conditions should closely match those in llama_model_quantize_internal in llama.cpp
            # Some tensor types are always in float32
            if data_qtype is False and (
                any(
                    cls.match_model_tensor_name(new_name, key, bid)
                    for key in (
                        gguf.MODEL_TENSOR.FFN_GATE_INP,
                        gguf.MODEL_TENSOR.FFN_GATE_INP_SHEXP,
                        gguf.MODEL_TENSOR.POS_EMBD,
                        gguf.MODEL_TENSOR.TOKEN_TYPES,
                        gguf.MODEL_TENSOR.SSM_CONV1D,
                        gguf.MODEL_TENSOR.SHORTCONV_CONV,
                        gguf.MODEL_TENSOR.TIME_MIX_FIRST,
                        gguf.MODEL_TENSOR.TIME_MIX_W1,
                        gguf.MODEL_TENSOR.TIME_MIX_W2,
                        gguf.MODEL_TENSOR.TIME_MIX_DECAY_W1,
                        gguf.MODEL_TENSOR.TIME_MIX_DECAY_W2,
                        gguf.MODEL_TENSOR.TIME_MIX_LERP_FUSED,
                        gguf.MODEL_TENSOR.POSNET_NORM1,
                        gguf.MODEL_TENSOR.POSNET_NORM2,
                        gguf.MODEL_TENSOR.V_ENC_EMBD_POS,
                        gguf.MODEL_TENSOR.A_ENC_EMBD_POS,
                        gguf.MODEL_TENSOR.ALTUP_CORRECT_COEF,
                        gguf.MODEL_TENSOR.ALTUP_PREDICT_COEF,
                        # Kimi KDA conv weights should be F32
                        gguf.MODEL_TENSOR.SSM_CONV1D_Q,
                        gguf.MODEL_TENSOR.SSM_CONV1D_K,
                        gguf.MODEL_TENSOR.SSM_CONV1D_V,
                    )
                )
                or not new_name.endswith(".weight")
            ):
                data_qtype = gguf.GGMLQuantizationType.F32

            if data_qtype is False and any(
                cls.match_model_tensor_name(new_name, key, bid)
                for key in (
                    gguf.MODEL_TENSOR.TOKEN_EMBD,
                    gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD,
                    gguf.MODEL_TENSOR.OUTPUT,
                    gguf.MODEL_TENSOR.ALTUP_ROUTER,
                    gguf.MODEL_TENSOR.LAUREL_L,
                    gguf.MODEL_TENSOR.LAUREL_R,
                )
            ):
                if cls.ftype in (
                    gguf.LlamaFileType.MOSTLY_TQ1_0,
                    gguf.LlamaFileType.MOSTLY_TQ2_0,
                ):
                    # TODO: use Q4_K and Q6_K
                    data_qtype = gguf.GGMLQuantizationType.F16
                    # data_qtype = gguf.GGMLQuantizationType.Q8_0  # llama.cpp:llama_tensor_get_type

            if cls.model_arch == gguf.MODEL_ARCH.MMPROJ and cls.quant_nontext_module is False:
                data_qtype = gguf.GGMLQuantizationType.F32
                layer_config_names = hf_names
            else:
                # get name by new_name (for experts),
                layer_config_names = tuple(get_moe_name(cls, source_name, new_name) for source_name in hf_names)
                # get data_qtype by layer_config
                fallback_qtype = (
                    cls._gguf_dtype_selector.select_qtype(new_name, n_dims)
                    if isinstance(data_qtype, bool)
                    else data_qtype
                )
                layer_config_qtype = resolve_restored_qtype(
                    cls.layer_config,
                    layer_config_names,
                    checkpoint_name,
                    new_name,
                    fallback_qtype,
                    (
                        cls._gguf_name_resolution_diagnostics
                        if restored.transform_kind != "extra" and new_name.endswith(".weight") and n_dims > 1
                        else []
                    ),
                    allow_recipe_fallback=not getattr(cls, "is_auto_scheme", False),
                )
                # # No override (data_qtype is False), or wants to be quantized (data_qtype is True)
                if layer_config_qtype is not None:
                    data_qtype = layer_config_qtype
                elif isinstance(data_qtype, bool):
                    data_qtype = fallback_qtype
            clean_weight_list.extend(layer_config_names)

            if isinstance(data_qtype, bool):
                if cls.ftype == gguf.LlamaFileType.ALL_F32:
                    data_qtype = gguf.GGMLQuantizationType.F32
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_F16:
                    data_qtype = gguf.GGMLQuantizationType.F16
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                    data_qtype = gguf.GGMLQuantizationType.BF16
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                    data_qtype = gguf.GGMLQuantizationType.Q8_0
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q4_0:
                    data_qtype = gguf.GGMLQuantizationType.Q4_0
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q4_1:
                    data_qtype = gguf.GGMLQuantizationType.Q4_1
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q5_0:
                    data_qtype = gguf.GGMLQuantizationType.Q5_0
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q5_1:
                    data_qtype = gguf.GGMLQuantizationType.Q5_1
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q2_K_S or cls.ftype == gguf.LlamaFileType.MOSTLY_Q2_K:
                    data_qtype = gguf.GGMLQuantizationType.Q2_K
                elif (
                    cls.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_S
                    or cls.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_M
                    or cls.ftype == gguf.LlamaFileType.MOSTLY_Q3_K_L
                ):
                    data_qtype = gguf.GGMLQuantizationType.Q3_K
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q4_K_S or cls.ftype == gguf.LlamaFileType.MOSTLY_Q4_K_M:
                    data_qtype = gguf.GGMLQuantizationType.Q4_K
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q5_K_S or cls.ftype == gguf.LlamaFileType.MOSTLY_Q5_K_M:
                    data_qtype = gguf.GGMLQuantizationType.Q5_K
                elif cls.ftype == gguf.LlamaFileType.MOSTLY_Q6_K:
                    data_qtype = gguf.GGMLQuantizationType.Q6_K
                else:
                    raise ValueError(f"Unknown file type: {cls.ftype.name}")

            if data_qtype.name.endswith("_K") and data_torch.shape[-1] % 256 != 0:
                if data_qtype in [
                    gguf.GGMLQuantizationType.Q2_K,
                    gguf.GGMLQuantizationType.Q3_K,
                    gguf.GGMLQuantizationType.Q4_K,
                ]:
                    data_qtype = gguf.GGMLQuantizationType.Q5_0
                elif data_qtype == gguf.GGMLQuantizationType.Q5_K:
                    data_qtype = gguf.GGMLQuantizationType.Q5_0
                elif data_qtype == gguf.GGMLQuantizationType.Q6_K:
                    data_qtype = gguf.GGMLQuantizationType.Q8_0

            from auto_round.export.export_to_gguf.config import GGML_QUANT_SIZES

            if data_qtype.name.lower() in GGML_QUANT_SIZES:
                block_size, type_size = GGML_QUANT_SIZES[data_qtype.name.lower()]
                if data_torch.shape[-1] % block_size != 0:
                    logger.warning(
                        f"{new_name}: Can't quantize tensor with shape {data_torch.shape} to {data_qtype.name},"
                        " fallback to F16"
                    )
                    data_qtype = gguf.GGMLQuantizationType.F16

            if n_dims > 1 and data_torch.numel() >= 100_000_000:
                pre_quant_shape = f"{{{', '.join(str(n) for n in reversed(data_torch.shape))}}}"
                logger.info(
                    "Start quantizing large tensor %s: %s --> %s, shape = %s",
                    new_name,
                    old_dtype,
                    data_qtype.name,
                    pre_quant_shape,
                )

            if isinstance(data_qtype, bool) or data_qtype in [
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.BF16,
                gguf.GGMLQuantizationType.F32,
            ]:
                # squeeze is necessary for reloading in transformers.
                data = data_torch.squeeze().cpu().numpy()

                # if data ends up empty, it means data_torch was a scalar tensor -> restore
                if len(data_torch.shape) == 0:
                    data = data_torch.cpu().numpy()
                try:
                    data = gguf.quants.quantize(data, data_qtype)
                except gguf.QuantError as e:
                    logger.warning("%s, %s", e, "falling back to F16")
                    data_qtype = gguf.GGMLQuantizationType.F16
                    data = gguf.quants.quantize(data, data_qtype)
            else:
                # for deepseek v2
                if hf_name.endswith("kv_b_proj.weight") and cls.model_arch.name in ("DEEPSEEK2", "GLM_DSA"):
                    layer_name = hf_name[: -len(".weight")]
                    module = get_module(cls.model, layer_name)
                    n_head_kv = cls.hparams["num_key_value_heads"]
                    v_head_dim = cls.hparams["v_head_dim"]
                    qk_nope_head_dim = cls.hparams["qk_nope_head_dim"]

                    attr_list = {"scale": None, "zp": None, "d_scale": None, "wmin": None, "d_wmin": None}
                    for attr in attr_list:
                        if hasattr(module, attr) or hasattr(module, "w_" + attr):
                            if attr in ["scale", "zp"]:
                                attr_tensor = getattr(module, attr)
                            else:
                                attr_tensor = getattr(module, "w_" + attr)
                            if attr_tensor is None or not isinstance(attr_tensor, torch.Tensor):
                                continue
                            kv_b = attr_tensor.view(n_head_kv, v_head_dim + qk_nope_head_dim, -1)
                            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
                            k_b = k_b.transpose(1, 2)

                            name_kb = modify_name.replace("kv_b_proj", "k_b_proj")
                            if new_name == cls.map_tensor_name(name_kb):
                                attr_list[attr] = k_b
                            else:
                                attr_list[attr] = v_b
                    attr_list["imatrix"] = module.imatrix if hasattr(module, "imatrix") else None
                    data = _quant_data_with_args(data_torch, data_qtype, device=device, **attr_list)

                # for MOE model
                elif len(data_torch.shape) == 3 and len(re.findall(r"\d+", hf_name)) == 2:
                    new_data = []
                    for idx, arr in enumerate(data_torch):
                        arr_name = hf_name.split(".")
                        for i in range(len(arr_name) - 1, -1, -1):
                            if arr_name[i].isdecimal() and int(arr_name[i]) == (data_torch.shape[0] - 1):
                                arr_name[i] = str(idx)
                        arr_name = ".".join(arr_name)
                        arr, data_qtype = _quant_data(
                            cls, arr, data_qtype, arr_name, modify_name, new_name, bid, device=device
                        )
                        new_data.append(arr)
                    data = np.array(new_data)
                    del new_data
                else:
                    data, data_qtype = _quant_data(
                        cls,
                        data_torch,
                        data_qtype,
                        hf_name,
                        modify_name,
                        new_name,
                        bid,
                        device=device,
                        use_layer_attrs=len(hf_names) == 1,
                    )

            shape = gguf.quant_shape_from_byte_shape(data.shape, data_qtype) if data.dtype == np.uint8 else data.shape

            # reverse shape to make it similar to the internal ggml dimension order
            shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

            # n_dims is implicit in the shape
            logger.info(
                f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype}" f" --> {data_qtype.name}, shape = {shape_str}"
            )
            if not (hasattr(cls, "current_packing_block") and cls.current_packing_block is not None):
                clear_memory(device_list=[cls.device])

            cls.gguf_writer.add_tensor(new_name, data, raw_dtype=data_qtype)

        if restored_outputs_completed:
            cls._gguf_written_checkpoint_names.add(restored.checkpoint_name)
            cls._gguf_written_hf_names.update(restored.hf_names)

        # # save cpu memory, but slow
        if cls.low_cpu_mem_usage:
            for weight_name in clean_weight_list:
                if cls.model_arch == gguf.MODEL_ARCH.GEMMA:
                    continue
                module = get_module(cls.model, ".".join(weight_name.split(".")[:-1]))
                for key in ["scale", "zp", "d_scale", "wmin", "d_wmin", "imatrix"]:
                    if hasattr(module, key):
                        setattr(module, key, None)
                    if hasattr(module, "w_" + key):
                        setattr(module, "w_" + key, None)
                if cls.model_arch == gguf.MODEL_ARCH.LLAMA and "embed_tokens.weight" in weight_name:
                    continue
                clean_module_parameter(module, weight_name.split(".")[-1])
    if not (hasattr(cls, "current_packing_block") and cls.current_packing_block is not None):
        _flush_name_resolution_diagnostics(cls)
