# Copyright (c) 2023 Intel Corporation
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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import copy
import json
import os
from collections.abc import Callable

import torch
import torch.nn as nn
from tqdm import tqdm

from auto_round.export.export_to_awq.utils import (
    WQLinear_GEMM,
    awq_gemm_kernel_supported,
    awq_layer_user_forced,
)
from auto_round.export.utils import (
    filter_quantization_config,
    is_immediate_saving_mode,
    release_layer_safely,
    save_model,
    save_pretrained_artifact,
)
from auto_round.logger import logger
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    compress_layer_names,
    copy_python_files_from_model_cache,
    extract_block_names_to_str,
    get_block_names,
    get_module,
    set_module,
    unsupported_meta_device,
)
from auto_round.utils.model import get_layer_names_in_block


def _is_supported_layer(module: torch.nn.Module) -> bool:
    """Check if module is a supported quantizable layer type."""
    return type(module) in SUPPORTED_LAYER_TYPES or module.__class__.__name__ in INNER_SUPPORTED_LAYER_TYPES


def _collect_modules_to_not_convert(
    model: torch.nn.Module,
    layer_config: dict,
    regex_config: dict,
    to_quant_block_names: list | None = None,
) -> list:
    """Collect all module names that should not be converted (not quantized).

    Args:
        model: The model to scan
        layer_config: Configuration dict for layers
        regex_config: Regex-based configuration for layers
        to_quant_block_names: List of block names to quantize (for MLLM models)

    Returns:
        List of module names to not convert
    """
    modules_to_not_convert = set()
    all_blocks = get_block_names(model, quant_vision=True)
    # 1. add non-quantized block directly
    if to_quant_block_names:
        all_block_names = extract_block_names_to_str(all_blocks).split(",")
        to_quant_set = set(
            to_quant_block_names.split(",") if isinstance(to_quant_block_names, str) else to_quant_block_names
        )
        non_quant_blocks = set(all_block_names) - to_quant_set
        modules_to_not_convert.update(non_quant_blocks)

    layers_in_blocks = set(get_layer_names_in_block(model, quant_block_list=all_blocks))

    # 2. Collect non-quantized layers from layer_config
    layers_from_block_patterns = set()
    for layer_name, layer_cfg in layer_config.items():
        if not check_to_quantized(layer_cfg) and not any(name in layer_name for name in modules_to_not_convert):
            layers_from_block_patterns.add(layer_name)
    modules_to_not_convert.update(layers_from_block_patterns)

    # 3. Scan full model for supported layers not in layer_config and not in blocks
    for module_name, module in model.named_modules():
        if _is_supported_layer(module):
            # If this layer is not in layer_config, it wasn't quantized
            if module_name not in layer_config and module_name not in layers_in_blocks:
                # Standalone layer outside blocks
                modules_to_not_convert.add(module_name)

    # 4. Add high-precision layers from regex_config (bits > 8)
    for regex_name, regex_cfg in regex_config.items():
        bits = regex_cfg.get("bits")
        if bits and int(bits) > 8:
            modules_to_not_convert.add(regex_name)

    return list(modules_to_not_convert)


def pack_layer(name, model, backend, device=None):
    layer = get_module(model, name)

    if type(layer) not in SUPPORTED_LAYER_TYPES:  ##already packed
        return

    bits = getattr(layer, "bits", None)
    if bits is None or bits > 8:
        return

    group_size = getattr(layer, "group_size", None)
    in_features = getattr(layer, "in_features", None)
    out_features = getattr(layer, "out_features", None)
    if group_size is None or in_features is None or out_features is None:
        # Layers missing AWQ packing attributes (e.g. Conv1D) stay in full precision.
        logger.warning_once(f"skipping {name}: its shape cannot be served by the AWQ GEMM kernel")
        return
    if not awq_gemm_kernel_supported(in_features, out_features, bits, group_size):
        if awq_layer_user_forced(layer):
            # The user explicitly configured this layer for quantization;
            # honor it and pack anyway.
            logger.warning_once(
                f"{name}: quantized per explicit `layer_config` but its shape "
                "cannot be served by the AWQ GEMM kernel; packing as configured"
            )
        else:
            # The GEMM kernel cannot serve this shape; keep it in full precision.
            logger.warning_once(f"skipping {name}: its shape cannot be served by the AWQ GEMM kernel")
            return
    sym = layer.sym
    linear_layer = get_module(model, name)
    scale, zp = linear_layer.scale, linear_layer.zp
    scale = scale.t().contiguous()
    if isinstance(zp, torch.Tensor):
        zp = zp.t().contiguous().to(torch.float32)
        if sym:
            zp = int(zp.flatten()[0])
    q_linear = WQLinear_GEMM.from_linear(
        linear=linear_layer,
        w_bit=bits,
        group_size=group_size,
        init_only=False,
        scales=scale,
        zeros=zp,
        device=device,
    )
    set_module(model, name, q_linear)
    # Note: release weight and bias explicitly, in case they are referenced elsewhere
    release_layer_safely(layer)


def save_quantized_as_autoawq(
    output_dir: str,
    model: torch.nn.Module = None,
    tokenizer: Callable | None = None,
    layer_config: dict | None = None,
    inplace: bool = True,
    device: str | torch.device = "cpu",
    serialization_dict: dict | None = None,
    **kwargs,
) -> torch.nn.Module:
    """Export the model to autogptq format to easily leverage cuda kernel."""
    to_quant_block_names = serialization_dict.get("to_quant_block_names", None)
    processor = kwargs.get("processor", None)
    image_processor = kwargs.get("image_processor", None)

    immediate_saving = is_immediate_saving_mode(model, serialization_dict)
    if output_dir is not None and os.path.exists(output_dir) and not immediate_saving:
        logger.warning(f"{output_dir} already exists, this may cause model conflict")

    logger.info("Saving quantized model to auto_awq format")
    save_pretrained_artifact(tokenizer, output_dir, artifact_name="tokenizer")

    if processor is not None and output_dir is not None:
        processor.save_pretrained(output_dir)
    if image_processor is not None and output_dir is not None:
        image_processor.save_pretrained(output_dir)

    if not unsupported_meta_device(model):
        if inplace:
            compressed_model = model.to("cpu")
        else:
            compressed_model = copy.deepcopy(model.to("cpu"))
    else:
        compressed_model = model

    names = list(layer_config.keys())

    # Layers the AWQ GEMM kernel cannot serve (e.g. out_features not divisible by
    # group_size, as in Gated-DeltaNet in_proj_ba) must stay in full precision;
    # packing them crashes serving stacks such as vLLM. They are reported via
    # modules_to_not_convert below. Layers the user explicitly configured for
    # quantization in layer_config are exempt: their setting is honored and they
    # are packed as requested. This also covers the split
    # quantize() -> save_quantized(format="auto_awq") flow, where format
    # resolution (and the fp16 downgrade in check_and_reset_format) happens only
    # after the layer has already been quantized.
    unservable_layers = set()
    meta_device = unsupported_meta_device(model)
    if not meta_device:
        for name in names:
            layer = get_module(compressed_model, name)
            if (
                type(layer) not in SUPPORTED_LAYER_TYPES
                or not hasattr(layer, "in_features")
                or not hasattr(layer, "out_features")
            ):
                continue
            cfg = layer_config.get(name, {})
            cfg_get = cfg.get if hasattr(cfg, "get") else (lambda *a, **k: None)
            bits = getattr(layer, "bits", None)
            if bits is None:
                bits = cfg_get("bits") or serialization_dict.get("bits")
            if not isinstance(bits, int) or bits > 8:
                continue
            group_size = getattr(layer, "group_size", None)
            if group_size is None:
                group_size = cfg_get("group_size") or serialization_dict.get("group_size")
            if not awq_gemm_kernel_supported(layer.in_features, layer.out_features, bits, group_size):
                if not awq_layer_user_forced(layer, cfg):
                    unservable_layers.add(name)
        if unservable_layers:
            logger.warning_once(
                f"{len(unservable_layers)} layer(s) cannot be served by the AWQ GEMM kernel "
                "(in/out features not divisible by group_size or out_features not divisible by 64) "
                "and are kept in full precision: "
                f"{compress_layer_names(sorted(unservable_layers))}"
            )

    backend = None
    if not meta_device:
        for name in tqdm(names, desc="packing", leave=True):
            if name in unservable_layers:
                continue
            pack_layer(name, compressed_model, backend, device)
    if output_dir is None:
        return model

    quantization_config = serialization_dict
    regex_config = quantization_config.pop("regex_config", {})  # awq do not support mixed bits config saving

    if output_dir is None:
        return compressed_model

    # Collect all modules that should not be converted (not quantized)
    modules_to_not_convert = _collect_modules_to_not_convert(
        compressed_model,
        layer_config,
        regex_config,
        to_quant_block_names=to_quant_block_names,
    )

    quantization_config["provider"] = "auto-round"
    quantization_config["quant_method"] = "awq"
    quantization_config["zero_point"] = not quantization_config["sym"]
    quantization_config["version"] = "gemm"
    if unservable_layers:
        modules_to_not_convert = sorted(set(modules_to_not_convert) | unservable_layers)
    quantization_config["modules_to_not_convert"] = modules_to_not_convert
    ##check module quantized in block, this may have bug for mixed precision quantization
    filter_quantization_config(quantization_config)
    if hasattr(compressed_model, "config"):
        compressed_model.config.quantization_config = quantization_config
    safe_serialization = kwargs.get("safe_serialization", True)
    # Force torch_dtype in the saved config to fp16 for vLLM CUDA kernel compatibility;
    # this does not change in-memory model weights.
    dtype = torch.float16
    save_model(
        compressed_model,
        output_dir,
        safe_serialization=safe_serialization,
        dtype=dtype,
        immediate_saving=immediate_saving,
    )

    return compressed_model
