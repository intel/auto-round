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

from auto_round.experimental.qmodules.base import QModuleBase


def _export_hadamard_transform_config(model, data_type):
    rotation_config = getattr(model, "_rotation_config", None)
    if rotation_config is None:
        return None

    algorithm = getattr(rotation_config, "algorithm", None)
    if algorithm is not None and algorithm != "hadamard":
        return None
    if isinstance(rotation_config, dict) and "hadamard_type" not in rotation_config:
        return None

    from auto_round.algorithms.transforms.hadamard.config import (
        RotationConfig,
        normalize_rotation_config,
    )
    from auto_round.algorithms.transforms.hadamard.dispatcher import (
        resolve_hadamard_backend,
    )

    rotation_config = RotationConfig.model_validate(normalize_rotation_config(rotation_config, data_type))
    if rotation_config.hadamard_type != "hadamard":
        raise NotImplementedError("llm_compressor Hadamard export currently supports only " "hadamard_type='hadamard'")
    if resolve_hadamard_backend(rotation_config, data_type) != "transform":
        raise NotImplementedError("llm_compressor Hadamard export requires the per-Linear " "transform backend")

    from compressed_tensors.base import TRANSFORM_CONFIG_NAME
    from compressed_tensors.offload import register_offload_module
    from compressed_tensors.transform import (
        TransformArgs,
        TransformConfig,
        TransformScheme,
    )
    from compressed_tensors.transform.apply import (
        _register_tied_transform_weights,
    )
    from compressed_tensors.transform.factory.base import TransformFactory

    online_args = TransformArgs(
        targets=["Linear"],
        location="input",
        ignore=["lm_head"],
    )
    offline_args = TransformArgs(
        targets=["Linear"],
        location="weight_input",
        inverse=True,
        ignore=["lm_head"],
    )
    scheme = TransformScheme(
        type="hadamard",
        apply=[online_args, offline_args],
        head_dim=rotation_config.block_size,
        precision=torch.float64,
    )
    transform_config = TransformConfig(config_groups={"v": scheme})

    factory = TransformFactory.from_scheme(scheme, name="v")
    transform_name = f"v_{online_args.location}"
    target_types = (torch.nn.Linear, QModuleBase)
    for name, module in model.named_modules():
        if "lm_head" in name:
            continue
        if (
            not isinstance(module, target_types)
            and not hasattr(module, "weight_packed")
            and not hasattr(module, "quantization_scheme")
        ):
            continue
        if transform_name in module._modules:
            continue
        transform = factory.create_transform(module, online_args)
        register_offload_module(module, transform_name, transform)

    _register_tied_transform_weights(model)
    setattr(model, TRANSFORM_CONFIG_NAME, transform_config)
    return transform_config.model_dump()


def _save_immediate_hadamard_weights(model, output_dir):
    transform_config = getattr(model, "transform_config", None)
    if transform_config is None:
        return

    matrix_name = next(
        (f"{name}.weight" for name, _ in model.named_modules() if name.endswith(".v_input")),
        None,
    )
    if matrix_name is None:
        raise ValueError("Hadamard transform_config is present but no online transform module was registered")

    from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix
    from safetensors import safe_open
    from safetensors.torch import save_file

    block_size = transform_config.config_groups["v"].head_dim
    matrix = deterministic_hadamard_matrix(block_size, torch.float64, "cpu").contiguous()
    transform_path = os.path.join(output_dir, "transform.safetensors")
    save_file({matrix_name: matrix}, transform_path)
    transform_size = os.path.getsize(transform_path)

    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        total_size = index.get("metadata", {}).get("total_size", 0)
    else:
        index = {}
        weight_map = {}
        total_size = 0
        for filename in sorted(os.listdir(output_dir)):
            if not filename.endswith(".safetensors") or filename == "transform.safetensors":
                continue
            path = os.path.join(output_dir, filename)
            total_size += os.path.getsize(path)
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weight_map[key] = filename

    if matrix_name not in weight_map:
        total_size += transform_size
    weight_map[matrix_name] = os.path.basename(transform_path)
    index["metadata"] = {**index.get("metadata", {}), "total_size": total_size}
    index["weight_map"] = weight_map
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
