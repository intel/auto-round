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

"""Quantize both Wan2.2 transformers to AutoRound SVDQuant MXFP4 for Nunchaku."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import torch
from diffusers import WanTransformer3DModel

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))

from auto_round.algorithms.transforms.svdquant import SVDQuantConfig
from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
from auto_round.algorithms.transforms.svdquant.smooth_adapters.base import SmoothSearchGroup
from auto_round.export.svdquant_adapters.wan import WAN_SVDQUANT_TARGET_MODULES, WanSVDQuantNunchakuAdapter
from auto_round.export.svdquant_nunchaku import SVDQuantExportConfig, save_svdquant_nunchaku_safetensors


def _set_mxfp4_scheme(module: torch.nn.Linear) -> None:
    module.data_type = "mx_fp"
    module.bits = 4
    module.group_size = 32
    module.sym = True
    module.act_data_type = "mx_fp"
    module.act_bits = 4
    module.act_group_size = 32
    module.act_sym = True
    module.act_dynamic = True


def _get_child(module: torch.nn.Module, path: str) -> torch.nn.Module:
    child = module
    for part in path.split("."):
        child = child[int(part)] if part.isdigit() else getattr(child, part)
    return child


def _set_child(module: torch.nn.Module, path: str, value: torch.nn.Module) -> None:
    parts = path.split(".")
    parent = module
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def _resolve_devices(value: str) -> tuple[torch.device, ...]:
    if value.strip().lower() == "auto":
        if not torch.cuda.is_available():
            return (torch.device("cpu"),)
        return tuple(torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count()))
    devices = tuple(torch.device(item.strip()) for item in value.split(",") if item.strip())
    if not devices:
        raise ValueError("devices must contain at least one device")
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA decomposition requested but CUDA is unavailable")
    return devices


@torch.inference_mode()
def _decompose_projection(
    linear: torch.nn.Linear,
    path: str,
    rank: int,
    residual_iters: int,
    device: torch.device,
    device_lock: Lock,
) -> torch.nn.Module:
    with device_lock:
        _set_mxfp4_scheme(linear)
        linear.to(device)
        transform = SVDQuantTransform(
            SVDQuantConfig(
                rank=rank,
                smooth_enabled=False,
                residual_iters=residual_iters,
                target_modules=[path],
                low_rank_dtype="bf16",
            )
        )
        try:
            group = SmoothSearchGroup(
                key=path,
                projection_names=(path,),
                projections=(linear,),
                projection_input_key=path,
                projection_input_module=linear,
                evaluation_input_key=path,
                evaluation_module=linear,
            )
            (wrapper,) = transform._decompose_group(group)
            return wrapper.to("cpu")
        finally:
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()


@torch.inference_mode()
def _decompose_blocks(
    model: WanTransformer3DModel,
    rank: int,
    devices: tuple[torch.device, ...],
    residual_iters: int,
) -> None:
    workers = min(len(devices), len(WAN_SVDQUANT_TARGET_MODULES))
    device_locks = tuple(Lock() for _ in devices)
    for index, block in enumerate(model.blocks, start=1):
        started = time.time()
        projections = []
        for path in WAN_SVDQUANT_TARGET_MODULES:
            linear = _get_child(block, path)
            if not isinstance(linear, torch.nn.Linear):
                raise TypeError(f"Wan projection {path!r} is not Linear: {type(linear).__name__}")
            projections.append(linear)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _decompose_projection,
                    linear,
                    path,
                    rank,
                    residual_iters,
                    devices[offset % len(devices)],
                    device_locks[offset % len(devices)],
                )
                for offset, (path, linear) in enumerate(zip(WAN_SVDQUANT_TARGET_MODULES, projections, strict=True))
            ]
            replacements = [future.result() for future in futures]
        for path, replacement in zip(WAN_SVDQUANT_TARGET_MODULES, replacements, strict=True):
            _set_child(block, path, replacement)
        device_names = ",".join(str(device) for device in devices)
        print(
            f"decomposed Wan block {index}/{len(model.blocks)} across {device_names} "
            f"in {time.time() - started:.2f}s",
            flush=True,
        )


def _prepare_pipeline(source: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in {"transformer", "transformer_2"}:
            continue
        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination, copy_function=os.link, dirs_exist_ok=True)
        elif item.name == "model_index.json":
            shutil.copy2(item, destination)
        else:
            os.link(item, destination)


def _quantize_component(
    source: Path,
    output: Path,
    *,
    rank: int,
    residual_iters: int,
    devices: tuple[torch.device, ...],
) -> None:
    print(f"loading {source}", flush=True)
    model = WanTransformer3DModel.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    _decompose_blocks(model, rank, devices, residual_iters)
    output.mkdir(parents=True, exist_ok=True)
    model.save_config(output)
    temporary = output / ".diffusion_pytorch_model.tmp.safetensors"
    adapter = WanSVDQuantNunchakuAdapter(config=dict(model.config), require_complete_model=True)
    try:
        save_svdquant_nunchaku_safetensors(
            model,
            os.fspath(temporary),
            config=SVDQuantExportConfig(runtime_loadable=True),
            adapter=adapter,
        )
        os.replace(temporary, output / "diffusion_pytorch_model.safetensors")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    del model
    gc.collect()
    for device in devices:
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Wan2.2 Diffusers pipeline directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output Diffusers/Nunchaku pipeline directory.")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--residual-iters", type=int, default=1)
    parser.add_argument(
        "--devices",
        default="auto",
        help="Comma-separated decomposition devices, or 'auto' to use every visible CUDA GPU.",
    )
    args = parser.parse_args()

    devices = _resolve_devices(args.devices)
    if args.rank <= 0 or args.rank % 16:
        raise ValueError("rank must be a positive multiple of 16 for Nunchaku")

    _prepare_pipeline(args.model, args.output)
    for component in ("transformer", "transformer_2"):
        _quantize_component(
            args.model / component,
            args.output / component,
            rank=args.rank,
            residual_iters=args.residual_iters,
            devices=devices,
        )

    model_index_path = args.output / "model_index.json"
    with model_index_path.open(encoding="utf-8") as handle:
        model_index = json.load(handle)
    for component in ("transformer", "transformer_2"):
        model_index[component] = ["nunchaku", "NunchakuWanTransformer3DModel"]
    with model_index_path.open("w", encoding="utf-8") as handle:
        json.dump(model_index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"saved Nunchaku Wan2.2 pipeline to {args.output}", flush=True)


if __name__ == "__main__":
    main()