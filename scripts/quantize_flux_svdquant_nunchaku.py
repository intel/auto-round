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

"""Quantize a Diffusers FLUX transformer and export a Nunchaku MXFP4 onefile."""

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import FluxTransformer2DModel

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))

from auto_round import SVDQuantConfig
from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
from auto_round.export.svdquant_adapters.flux import (
    FLUX_SVDQUANT_TARGET_MODULES,
    FluxSVDQuantNunchakuAdapter,
)
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


def _limit_blocks(model: FluxTransformer2DModel, double_blocks: int, single_blocks: int) -> None:
    if double_blocks >= 0:
        model.transformer_blocks = model.transformer_blocks[:double_blocks]
    if single_blocks >= 0:
        model.single_transformer_blocks = model.single_transformer_blocks[:single_blocks]


@torch.inference_mode()
def _decompose_blocks(model: FluxTransformer2DModel, rank: int, device: torch.device) -> None:
    transform = SVDQuantTransform(
        SVDQuantConfig(
            rank=rank,
            smooth_enabled=False,
            residual_iters=1,
            target_modules=list(FLUX_SVDQUANT_TARGET_MODULES),
            low_rank_dtype="bf16",
        )
    )
    blocks = [*model.transformer_blocks, *model.single_transformer_blocks]
    for index, block in enumerate(blocks, start=1):
        for module in block.modules():
            if isinstance(module, torch.nn.Linear):
                _set_mxfp4_scheme(module)
        block.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.time()
        transform.pre_quantize_block(SimpleNamespace(block=block))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        block.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"decomposed block {index}/{len(blocks)} in {time.time() - started:.2f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Diffusers FLUX transformer directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output Nunchaku onefile safetensors.")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--device", default="cuda:0", help="Device used for SVD decomposition and fusion.")
    parser.add_argument("--double-blocks", type=int, default=-1, help="Use -1 for all double-stream blocks.")
    parser.add_argument("--single-blocks", type=int, default=-1, help="Use -1 for all single-stream blocks.")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA decomposition requested but CUDA is unavailable")

    print(f"loading {args.model}", flush=True)
    model = FluxTransformer2DModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    _limit_blocks(model, args.double_blocks, args.single_blocks)
    model_config = dict(model.config)
    model_config["num_layers"] = len(model.transformer_blocks)
    model_config["num_single_layers"] = len(model.single_transformer_blocks)

    _decompose_blocks(model, args.rank, device)

    adapter = FluxSVDQuantNunchakuAdapter(
        config=model_config,
        decomposition_device=device,
        require_complete_model=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.safetensors")
    try:
        save_svdquant_nunchaku_safetensors(
            model,
            os.fspath(temporary),
            config=SVDQuantExportConfig(runtime_loadable=True),
            adapter=adapter,
        )
        os.replace(temporary, args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
