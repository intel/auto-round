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

"""Coverage for the Omni-specific tensor layout, independent of model architecture."""

import json
from dataclasses import replace

import torch
from safetensors.torch import load_file

from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
from auto_round.data_type.nvfp import nv_fp4
from auto_round.export.svdquant_nunchaku import IdentitySVDQuantModelAdapter
from auto_round.export.svdquant_omni import save_svdquant_omni


def test_omni_tensor_layout_and_adapter_mapping(tmp_path):
    model = torch.nn.Module()
    model.config = {"_name_or_path": tmp_path}
    residual = torch.nn.Linear(32, 16, dtype=torch.bfloat16)
    down = torch.nn.Linear(32, 8, bias=False, dtype=torch.bfloat16)
    up = torch.nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
    model.projection = SVDQuantLinear(residual, down, up, torch.full((32,), 2, dtype=torch.bfloat16))
    model.unquantized = torch.nn.Linear(32, 16)

    class RenameAdapter(IdentitySVDQuantModelAdapter):
        def map_modules(self, model, records):
            return [replace(record, prefix="runtime.projection") for record in super().map_modules(model, records)]

    for adapter, prefix in ((None, "projection"), (RenameAdapter(), "runtime.projection")):
        if adapter is not None:
            # Also exercise export of tuned residuals through the same ABI.
            residual.data_type, residual.bits, residual.group_size = "nv_fp4", 4, 16
            residual.weight_global_scale = torch.tensor([4.0])
            quantized, residual.scale, _ = nv_fp4(residual.weight.float(), global_scale=residual.weight_global_scale)
            residual.weight.data = quantized.bfloat16()
        output = tmp_path / prefix
        tensors = load_file(save_svdquant_omni(model, output, adapter=adapter))
        assert set(tensors) == {
            *(
                f"{prefix}.{suffix}"
                for suffix in (
                    "qweight",
                    "wscales",
                    "wtscale",
                    "wcscales",
                    "proj_down",
                    "proj_up",
                    "smooth_factor",
                    "bias",
                )
            ),
            "unquantized.weight",
            "unquantized.bias",
        }
        assert tensors[f"{prefix}.qweight"].shape == (16, 16)
        assert tensors[f"{prefix}.qweight"].dtype == torch.int8
        assert tensors[f"{prefix}.wscales"].shape == (2, 16)
        assert tensors[f"{prefix}.wscales"].dtype == torch.float8_e4m3fn
        assert tensors[f"{prefix}.wtscale"].shape == (1,)
        assert tensors[f"{prefix}.wtscale"].dtype == torch.bfloat16
        if adapter is not None:
            torch.testing.assert_close(tensors[f"{prefix}.wtscale"], torch.tensor([0.25], dtype=torch.bfloat16))
            torch.testing.assert_close(tensors[f"{prefix}.wscales"].float(), residual.scale.reshape(16, 2).T)
        torch.testing.assert_close(tensors[f"{prefix}.wcscales"], torch.ones(16, dtype=torch.bfloat16))
        torch.testing.assert_close(tensors[f"{prefix}.proj_down"], (down.weight.float() * 2).T.bfloat16())
        torch.testing.assert_close(tensors[f"{prefix}.proj_up"], up.weight)
        torch.testing.assert_close(tensors[f"{prefix}.smooth_factor"], torch.full((32,), 0.5, dtype=torch.bfloat16))
        torch.testing.assert_close(tensors[f"{prefix}.bias"], residual.bias)
        torch.testing.assert_close(tensors["unquantized.weight"], model.unquantized.weight)
        config = json.loads((output / "config.json").read_text())
        quantization = json.loads((output / "quantization_config.json").read_text())
        assert config["quantization_config"] == quantization
        assert quantization == {
            "quant_method": "svdquant",
            "precision": "nvfp4",
            "rank": 8,
            "act_unsigned": False,
            "modules_to_not_convert": ["unquantized"],
        }
