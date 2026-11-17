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

"""Residual overrides must obey each runtime's rules without mutating user input."""

import copy
from types import SimpleNamespace

import pytest
import torch

from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
from auto_round.formats import get_formats
from auto_round.schemes import PRESET_SCHEMES


@pytest.mark.parametrize(
    "format_name,preset",
    [("svdquant_nunchaku", "MXFP4"), ("svdquant_omni", "MXFP4"), ("svdquant_omni", "NVFP4")],
)
@pytest.mark.parametrize("layer_name", ["projection", "projection.residual_linear"])
@pytest.mark.parametrize("override_kind", ["preset", "scheme", "dict", "partial"])
def test_residual_override_preserves_input_and_runtime_validation(format_name, preset, layer_name, override_kind):
    model = torch.nn.Module()
    model.projection = SVDQuantLinear(
        torch.nn.Linear(32, 16),
        torch.nn.Linear(32, 8, bias=False),
        torch.nn.Linear(8, 16, bias=False),
        torch.ones(32),
    )
    scheme = PRESET_SCHEMES[preset].copy()
    override = {
        "preset": preset,
        "scheme": scheme,
        "dict": {"scheme": preset, "act_dynamic": True},
        "partial": {"act_dynamic": True},
    }[override_kind]
    layer_config = {layer_name: override}
    original = copy.deepcopy(layer_config)
    output_format = get_formats(format_name, SimpleNamespace(scheme=preset))[0]
    ctx = SimpleNamespace(model=model, layer_config=layer_config, quant_block_list=[["blocks.0"]])

    result = output_format.check_and_reset_format(scheme, ctx)

    assert result == (None, scheme, layer_config, ctx.quant_block_list)
    assert layer_config == original
    # A per-layer override must never turn a residual into an unsupported W4A16 layer.
    ctx.layer_config = {layer_name: {"act_bits": 16}}
    with pytest.raises(ValueError, match="incompatible residual scheme"):
        output_format.check_and_reset_format(scheme, ctx)
