"""Regression tests for FP8 overrides in an INT AutoRound checkpoint."""

from unittest.mock import patch

import pytest
import torch

from auto_round.export.export_to_autoround.export import pack_layer
from auto_round.export.export_to_autoround.export_to_fp8 import FP8BlockQLinear


@pytest.mark.parametrize("data_type", ["fp", "block_fp8", "fp8"])
def test_int_backend_packs_fp8_block_override(data_type):
    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
    model.dtype = torch.float32
    layer = model[0]
    layer.bits = 8
    layer.data_type = data_type
    layer.group_size = (128, 128)
    layer.act_bits = 8
    layer.sym = True
    layer.scale = torch.ones(1, 1)
    layer.zp = None
    with torch.no_grad():
        layer.weight.fill_(1)

    with patch("auto_round.export.export_to_autoround.export.pack_qact_layer") as int_packer:
        pack_layer("0", model, "auto_round:auto_gptq", device="cpu")

    int_packer.assert_not_called()
    assert isinstance(model[0], FP8BlockQLinear)
    assert model[0].weight.dtype == torch.float8_e4m3fn
    assert model[0].weight_scale_inv.shape == (1, 1)
    torch.testing.assert_close(model[0].weight.float(), torch.ones(128, 128))


def test_integer_activation_packing_is_unchanged():
    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
    model[0].data_type = "int"
    model[0].bits = 8
    model[0].act_bits = 8
    with patch("auto_round.export.export_to_autoround.export.pack_qact_layer") as int_packer:
        pack_layer("0", model, "auto_round:auto_gptq", device="cpu")
    int_packer.assert_called_once_with("0", model)