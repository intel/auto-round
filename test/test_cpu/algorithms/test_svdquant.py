import types

import torch

from auto_round.algorithms.pipeline import QuantizationPipeline
from auto_round.algorithms.quantization.rtn.config import RTNConfig


def test_svdquant_config_is_pipeline_preprocessor():
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    pipeline = QuantizationPipeline.from_configs([SVDQuantConfig(rank=8), RTNConfig()])

    assert len(pipeline.preprocessors) == 1
    assert isinstance(pipeline.preprocessors[0], SVDQuantTransform)
    assert pipeline.block_quantizer.__class__.__name__ in {"RTNQuantizer", "OptimizedRTNQuantizer"}


def test_svdquant_linear_matches_manual_reference():
    from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear

    residual = torch.nn.Linear(3, 2, bias=True)
    lora_down = torch.nn.Linear(3, 1, bias=False)
    lora_up = torch.nn.Linear(1, 2, bias=False)
    smooth = torch.tensor([2.0, 0.5, 1.0])

    with torch.no_grad():
        residual.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]))
        residual.bias.copy_(torch.tensor([0.25, -0.5]))
        lora_down.weight.copy_(torch.tensor([[0.5, -1.0, 2.0]]))
        lora_up.weight.copy_(torch.tensor([[3.0], [-2.0]]))

    layer = SVDQuantLinear(residual, lora_down, lora_up, smooth)
    x = torch.tensor([[1.0, 2.0, -1.0], [0.0, -2.0, 3.0]])

    x_hat = x * smooth
    expected = residual(x_hat) + lora_up(lora_down(x_hat))

    torch.testing.assert_close(layer(x), expected)


def test_svdquant_transform_replaces_linear_with_residual_branch():
    from auto_round.algorithms.transforms.svdquant.apply import SVDQuantTransform
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
    from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear

    block = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=True))
    ctx = types.SimpleNamespace(block=block, block_name="model.layers.0")
    transform = SVDQuantTransform(SVDQuantConfig(rank=2, smooth_alpha=0.5, low_rank_dtype="float32"))

    transform.pre_quantize_block(ctx)

    assert isinstance(block[0], SVDQuantLinear)
    assert isinstance(block[0].residual_linear, torch.nn.Linear)
    assert block[0].residual_linear.weight.shape == (3, 4)
    assert block[0].lora_down.out_features == 2
    assert block[0].lora_up.in_features == 2
