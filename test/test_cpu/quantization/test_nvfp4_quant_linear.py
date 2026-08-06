from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers.quantizers.auto import AutoHfQuantizer

from auto_round.data_type.nvfp import calculate_gparam, fp4_v2
from auto_round.data_type.utils import get_quant_func
from auto_round.experimental import qmodules as ar_qmodules
from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear as _FPLinear
from auto_round.export.export_to_llmcompressor.config import initialize_nvfp4_e5m3_quantization
from auto_round.export.formats import BackendDataType
from auto_round.schemes import PRESET_SCHEMES

QMODULE_MAPPING = {
    BackendDataType.NVFP4.value: ar_qmodules.NVFP4QuantLinear,
}


@pytest.fixture(autouse=True)
def fixed_seed():
    """Ensure deterministic RNG for every test."""
    seed = 42
    print("\nSetting fixed random seed for test:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield
    # (Optional) cleanup or reset after test


def test_calculate_gparam_with_float8_input():
    tensor = torch.tensor([[-2.0, 0.0, 1.0]], dtype=torch.float32).to(torch.float8_e4m3fn)

    global_scale = calculate_gparam(tensor)

    assert global_scale.dtype == torch.float32
    assert torch.isfinite(global_scale)


def test_nvfp4_e5m3_compressed_tensors_loading_uses_no_global_scales():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="tiny")
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.ModuleDict({"fc": torch.nn.Linear(16, 8)})])

    quantizer = AutoHfQuantizer.from_config(initialize_nvfp4_e5m3_quantization([]))
    model = quantizer._process_model_before_weight_loading(TinyModel())
    layer = model.model.layers[0]["fc"]

    assert isinstance(layer, ar_qmodules.NVFP4E5M3QuantLinear)
    assert set(layer.state_dict()) == {"bias", "weight_packed", "weight_scale"}
    assert quantizer._process_model_after_weight_loading(model) is model


def test_nvfp4_e5m3_qdq_input_uses_reference_fallback_on_cpu():
    config = PRESET_SCHEMES["NVFP4_E5M3"]
    layer = ar_qmodules.NVFP4E5M3QuantLinear(16, 8, config, dtype=torch.float32)
    activation = torch.randn(2, 3, 16)

    expected, _, _ = fp4_v2(activation, bits=config.act_bits, group_size=config.act_group_size)

    assert torch.equal(layer.qdq_input(activation), expected)


def test_nvfp4_e5m3_forward_does_not_cache_dequantized_weight_by_default():
    config = PRESET_SCHEMES["NVFP4_E5M3"]
    layer = ar_qmodules.NVFP4E5M3QuantLinear(16, 8, config, dtype=torch.float32)
    activation = torch.randn(2, 16)
    dequantized_weight = torch.randn(8, 16)

    with patch.object(layer, "dequant_weight_online", return_value=dequantized_weight) as dequant_weight_online:
        layer(activation)
        layer(activation)

    assert layer._cached_weight is None
    assert dequant_weight_online.call_count == 2


def test_nvfp4_e5m3_forward_caches_dequantized_weight_when_enabled():
    config = PRESET_SCHEMES["NVFP4_E5M3"]
    layer = ar_qmodules.NVFP4E5M3QuantLinear(16, 8, config, dtype=torch.float32, cache_weight=True)
    activation = torch.randn(2, 16)
    dequantized_weight = torch.randn(8, 16)

    with patch.object(layer, "dequant_weight_online", return_value=dequantized_weight) as dequant_weight_online:
        layer(activation)
        layer(activation)

    dequant_weight_online.assert_called_once_with()
    assert layer.weight_packed is None
    assert layer.weight_scale is None


def test_nvfp4_e5m3_cannot_clear_released_quantized_weight_buffers():
    config = PRESET_SCHEMES["NVFP4_E5M3"]
    layer = ar_qmodules.NVFP4E5M3QuantLinear(16, 8, config, dtype=torch.float32, cache_weight=True)

    layer(torch.randn(2, 16))

    with pytest.raises(RuntimeError, match="quantized weight buffers have been released"):
        layer.clear_weight_cache()


def test_cute_nvfp4_e5m3_does_not_cache_dequantized_weight_by_default(monkeypatch):
    monkeypatch.delenv("AR_NVFP4_E5M3_CACHE_HP_WEIGHT", raising=False)
    config = PRESET_SCHEMES["NVFP4_E5M3"]
    layer = ar_qmodules.CuteNVFP4E5M3QuantLinear(16, 8, config, dtype=torch.float32)
    activation = torch.randn(2, 16)
    dequantized_weight = torch.randn(8, 16)

    with patch.object(layer, "dequant_weight_online", return_value=dequantized_weight) as dequant_weight_online:
        layer(activation)
        layer(activation)

    assert not layer.cache_weight
    assert dequant_weight_online.call_count == 2


def test_nvfp4_e5m3_cache_weight_environment_override(monkeypatch):
    config = PRESET_SCHEMES["NVFP4_E5M3"]
    monkeypatch.setenv("AR_NVFP4_E5M3_CACHE_HP_WEIGHT", "1")
    assert ar_qmodules.NVFP4E5M3QuantLinear(16, 8, config, dtype=torch.float32).cache_weight
    assert ar_qmodules.CuteNVFP4E5M3QuantLinear(16, 8, config, dtype=torch.float32).cache_weight
    monkeypatch.setenv("AR_NVFP4_E5M3_CACHE_HP_WEIGHT", "0")
    assert not ar_qmodules.CuteNVFP4E5M3QuantLinear(16, 8, config, dtype=torch.float32).cache_weight


def test_nvfp4_e5m3_torch_forward_does_not_call_cute():
    config = PRESET_SCHEMES["NVFP4_E5M3"]
    layer = ar_qmodules.NVFP4E5M3QuantLinear(16, 8, config, dtype=torch.float32)
    activation = torch.randn(2, 16)

    with patch(
        "auto_round.experimental.qmodules.nvfp4_e5m3.try_cute_nvfp4_e5m3_linear",
    ) as cute_linear:
        output = layer(activation)

    assert output.shape == (2, 8)
    cute_linear.assert_not_called()


def test_nvfp4_e5m3_cute_forward_uses_fused_output_when_available():
    config = PRESET_SCHEMES["NVFP4_E5M3"]
    layer = ar_qmodules.CuteNVFP4E5M3QuantLinear(16, 8, config, dtype=torch.float32)
    activation = torch.randn(2, 16)
    fused_output = torch.randn(2, 8)

    with patch("auto_round.experimental.qmodules.nvfp4_e5m3.try_cute_nvfp4_e5m3_linear", return_value=fused_output):
        assert layer(activation) is fused_output


def test_cute_nvfp4_e5m3_uses_cached_weight_when_enabled():
    config = PRESET_SCHEMES["NVFP4_E5M3"]
    layer = ar_qmodules.CuteNVFP4E5M3QuantLinear(16, 8, config, dtype=torch.float32, cache_weight=True)
    activation = torch.randn(2, 16)
    dequantized_weight = torch.randn(8, 16)

    with (
        patch.object(layer, "dequant_weight_online", return_value=dequantized_weight) as dequant_weight_online,
        patch("auto_round.experimental.qmodules.nvfp4_e5m3.try_cute_nvfp4_e5m3_linear") as cute_linear,
    ):
        layer(activation)
        layer(activation)

    dequant_weight_online.assert_called_once_with()
    cute_linear.assert_not_called()


@pytest.mark.parametrize("scheme", [BackendDataType.NVFP4.value])
@torch.inference_mode()
def test_nvfp4_quantlinear_from_original_and_forward(scheme):
    """
    Test NVFP4 quantization schemes by creating quantized layers
    from an original torch.nn.Linear layer and validating their forward pass.
    """

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define layer dimensions
    in_features = 64
    out_features = 512

    # Create an original torch.nn.Linear layer
    original_layer = torch.nn.Linear(in_features, out_features, bias=False)

    # Select the quantization scheme
    config = PRESET_SCHEMES[scheme.upper()]

    # Define weight scale shape
    weight_scale_shape = (out_features, in_features // config.group_size)

    # Quantize the weights using the quantization function
    weight_qdq_func, _ = get_quant_func(dtype=config.data_type, bits=config.bits, sym=True)

    weight_global_scale = calculate_gparam(original_layer.weight, config.group_size)
    weight_qdq, weight_scale, _ = weight_qdq_func(
        original_layer.weight, bits=config.bits, group_size=config.group_size, global_scale=weight_global_scale
    )

    # Generate a random input tensor
    input_tensor = torch.randn((4, in_features), dtype=torch.float32)
    input_global_scale = calculate_gparam(input_tensor, config.act_group_size)

    weight_scale = weight_scale.reshape(weight_scale_shape)

    # Pack the weights using the QuantLinear class
    kwargs = {"act_bits": config.act_bits}
    nvfp4_lin = _FPLinear(
        bits=config.bits,
        group_size=config.group_size,
        infeatures=in_features,
        outfeatures=out_features,
        bias=original_layer.bias is not None,
        data_type=config.data_type,
        **kwargs,
    )

    nvfp4_lin.pack(
        linear=original_layer,
        scales=weight_scale,
        global_scale=weight_global_scale,
        input_global_scale=input_global_scale,
    )

    # Create an NVFP4QuantLinear layer from the original layer
    QuantLinearClass = QMODULE_MAPPING[scheme]
    nvfp4_layer = QuantLinearClass.from_original(
        config=config,
        original_layer=original_layer,
    )

    # Copy the packed weights and scales to the quantized layer
    packed_weight = nvfp4_lin.weight_packed if config.bits == 4 else nvfp4_lin.weight
    nvfp4_layer.weight_packed.data.copy_(packed_weight)
    nvfp4_layer.weight_scale.data.copy_(nvfp4_lin.weight_scale)
    nvfp4_layer.weight_global_scale.data.copy_(nvfp4_lin.weight_global_scale)
    nvfp4_layer.input_global_scale.data.copy_(nvfp4_lin.input_global_scale)

    # Validate layer attributes
    assert nvfp4_layer.in_features == original_layer.in_features
    assert nvfp4_layer.out_features == original_layer.out_features

    # Perform a forward pass with both layers
    original_output = original_layer(input_tensor)
    nvfp4_output = nvfp4_layer(input_tensor)

    # Compute the difference between the outputs
    diff = nvfp4_output - original_output
    # Note: Remove NaN values, as we might get NaN when casting scales to FP8

    diff = diff[~torch.isnan(diff)]
    diff_amax = diff.abs().max()

    # Print the maximum difference for debugging
    print(f"Scheme: {scheme}, Max Difference: {diff_amax}")

    # Assert that the outputs are close within a tolerance
    assert diff_amax < 5e-1, f"Outputs differ too much for scheme {scheme}!"
