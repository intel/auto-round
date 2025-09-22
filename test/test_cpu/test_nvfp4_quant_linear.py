import pytest
import torch

from auto_round.data_type.nvfp import calculate_gparam
from auto_round.data_type.utils import get_quant_func
from auto_round.experimental import qmodules as ar_qmodules
from auto_round.export.export_to_autoround import AutoRoundFormat
from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear as _FPLinear
from auto_round.schemes import PRESET_SCHEMES

mx_schemes = [AutoRoundFormat.NVFP4.value]
QMODULE_MAPPING = {
    AutoRoundFormat.NVFP4.value: ar_qmodules.NVFP4QuantLinear,
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


@pytest.mark.parametrize("scheme", [AutoRoundFormat.NVFP4.value])
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

    # Create an MXQuantLinear layer from the original layer
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
