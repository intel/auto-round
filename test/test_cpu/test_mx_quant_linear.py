import pytest
import torch

from auto_round.data_type.utils import get_quant_func
from auto_round.experimental import qmodules as ar_qmodules
from auto_round.export.export_to_autoround import AutoRoundExportFormat
from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear as _MXFPLinear
from auto_round.schemes import PRESET_SCHEMES

mx_schemes = [AutoRoundExportFormat.MXFP8.value, AutoRoundExportFormat.MXFP4.value]
QMODULE_MAPPING = {
    AutoRoundExportFormat.MXFP8.value: ar_qmodules.MXFP8QuantLinear,
    AutoRoundExportFormat.MXFP4.value: ar_qmodules.MXFP4QuantLinear,
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


@pytest.mark.parametrize("scheme", mx_schemes)
@torch.inference_mode()
def test_mxquantlinear_from_original_and_forward(scheme):
    """
    Test MXFP4 and MXFP8 quantization schemes by creating quantized layers
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
    qdq_func, _ = get_quant_func(dtype=config.act_data_type, bits=config.act_bits, sym=True)
    qdq_weight, shared_exp, _ = qdq_func(
        tensor=original_layer.weight, bits=config.act_bits, group_size=config.act_group_size
    )
    shared_exp = shared_exp.reshape(weight_scale_shape)

    # Pack the weights using the QuantLinear class
    mxfp_lin = _MXFPLinear(
        bits=config.bits,
        group_size=config.group_size,
        infeatures=in_features,
        outfeatures=out_features,
        bias=original_layer.bias is not None,
        data_type=config.data_type,
    )
    mxfp_lin.pack(linear=original_layer, scales=shared_exp)

    # Create an MXQuantLinear layer from the original layer
    QuantLinearClass = QMODULE_MAPPING[scheme]
    mxfp_layer = QuantLinearClass.from_original(
        config=config,
        original_layer=original_layer,
    )

    # Copy the packed weights and scales to the quantized layer
    packed_weight = mxfp_lin.weight_packed if config.bits == 4 else mxfp_lin.weight
    if config.bits == 4:
        mxfp_layer.weight_packed.data.copy_(packed_weight)
    elif config.bits == 8:
        mxfp_layer.weight.data.copy_(packed_weight)
    else:
        raise ValueError("Only 4-bit and 8-bit quantization are supported.")
    mxfp_layer.weight_scale.data.copy_(mxfp_lin.weight_scale)

    # Validate layer attributes
    assert mxfp_layer.in_features == original_layer.in_features
    assert mxfp_layer.out_features == original_layer.out_features

    # Generate a random input tensor
    input_tensor = torch.randn((4, in_features), dtype=torch.float32)

    # Perform a forward pass with both layers
    original_output = original_layer(input_tensor)
    mx_output = mxfp_layer(input_tensor)

    # Compute the difference between the outputs
    diff = mx_output - original_output
    # Note: Remove NaN values, as we might get NaN when casting scales to FP8
    diff = diff[~torch.isnan(diff)]
    diff_amax = diff.abs().max()

    # Print the maximum difference for debugging
    print(f"Scheme: {scheme}, Max Difference: {diff_amax}")

    # Assert that the outputs are close within a tolerance
    assert diff_amax < 5e-1, f"Outputs differ too much for scheme {scheme}!"
