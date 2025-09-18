import torch

from auto_round import schemes as ar_schemes
from auto_round.data_type.utils import get_quant_func
from auto_round.experimental.qmodules.mx import MXQuantLinear
from auto_round.schemes import QuantizationScheme

__all__ = ["MXQuantLinear"]

SUPPORTED_HIGHER_DTYPE = [torch.bfloat16, torch.float16, torch.float32]
E8M0_EXPONENT_BIAS = 127

scheme = "MXFP8"


@torch.inference_mode()
def test_mxquantlinear_from_original_and_forward():
    # Initialize the weights and bias of the original layer for reproducibility
    torch.manual_seed(42)

    # Create an original torch.nn.Linear layer
    in_features = 64
    out_features = 512
    original_layer = torch.nn.Linear(in_features, out_features, bias=False)
    config = ar_schemes.MXFP4
    config = ar_schemes.MXFP8
    weight_scale_shape = (out_features, in_features // 32)
    qdq_func, _ = get_quant_func(dtype=config.act_data_type, bits=config.act_bits, sym=True)
    qdq_weight, shared_exp, _ = qdq_func(
        tensor=original_layer.weight.t(), bits=config.act_bits, group_size=config.act_group_size
    )

    shared_exp = shared_exp.reshape(weight_scale_shape)
    from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear as _MXFPLinear

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

    mxfp_layer = MXQuantLinear.from_original(
        config=config,
        original_layer=original_layer,
    )
    packed_weight = mxfp_lin.weight_packed if config.bits == 4 else mxfp_lin.weight
    mxfp_layer.weight.data.copy_(packed_weight)
    mxfp_layer.weight_scale.data.copy_(mxfp_lin.weight_scale)

    # Ensure the weights and bias are correctly transferred
    assert mxfp_layer.in_features == original_layer.in_features
    assert mxfp_layer.out_features == original_layer.out_features

    # Generate a random input tensor
    input_tensor = torch.randn((4, in_features), dtype=torch.float32)

    # Perform a forward pass with both layers
    original_output = original_layer(input_tensor)
    mx_output = mxfp_layer(input_tensor)
    diff = mx_output - original_output
    diff_amax = diff.abs().max()
    print(diff)
    # # Compare the outputs
    # assert torch.allclose(mx_output, original_output, atol=1e-6), "Outputs do not match!"


test_mxquantlinear_from_original_and_forward()
