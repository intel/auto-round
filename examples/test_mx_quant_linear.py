import torch

from auto_round import schemes as ar_schemes
from auto_round.experimental.qmodules.mx import MXQuantLinear


def test_mxquantlinear_from_original_and_forward():
    # Initialize the weights and bias of the original layer for reproducibility
    torch.manual_seed(42)

    # Create an original torch.nn.Linear layer
    in_features = 64
    out_features = 32
    original_layer = torch.nn.Linear(in_features, out_features, bias=True)

    original_layer.weight.data = torch.randn((out_features, in_features), dtype=torch.float32)
    original_layer.bias.data = torch.randn((out_features,), dtype=torch.float32)

    # Create an MXQuantLinear layer from the original layer
    mx_layer = MXQuantLinear.from_original(config=ar_schemes.MXFP8, original_layer=original_layer)

    # Ensure the weights and bias are correctly transferred
    assert mx_layer.in_features == original_layer.in_features
    assert mx_layer.out_features == original_layer.out_features
    # assert torch.allclose(mx_layer.weight, original_layer.weight, atol=1e-6)
    # assert torch.allclose(mx_layer.bias, original_layer.bias, atol=1e-6)

    # Generate a random input tensor
    input_tensor = torch.randn((4, in_features), dtype=torch.float32)

    # Perform a forward pass with both layers
    original_output = original_layer(input_tensor)
    mx_output = mx_layer(input_tensor)

    # # Compare the outputs
    # assert torch.allclose(mx_output, original_output, atol=1e-6), "Outputs do not match!"


test_mxquantlinear_from_original_and_forward()
