import pytest
import torch

from auto_round.export.export_to_autoround.qlinear_fp import FLOAT_TO_E2M1, pack_fp4_to_uint8


# Random sampling from FLOAT_TO_E2M1
def _create_random_e2m1_tensor(shape):
    """Create a tensor of the given shape with random values from FLOAT_TO_E2M1."""
    # Create a tensor of indices randomly selected from 0 to len(FLOAT_TO_E2M1)-1
    indices = torch.randint(0, len(FLOAT_TO_E2M1), shape)

    # Map the indices to their corresponding values
    e2m1_tensor = torch.tensor(FLOAT_TO_E2M1, dtype=torch.float32)[indices]
    return e2m1_tensor


def pack_fp4_to_uint8_old(x: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor with values in the fp4 range into uint8.
    As there are 16 valid fp4 values, two fp4 values can be
    packed into one uint8. Each fp4 value is mapped to its
    particular index (e.g. 0.5 is mapped to index 1, 6.0 is mapped
    to index 7) which is then represented using 4 bits. Consecutive
    pairs of 4 bits are then packed into an uint8.

    :param x: tensor to pack
    returns: a packed tensor in uint8
    """

    m, n = x.shape
    device = x.device

    # Create lookup table for FP4 values to indices
    # Map the absolute values to 0-7 indices
    kE2M1 = torch.tensor(FLOAT_TO_E2M1, device=device, dtype=x.dtype)

    # Find closest valid FP4 value index for each element
    abs_x = torch.abs(x)
    abs_indices = torch.zeros_like(abs_x, dtype=torch.long)
    for i, val in enumerate(kE2M1):  # TODO any optimize?
        abs_indices = torch.where(torch.isclose(abs_x, val), i, abs_indices)

    # Apply sign bit (bit 3) to get final 4-bit representation
    indices = abs_indices + (torch.signbit(x) << 3).to(torch.long)

    # Reshape to prepare for packing pairs of values
    indices = indices.reshape(-1)

    # Handle odd length by padding if necessary
    if indices.numel() % 2 != 0:
        indices = torch.cat([indices, torch.zeros(1, dtype=torch.long, device=device)])

    # Reshape to pair consecutive elements
    indices = indices.reshape(-1, 2)

    # Pack pairs of 4-bit values into 8-bit values
    packed = (indices[:, 0] | (indices[:, 1] << 4)).to(torch.uint8)

    return packed.reshape(m, n // 2)


qwen_weight_shapes = [
    torch.Size([2048, 768]),
    torch.Size([768, 2048]),
    torch.Size([128, 2048]),
    torch.Size([512, 2048]),
    torch.Size([4096, 2048]),
    torch.Size([151936, 2048]),
    torch.Size([2048, 4096]),
]


@pytest.mark.parametrize("shape", qwen_weight_shapes)
def test_packing_fp4(shape):
    with torch.device("cuda"):
        M, N = shape
        random_tensor = _create_random_e2m1_tensor((M, N))
        # Pack the tensor using the packing function
        packed_tensor = pack_fp4_to_uint8(random_tensor)
        packed_tensor_old = pack_fp4_to_uint8_old(random_tensor)
        # check equal
        assert torch.equal(packed_tensor, packed_tensor_old), "Packed tensors are not equal"
