import ast
import inspect

import pytest
import torch

import auto_round.export.svdquant_mxfp4 as codec_module
from auto_round.data_type.mxfp import quant_element
from auto_round.export.svdquant_mxfp4 import (
    decode_e2m1,
    decode_ue8m0,
    encode_e2m1,
    encode_ue8m0,
    pack_nibbles,
    unpack_nibbles,
)


E2M1_CODEBOOK = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def test_e2m1_roundtrips_every_logical_code():
    codes = torch.arange(16, dtype=torch.uint8)
    scales = torch.tensor(2.0)

    decoded = decode_e2m1(codes, scales, dtype=torch.float32)

    torch.testing.assert_close(decoded, E2M1_CODEBOOK * 2)
    assert torch.equal(encode_e2m1(decoded, scales), codes)
    assert torch.signbit(decoded[8])


def test_e2m1_raw_values_follow_autoround_rounding_and_saturate():
    normalized = torch.tensor(
        [0.25, 0.4, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 100.0]
    )
    normalized = torch.cat((normalized, -normalized))
    scales = torch.tensor(2.0)
    values = normalized * scales
    expected = quant_element(normalized, ebits=2, mbits=3, max_norm=6.0) * scales

    actual = decode_e2m1(encode_e2m1(values, scales), scales, dtype=torch.float32)

    torch.testing.assert_close(actual, expected)


def test_e2m1_float32_extreme_ratios_saturate_and_equal_edges_encode_one():
    finfo = torch.finfo(torch.float32)
    values = torch.tensor([finfo.max, -finfo.max, finfo.max, -finfo.max, finfo.tiny, -finfo.tiny])
    scales = torch.tensor([finfo.tiny, finfo.tiny, finfo.max, finfo.max, finfo.tiny, finfo.tiny])

    codes = encode_e2m1(values, scales)

    assert torch.equal(codes, torch.tensor([7, 15, 2, 10, 2, 10], dtype=torch.uint8))


def test_e2m1_float64_extremes_saturate_preserve_sign_and_encode_equal_huge_values():
    finfo = torch.finfo(torch.float64)
    values = torch.tensor(
        [finfo.max, -finfo.max, finfo.max, -finfo.max, finfo.tiny, -finfo.tiny, finfo.tiny, -finfo.tiny],
        dtype=torch.float64,
    )
    scales = torch.tensor(
        [finfo.tiny, finfo.tiny, finfo.max, finfo.max, finfo.tiny, finfo.tiny, finfo.max, finfo.max],
        dtype=torch.float64,
    )

    codes = encode_e2m1(values, scales)

    assert torch.equal(codes, torch.tensor([7, 15, 2, 10, 2, 10, 0, 8], dtype=torch.uint8))


def test_ue8m0_encodes_invalid_values_and_exponent_boundaries():
    scales = torch.tensor(
        [0.0, -1.0, torch.nan, torch.inf, -torch.inf, 2.0**-128, 2.0**-127, 1.0, 1.01, 2.0**127],
        dtype=torch.float64,
    )

    codes = encode_ue8m0(scales)

    assert torch.equal(codes, torch.tensor([127, 127, 127, 127, 127, 0, 0, 127, 128, 254], dtype=torch.uint8))


def test_ue8m0_decode_and_roundtrip_cover_full_encoded_range():
    codes = torch.arange(255, dtype=torch.uint8)

    decoded = decode_ue8m0(codes)

    assert decoded.dtype == torch.float32
    assert torch.equal(encode_ue8m0(decoded), codes)
    assert torch.isinf(decode_ue8m0(torch.tensor([255], dtype=torch.uint8))).item()


def test_nibble_packing_roundtrips_even_last_dimension_low_nibble_first():
    codes = torch.tensor([[0, 1, 2, 3], [15, 14, 13, 12]], dtype=torch.uint8)

    packed = pack_nibbles(codes)

    assert packed.dtype == torch.uint8
    assert torch.equal(packed, torch.tensor([[0x10, 0x32], [0xEF, 0xCD]], dtype=torch.uint8))
    assert torch.equal(unpack_nibbles(packed), codes)


def test_nibble_packing_zero_pads_and_trims_odd_last_dimension():
    codes = torch.tensor([1, 2, 15], dtype=torch.uint8)

    packed = pack_nibbles(codes)

    assert torch.equal(packed, torch.tensor([0x21, 0x0F], dtype=torch.uint8))
    assert torch.equal(unpack_nibbles(packed), torch.tensor([1, 2, 15, 0], dtype=torch.uint8))
    assert torch.equal(unpack_nibbles(packed, logical_count=3), codes)


def test_e2m1_accepts_group_aligned_scales_and_preserves_shape_and_requested_dtype():
    normalized = E2M1_CODEBOOK[:8].reshape(1, 2, 4).expand(2, -1, -1)
    scales = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    values = normalized * scales.unsqueeze(-1)

    codes = encode_e2m1(values, scales)
    decoded = decode_e2m1(codes, scales, dtype=torch.float64)

    assert codes.shape == values.shape
    assert codes.dtype == torch.uint8
    assert decoded.shape == values.shape
    assert decoded.dtype == torch.float64
    torch.testing.assert_close(decoded, values.to(torch.float64))


@pytest.mark.parametrize(
    "scales",
    [
        torch.tensor(0.0),
        torch.tensor(-1.0),
        torch.tensor(torch.nan),
        torch.tensor(torch.inf),
        torch.ones(3, 2),
        torch.tensor(1),
    ],
)
def test_e2m1_rejects_malformed_scales(scales):
    with pytest.raises(ValueError, match="scales"):
        encode_e2m1(torch.ones(2, 4), scales)


@pytest.mark.parametrize(
    "values",
    [torch.tensor([1]), torch.tensor([torch.nan]), torch.tensor([torch.inf])],
)
def test_e2m1_rejects_malformed_values(values):
    with pytest.raises(ValueError, match="values"):
        encode_e2m1(values, torch.tensor(1.0))


@pytest.mark.parametrize(
    "codes",
    [
        torch.tensor([-1]),
        torch.tensor([16]),
        torch.tensor([1.0]),
        torch.tensor([True]),
    ],
)
def test_e2m1_rejects_malformed_codes(codes):
    with pytest.raises(ValueError, match="codes"):
        decode_e2m1(codes, torch.tensor(1.0))


def test_e2m1_rejects_non_floating_decode_dtype():
    with pytest.raises(ValueError, match="dtype"):
        decode_e2m1(torch.tensor([0], dtype=torch.uint8), torch.tensor(1.0), dtype=torch.int32)


def test_ue8m0_rejects_malformed_tensor_dtypes():
    with pytest.raises(ValueError, match="scales"):
        encode_ue8m0(torch.tensor([1]))
    with pytest.raises(ValueError, match="codes"):
        decode_ue8m0(torch.tensor([127.0]))


@pytest.mark.parametrize("codes", [torch.tensor([16]), torch.tensor([-1]), torch.tensor([1.0])])
def test_pack_nibbles_rejects_malformed_codes(codes):
    with pytest.raises(ValueError, match="codes"):
        pack_nibbles(codes)


def test_nibble_helpers_reject_scalar_tensors_and_invalid_trim_counts():
    with pytest.raises(ValueError, match="dimension"):
        pack_nibbles(torch.tensor(1, dtype=torch.uint8))
    with pytest.raises(ValueError, match="dimension"):
        unpack_nibbles(torch.tensor(1, dtype=torch.uint8))

    packed = torch.tensor([0x21, 0x03], dtype=torch.uint8)
    for logical_count in (-1, 5, 1.5, True):
        with pytest.raises(ValueError, match="logical_count"):
            unpack_nibbles(packed, logical_count=logical_count)


def test_codec_has_no_deepcompressor_or_nunchaku_imports():
    tree = ast.parse(inspect.getsource(codec_module))
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert not any(name.startswith(("deepcompressor", "nunchaku")) for name in imported_modules)
