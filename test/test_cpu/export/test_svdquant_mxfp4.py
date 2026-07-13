import ast
import hashlib
import inspect

import pytest
import torch

import auto_round.export.svdquant_mxfp4 as codec_module
from auto_round.data_type.mxfp import quant_element, quant_mx
from auto_round.export.svdquant_mxfp4 import (
    NunchakuMXFP4Packer,
    PackedMXFP4,
    decode_e2m1,
    decode_ue8m0,
    encode_e2m1,
    encode_ue8m0,
    pack_lowrank_weight,
    pack_nibbles,
    unpack_lowrank_weight,
    unpack_nibbles,
)


E2M1_CODEBOOK = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("down,shape", [(True, (3, 17)), (False, (17, 3))])
def test_lowrank_physical_pack_roundtrips_values_with_128_aligned_feature_axis(dtype, down, shape):
    logical = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape).to(dtype)

    packed = pack_lowrank_weight(logical, down=down)
    padded = unpack_lowrank_weight(packed, down=down)

    assert packed.shape == (128, 16)
    assert packed.dtype == dtype
    assert padded.shape == ((16, 128) if down else (128, 16))
    torch.testing.assert_close(padded[: shape[0], : shape[1]], logical)
    assert torch.count_nonzero(padded[shape[0] :, :]) == 0
    assert torch.count_nonzero(padded[: shape[0], shape[1] :]) == 0


@pytest.mark.parametrize(
    "down,shape,expected_hash",
    [
        (True, (3, 17), "f62c895e44a7139fc942941b1244857d65143dfe52ad852e8847339aa6119029"),
        (False, (17, 3), "0690f5c24d1ca25ad4f7714d9ce6b626b792ac89e46986d7a73dd0f327b163b4"),
    ],
)
def test_lowrank_physical_pack_matches_independent_fixed_fixture(down, shape, expected_hash):
    logical = torch.arange(shape[0] * shape[1], dtype=torch.float16).reshape(shape)

    packed = pack_lowrank_weight(logical, down=down)

    assert packed.shape == (128, 16)
    assert hashlib.sha256(bytes(packed.view(torch.uint8).flatten().tolist())).hexdigest() == expected_hash


@pytest.mark.parametrize(
    "weight,down,message",
    [
        (torch.ones(16), False, "2D"),
        (torch.ones(2, 2), False, "dtype"),
        (torch.empty(0, 2, dtype=torch.float16), False, "non-empty"),
        (torch.tensor([[torch.inf]], dtype=torch.float16), False, "finite"),
        (torch.ones(2, 2, dtype=torch.float16), 1, "down"),
    ],
)
def test_lowrank_pack_rejects_invalid_inputs(weight, down, message):
    with pytest.raises(ValueError, match=message):
        pack_lowrank_weight(weight, down=down)


def test_lowrank_unpack_rejects_non_block_aligned_physical_shape():
    with pytest.raises(ValueError, match="divisible by 128 and 16"):
        unpack_lowrank_weight(torch.ones(16, 15, dtype=torch.bfloat16), down=False)


def test_pack_residual_returns_immutable_aligned_physical_tensors():
    weight = torch.linspace(-8.0, 8.0, steps=128 * 128).reshape(128, 128)

    packed = NunchakuMXFP4Packer().pack_residual(weight)

    assert isinstance(packed, PackedMXFP4)
    assert packed.logical_shape == (128, 128)
    assert packed.padded_shape == (128, 128)
    assert packed.qweight.shape == (128, 64)
    assert packed.qweight.dtype == torch.int8
    assert packed.wscales.shape == (4, 128)
    assert packed.wscales.dtype == torch.uint8
    with pytest.raises(AttributeError):
        packed.logical_shape = (1, 1)


@pytest.mark.parametrize("shape", [(128, 128), (7, 65)])
def test_pack_unpack_residual_matches_autoround_rtn_qdq(shape):
    generator = torch.Generator().manual_seed(20260713)
    weight = torch.randn(shape, generator=generator, dtype=torch.float32) * 3
    expected, _, _ = quant_mx(weight, bits=4, group_size=32, data_type="mx_fp4e2m1")
    packer = NunchakuMXFP4Packer()

    packed = packer.pack_residual(weight)
    actual = packer.unpack_residual(
        packed.qweight, packed.wscales, packed.logical_shape, dtype=torch.float32
    )

    assert actual.shape == shape
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_physical_reorders_roundtrip_exact_codes_and_match_reference_fixture():
    """Fixture generated from the reference packers at source commit 0abaaf0.

    The logical patterns exercise every E2M1 nibble and nontrivial UE8M0 byte;
    constants lock the documented MMA tile permutation and little-endian nibble
    placement without importing the reference package in the test suite.
    """

    packer = NunchakuMXFP4Packer()
    n = torch.arange(128).reshape(128, 1)
    k = torch.arange(128).reshape(1, 128)
    logical_codes = ((n * 3 + k * 5 + n // 7 + k // 11) % 16).to(torch.uint8)
    groups = torch.arange(4).reshape(1, 4)
    logical_scales = ((127 + n * 7 + groups * 11) % 255).to(torch.uint8)

    qweight = packer._pack_weight_codes(logical_codes)
    wscales = packer._pack_scale_codes(logical_scales)

    assert hashlib.sha256(bytes(qweight.view(torch.uint8).flatten().tolist())).hexdigest() == (
        "9ea940ceb244d16f7de4b1daf9e39f6403de13aaf7d8ec17635600d8531f0e61"
    )
    assert hashlib.sha256(bytes(wscales.flatten().tolist())).hexdigest() == (
        "13ec93403839f512d4300d813bd160f63940363868fba69034c01142595d7340"
    )
    assert qweight.view(torch.uint8).flatten()[:16].tolist() == [
        80, 250, 148, 62, 130, 45, 199, 97, 233, 131, 45, 199, 27, 182, 80, 250
    ]
    assert wscales.flatten()[:16].tolist() == [
        127, 138, 96, 107, 65, 76, 34, 45, 183, 194, 152, 163, 121, 132, 90, 101
    ]
    assert torch.equal(packer._unpack_weight_codes(qweight), logical_codes)
    assert torch.equal(packer._unpack_scale_codes(wscales), logical_scales)


def test_unaligned_pack_zero_fills_codes_and_identity_fills_scale_padding():
    packer = NunchakuMXFP4Packer()
    packed = packer.pack_residual(torch.ones(7, 65))

    assert packed.logical_shape == (7, 65)
    assert packed.padded_shape == (128, 128)
    assert packed.qweight.shape == (128, 64)
    assert packed.wscales.shape == (4, 128)
    codes = packer._unpack_weight_codes(packed.qweight)
    scales = packer._unpack_scale_codes(packed.wscales)
    assert torch.count_nonzero(codes[7:]) == 0
    assert torch.count_nonzero(codes[:7, 96:]) == 0
    assert torch.equal(scales[7:], torch.full_like(scales[7:], 127))
    assert torch.equal(scales[:7, 3:], torch.full_like(scales[:7, 3:], 127))


@pytest.mark.parametrize(
    "weight, message",
    [
        (torch.ones(32), "2D floating-point"),
        (torch.ones(2, 3, 4), "2D floating-point"),
        (torch.ones(2, 32, dtype=torch.int32), "2D floating-point"),
        (torch.tensor([[torch.nan]]), "finite"),
        (torch.empty(0, 32), "non-empty"),
    ],
)
def test_pack_residual_rejects_invalid_weights(weight, message):
    with pytest.raises(ValueError, match=message):
        NunchakuMXFP4Packer().pack_residual(weight)


def test_pack_residual_rejects_wrong_group_size_and_warp_layout():
    with pytest.raises(ValueError, match="group_size must be 32"):
        NunchakuMXFP4Packer().pack_residual(torch.ones(2, 32), group_size=16)
    with pytest.raises(ValueError, match="warp_n must be 128"):
        NunchakuMXFP4Packer(warp_n=64)


def test_unpack_residual_rejects_wrong_shapes_dtypes_and_logical_bounds():
    packer = NunchakuMXFP4Packer()
    packed = packer.pack_residual(torch.ones(7, 65))

    invalid_calls = [
        (packed.qweight.to(torch.uint8), packed.wscales, (7, 65), torch.float32, "qweight.*int8"),
        (packed.qweight[:, :-1], packed.wscales, (7, 65), torch.float32, "qweight shape"),
        (packed.qweight, packed.wscales.to(torch.int16), (7, 65), torch.float32, "wscales.*uint8"),
        (packed.qweight, packed.wscales[:-1], (7, 65), torch.float32, "wscales shape"),
        (packed.qweight, packed.wscales, (129, 65), torch.float32, "logical_shape"),
        (packed.qweight, packed.wscales, (7,), torch.float32, "logical_shape"),
        (packed.qweight, packed.wscales, (7, 65), torch.int32, "dtype"),
    ]
    for qweight, wscales, logical_shape, dtype, message in invalid_calls:
        with pytest.raises(ValueError, match=message):
            packer.unpack_residual(qweight, wscales, logical_shape, dtype)


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


def test_e2m1_reviewer_value_uses_autoround_float32_normalization():
    values = torch.tensor([435.5967712], dtype=torch.float64)
    scales = torch.tensor([1742.387207], dtype=torch.float64)
    normalized = values.to(torch.float32) / scales.to(torch.float32)
    expected = quant_element(normalized.clamp(min=-6.0, max=6.0), ebits=2, mbits=3, max_norm=6.0)

    codes = encode_e2m1(values, scales)
    actual = decode_e2m1(codes, torch.ones_like(scales), dtype=torch.float32)

    torch.testing.assert_close(actual, expected)


def test_e2m1_values_around_every_tie_match_float32_quant_element():
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
    below = torch.nextafter(boundaries, torch.full_like(boundaries, -torch.inf))
    above = torch.nextafter(boundaries, torch.full_like(boundaries, torch.inf))
    normalized_targets = torch.cat((below, boundaries, above))
    normalized_targets = torch.cat((normalized_targets, -normalized_targets))
    scales = torch.full(normalized_targets.shape, 4.0, dtype=torch.float64)
    values = normalized_targets.to(torch.float64) * scales
    normalized32 = values.to(torch.float32) / scales.to(torch.float32)
    expected = quant_element(normalized32.clamp(min=-6.0, max=6.0), ebits=2, mbits=3, max_norm=6.0)

    codes = encode_e2m1(values, scales)
    actual = decode_e2m1(codes, torch.ones_like(scales), dtype=torch.float32)

    torch.testing.assert_close(actual, expected)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


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


def test_ue8m0_valid_export_codes_roundtrip_and_reserved_code_255_decodes():
    codes = torch.arange(255, dtype=torch.uint8)

    decoded = decode_ue8m0(codes)

    assert decoded.dtype == torch.float32
    assert torch.equal(encode_ue8m0(decoded), codes)
    largest_finite_scale = torch.tensor([torch.finfo(torch.float64).max], dtype=torch.float64)
    assert encode_ue8m0(largest_finite_scale).item() == 254
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


def test_e2m1_group_aligned_scale_rank_wins_over_broadcast_collision():
    rows = torch.arange(32).reshape(32, 1)
    groups = torch.arange(32).reshape(1, 32)
    scales = torch.pow(2.0, ((rows + 2 * groups) % 5).to(torch.float32))
    values = scales.unsqueeze(-1).expand(32, 32, 32).contiguous()

    codes = encode_e2m1(values, scales)
    decoded = decode_e2m1(codes, scales, dtype=torch.float32)

    assert torch.equal(codes, torch.full_like(codes, 2))
    torch.testing.assert_close(decoded, values)


def test_e2m1_group_size_32_codec_matches_autoround_quant_mx_qdq():
    weight = torch.linspace(-7.0, 7.0, steps=3 * 64, dtype=torch.float32).reshape(3, 64)
    expected, shared_exponent, _ = quant_mx(
        weight,
        bits=4,
        group_size=32,
        data_type="mx_fp4e2m1",
    )
    grouped_weight = weight.reshape(3, 2, 32)
    scales = torch.exp2(shared_exponent.reshape(3, 2).to(torch.float32))

    codes = encode_e2m1(grouped_weight, scales)
    actual = decode_e2m1(codes, scales, dtype=torch.float32).reshape_as(weight)

    torch.testing.assert_close(actual, expected)


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


def test_e2m1_rejects_ambiguous_scale_ranks():
    with pytest.raises(ValueError, match="scales rank 1.*tensor rank 3"):
        encode_e2m1(torch.ones(2, 3, 4), torch.ones(3))


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_e2m1_decode_supports_runtime_float_dtypes(dtype):
    decoded = decode_e2m1(torch.tensor([1, 6, 9, 14], dtype=torch.uint8), torch.tensor(2.0), dtype=dtype)

    assert decoded.dtype == dtype
    assert torch.equal(decoded, torch.tensor([1.0, 8.0, -1.0, -8.0], dtype=dtype))


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="PyTorch build has no float8 dtype")
def test_e2m1_rejects_float8_decode_dtype():
    with pytest.raises(ValueError, match="dtype.*float16.*bfloat16.*float32.*float64"):
        decode_e2m1(torch.tensor([0], dtype=torch.uint8), torch.tensor(1.0), dtype=torch.float8_e4m3fn)


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
