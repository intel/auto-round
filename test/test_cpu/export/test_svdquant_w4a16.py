import ast
import hashlib
import inspect
from dataclasses import replace

import pytest
import torch

import auto_round.export.svdquant_w4a16 as codec_module
from auto_round.export.svdquant_w4a16 import (
    PackedW4A16,
    dequantize_adanorm_w4a16,
    pack_adanorm_w4a16,
    quantize_adanorm_w4a16_rtn,
    unpack_adanorm_w4a16,
)


def _representable_fixture(dtype=torch.float16):
    rows = torch.arange(12).reshape(12, 1)
    columns = torch.arange(1024).reshape(1, 1024)
    signed = ((rows * 5 + columns * 3) % 15 - 7).to(torch.float32)
    scales = (torch.arange(12 * 16).reshape(12, 16) % 13 + 1).to(dtype) / 8
    weight = (signed.reshape(12, 16, 64) * scales.unsqueeze(-1)).reshape(12, 1024).to(dtype)
    return weight, scales, signed.to(torch.int8)


def test_pack_returns_runtime_layout_and_unpack_recovers_channel_major_codes():
    weight, scales, signed = _representable_fixture()

    packed = pack_adanorm_w4a16(weight, scales, splits=3)

    assert isinstance(packed, PackedW4A16)
    assert packed.qweight.shape == (3, 512)
    assert packed.qweight.dtype == torch.int32
    assert packed.wscales.shape == (16, 12)
    assert packed.wzeros.shape == (16, 12)
    assert packed.bias.shape == (12,)
    assert packed.dtype == torch.float16
    assert packed.logical_shape == (12, 1024)
    expected_codes = signed.reshape(3, 4, 1024).permute(1, 0, 2).reshape(12, 1024)
    assert torch.equal(unpack_adanorm_w4a16(packed), expected_codes)


@pytest.mark.parametrize(
    "weight,scale,bias,splits,group_size,message",
    [
        (torch.ones(12, 1024), torch.ones(12, 16), None, 3, 64, "BF16 or FP16"),
        (
            torch.ones(12, 1024, dtype=torch.float16),
            torch.ones(12, 16, dtype=torch.int32),
            None,
            3,
            64,
            "scale.*floating",
        ),
        (
            torch.ones(12, 1023, dtype=torch.float16),
            torch.ones(12, 16, dtype=torch.float16),
            None,
            3,
            64,
            "K.*divisible",
        ),
        (torch.ones(10, 1024, dtype=torch.float16), torch.ones(10, 16, dtype=torch.float16), None, 3, 64, "O.*splits"),
        (torch.ones(9, 1024, dtype=torch.float16), torch.ones(9, 16, dtype=torch.float16), None, 3, 64, "O.*4"),
        (torch.ones(12, 512, dtype=torch.float16), torch.ones(12, 8, dtype=torch.float16), None, 3, 64, "G.*16"),
        (
            torch.ones(12, 1024, dtype=torch.float16),
            torch.ones(12, 15, dtype=torch.float16),
            None,
            3,
            64,
            "scale shape",
        ),
        (
            torch.ones(12, 1024, dtype=torch.float16),
            torch.zeros(12, 16, dtype=torch.float16),
            None,
            3,
            64,
            "positive finite",
        ),
        (
            torch.ones(12, 1024, dtype=torch.float16),
            torch.ones(12, 16, dtype=torch.float16),
            torch.ones(11),
            3,
            64,
            "bias",
        ),
        (
            torch.ones(12, 1024, dtype=torch.float16) * 8,
            torch.ones(12, 16, dtype=torch.float16),
            None,
            3,
            64,
            r"\[-7, 7\]",
        ),
    ],
)
def test_pack_rejects_invalid_contract_inputs(weight, scale, bias, splits, group_size, message):
    with pytest.raises(ValueError, match=message):
        pack_adanorm_w4a16(weight, scale, bias=bias, splits=splits, group_size=group_size)


@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float64])
def test_pack_rejects_scale_dtype_that_does_not_match_weight(scale_dtype):
    weight = torch.ones(12, 1024, dtype=torch.float16)
    scale = torch.ones(12, 16, dtype=scale_dtype)

    with pytest.raises(ValueError, match="scale dtype must exactly match weight dtype"):
        pack_adanorm_w4a16(weight, scale)


@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.float64])
def test_pack_rejects_bias_dtype_that_does_not_match_weight(bias_dtype):
    weight = torch.ones(12, 1024, dtype=torch.bfloat16)
    scale = torch.ones(12, 16, dtype=torch.bfloat16)
    bias = torch.zeros(12, dtype=bias_dtype)

    with pytest.raises(ValueError, match="bias dtype must exactly match weight dtype"):
        pack_adanorm_w4a16(weight, scale, bias=bias)


@pytest.mark.parametrize(
    "dtype,bias_value",
    [(torch.float16, 2048), (torch.bfloat16, 256)],
)
def test_pack_rejects_unrepresentable_adanorm_identity_bias_offset(dtype, bias_value):
    weight = torch.zeros(12, 1024, dtype=dtype)
    scale = torch.ones(12, 16, dtype=dtype)
    bias = torch.full((12,), bias_value, dtype=dtype)

    with pytest.raises(ValueError, match=r"AdaNorm bias identity offset \+1 must be exactly representable"):
        pack_adanorm_w4a16(weight, scale, bias=bias)


@pytest.mark.parametrize(
    "dtype,bias_value,expected_offset",
    [(torch.float16, 2047, 2048), (torch.bfloat16, 255, 256)],
)
def test_pack_accepts_exactly_representable_adanorm_identity_bias_boundary(dtype, bias_value, expected_offset):
    weight = torch.zeros(12, 1024, dtype=dtype)
    scale = torch.ones(12, 16, dtype=dtype)
    bias = torch.full((12,), bias_value, dtype=dtype)

    packed = pack_adanorm_w4a16(weight, scale, bias=bias)

    assert torch.count_nonzero(packed.bias == expected_offset) == 4
    assert torch.count_nonzero(packed.bias == bias_value) == 8


def test_pack_rejects_scale_whose_emitted_zero_overflows_source_dtype():
    weight = torch.zeros(12, 1024, dtype=torch.float16)
    scale = torch.full((12, 16), 10000, dtype=torch.float16)

    with pytest.raises(ValueError, match="wzeros must remain finite"):
        pack_adanorm_w4a16(weight, scale)


def test_flux_adanorm_dimensions_satisfy_runtime_shape_formulas_without_allocating_tensors():
    for out_features, splits in ((3072 * 3, 3), (3072 * 6, 6)):
        in_features = 3072
        groups = in_features // 64
        assert out_features % splits == out_features % 4 == 0
        assert groups == 48 and groups % 16 == 0
        assert (out_features // 4, in_features // 2) == (out_features // 4, 1536)
        assert (groups, out_features) == (48, out_features)


@pytest.mark.parametrize("splits", [3, 6])
def test_pack_reorders_adanorm_fields_and_adds_each_identity_offset_once(splits):
    weight, scales, _ = _representable_fixture(dtype=torch.bfloat16)
    scale4d = scales.reshape(12, 1, 16, 1)
    bias = torch.arange(12, dtype=torch.bfloat16) / 4

    packed = pack_adanorm_w4a16(weight, scale4d, bias=bias, splits=splits)

    channels = 12 // splits
    expected_weight = weight.reshape(splits, channels, 1024).transpose(0, 1).reshape(12, 1024)
    expected_scales = scales.reshape(splits, channels, 16).transpose(0, 1).reshape(12, 16)
    expected_bias = bias.reshape(splits, channels).transpose(0, 1).clone()
    identity_fields = {1, splits - 2}
    for field in identity_fields:
        expected_bias[:, field] += 1

    torch.testing.assert_close(dequantize_adanorm_w4a16(packed), expected_weight)
    assert torch.equal(packed.wscales, expected_scales.t().contiguous())
    assert torch.equal(packed.wzeros, (-7 * expected_scales).t().contiguous())
    assert torch.equal(packed.bias, expected_bias.reshape(-1))


@pytest.mark.parametrize("splits", [3, 6])
def test_chunked_pack_matches_unchunked_bytes(splits):
    weight, scales, _ = _representable_fixture(dtype=torch.bfloat16)
    bias = torch.arange(12, dtype=torch.bfloat16) / 4

    chunked = pack_adanorm_w4a16(weight, scales, bias=bias, splits=splits, chunk_rows=4)
    unchunked = pack_adanorm_w4a16(weight, scales, bias=bias, splits=splits, chunk_rows=None)

    for name in ("qweight", "wscales", "wzeros", "bias"):
        assert torch.equal(getattr(chunked, name), getattr(unchunked, name))


def test_rtn_uses_absmax_over_seven_and_identity_scale_for_zero_groups():
    weight = torch.zeros(12, 1024, dtype=torch.bfloat16)
    weight[:, 64:128] = torch.linspace(-3, 3.5, 64, dtype=torch.bfloat16)

    packed = quantize_adanorm_w4a16_rtn(weight, splits=3)

    logical_scales = torch.ones(12, 16, dtype=torch.bfloat16)
    logical_scales[:, 1] = 0.5
    channel_scales = logical_scales.reshape(3, 4, 16).transpose(0, 1).reshape(12, 16)
    assert torch.equal(packed.wscales, channel_scales.t().contiguous())
    qdq = dequantize_adanorm_w4a16(packed)
    assert torch.count_nonzero(qdq[:, :64]) == 0
    assert torch.count_nonzero(qdq[:, 128:]) == 0
    assert unpack_adanorm_w4a16(packed).min() >= -7
    assert unpack_adanorm_w4a16(packed).max() <= 7


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rtn_preserves_nonzero_groups_at_smallest_subnormal(dtype):
    smallest_subnormal = torch.nextafter(torch.tensor(0, dtype=dtype), torch.tensor(1, dtype=dtype))
    weight = torch.zeros(12, 1024, dtype=dtype)
    weight[:, 0] = smallest_subnormal

    packed = quantize_adanorm_w4a16_rtn(weight)

    assert bool(torch.isfinite(packed.wscales).all())
    assert bool((packed.wscales > 0).all())
    assert torch.count_nonzero(dequantize_adanorm_w4a16(packed)) == 12


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rtn_keeps_emitted_tensors_finite_at_maximum_source_value(dtype):
    weight = torch.full((12, 1024), torch.finfo(dtype).max, dtype=dtype)

    packed = quantize_adanorm_w4a16_rtn(weight)

    assert bool(torch.isfinite(packed.wscales).all())
    assert bool(torch.isfinite(packed.wzeros).all())
    assert bool(torch.isfinite(packed.bias).all())
    assert unpack_adanorm_w4a16(packed).max() == 7


def test_chunked_rtn_matches_unchunked_bytes():
    weight, _, _ = _representable_fixture(dtype=torch.float16)

    chunked = quantize_adanorm_w4a16_rtn(weight, splits=3, chunk_rows=4)
    unchunked = quantize_adanorm_w4a16_rtn(weight, splits=3, chunk_rows=None)

    for name in ("qweight", "wscales", "wzeros", "bias"):
        assert torch.equal(getattr(chunked, name), getattr(unchunked, name))


@pytest.mark.parametrize(
    "splits,expected",
    [
        (
            3,
            {
                "qweight": "615f5234fb03a7e5e6934134f63afc10f43dc58d4fe1f7d4d0d60130bde21246",
                "wscales": "e946419d02f5cf3935bbe9658719b66eaac1734c0dee36b2924221b7cc179c60",
                "wzeros": "83c884513a32ea173f9a044040b971871443c2a6ed2f5bda83016645befa92fa",
                "bias": "3327fe77502139f6f9169f331587fdcc1c5286b1058472d6595325f19eca02b4",
            },
        ),
        (
            6,
            {
                "qweight": "313bbd934935aa9a5e555455293e71c25bb1edd8f4c06038a61c930fd04004cf",
                "wscales": "7d2b98f262f9ec596f7bb41f4e140f616a4d7b212f9c65b1ccfcc83cc80fbbb2",
                "wzeros": "f236afd430b46be100b9d830ccbe997258a0c88a36ee78148b0a253531d4b00d",
                "bias": "5cfcc675cceaaaca4ef0233ebd354750cd173c9ee63f5477945c46c7965dcf91",
            },
        ),
    ],
)
def test_pack_matches_independent_bf16_adanorm_fixture(splits, expected):
    """Fixture generated from DeepCompressor commit 0abaaf0.

    Algorithm sources: ``deepcompressor/backend/tinychat/utils.py::pack_w4``
    and ``deepcompressor/backend/nunchaku/utils.py::convert_to_nunchaku_w4x16_linear_weight``.
    The deterministic generator below defines signed codes, power-of-two BF16
    scales, and bias directly; the constants were produced independently.
    """

    rows = torch.arange(12).reshape(12, 1)
    columns = torch.arange(1024).reshape(1, 1024)
    groups = torch.arange(16).reshape(1, 16)
    signed = (rows * 5 + columns * 3 + rows // 2 + columns // 7) % 15 - 7
    scales = torch.exp2((((rows + groups * 2) % 7) - 8).float()).to(torch.bfloat16)
    weight = (signed.reshape(12, 16, 64) * scales.unsqueeze(-1)).reshape(12, 1024).to(torch.bfloat16)
    bias = ((torch.arange(12) - 4) / 16).to(torch.bfloat16)

    packed = pack_adanorm_w4a16(weight, scales, bias=bias, splits=splits, chunk_rows=4)

    for name, expected_hash in expected.items():
        tensor = getattr(packed, name)
        actual_hash = hashlib.sha256(bytes(tensor.contiguous().view(torch.uint8).flatten().tolist())).hexdigest()
        assert actual_hash == expected_hash


def test_codec_has_no_deepcompressor_or_nunchaku_imports():
    tree = ast.parse(inspect.getsource(codec_module))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert "deepcompressor" not in imported_roots
    assert "nunchaku" not in imported_roots


def test_codec_rejects_non_little_endian_hosts(monkeypatch):
    weight, scales, _ = _representable_fixture()
    packed = pack_adanorm_w4a16(weight, scales)
    monkeypatch.setattr(codec_module.sys, "byteorder", "big")

    with pytest.raises(ValueError, match="little-endian"):
        pack_adanorm_w4a16(weight, scales)
    with pytest.raises(ValueError, match="little-endian"):
        unpack_adanorm_w4a16(packed)


def test_unpack_and_qdq_reject_invalid_packed_payloads():
    weight, scales, _ = _representable_fixture()
    packed = pack_adanorm_w4a16(weight, scales)
    invalid_payloads = [
        (replace(packed, qweight=packed.qweight.to(torch.int64)), "qweight dtype"),
        (replace(packed, qweight=packed.qweight[:, :-1]), "qweight shape"),
        (replace(packed, wscales=packed.wscales.to(torch.float32)), "wscales dtype"),
        (replace(packed, wzeros=packed.wzeros[:-1]), "wzeros shape"),
        (replace(packed, bias=packed.bias[:-1]), "bias shape"),
        (replace(packed, dtype=torch.float32), "payload dtype"),
        (replace(packed, logical_shape=(12,)), "logical_shape"),
        (replace(packed, splits=1), "splits"),
        (replace(packed, splits=3.0), "splits"),
        (replace(packed, group_size=32), "group_size"),
        (replace(packed, group_size=64.0), "group_size"),
        (replace(packed, qweight=packed.qweight.to("meta")), "same device"),
        (replace(packed, wscales=torch.full_like(packed.wscales, torch.inf)), "wscales.*finite"),
        (replace(packed, wzeros=torch.full_like(packed.wzeros, torch.inf)), "wzeros.*finite"),
        (replace(packed, wzeros=torch.zeros_like(packed.wzeros)), "wzeros must equal"),
        (replace(packed, bias=torch.full_like(packed.bias, torch.inf)), "bias.*finite"),
        (replace(packed, qweight=torch.full_like(packed.qweight, -1)), "codes outside"),
    ]

    for invalid, message in invalid_payloads:
        for decode in (unpack_adanorm_w4a16, dequantize_adanorm_w4a16):
            with pytest.raises(ValueError, match=message):
                decode(invalid)
