import pytest
import torch

from auto_round.data_type.utils import get_quant_func
from auto_round.schemes import parse_scheme, preset_name_to_scheme, scheme_to_preset_name


@pytest.mark.parametrize(
    ("name", "kind", "dtype", "symmetric", "scale_dtype"),
    [
        ("EXP-FP4-BF16", "fp4", "exp_fp4_bf16s", True, torch.bfloat16),
        ("EXP-FP4-FP16", "fp4", "exp_fp4_fp16s", True, torch.float16),
        ("EXP-INT4-UBF16", "int4", "exp_int4_bf16s", True, torch.bfloat16),
        ("EXP-INT4-UFP16", "int4", "exp_int4_fp16s", True, torch.float16),
        ("EXP-UINT4-ZP-BF16", "int4_zp", "exp_int4_zp_bf16s", False, torch.bfloat16),
        ("EXP-UINT4-ZP-FP16", "int4_zp", "exp_int4_zp_fp16s", False, torch.float16),
        ("EXP-INT4-BF16", "sint4", "exp_sint4_bf16s", True, torch.bfloat16),
        ("EXP-INT4-FP16", "sint4", "exp_sint4_fp16s", True, torch.float16),
    ],
)
def test_exp_scale_presets_and_rtn(name, kind, dtype, symmetric, scale_dtype):
    scheme = preset_name_to_scheme(name)
    assert scheme.bits == 4
    assert scheme.group_size == 16
    assert scheme.data_type == dtype
    assert scheme.sym is symmetric
    assert scheme_to_preset_name(scheme) == name

    quant_func, selected = get_quant_func(
        dtype=dtype, bits=4, sym=symmetric, group_size=16, iters=0
    )
    assert selected == f"rtn_{dtype}"

    torch.manual_seed(41)
    weight = torch.randn(4, 16)
    qdq, scale, zero_point = quant_func(weight, bits=4, group_size=16, scale_dtype=torch.float32)
    assert qdq.shape == weight.shape
    assert torch.isfinite(qdq).all()
    assert torch.isfinite(scale).all()
    assert scale.dtype == scale_dtype
    if kind == "int4":
        assert zero_point is None
        assert (qdq.abs() <= 7 * scale.float() + 1e-3).all()
    elif kind == "int4_zp":
        assert zero_point is not None


@pytest.mark.parametrize(
    "name",
    [
        "EXP-FP4-UE5M3",
        "EXP-INT4-E4M3",
        "EXP-INT4-UE5M3",
        "EXP-UINT4-ZP-UE5M3",
        "EXP-FP4-BF16",
        "EXP-FP4-FP16",
        "EXP-INT4-UBF16",
        "EXP-INT4-UFP16",
        "EXP-UINT4-ZP-BF16",
        "EXP-UINT4-ZP-FP16",
        "EXP-INT4-BF16",
        "EXP-INT4-FP16",
    ],
)
@pytest.mark.parametrize("case", [str.lower, str.swapcase])
def test_exp_scheme_names_are_case_insensitive(name, case):
    expected = preset_name_to_scheme(name)
    actual = preset_name_to_scheme(case(name))
    assert actual == expected

    parsed, is_auto_scheme, attrs = parse_scheme(case(name), {})
    assert parsed == name
    assert is_auto_scheme is False
    assert attrs["data_type"] == expected.data_type


@pytest.mark.parametrize(
    ("legacy_name", "canonical_name"),
    [
        ("EXP-FP4", "EXP-FP4-UE5M3"),
        ("EXP-UINT4-ZP-E5M3", "EXP-UINT4-ZP-UE5M3"),
        ("EXP-NVSINT4", "EXP-INT4-E4M3"),
        ("EXP-NVINT4", "EXP-INT4-UE5M3"),
        ("EXP-NVINT4-ZP", "EXP-UINT4-ZP-UE5M3"),
        ("EXP-FP4-BF16S", "EXP-FP4-BF16"),
        ("EXP-FP4-FP16S", "EXP-FP4-FP16"),
        ("EXP-INT4-BF16S", "EXP-INT4-UBF16"),
        ("EXP-INT4-FP16S", "EXP-INT4-UFP16"),
        ("EXP-INT4-ZP-BF16S", "EXP-UINT4-ZP-BF16"),
        ("EXP-INT4-ZP-FP16S", "EXP-UINT4-ZP-FP16"),
        ("EXP-SINT4-BF16S", "EXP-INT4-BF16"),
        ("EXP-SINT4-FP16S", "EXP-INT4-FP16"),
    ],
)
def test_exp_scale_legacy_aliases_resolve_to_canonical_names(legacy_name, canonical_name):
    legacy_scheme = preset_name_to_scheme(legacy_name)
    canonical_scheme = preset_name_to_scheme(canonical_name)

    assert legacy_scheme == canonical_scheme
    # EXP-FP4-UE5M3 shares its config with NVFP4_E5M3, which reverse lookup returns first.
    if canonical_name != "EXP-FP4-UE5M3":
        assert scheme_to_preset_name(legacy_scheme) == canonical_name
    assert parse_scheme(legacy_name.lower(), {})[0] == canonical_name


@pytest.mark.parametrize(
    ("name", "scale_dtype"),
    [("EXP-INT4-UBF16", torch.bfloat16), ("EXP-INT4-UFP16", torch.float16)],
)
def test_exp_int4_uses_signed_levels_without_zero_point(name, scale_dtype):
    scheme = preset_name_to_scheme(name)
    quant_func, _ = get_quant_func(
        dtype=scheme.data_type, bits=4, sym=scheme.sym, group_size=16, iters=0
    )
    weight = torch.zeros(1, 16)
    weight[0, 0] = -1
    weight[0, 1] = 1

    qdq, scale, zero_point = quant_func(weight, bits=4, group_size=16)

    assert scale.dtype == scale_dtype
    assert zero_point is None
    expected = torch.tensor([-7.0, 7.0]) * scale.float().reshape(-1)[0]
    torch.testing.assert_close(qdq[0, :2], expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "name",
    [
        "EXP-INT4-E4M3",
        "EXP-INT4-UE5M3",
        "EXP-UINT4-ZP-UE5M3",
        "EXP-FP4-BF16",
        "EXP-FP4-FP16",
        "EXP-INT4-UBF16",
        "EXP-INT4-UFP16",
        "EXP-UINT4-ZP-BF16",
        "EXP-UINT4-ZP-FP16",
        "EXP-INT4-BF16",
        "EXP-INT4-FP16",
    ],
)
def test_exp_presets_quantize_activations_like_weights(name):
    scheme = preset_name_to_scheme(name)
    assert scheme.act_bits == 4
    assert scheme.act_group_size == scheme.group_size == 16
    assert scheme.act_data_type == scheme.data_type
    assert scheme.act_sym is scheme.sym


def test_exp_nvint4_uses_symmetric_seven_levels():
    from auto_round.data_type.exp import quant_tensor_exp_nvint4

    weight = torch.zeros(2, 16)
    weight[0, 0], weight[0, 1] = 1.0, -0.5
    weight[1, 0], weight[1, 1] = -1.0, 0.5
    qdq, scale, _ = quant_tensor_exp_nvint4(weight, bits=4, group_size=16)
    codes = torch.round(qdq / scale)
    assert codes.max() == 7 and codes.min() == -7


def test_exp_fp4_fake_backend_small_model_forward():
    from types import SimpleNamespace

    from transformers import LlamaConfig, LlamaForCausalLM

    from auto_round.experimental.qmodules.fake import FakeActQuantLinear
    from auto_round.inference.convert_model import convert_hf_model

    scheme = preset_name_to_scheme("EXP-FP4-BF16")
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        vocab_size=64,
    )
    model = LlamaForCausalLM(config)
    layer = model.model.layers[0].self_attn.q_proj
    weight_quantizer, _ = get_quant_func(
        dtype=scheme.data_type, bits=scheme.bits, sym=scheme.sym, group_size=scheme.group_size, iters=0
    )
    quantized_weight, scale, _ = weight_quantizer(layer.weight.data, bits=4, group_size=16)
    layer.weight.data.copy_(quantized_weight)
    assert scale.dtype == torch.bfloat16

    quantization_config = scheme.to_dict()
    quantization_config.update({"quant_method": "auto-round", "packing_format": "auto_round:fake"})
    model.config.quantization_config = SimpleNamespace(**quantization_config)
    model, used_backends = convert_hf_model(model, target_device="cpu")

    fake_layer = model.model.layers[0].self_attn.q_proj
    assert used_backends == ["auto_round:fake"]
    assert isinstance(fake_layer, FakeActQuantLinear)
    hidden = torch.randn(2, 32)
    assert not torch.equal(fake_layer.qdq_input(hidden), hidden)
    logits = model(input_ids=torch.randint(0, 64, (1, 4))).logits
    assert logits.shape == (1, 4, 64)
    assert torch.isfinite(logits).all()