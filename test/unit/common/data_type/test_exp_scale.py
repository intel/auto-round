import pytest
import torch

from auto_round.data_type.utils import get_quant_func
from auto_round.schemes import preset_name_to_scheme


@pytest.mark.parametrize(
    ("name", "kind", "dtype", "symmetric", "scale_dtype"),
    [
        ("EXP-FP4-BF16S", "fp4", "exp_fp4_bf16s", True, torch.bfloat16),
        ("EXP-FP4-FP16S", "fp4", "exp_fp4_fp16s", True, torch.float16),
        ("EXP-INT4-BF16S", "int4", "exp_int4_bf16s", True, torch.bfloat16),
        ("EXP-INT4-FP16S", "int4", "exp_int4_fp16s", True, torch.float16),
        ("EXP-INT4-ZP-BF16S", "int4_zp", "exp_int4_zp_bf16s", False, torch.bfloat16),
        ("EXP-INT4-ZP-FP16S", "int4_zp", "exp_int4_zp_fp16s", False, torch.float16),
        ("EXP-SINT4-BF16S", "sint4", "exp_sint4_bf16s", True, torch.bfloat16),
        ("EXP-SINT4-FP16S", "sint4", "exp_sint4_fp16s", True, torch.float16),
    ],
)
def test_exp_scale_presets_and_rtn(name, kind, dtype, symmetric, scale_dtype):
    scheme = preset_name_to_scheme(name)
    assert scheme.bits == 4
    assert scheme.group_size == 16
    assert scheme.data_type == dtype
    assert scheme.sym is symmetric

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
    ("name", "scale_dtype"),
    [("EXP-INT4-BF16S", torch.bfloat16), ("EXP-INT4-FP16S", torch.float16)],
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


def test_exp_scale_presets_keep_fake_activation_config_isolated():
    fp4 = preset_name_to_scheme("EXP-FP4-BF16S")
    assert fp4.act_data_type == "nvfp4_v2"
    assert fp4.act_bits == 4

    int4 = preset_name_to_scheme("EXP-INT4-BF16S")
    int4_zp = preset_name_to_scheme("EXP-INT4-ZP-BF16S")
    sint4 = preset_name_to_scheme("EXP-SINT4-BF16S")
    assert int4.act_bits == sint4.act_bits == 16
    assert int4.sym is True
    assert int4_zp.sym is False
    assert int4.data_type != sint4.data_type


def test_exp_fp4_fake_backend_small_model_forward():
    from types import SimpleNamespace

    from transformers import LlamaConfig, LlamaForCausalLM

    from auto_round.experimental.qmodules.fake import FakeActQuantLinear
    from auto_round.inference.convert_model import convert_hf_model

    scheme = preset_name_to_scheme("EXP-FP4-BF16S")
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