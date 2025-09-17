import pytest
import torch
from _test_helpers import is_pytest_mode_compile, is_pytest_mode_lazy

from auto_round.utils import is_hpu_supported


def run_opt_125m_on_hpu():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from auto_round import AutoRound

    model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    bits, group_size, sym = 4, 128, False
    autoround = AutoRound(
        model,
        tokenizer,
        bits=bits,
        group_size=group_size,
        sym=sym,
        iters=2,
        seqlen=2,
    )
    q_model, qconfig = autoround.quantize()
    assert q_model is not None, "Expected q_model to be not None"


@pytest.mark.skipif(not is_hpu_supported(), reason="HPU is not supported")
@pytest.mark.skipif(not is_pytest_mode_lazy(), reason="Only for lazy mode")
def test_opt_125m_lazy_mode():
    run_opt_125m_on_hpu()


@pytest.mark.skipif(not is_hpu_supported(), reason="HPU is not supported")
@pytest.mark.skipif(not is_pytest_mode_compile(), reason="Only for compile mode")
def test_opt_125m_compile_mode():
    torch._dynamo.reset()
    run_opt_125m_on_hpu()


def test_import():
    from auto_round import AutoRound
    from auto_round.export.export_to_itrex.export import WeightOnlyLinear, save_quantized_as_itrex


@pytest.mark.parametrize(
    "data_type",
    ["fp8_to_int_sym"],
)
def test_w4a8(data_type):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from auto_round import AutoRound

    model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    autoround = AutoRound(
        model,
        tokenizer,
        bits=4,
        group_size=128,
        iters=2,
        seqlen=2,
        data_type=data_type,
        act_data_type="fp8_sym",
        act_bits=8,
        nsamples=1,
        act_dynamic=False,
    )
    q_model, qconfig = autoround.quantize()
    assert q_model is not None, "Expected q_model to be not None"
