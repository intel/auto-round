import shutil

import pytest
import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Llama4ForConditionalGeneration
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

from auto_round import AutoRound


def quantize_model(model, output_dir, scheme, iters=0, ignore_layers="self_attn,router,lm_head,mlp.gate"):
    """Helper function to quantize the model with the given scheme."""
    disable_opt_rtn = True if iters == 0 else False
    autoround = AutoRound(
        model,
        scheme=scheme,
        nsamples=2,
        iters=iters,
        seqlen=32,
        ignore_layers=ignore_layers,
        disable_opt_rtn=disable_opt_rtn,
    )
    quantized_model, save_folder = autoround.quantize_and_save(format="auto_round", output_dir=output_dir)
    return quantized_model


def count_modules_by_type(model, target_module_name_or_class):
    """Helper function to count modules of a specific type in the model."""
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(target_module_name_or_class, str):
            if target_module_name_or_class == module.__class__.__name__:
                cnt += 1
        else:
            if isinstance(module, target_module_name_or_class):
                cnt += 1
    return cnt


@pytest.mark.parametrize("scheme", ["MXFP4", "MXFP8"])
def test_gptoss(scheme, tiny_gpt_oss_model_path):
    config = AutoConfig.from_pretrained(tiny_gpt_oss_model_path, trust_remote_code=True)
    output_dir = "./saved"
    quantized_model = quantize_model(tiny_gpt_oss_model_path, output_dir, scheme, ignore_layers="self_attn,lm_head")

    # Ensure the quantized model is not None
    assert quantized_model is not None, "Quantized model should not be None."
    from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear
    from auto_round.modeling.fused_moe.gpt_oss import GPTOssSingleExpert

    single_expert_cnt = count_modules_by_type(quantized_model, GPTOssSingleExpert)
    single_expert_cnt += count_modules_by_type(quantized_model, "_ExpertContainer")  # for Transformers >= 5.0.0
    quant_linear_cnt = count_modules_by_type(quantized_model, QuantLinear)
    assert (
        single_expert_cnt == config.num_local_experts
    ), f"Expected {config.num_local_experts} GPTOssSingleExpert modules, found {single_expert_cnt}."
    assert (
        quant_linear_cnt == config.num_hidden_layers * 3 * config.num_local_experts
    ), f"Expected {config.num_hidden_layers * 3 * config.num_local_experts} QuantLinear modules, found {quant_linear_cnt}."

    if scheme == "MXFP4":
        loaded_model = GptOssForCausalLM.from_pretrained(output_dir)
        for n, m in quantized_model.named_modules():
            if m.__class__.__name__ == "QuantLinear":
                loaded_m = loaded_model.get_submodule(n)
                assert (loaded_m.weight_packed.to("cpu") == m.weight_packed.to("cpu")).all()
    inp = torch.randint(0, 100, (1, 32))
    with torch.inference_mode():
        loaded_out = loaded_model(inp)
    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)


def test_llama4(tiny_llama4_model_path):
    output_dir = "./tmp/test_quantized_llama4"
    quantized_model = quantize_model(tiny_llama4_model_path, output_dir, "MXFP4", ignore_layers="self_attn,lm_head")

    # Ensure the quantized model is not None
    assert quantized_model is not None, "Quantized model should not be None."

    loaded_model = Llama4ForConditionalGeneration.from_pretrained(output_dir)
    for n, m in quantized_model.named_modules():
        if m.__class__.__name__ == "QuantLinear":
            loaded_m = loaded_model.get_submodule(n)
            assert (loaded_m.weight_packed.to("cpu") == m.weight_packed.to("cpu")).all()
    inp = torch.randint(0, 100, (1, 32))
    with torch.inference_mode():
        loaded_out = loaded_model(inp)
    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)


def test_qwen3_vl_moe_mxfp(tiny_qwen3_vl_moe_model_path):
    output_dir = "/tmp/test_quantized_qwen3_vl_moe"
    from auto_round.utils import llm_load_model

    model = llm_load_model(tiny_qwen3_vl_moe_model_path)
    print(model)

    autoround = AutoRound(
        tiny_qwen3_vl_moe_model_path,
        scheme="MXFP4",
        nsamples=2,
        seqlen=32,
        iters=0,
        disable_opt_rtn=True,
        ignore_layers="self_attn,lm_head, mlp.gate",
    )
    quantized_model, _ = autoround.quantize_and_save(format="auto_round", output_dir=output_dir)
    assert quantized_model is not None, "Quantized model should not be None."
    loaded_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(output_dir, device_map="cpu")

    for n, m in quantized_model.named_modules():
        if m.__class__.__name__ == "QuantLinear":
            loaded_m = loaded_model.get_submodule(n)
            assert (loaded_m.weight_packed.to("cpu") == m.weight_packed.to("cpu")).all()
    inp = torch.randint(0, 100, (1, 32))
    with torch.inference_mode():
        loaded_out = loaded_model(inp)
    shutil.rmtree(output_dir, ignore_errors=True)


class _FakeMoELinear(nn.Module):
    """Mimics the fused MoELinear used in Step3p5MoEMLP."""

    def __init__(self, num_experts, in_features, out_features):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(num_experts, out_features, in_features))


class Step3p5MoEMLP(nn.Module):
    """Mock of the original Step3p5MoEMLP matched by class name."""

    def __init__(self, hidden_size=32, intermediate_size=64, num_experts=4, top_k=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.limit = None
        self.need_fp32_gate = False
        self.routed_scaling_factor = 1.0
        self.use_moe_router_bias = False
        self.custom_routing_function = None
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.gate_proj = _FakeMoELinear(num_experts, hidden_size, intermediate_size)
        self.up_proj = _FakeMoELinear(num_experts, hidden_size, intermediate_size)
        self.down_proj = _FakeMoELinear(num_experts, intermediate_size, hidden_size)


def test_step3p5_moe_replacement():
    """Verify that apply_replacements swaps Step3p5MoEMLP → LinearStep3p5MoEMLP."""
    from auto_round.modeling.fused_moe.replace_modules import apply_replacements, materialize_model_
    from auto_round.modeling.fused_moe.step3_5_moe import LinearStep3p5MoEMLP, Step3p5ExpertMLP

    num_experts, hidden_size = 4, 32
    moe = Step3p5MoEMLP(hidden_size=hidden_size, num_experts=num_experts)
    # wrap in a model with config.model_type so the registry can find it
    model = nn.Module()
    model.config = type("Cfg", (), {"model_type": "step3p5"})()
    model.moe = moe
    model.add_module("moe", moe)

    orig_weights = {k: moe.get_parameter(k).clone() for k in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")}

    apply_replacements(model, auto_detect_moe=False)
    assert isinstance(model.moe, LinearStep3p5MoEMLP)
    assert count_modules_by_type(model, Step3p5ExpertMLP) == num_experts

    materialize_model_(model)
    for i, expert in enumerate(model.moe.experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w = getattr(expert, proj).weight
            assert w.device.type != "meta"
            assert torch.equal(w, orig_weights[f"{proj}.weight"][i])

    out = model.moe(torch.randn(1, 4, hidden_size))
    assert out.shape == (1, 4, hidden_size)
