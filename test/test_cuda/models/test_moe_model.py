import shutil

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Llama4ForConditionalGeneration
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3ForCausalLM, Qwen3MLP
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

from auto_round import AutoRound
from auto_round.modelling.replace_modules import ReplacementModuleBase
from auto_round.utils import LazyImport, logger, unsupported_meta_device


@pytest.fixture
def setup_gpt_oss():
    """Fixture to set up the GPT-OSS model and tokenizer."""
    model_name = "/models/gpt-oss-20b-BF16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_hidden_layers = 1  # Reduce layers for testing
    model = GptOssForCausalLM(config)
    output_dir = "test_quantized_gpt_oss"
    return model, tokenizer, output_dir, config


@pytest.fixture
def setup_glm4_moe_lite():
    """Fixture to set up the glm4_moe_lite model and tokenizer."""
    model_name = "/dataset/GLM-4.7-Flash/"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2  # Reduce layers for testing
    model = AutoModelForCausalLM.from_config(config)
    output_dir = "/tmp/test_quantized_glm4_moe_lite"
    return model, tokenizer, output_dir, config


@pytest.fixture
def setup_llama4():
    """Fixture to set up the llama4 model and tokenizer."""
    model_name = "/dataset/Llama-4-Scout-17B-16E-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.vision_config.num_hidden_layers = 1  # Reduce layers for testing
    config.text_config.num_hidden_layers = 1
    model = Llama4ForConditionalGeneration(config)
    output_dir = "test_quantized_llama4"
    return model, tokenizer, output_dir, config


@pytest.fixture
def setup_qwen3_vl_moe():
    """Fixture to set up the qwen3_vl_moe model and tokenizer."""
    model_name = "/models/Qwen3-VL-30B-A3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.vision_config.num_hidden_layers = 1
    config.text_config.num_hidden_layers = 1
    config.num_hidden_layers = 1  # Reduce layers for testing
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLMoeForConditionalGeneration(config)
    output_dir = "test_quantized_qwen3_vl_moe"
    return model, tokenizer, processor, output_dir, config


def quantize_model(model, tokenizer, output_dir, scheme, iters=0):
    """Helper function to quantize the model with the given scheme."""
    autoround = AutoRound(
        model,
        tokenizer,
        scheme=scheme,
        nsamples=2,
        iters=iters,
        seqlen=32,
        ignore_layers="self_attn,router,lm_head,mlp.gate",
    )
    quantized_model, save_folder = autoround.quantize_and_save(format="auto_round", output_dir=output_dir)
    return quantized_model


def test_gptoss(setup_gpt_oss):
    model, tokenizer, output_dir, config = setup_gpt_oss

    # Below parameter is set to be same as the full model
    # Remove it to avoid mismatch during quantized model loading
    delattr(model.config, "layer_types")

    quantized_model = quantize_model(model, tokenizer, output_dir, "MXFP4")

    # Ensure the quantized model is not None
    assert quantized_model is not None, "Quantized model should not be None."

    loaded_model = GptOssForCausalLM.from_pretrained(output_dir)
    quantized_model.to("cuda")
    loaded_model.to("cuda")
    for n, m in quantized_model.named_modules():
        if m.__class__.__name__ == "QuantLinear":
            loaded_m = loaded_model.get_submodule(n)
            assert (loaded_m.weight_packed == m.weight_packed).all()

    inp = torch.randint(0, 100, (1, 64)).to("cuda")
    with torch.inference_mode():
        loaded_out = loaded_model(inp)

    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)


def test_llama4(setup_llama4):
    model, tokenizer, output_dir, config = setup_llama4

    # Below parameters are set to be same as the full model
    # Remove them to avoid mismatch during quantized model loading
    model.config.text_config.no_rope_layers = []
    delattr(model.config.text_config, "moe_layers")
    delattr(model.config.text_config, "layer_types")

    quantized_model = quantize_model(model, tokenizer, output_dir, "MXFP4")

    # Ensure the quantized model is not None
    assert quantized_model is not None, "Quantized model should not be None."

    loaded_model = Llama4ForConditionalGeneration.from_pretrained(output_dir)
    quantized_model.to("cuda")
    loaded_model.to("cuda")
    for n, m in quantized_model.named_modules():
        if m.__class__.__name__ == "QuantLinear":
            loaded_m = loaded_model.get_submodule(n)
            assert (loaded_m.weight_packed == m.weight_packed).all()

    inp = torch.randint(0, 100, (1, 64)).to("cuda")
    with torch.inference_mode():
        loaded_out = loaded_model(inp)

    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)


class NewQwen3MLP(ReplacementModuleBase):
    def __init__(self, original: Qwen3MLP, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.new_gate_proj = original.gate_proj
        self.new_up_proj = original.up_proj
        self.new_down_proj = original.down_proj
        self.act_fn = nn.GELU()

    def forward(self, x):
        gate = self.new_gate_proj(x)
        up = self.new_up_proj(x)
        gate_act = self.act_fn(gate)
        act = gate_act * up
        out = self.new_down_proj(act)
        return out

    @classmethod
    def original_module_class(cls) -> str:
        return "Qwen3MLP"

    @classmethod
    def from_original(cls, original: Qwen3MLP, config: Qwen3Config):
        return cls(original, config)


@pytest.fixture
def setup_qwen3():
    """Fixture to set up the qwen3 model and tokenizer."""
    model_name = "/models/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2  # Reduce layers for testing
    config.layer_types = config.layer_types[:2]  # Reduce layers for testing
    model = Qwen3ForCausalLM(config)
    output_dir = "test_quantized_qwen3"
    return model, tokenizer, output_dir, config


def test_qwen3_vl_moe_mxfp(setup_qwen3_vl_moe):
    model, tokenizer, processor, output_dir, config = setup_qwen3_vl_moe
    autoround = AutoRound(
        model,
        tokenizer=tokenizer,
        processor=processor,
        scheme="MXFP4",
        nsamples=2,
        seqlen=32,
        iters=1,
        ignore_layers="self_attn,lm_head,mlp.gate",
    )
    quantized_model, _ = autoround.quantize_and_save(format="auto_round", output_dir=output_dir)
    assert quantized_model is not None, "Quantized model should not be None."
    loaded_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(output_dir)
    loaded_model.to("cuda")
    quantized_model.to("cuda")
    for n, m in quantized_model.named_modules():
        if m.__class__.__name__ == "QuantLinear":
            loaded_m = loaded_model.get_submodule(n)
            assert (loaded_m.weight_packed == m.weight_packed).all()

    inp = torch.randint(0, 100, (1, 64)).to("cuda")
    with torch.inference_mode():
        loaded_out = loaded_model(inp)

    # test generation
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(device=loaded_model.device)
    print(tokenizer.decode(loaded_model.generate(**inputs, max_new_tokens=50)[0]))
    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)


def has_module(model: torch.nn.Module, module_name: str) -> bool:
    for n, m in model.named_modules():
        if module_name in n:
            return True
    return False


def test_register_module_out_of_tree_base():
    from auto_round.modelling.replace_modules import ReplacementModuleBase

    for name, subclass in ReplacementModuleBase._replacement_registry.items():
        if name == "Qwen3MLP":
            assert subclass == NewQwen3MLP, "Qwen3MLP not registered correctly."


def test_register_module_out_of_tree_model(setup_qwen3):
    model, tokenizer, output_dir, config = setup_qwen3
    quantized_model = quantize_model(model, tokenizer, output_dir, "MXFP4")
    # Ensure the quantized model is not None
    assert quantized_model is not None, "Quantized model should not be None."
    check_module_names = ["new_gate_proj", "new_up_proj", "new_down_proj"]
    for name in check_module_names:
        assert has_module(quantized_model, name), f"Module {name} not found in quantized model."
    loaded_model = Qwen3ForCausalLM.from_pretrained(output_dir)
    quantized_model.to("cuda")

    for name in check_module_names:
        assert has_module(loaded_model, name), f"Module {name} not found in loaded model."
    loaded_model.to("cuda")


def test_glm4_moe_lite(setup_glm4_moe_lite):
    model, tokenizer, output_dir, config = setup_glm4_moe_lite
    autoround = AutoRound(
        model,
        tokenizer=tokenizer,
        scheme="W4A16",
        nsamples=2,
        seqlen=32,
        iters=1,
        ignore_layers="shared_experts,layers.0.mlp",
    )
    quantized_model, _ = autoround.quantize_and_save(format="auto_round", output_dir=output_dir)
    assert quantized_model is not None, "Quantized model should not be None."
    loaded_model = AutoModelForCausalLM.from_pretrained(output_dir, device_map="auto")

    for n, m in quantized_model.named_modules():
        if m.__class__.__name__ == "QuantLinear":
            loaded_m = loaded_model.get_submodule(n)
            assert (loaded_m.qweight.to("cuda") == m.qweight.to("cuda")).all()
    # test generation
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(device=loaded_model.device)
    print(tokenizer.decode(loaded_model.generate(**inputs, max_new_tokens=50)[0]))
    # test vllm loading
    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # Create an LLM.
    QUANTIZATION = "auto-round"  # quantized_model_path
    llm = LLM(model=output_dir, quantization=QUANTIZATION, trust_remote_code=True, tensor_parallel_size=4)
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # if "France" in prompt:
        assert "!!!" not in generated_text
        print(f"{prompt}: {generated_text}")
    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)
