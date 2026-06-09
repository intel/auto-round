import os

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.calib_dataset import get_dataset

from ...envs import multi_card

recipe_str = """
quant_stage:
    quant_modifiers:
        AutoRoundModifier:
            ignore: ["lm_head"]
            iters: 10
            config_groups:
                group_0:
                    targets:
                        - "Linear"
                    input_activations: null
                    output_activations: null
                    weights:
                        num_bits: 4
                        type: "int"
                        symmetric: true
                        strategy: group
                        group_size: 128
"""

recipe_modifier_full = AutoRoundModifier(
    ignore=["lm_head"],
    iters=10,
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(num_bits=4, strategy="group", group_size=128),
        )
    },
)
recipe_modifier_nvfp4 = AutoRoundModifier(
    ignore=["lm_head"],
    iters=2,
    scheme="NVFP4",
)

recipe_modifier_mxfp4 = AutoRoundModifier(
    ignore=["lm_head"],
    iters=0,
    scheme="MXFP4",
)

w8a8_dynamic_recipe_modifier = AutoRoundModifier(
    ignore=["lm_head"],
    iters=0,
    enable_torch_compile=False,
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(num_bits=8, type="float", strategy="channel"),
            input_activations=QuantizationArgs(num_bits=8, type="float", strategy="token", dynamic=True),
        )
    },
)

w8a8_static_recipe_modifier = AutoRoundModifier(
    ignore=["lm_head"],
    iters=0,
    enable_torch_compile=False,
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(num_bits=8, type="float", strategy="tensor"),
            input_activations=QuantizationArgs(num_bits=8, type="float", strategy="tensor"),
        )
    },
)


@pytest.mark.parametrize(
    "recipe",
    [
        recipe_str,
        recipe_modifier_full,
        recipe_modifier_nvfp4,
        recipe_modifier_mxfp4,
    ],
)
def test_oneshot_application(tiny_tiny_llama_model_path, recipe, tmp_path):
    output = tmp_path / "oneshot_output"
    model = tiny_tiny_llama_model_path
    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset = get_dataset(
        tokenizer=tokenizer,
        seqlen=1024,
        nsamples=32,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    oneshot(
        model=model,
        dataset=dataset,
        output_dir=output,
        recipe=recipe,
    )
    model_loaded = AutoModelForCausalLM.from_pretrained(output, device_map=device)

    # Check that the model is quantized
    # decompress() will attach a quantization_config to the model
    # as we decompress right away
    quantization_config = model_loaded.config.quantization_config.quantization_config
    assert quantization_config is not None

    # check config is set properly
    assert "lm_head" in quantization_config.ignore
    assert len(quantization_config.config_groups) == 1
    quant_scheme = quantization_config.config_groups["group_0"]
    assert isinstance(quant_scheme, QuantizationScheme)

    weight_args = quantization_config.config_groups["group_0"].weights
    assert isinstance(weight_args, QuantizationArgs)
    assert weight_args.num_bits == 4

    # Check a specific layer is quantized
    targeted_linear_layer = model_loaded.model.layers[2].self_attn.q_proj
    assert hasattr(targeted_linear_layer, "quantization_scheme")

    # Check lm-head is not quantized
    not_targeted = model_loaded.lm_head
    assert not hasattr(not_targeted, "quantization_scheme")


@multi_card
def test_oneshot_with_device_ids(tiny_tiny_llama_model_path, tmp_path):
    output = tmp_path / "oneshot_output"
    model = tiny_tiny_llama_model_path
    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset = get_dataset(
        tokenizer=tokenizer,
        seqlen=512,
        nsamples=4,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    recipe = AutoRoundModifier(
        ignore=["lm_head"],
        iters=10,
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(num_bits=4, strategy="group", group_size=128),
            )
        },
        device_ids="0,1",
    )

    oneshot(
        model=model,
        dataset=dataset,
        output_dir=output,
        recipe=recipe,
    )
    model_loaded = AutoModelForCausalLM.from_pretrained(output, device_map=device)

    # Check that the model is quantized
    # decompress() will attach a quantization_config to the model
    # as we decompress right away
    quantization_config = model_loaded.config.quantization_config.quantization_config
    assert quantization_config is not None

    # check config is set properly
    assert "lm_head" in quantization_config.ignore
    assert len(quantization_config.config_groups) == 1
    quant_scheme = quantization_config.config_groups["group_0"]
    assert isinstance(quant_scheme, QuantizationScheme)

    weight_args = quantization_config.config_groups["group_0"].weights
    assert isinstance(weight_args, QuantizationArgs)
    assert weight_args.num_bits == 4

    # Check a specific layer is quantized
    targeted_linear_layer = model_loaded.model.layers[2].self_attn.q_proj
    assert hasattr(targeted_linear_layer, "quantization_scheme")

    # Check lm-head is not quantized
    not_targeted = model_loaded.lm_head
    assert not hasattr(not_targeted, "quantization_scheme")


@pytest.mark.parametrize(
    "recipe",
    [w8a8_dynamic_recipe_modifier, w8a8_static_recipe_modifier],
)
def test_rtn_oneshot(recipe, tmp_path, tiny_tiny_llama_model_path):
    output = tmp_path / "oneshot_output"
    model = tiny_tiny_llama_model_path
    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset = get_dataset(
        tokenizer=tokenizer,
        seqlen=1024,
        nsamples=32,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    oneshot(
        model=model,
        dataset=dataset,
        output_dir=output,
        recipe=recipe,
    )
    model_loaded = AutoModelForCausalLM.from_pretrained(output, device_map=device)

    quantization_config = model_loaded.config.quantization_config.quantization_config
    assert quantization_config is not None

    # check config is set properly
    assert "lm_head" in quantization_config.ignore
    assert len(quantization_config.config_groups) == 1
    quant_scheme = quantization_config.config_groups["group_0"]
    assert isinstance(quant_scheme, QuantizationScheme)

    weight_args = quantization_config.config_groups["group_0"].weights
    act_args = quantization_config.config_groups["group_0"].input_activations
    assert isinstance(weight_args, QuantizationArgs)
    assert weight_args.num_bits == recipe.config_groups["group_0"].weights.num_bits
    assert weight_args.strategy == recipe.config_groups["group_0"].weights.strategy
    if act_args is not None:
        assert act_args.num_bits == recipe.config_groups["group_0"].input_activations.num_bits
        assert act_args.strategy == recipe.config_groups["group_0"].input_activations.strategy

    # Check a specific layer is quantized
    targeted_linear_layer = model_loaded.model.layers[2].self_attn.q_proj
    assert hasattr(targeted_linear_layer, "quantization_scheme")

    # Check lm-head is not quantized
    not_targeted = model_loaded.lm_head
    assert not hasattr(not_targeted, "quantization_scheme")


# ---------------------------------------------------------------------------
# AWQ W8A8 llm_compressor export config args
# ---------------------------------------------------------------------------
def test_llmc_awq_w8a8_export_config_args(tiny_opt_model_path, tmp_path):
    """W8A8 AWQ → llm_compressor: verify compressed-tensors metadata fields."""
    save_dir = str(tmp_path / "saved")
    ar = AutoRound(
        tiny_opt_model_path,
        scheme="INT8",
        algorithm="awq",
        nsamples=2,
        seqlen=32,
        batch_size=2,
    )
    _, save_path = ar.quantize_and_save(output_dir=save_dir, format="llm_compressor")

    config = AutoConfig.from_pretrained(save_path, trust_remote_code=True)
    qconfig = config.quantization_config

    assert qconfig["quant_method"] == "compressed-tensors"

    group0 = qconfig["config_groups"]["group_0"]
    # Weight args
    assert group0["weights"]["num_bits"] == 8
    assert group0["weights"]["type"] == "int"
    assert group0["weights"]["symmetric"] is True
    # Activation args
    assert group0["input_activations"]["num_bits"] == 8
    # Targets
    targets = group0.get("targets")
    assert targets is not None and len(targets) > 0

    # QuantLinear check: verify saved weights are int8 with per-channel scales
    from safetensors import safe_open

    st_files = [f for f in os.listdir(save_path) if f.endswith(".safetensors")]
    assert len(st_files) > 0, f"No safetensors files in {save_path}"
    with safe_open(os.path.join(save_path, st_files[0]), framework="pt") as f:
        weight = f.get_tensor("model.decoder.layers.0.self_attn.k_proj.weight")
        assert weight.dtype == torch.int8, f"Expected int8 weight, got {weight.dtype}"
        scale = f.get_tensor("model.decoder.layers.0.self_attn.k_proj.weight_scale")
        assert scale.shape[1] == 1, f"Expected per-channel scale shape (out, 1), got {scale.shape}"
