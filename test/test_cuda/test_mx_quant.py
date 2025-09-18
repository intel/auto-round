import shutil
import tempfile

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round import schemes as ar_schemes
from auto_round.experimental import qmodules as ar_qmodules
from auto_round.export.export_to_autoround import AutoRoundFormat
from auto_round.export.export_to_autoround import qlinear_fp as ar_qlinear_fp

mx_schemes = [AutoRoundFormat.MXFP8.value, AutoRoundFormat.MXFP4.value]
QMODULE_MAPPING = {
    AutoRoundFormat.MXFP8.value: ar_qmodules.MXFP8QuantLinear,
    AutoRoundFormat.MXFP4.value: ar_qmodules.MXFP4QuantLinear,
}


def has_module(model: torch.nn.Module, target_module_type: torch.nn.Module) -> bool:
    """Check if the model contains a specific module type."""
    for _, module in model.named_modules():
        if isinstance(module, target_module_type):
            return True
    return False


@pytest.mark.parametrize("scheme", mx_schemes)
@torch.inference_mode()
def test_e2e_quant_and_infer(scheme):
    # Use a temporary directory for saving the quantized model
    with tempfile.TemporaryDirectory() as temp_dir:
        model_name = "/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct/"

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype="auto",
            trust_remote_code=True,
        )

        # Initialize AutoRound for quantization
        autoround = AutoRound(
            model,
            tokenizer,
            scheme=scheme,
            iters=0,
            nsamples=2,
        )

        # Quantize and save the model to the temporary directory
        quantized_model_path = f"{temp_dir}/tmp_autoround_{scheme}"
        autoround.quantize_and_save(format="auto_round", output_dir=quantized_model_path)

        # Perform inference with the quantized model
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            torch_dtype="auto",
            # low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
        assert has_module(model, QMODULE_MAPPING[scheme]), f"Expected {QMODULE_MAPPING[scheme].__name__} in the model."

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        prompt = "Ai is "

        # Tokenize the input prompt
        encode = tokenizer.encode(prompt, return_tensors="pt")

        # Generate output tokens
        output_tokens = model.generate(
            encode,
            max_length=30,
        )
        output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Print and validate the output
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
        assert output is not None, "Output should not be None"
