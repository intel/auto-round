import shutil
import tempfile

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round import schemes as ar_schemes
from auto_round.experimental import qmodules as ar_qmodules
from auto_round.export.export_to_autoround import qlinear_fp as ar_qlinear_fp
from auto_round.formats import AutoRoundExportFormat
from auto_round.testing_utils import has_module

from ..helpers import get_model_path

testing_schemes = [
    AutoRoundExportFormat.MXFP8.value,
    AutoRoundExportFormat.MXFP4.value,
    AutoRoundExportFormat.NVFP4.value,
]
QMODULE_MAPPING = {
    AutoRoundExportFormat.MXFP8.value: ar_qmodules.MXFP8QuantLinear,
    AutoRoundExportFormat.MXFP4.value: ar_qmodules.MXFP4QuantLinear,
    AutoRoundExportFormat.NVFP4.value: ar_qmodules.NVFP4QuantLinear,
}


@pytest.mark.parametrize("scheme", testing_schemes)
@torch.inference_mode()
def test_e2e_quant_and_infer(scheme, tiny_qwen_model_path):
    # Use a temporary directory for saving the quantized model
    with tempfile.TemporaryDirectory() as temp_dir:

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            tiny_qwen_model_path,
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
