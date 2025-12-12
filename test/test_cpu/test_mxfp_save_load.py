import shutil
import tempfile

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from auto_round import AutoRound
from auto_round import schemes as ar_schemes
from auto_round.experimental import qmodules as ar_qmodules
from auto_round.export.export_to_autoround import AutoRoundExportFormat
from auto_round.export.export_to_autoround import qlinear_fp as ar_qlinear_fp
from auto_round.inference.backend import MX_TENSOR_DATA_TYPES
from auto_round.testing_utils import has_module

testing_scheme_name_lst = [
    AutoRoundExportFormat.MXFP8.value,
    AutoRoundExportFormat.MXFP4.value,
]
QMODULE_MAPPING = {
    AutoRoundExportFormat.MXFP8.value: ar_qmodules.MXFP8QuantLinear,
    AutoRoundExportFormat.MXFP4.value: ar_qmodules.MXFP4QuantLinear,
}
SCHEMES_MAPPING = {
    AutoRoundExportFormat.MXFP8.value: ar_schemes.MXFP8,
    AutoRoundExportFormat.MXFP4.value: ar_schemes.MXFP4,
}


@pytest.mark.parametrize("scheme_name", testing_scheme_name_lst)
@pytest.mark.parametrize("weight_data_type", MX_TENSOR_DATA_TYPES)
@pytest.mark.parametrize("act_data_type", MX_TENSOR_DATA_TYPES)
@torch.inference_mode()
def test_e2e_quant_and_load(scheme_name, weight_data_type, act_data_type):
    # Use a temporary directory for saving the quantized model
    with tempfile.TemporaryDirectory() as temp_dir:
        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen2.5-0.5B-Instruct"
        config = AutoConfig.from_pretrained(model_name)
        config.num_hidden_layers = 2  # Use a smaller model for testing
        # Fix configuration validation issues
        config.layer_types = config.layer_types[: config.num_hidden_layers]

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = Qwen2ForCausalLM(config)
        scheme = SCHEMES_MAPPING[scheme_name]
        scheme.data_type = weight_data_type
        scheme.act_data_type = act_data_type
        # Initialize AutoRound for quantization
        autoround = AutoRound(
            model,
            tokenizer,
            scheme=scheme,
            iters=0,
            nsamples=2,
        )

        # Quantize and save the model to the temporary directory
        quantized_model_path = f"{temp_dir}/tmp_autoround"
        autoround.quantize_and_save(format="auto_round", output_dir=quantized_model_path)

        # Perform inference with the quantized model
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            torch_dtype="auto",
        )
        model.eval()
        assert has_module(
            model, QMODULE_MAPPING[scheme_name]
        ), f"Expected {QMODULE_MAPPING[scheme_name].__name__} in the model."
