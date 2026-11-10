import tempfile

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round import schemes as ar_schemes
from auto_round.experimental import qmodules as ar_qmodules
from auto_round.export.formats import BackendDataType
from auto_round.inference.backend import MX_TENSOR_DATA_TYPES

from ...envs import has_module

testing_scheme_name_lst = [
    BackendDataType.MXFP8.value,
    BackendDataType.MXFP4.value,
]
QMODULE_MAPPING = {
    BackendDataType.MXFP8.value: ar_qmodules.MXFP8QuantLinear,
    BackendDataType.MXFP4.value: ar_qmodules.MXFP4QuantLinear,
}
SCHEMES_MAPPING = {
    BackendDataType.MXFP8.value: ar_schemes.MXFP8,
    BackendDataType.MXFP4.value: ar_schemes.MXFP4,
}
MX_TENSOR_DATA_TYPES_FP = [i for i in MX_TENSOR_DATA_TYPES if "int" not in i]


@pytest.mark.parametrize("scheme_name", testing_scheme_name_lst)
@pytest.mark.parametrize("weight_data_type", MX_TENSOR_DATA_TYPES_FP)
@pytest.mark.parametrize("act_data_type", MX_TENSOR_DATA_TYPES_FP)
@torch.inference_mode()
def test_e2e_quant_and_load(scheme_name, weight_data_type, act_data_type, tiny_qwen_model_path):
    # Use a temporary directory for saving the quantized model
    with tempfile.TemporaryDirectory() as temp_dir:
        model_name = tiny_qwen_model_path
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
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
        _, quantized_model_path = autoround.quantize_and_save(format="auto_round", output_dir=quantized_model_path)

        # Perform inference with the quantized model
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            torch_dtype="auto",
        )
        model.eval()
        assert has_module(
            model, QMODULE_MAPPING[scheme_name]
        ), f"Expected {QMODULE_MAPPING[scheme_name].__name__} in the model."
