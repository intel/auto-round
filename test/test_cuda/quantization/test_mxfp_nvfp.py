import copy
import shutil
import tempfile

import pytest
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round import schemes as ar_schemes
from auto_round.experimental import qmodules as ar_qmodules
from auto_round.export.export_to_autoround import qlinear_fp as ar_qlinear_fp
from auto_round.formats import AutoRoundExportFormat

from ...envs import has_module, require_awq, require_optimum
from ...helpers import get_model_path, save_tiny_model

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
            disable_opt_rtn=True,
        )

        # Quantize and save the model to the temporary directory
        quantized_model_path = f"{temp_dir}/tmp_autoround_{scheme}"
        _, quantized_model_path = autoround.quantize_and_save(format="auto_round", output_dir=quantized_model_path)

        # Perform inference with the quantized model
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            torch_dtype="auto",
        )
        model.eval()
        assert has_module(model, QMODULE_MAPPING[scheme]), f"Expected {QMODULE_MAPPING[scheme].__name__} in the model."

        # Skip accuracy check for tiny model.

        # tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        # prompt = "The capital of France is"
        # encode = tokenizer.encode(prompt, return_tensors="pt")
        # output_tokens = model.generate(
        #     encode,
        #     max_length=10,
        # )
        # output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        # assert "paris" in output.lower(), f"Expected 'Paris' in the output, but got: {output}"


class TestAutoRound:

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        yield
        shutil.rmtree("runs", ignore_errors=True)

    def test_nvfp4_moe_actmax_rtn(self, tiny_deepseek_v2_model_path, dataloader):
        # model_name = "/data0/deepseek-ai/DeepSeek-V2-Lite"
        scheme = "nvfp4"
        autoround = AutoRound(
            tiny_deepseek_v2_model_path,
            scheme=scheme,
            iters=0,
            seqlen=2,
            nsamples=2,
            dataset=dataloader,
            trust_remote_code=False,
        )
        autoround.quantize()
        quantized_model_path = self.save_dir
        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", device_map="auto")
        print(model)
        assert model is not None, "Failed to load the quantized model."

    @pytest.mark.skip_ci(reason="Cannot test all case in CI; time-consuming")
    def test_nvfp4_moe_actmax_ar(self, tiny_deepseek_v2_model_path, dataloader):
        scheme = "nvfp4"
        autoround = AutoRound(
            tiny_deepseek_v2_model_path,
            scheme=scheme,
            iters=1,
            seqlen=2,
            nsamples=2,
            dataset=dataloader,
            trust_remote_code=False,
        )
        autoround.quantize()
        quantized_model_path = self.save_dir
        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")

    @pytest.mark.skip_ci(reason="Only tiny model is suggested")
    @pytest.mark.skip_ci(reason="Time-consuming; Accuracy evaluation")
    def test_qwen_moe_quant_infer(self, dataloader):
        model_name = get_model_path("Qwen/Qwen1.5-MoE-A2.7B")
        layer_config = {
            "layers\.(?:[3-9]|1[0-9]|2[0-3])": {"bits": 16, "act_bits": 16},
        }
        scheme = "nvfp4"
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=1,
            seqlen=3,
            nsamples=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, inplace=False, format="auto_round"
        )
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        from ...helpers import evaluate_accuracy

        evaluate_accuracy(model, tokenizer, threshold=0.49, batch_size=16, task="piqa", limit=10)
