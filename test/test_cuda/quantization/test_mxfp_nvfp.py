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
        autoround.quantize_and_save(format="auto_round", output_dir=quantized_model_path)

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
    save_dir = "./saved"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("./saved", ignore_errors=True)
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
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @pytest.mark.skip_ci(reason="Cannot test all case in CI; time-consuming")
    def test_fp8input_mxfp4_llmcompressor_format(self, dataloader, tiny_fp8_qwen_model_path, mock_fp8_capable_device):
        scheme = "mxfp4"
        ar = AutoRound(
            model=tiny_fp8_qwen_model_path,
            iters=1,
            seqlen=2,
            scheme=scheme,
            dataset=dataloader,
        )
        print(ar.model)
        compressed_model, _ = ar.quantize_and_save(output_dir=self.save_dir, format="llm_compressor")
        tmp_layer = compressed_model.model.layers[1].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_packed")
            and tmp_layer.weight_scale.dtype is torch.uint8
            and tmp_layer.weight_scale.shape[0] == 2048
        ), "Illegal MXFP4 packing name or data_type or shape"
        quantization_config = AutoConfig.from_pretrained(self.save_dir, trust_remote_code=True).quantization_config
        assert (
            quantization_config["format"] == "float-quantized"
            and quantization_config["config_groups"]["group_0"]["weights"]["is_mx"] is True
            and quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] == 4
        ), f"Invalid MXFP4 quantization configuration: {quantization_config}"
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.skip_ci(reason="Cannot test all case in CI; time-consuming")
    def test_nvfp4_llmcompressor_format(self, tiny_opt_model_path, dataloader):
        scheme = "nvfp4"
        autoround = AutoRound(
            tiny_opt_model_path,
            scheme=scheme,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="llm_compressor")
        tmp_layer = compressed_model.model.decoder.layers[1].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_global_scale")
            and hasattr(tmp_layer, "input_global_scale")
            and tmp_layer.weight_packed.dtype is torch.uint8
            and tmp_layer.weight_scale.dtype is torch.float8_e4m3fn
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal NVFP4 packing name or data_type or shape"
        quantization_config = AutoConfig.from_pretrained(
            quantized_model_path, trust_remote_code=True
        ).quantization_config
        assert (
            quantization_config["format"] == "nvfp4-pack-quantized"
            and quantization_config["config_groups"]["group_0"]["input_activations"]["num_bits"] == 4
        ), f"Invalid NVFP4 quantization configuration: {quantization_config}"
        shutil.rmtree(quantized_model_path, ignore_errors=True)

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
        shutil.rmtree(quantized_model_path, ignore_errors=True)

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
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        from ...helpers import evaluate_accuracy

        evaluate_accuracy(model, tokenizer, threshold=0.49, batch_size=16, task="piqa", limit=10)
        shutil.rmtree(quantized_model_path, ignore_errors=True)
