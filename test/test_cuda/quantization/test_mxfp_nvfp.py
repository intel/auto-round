import copy
import shutil

import pytest
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ...envs import require_awq, require_optimum
from ...helpers import get_model_path, save_tiny_model


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

    def test_fp8input_mxfp4_llmcompressor_format(self, dataloader, mock_fp8_capable_device):
        model_name = get_model_path("Qwen/Qwen3-0.6B-FP8")
        tiny_model_path = "./tmp/tiny_qwen3_fp8"
        save_tiny_model(model_name, tiny_model_path)
        scheme = "mxfp4"
        ar = AutoRound(
            model=tiny_model_path,
            iters=0,
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
        shutil.rmtree(quantized_model_path, ignore_errors=True)

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

    @pytest.mark.skip_ci(reason="OOM")
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
