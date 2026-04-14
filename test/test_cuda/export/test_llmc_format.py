import shutil

import pytest
import torch
import transformers

from auto_round import AutoRound
from auto_round import schemes as ar_schemes

from ...helpers import eval_generated_prompt, get_model_path, is_cuda_support_fp8


class TestAutoRound:

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

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
        compressed_model, quantized_model_path = ar.quantize_and_save(output_dir=self.save_dir, format="llm_compressor")
        tmp_layer = compressed_model.model.layers[1].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_packed")
            and tmp_layer.weight_scale.dtype is torch.uint8
            and tmp_layer.weight_scale.shape[0] == 2048
        ), "Illegal MXFP4 packing name or data_type or shape"
        quantization_config = transformers.AutoConfig.from_pretrained(
            quantized_model_path, trust_remote_code=True
        ).quantization_config
        assert (
            quantization_config["format"] == "mxfp4-pack-quantized"
            and quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] == 4
        ), f"Invalid MXFP4 quantization configuration: {quantization_config}"

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
        compressed_model, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="llm_compressor"
        )
        tmp_layer = compressed_model.model.decoder.layers[1].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_global_scale")
            and hasattr(tmp_layer, "input_global_scale")
            and tmp_layer.weight_packed.dtype is torch.uint8
            and tmp_layer.weight_scale.dtype is torch.float8_e4m3fn
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal NVFP4 packing name or data_type or shape"
        quantization_config = transformers.AutoConfig.from_pretrained(
            quantized_model_path, trust_remote_code=True
        ).quantization_config
        assert (
            quantization_config["format"] == "nvfp4-pack-quantized"
            and quantization_config["config_groups"]["group_0"]["input_activations"]["num_bits"] == 4
        ), f"Invalid NVFP4 quantization configuration: {quantization_config}"

    @pytest.mark.skip_ci(reason="Cannot test all case in CI; time-consuming")
    def test_fp8_block_llm_compressor_format(self, tiny_qwen_model_path, dataloader):
        model_name = get_model_path("Qwen/Qwen3-0.6B")

        scheme = "FP8_BLOCK"
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=0,
            disable_opt_rtn=True,
        )
        quantized_model_path = self.save_dir
        compressed_model, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="llm_compressor"
        )
        tmp_layer = compressed_model.model.layers[1].self_attn.q_proj
        assert hasattr(tmp_layer, "weight_scale")
        assert tmp_layer.weight.dtype is torch.float8_e4m3fn
        assert list(tmp_layer.weight_scale.shape) == [16, 8]
        assert compressed_model.config.quantization_config["quant_method"] == "compressed-tensors"
        if is_cuda_support_fp8():
            eval_generated_prompt(quantized_model_path, device="cuda")
