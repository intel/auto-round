import shutil

import torch

from auto_round import AutoRound

from ...helpers import eval_generated_prompt, get_model_path, is_cuda_support_fp8


class TestAutoRoundBlockFP:
    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("Qwen/Qwen3-0.6B")
        self.save_dir = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_fp8_block_fp8_format(self):
        model_name = self.model_name

        scheme = "FP8_BLOCK"
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=2,
            seqlen=2,
        )
        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="fp8")
        tmp_layer = compressed_model.model.layers[1].self_attn.q_proj
        assert hasattr(tmp_layer, "weight_scale_inv")
        assert tmp_layer.weight.dtype is torch.float8_e4m3fn
        assert list(tmp_layer.weight_scale_inv.shape) == [16, 8]
        assert compressed_model.config.quantization_config["quant_method"] == "fp8"
        assert compressed_model.config.quantization_config["weight_block_size"] == (128, 128)
        if is_cuda_support_fp8():
            eval_generated_prompt(quantized_model_path, device="cuda")
        shutil.rmtree(quantized_model_path, ignore_errors=True)
