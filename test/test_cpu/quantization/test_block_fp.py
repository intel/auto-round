import shutil

import pytest
import torch

from auto_round import AutoRound

from ...helpers import get_model_path


class TestAutoRoundBlockFP:
    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("facebook/opt-125m")
        self.save_dir = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_fp8_block_autoround_format(self):
        model_name = self.model_name

        scheme = "FP8_BLOCK"
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=2,
            seqlen=2,
        )
        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        tmp_layer = compressed_model.model.decoder.layers[1].self_attn.q_proj
        assert hasattr(tmp_layer, "weight_scale_inv")
        assert tmp_layer.weight.dtype is torch.float8_e4m3fn
        assert list(tmp_layer.weight_scale_inv.shape) == [6, 6]
        assert compressed_model.config.quantization_config["quant_method"] == "auto_round:fp8"
        assert compressed_model.config.quantization_config["group_size"] == [128, 128]
        shutil.rmtree(quantized_model_path, ignore_errors=True)

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
        tmp_layer = compressed_model.model.decoder.layers[1].self_attn.q_proj
        assert hasattr(tmp_layer, "weight_scale_inv")
        assert tmp_layer.weight.dtype is torch.float8_e4m3fn
        assert list(tmp_layer.weight_scale_inv.shape) == [6, 6]
        assert compressed_model.config.quantization_config["quant_method"] == "fp8"
        assert compressed_model.config.quantization_config["weight_block_size"] == [128, 128]
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_fp8_block_llm_compressor_format(self):
        model_name = self.model_name

        scheme = "FP8_BLOCK"
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=2,
            seqlen=2,
        )
        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="llm_compressor")
        tmp_layer = compressed_model.model.decoder.layers[1].self_attn.q_proj
        assert hasattr(tmp_layer, "weight_scale")
        assert tmp_layer.weight.dtype is torch.float8_e4m3fn
        assert list(tmp_layer.weight_scale.shape) == [6, 6]
        assert compressed_model.config.quantization_config["quant_method"] == "compressed-tensors"
        shutil.rmtree(quantized_model_path, ignore_errors=True)
