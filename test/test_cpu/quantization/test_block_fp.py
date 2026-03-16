import shutil

import pytest
import torch

from auto_round import AutoRound
from math import ceil
from auto_round.data_type.fp8 import quant_block_fp_sym
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad

from ...helpers import evaluate_accuracy, get_model_path


class TestAutoRoundBlockFP:
    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("Qwen/Qwen2.5-1.5B-Instruct")
        self.save_dir = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_invalid_scheme(self):
        model_name = self.model_name

        with pytest.raises(ValueError):
            scheme = {
                "bits": 8,
                "group_size": (128, 128),
                "data_type": "int",
                "act_bits": 16,
            }
            autoround = AutoRound(
                model_name,
                scheme=scheme,
                iters=2,
                seqlen=2,
            )

        with pytest.raises(NotImplementedError):
            scheme = {
                "bits": 8,
                "group_size": (128, 128),
                "data_type": "fp",
                "act_bits": 8,
                "act_data_type": "fp",
                "act_group_size": 128,
                "act_dynamic": False,
            }
            autoround = AutoRound(
                model_name,
                scheme=scheme,
                iters=2,
                seqlen=2,
            )

        with pytest.raises(ValueError):
            scheme = {
                "bits": 8,
                "group_size": (128, 128),
                "data_type": "fp",
                "act_bits": 8,
                "act_data_type": "fp",
                "act_group_size": (128, 128),
                "act_dynamic": True,
            }
            autoround = AutoRound(
                model_name,
                scheme=scheme,
                iters=2,
                seqlen=2,
            )

    def test_block_fp8_quant(self):
        data = torch.randn(256, 240)
        group_size = (128,128)
        reshaped_data, orig_shape, pad_len = reshape_pad_tensor_by_group_size(data, group_size)
        assert list(reshaped_data.shape) == [2, 2, 128, 128]
        assert list(orig_shape) == [256, 240]
        assert pad_len == (0, 16)

        qdq_data, scale, _ = quant_block_fp_sym(data)
        M = ceil(data.shape[0] / 128)
        N = ceil(data.shape[1] / 128)
        scale_ref = torch.zeros(M, N)

        max_val = torch.finfo(torch.float8_e4m3fn).max
        for i in range(M):
            for j in range(N):
                scale_ref[i, j] = data[i * 128: (i + 1) * 128, j * 128: (j + 1) * 128].abs().max() / max_val
        assert (scale == scale_ref).all()

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
        tmp_layer = compressed_model.model.layers[1].self_attn.q_proj
        assert hasattr(tmp_layer, "weight_scale_inv")
        assert tmp_layer.weight.dtype is torch.float8_e4m3fn
        assert list(tmp_layer.weight_scale_inv.shape) == [12, 12]
        assert compressed_model.config.quantization_config["quant_method"] == "fp8"
        assert compressed_model.config.quantization_config["group_size"] == (128, 128)
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
        tmp_layer = compressed_model.model.layers[1].self_attn.q_proj
        assert hasattr(tmp_layer, "weight_scale_inv")
        assert tmp_layer.weight.dtype is torch.float8_e4m3fn
        assert list(tmp_layer.weight_scale_inv.shape) == [12, 12]
        assert compressed_model.config.quantization_config["quant_method"] == "fp8"
        assert compressed_model.config.quantization_config["weight_block_size"] == (128, 128)
        evaluate_accuracy(quantized_model_path, threshold=0.55, batch_size=32, limit=100)
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
        tmp_layer = compressed_model.model.layers[1].self_attn.q_proj
        assert hasattr(tmp_layer, "weight_scale")
        assert tmp_layer.weight.dtype is torch.float8_e4m3fn
        assert list(tmp_layer.weight_scale.shape) == [12, 12]
        assert compressed_model.config.quantization_config["quant_method"] == "compressed-tensors"
        evaluate_accuracy(quantized_model_path, threshold=0.55, batch_size=32, limit=100)
        shutil.rmtree(quantized_model_path, ignore_errors=True)
