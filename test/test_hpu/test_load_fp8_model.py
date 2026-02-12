import os
import shutil

import pytest
import torch

from auto_round import AutoRound


class TestAutoRound:
    save_dir = "./saved"

    def check_nan_inf_in_tensor(self, tensor, name=""):
        return torch.isnan(tensor).any() or torch.isinf(tensor).any()

    def test_small_model_rtn_generation(self):
        model_name = "Qwen/Qwen3-0.6B-FP8"
        ar = AutoRound(model_name, iters=0, scheme="FP8_STATIC", nsamples=16)
        model, folder = ar.quantize_and_save(output_dir=self.save_dir, format="llm_compressor")
        # all linears except lm_head should be quantized to FP8
        fp8_linear_count = 0
        for name, module in model.named_modules():
            if "FP8QLinear" in type(module).__name__:
                assert module.weight.dtype == torch.float8_e4m3fn, f"{name} is not in FP8"
                assert not self.check_nan_inf_in_tensor(module.weight.to(torch.float32)), (
                    f"{name} has NaN or Inf in weights"
                )
                fp8_linear_count += 1
        assert fp8_linear_count > 0, "No FP8 linear layer found in the quantized model"
        shutil.rmtree(self.save_dir, ignore_errors=True)
