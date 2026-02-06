import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.eval.evaluation import simple_evaluate_user_model

from ..helpers import get_model_path, model_infer


class TestAutoRoundARKBackend:

    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("facebook/opt-125m")
        self.save_folder = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def main_op(self, format, bits, group_size, sym, dtype, device, fast_cfg=True, tar_acc=0.27):
        limit = 100
        if device == "xpu":
            limit = 1000
            if not torch.xpu.is_available():
                pytest.skip("No XPU device")
            if sym is False:
                pytest.skip("No asym support for XPU")

        # Skip tests in CI based on environment variables, workaround for ark LD_PRELOAD issue
        if device == "cpu" and os.environ.get("SKIP_CPU"):
            pytest.skip("Skip CPU test in CI")
        if device == "xpu" and os.environ.get("SKIP_XPU"):
            pytest.skip("Skip XPU test in CI")

        model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if fast_cfg:
            autoround = AutoRound(
                model, tokenizer, bits=bits, group_size=group_size, sym=sym, iters=0, nsamples=1, disable_opt_rtn=True
            )
        else:
            autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)
        quantized_model_path = self.save_folder
        autoround.quantize_and_save(output_dir=quantized_model_path, format=format)  ##will convert to gptq model

        quantization_config = AutoRoundConfig(backend="ark")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, dtype=dtype, device_map=device, quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=32, tasks="lambada_openai", limit=limit)
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > tar_acc
        torch.xpu.empty_cache()
        shutil.rmtree(self.save_folder, ignore_errors=True)

    @pytest.mark.parametrize("format", ["auto_round", "auto_round:gptqmodel"])
    @pytest.mark.parametrize("bits, group_size, sym", [(4, 128, True), (8, 128, True)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("device", ["cpu", "xpu"])
    def test_formats(self, format, bits, group_size, sym, dtype, device):
        self.main_op(format, bits, group_size, sym, dtype, device)

    @pytest.mark.parametrize("format", ["auto_round:auto_awq"])
    @pytest.mark.parametrize("bits, group_size, sym", [(4, 32, True)])
    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("device", ["cpu", "xpu"])
    def test_awq_fp16(self, format, bits, group_size, sym, dtype, device):
        self.main_op(format, bits, group_size, sym, dtype, device)

    @pytest.mark.parametrize("format", ["auto_round"])
    @pytest.mark.parametrize("bits, group_size, sym", [(2, 32, False)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.skip(reason="temp skip, this test can't work with ark 0.9 without oneapi toolkit")
    def test_other_bits(self, format, bits, group_size, sym, dtype, device):
        self.main_op(format, bits, group_size, sym, dtype, device, False, 0.2)


if __name__ == "__main__":
    p = TestAutoRoundARKBackend()
    p.setup_class()
    p.test_formats("auto_round:auto_awq", 4, 64, True, torch.bfloat16, "xpu")
    p.teardown_class()
