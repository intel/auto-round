import shutil
import sys

import pytest

sys.path.insert(0, "../..")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AutoRoundConfig
from auto_round.eval.evaluation import simple_evaluate_user_model
from auto_round.testing_utils import require_autogptq, require_gptqmodel


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRoundARKBackend:

    @classmethod
    def setup_class(self):
        self.model_name = "facebook/opt-125m"
        self.save_folder = "./saved"
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def model_infer(self, model, tokenizer):
        prompts = [
            "Hello,my name is",
            # "The president of the United States is",
            # "The capital of France is",
            # "The future of AI is",
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=False, truncation=True)

        outputs = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            do_sample=False,  ## change this to follow official usage
            max_new_tokens=5,
        )
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], outputs)]

        decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for i, prompt in enumerate(prompts):
            print(f"Prompt: {prompt}")
            print(f"Generated: {decoded_outputs[i]}")
            print("-" * 50)
        return decoded_outputs[0]

    def main_op(self, format, bits, group_size, sym, dtype, device, fast_cfg=True, tar_acc=0.28):
        limit = 100
        if device == "xpu":
            limit = 1000
            if not torch.xpu.is_available():
                pytest.skip("No XPU device")
            if sym is False:
                pytest.skip("No asym support for XPU")
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
        self.model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=32, tasks="lambada_openai", limit=limit)
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > tar_acc
        torch.xpu.empty_cache()
        shutil.rmtree(self.save_folder, ignore_errors=True)

    @pytest.mark.parametrize("format", ["auto_round", "auto_round:auto_awq", "auto_round:gptqmodel"])
    @pytest.mark.parametrize("bits, group_size, sym", [(4, 128, True), (8, 128, True)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("device", ["cpu", "xpu"])
    def test_formats(self, format, bits, group_size, sym, dtype, device):
        self.main_op(format, bits, group_size, sym, dtype, device)

    @pytest.mark.parametrize("format", ["auto_round"])
    @pytest.mark.parametrize("bits, group_size, sym", [(4, 32, True)])
    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("device", ["cpu", "xpu"])
    def test_fp16(self, format, bits, group_size, sym, dtype, device):
        self.main_op(format, bits, group_size, sym, dtype, device)

    @pytest.mark.parametrize("format", ["auto_round"])
    @pytest.mark.parametrize("bits, group_size, sym", [(2, 32, False)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_other_bits(self, format, bits, group_size, sym, dtype, device):
        self.main_op(format, bits, group_size, sym, dtype, device, False, 0.2)


if __name__ == "__main__":
    p = TestAutoRoundARKBackend()
    p.setup_class()
    p.test_formats('auto_round:auto_awq',4,32,True,torch.bfloat16,'xpu')
    p.teardown_class()
