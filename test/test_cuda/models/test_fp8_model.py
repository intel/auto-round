import os
import shutil

import pytest
import torch
from packaging import version
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.utils import llm_load_model
from auto_round.utils.weight_handler import (
    ModuleWeightType,
    check_and_mark_quantized_module,
    convert_module_to_hp_if_necessary,
)

from ...helpers import evaluate_accuracy, generate_prompt, get_model_path, get_tiny_model, transformers_version


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

    def test_small_model_rtn_generation(self, mock_fp8_capable_device, tiny_fp8_qwen_model_path):
        ar = AutoRound(tiny_fp8_qwen_model_path, iters=0, disable_opt_rtn=True)
        _, quantized_model_path = ar.quantize_and_save(output_dir=self.save_dir)
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        generate_prompt(model, tokenizer)

    def test_gguf_imatrix(self, mock_fp8_capable_device, tiny_fp8_qwen_model_path):
        ar = AutoRound(tiny_fp8_qwen_model_path, iters=0)
        _, quantized_model_path = ar.quantize_and_save(format="gguf:q2_k_s", output_dir=self.save_dir)
        # from llama_cpp import Llama
        #
        # gguf_file = os.listdir("saved/Qwen3-0.6B-FP8/-gguf")[0]
        # llm = Llama(f"saved/Qwen2.5-0.5B-Instruct-gguf/{gguf_file}", n_gpu_layers=-1)
        # output = llm("There is a girl who likes adventure,", max_tokens=32)
        # print(output)
        # shutil.rmtree("./saved", ignore_errors=True)
        # model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", trust_remote_code=True)
        # tokenizer = AutoTokenizer.from_pretrained(quantized_model_path    )
        # text = "There is a girl who likes adventure,"
        # inputs = tokenizer(text, return_tensors="pt").to(model.device)
        # print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    @pytest.mark.skip_ci(reason="Triton issue; time-consuming")
    def test_small_model_rtn(self, mock_fp8_capable_device):
        model_name = get_model_path("Qwen/Qwen3-0.6B-FP8")
        ar = AutoRound(model=model_name, iters=0)
        _, folder = ar.quantize_and_save(output_dir=self.save_dir)
        evaluate_accuracy(self.save_dir, threshold=0.25)

    @pytest.mark.skip_ci(reason="Triton issue; time-consuming")
    def test_small_model_iters1(self, mock_fp8_capable_device):
        model_name = get_model_path("Qwen/Qwen3-0.6B-FP8")
        ar = AutoRound(model=model_name, iters=1)
        _, folder = ar.quantize_and_save(output_dir=self.save_dir)
        evaluate_accuracy(self.save_dir, threshold=0.25)

    @pytest.mark.skip_ci(reason="Triton issue; time-consuming")
    def test_medium_model_rtn(self, mock_fp8_capable_device):
        model_name = get_model_path("Qwen/Qwen3-0.6B-FP8")
        ar = AutoRound(model=model_name, iters=0)
        _, folder = ar.quantize_and_save(output_dir=self.save_dir)
        evaluate_accuracy(self.save_dir, threshold=0.33)

    @pytest.mark.skip_ci(reason="Triton issue; time-consuming")
    def test_medium_model_rtn_with_lm_head(self, mock_fp8_capable_device):
        model_name = get_model_path("Qwen/Qwen3-0.6B-FP8")
        layer_config = {"lm_head": {"bits": 4}}
        ar = AutoRound(model=model_name, iters=0, layer_config=layer_config)
        _, folder = ar.quantize_and_save(output_dir=self.save_dir)
        evaluate_accuracy(self.save_dir, threshold=0.33)

    def test_fp8_model_gguf_q4(self, mock_fp8_capable_device, tiny_fp8_qwen_model_path):
        from llama_cpp import Llama

        ar = AutoRound(tiny_fp8_qwen_model_path, iters=0, disable_opt_rtn=True)
        _, quantized_model_path = ar.quantize_and_save(output_dir=self.save_dir, format="gguf:q4_0")
        for file in os.listdir(quantized_model_path):
            if file.endswith(".gguf"):
                gguf_file = file
        llm = Llama(f"{quantized_model_path}/{gguf_file}", n_gpu_layers=-1)
        output = llm("There is a girl who likes adventure,", max_tokens=32)
        print(output)

    @pytest.mark.skip_ci(reason="Not necessary to test all options in CI")
    def test_fp8_model_gguf_q3(self, mock_fp8_capable_device, tiny_fp8_qwen_model_path):
        from llama_cpp import Llama

        ar = AutoRound(tiny_fp8_qwen_model_path, iters=1)
        _, quantized_model_path = ar.quantize_and_save(output_dir=self.save_dir, format="gguf:q3_k_s")
        for file in os.listdir(quantized_model_path):
            if file.endswith(".gguf"):
                gguf_file = file
        llm = Llama(f"{quantized_model_path}/{gguf_file}", n_gpu_layers=-1)
        output = llm("There is a girl who likes adventure,", max_tokens=32)
        print(output)

    @pytest.mark.skip_ci(reason="Not necessary to test all options in CI")
    @pytest.mark.parametrize("scheme", ["MXFP4", "NVFP4"])
    def test_diff_datatype(self, scheme, tiny_fp8_qwen_model_path, mock_fp8_capable_device):
        model_name = tiny_fp8_qwen_model_path
        print(f"Testing scheme: {scheme}")
        ar = AutoRound(model_name, iters=0, scheme=scheme, disable_opt_rtn=True, nsamples=2)
        _, quantized_model_path = ar.quantize_and_save(output_dir=self.save_dir)
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", trust_remote_code=True)
        assert model is not None, f"Failed to load model for scheme {scheme}"


def test_qwen3_fp8_moe_mxfp(tiny_fp8_qwen_moe_model_path, mock_fp8_capable_device):
    output_dir = "./tmp"
    autoround = AutoRound(
        tiny_fp8_qwen_moe_model_path,
        scheme="MXFP4",
        nsamples=2,
        seqlen=32,
        iters=0,
    )
    quantized_model, quantized_model_path = autoround.quantize_and_save(format="auto_round", output_dir=output_dir)
    assert quantized_model is not None, "Quantized model should not be None."
    loaded_model = AutoModelForCausalLM.from_pretrained(quantized_model_path)
    for n, m in quantized_model.named_modules():
        if m.__class__.__name__ == "QuantLinear":
            loaded_m = loaded_model.get_submodule(n)
            assert (loaded_m.weight_packed == m.weight_packed).all()
    # Expect all linear in experts are quantized
    for n, m in quantized_model.named_modules():
        if "experts" in m.__class__.__name__.lower():
            for sub_n, sub_m in m.named_modules():
                assert sub_m.__class__.__name__ == "QuantLinear", f"Module {n}.{sub_n} is not quantized."
    shutil.rmtree(output_dir, ignore_errors=True)


# requires GPU to load FP8Linear
class TestFP8Linear:
    def test_fp8_input(self, mock_fp8_capable_device):
        model = get_tiny_model(get_model_path("Qwen/Qwen3-0.6B-FP8"))
        assert (
            type(model.model.layers[0].mlp.up_proj).__name__ == "FP8Linear"
        ), "Model does not contain FP8Linear layers"
        detected_types = check_and_mark_quantized_module(model)
        assert ModuleWeightType.FP8 in detected_types
        model = convert_module_to_hp_if_necessary(model)
        assert type(model.model.layers[0].mlp.up_proj) is torch.nn.Linear, "FP8Linear layer was not converted to Linear"
