import copy
import shutil

import pytest
import torch
import transformers
from lm_eval.utils import make_table  # pylint: disable=E0401
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.versions import require_version

from auto_round import AutoRound, AutoRoundAdam
from auto_round.testing_utils import require_awq, require_gptqmodel, require_optimum, require_package_version_ut

from ...helpers import evaluate_accuracy, get_model_path


class TestMainFunc:
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

    @require_gptqmodel
    @require_optimum
    def test_backend(self):
        model_name = get_model_path("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        autoround = AutoRound(model, tokenizer, bits=4, group_size=128)
        autoround.quantize()

        ##test auto_round format
        autoround.save_quantized(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        evaluate_accuracy(model_args, threshold=0.35, batch_size="auto")
        shutil.rmtree("./saved", ignore_errors=True)

        ##test auto_gptq format
        autoround.save_quantized(self.save_dir, format="auto_gptq", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        evaluate_accuracy(model_args, threshold=0.35, batch_size="auto")
        shutil.rmtree("./saved", ignore_errors=True)

    @require_optimum
    @require_awq
    @require_package_version_ut("transformers", "<4.57.0")
    def test_backend_awq(self):
        model_name = get_model_path("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        autoround = AutoRound(model, tokenizer, bits=4, group_size=128)
        autoround.quantize()

        ##test auto_awq format
        autoround.save_quantized(self.save_dir, format="auto_awq", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        evaluate_accuracy(model_args, threshold=0.35, batch_size="auto")
        shutil.rmtree("./saved", ignore_errors=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @require_gptqmodel
    def test_ignore_layers(self):
        model_name = get_model_path("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        from auto_round.compressors.utils import get_fp_layer_names

        layer_names = get_fp_layer_names(model, "model.decoder.layers.0,model.decoder.layers.1")
        layer_configs = {}
        for name in layer_names:
            layer_configs[name] = {"bits": 16}
        autoround = AutoRound(model, tokenizer, bits=4, group_size=128)
        autoround.quantize()

        ##test auto_round format
        autoround.save_quantized(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        evaluate_accuracy(model_args, threshold=0.35, batch_size="auto")
        shutil.rmtree("./saved", ignore_errors=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @require_awq
    @require_package_version_ut("transformers", "<4.57.0")
    def test_ignore_layers_awq(self):
        model_name = get_model_path("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        from auto_round.compressors.utils import get_fp_layer_names

        layer_names = get_fp_layer_names(model, "model.decoder.layers.0,model.decoder.layers.1")
        layer_configs = {}
        for name in layer_names:
            layer_configs[name] = {"bits": 16}
        autoround = AutoRound(model, tokenizer, bits=4, group_size=128)
        autoround.quantize()

        ##test auto_awq format
        autoround.save_quantized(self.save_dir, format="auto_awq", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        evaluate_accuracy(model_args, threshold=0.35, batch_size="auto")
        shutil.rmtree("./saved", ignore_errors=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_undivided_group_size_tuning(self, tiny_opt_model_path):
        model = AutoModelForCausalLM.from_pretrained(tiny_opt_model_path, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path)

        autoround = AutoRound(model, tokenizer, bits=4, group_size=127, nsamples=2, iters=2)
        autoround.quantize()

    @require_gptqmodel
    def test_adam(self):
        model_name = get_model_path("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        autoround = AutoRoundAdam(model, tokenizer, bits=4, group_size=128)
        autoround.quantize()

        ##test auto_round format
        autoround.save_quantized(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        evaluate_accuracy(model_args, threshold=0.34, batch_size="auto")
        shutil.rmtree("./saved", ignore_errors=True)

    def test_autoround_asym(self):  ##need to install false
        try:
            from autoround_exllamav2_kernels import gemm_half_q_half, make_q_matrix
        except ImportError as e:
            print("skip autoround asym test, as autoround is not installed from source")
            return
        model_name = get_model_path("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        autoround = AutoRound(model, tokenizer, bits=4, group_size=128, sym=False)
        autoround.quantize()

        ##test auto_round format
        autoround.save_quantized(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        evaluate_accuracy(model_args, threshold=0.35, batch_size="auto")
        shutil.rmtree("./saved", ignore_errors=True)

    def test_attention_mask_lm_head(self, tiny_qwen_moe_model_path):
        from transformers import AutoTokenizer

        # model_name = "/models/Qwen3-8B"
        # model_name = "/models/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_moe_model_path)
        text = ["haha", "hello world"]
        res = tokenizer(text, return_tensors="pt", max_length=8, padding="max_length", truncation=True)
        res.data.pop("attention_mask")
        data = [res.data]

        text = ["qudd", "hfd"]
        res = tokenizer(text, return_tensors="pt", max_length=8, padding="max_length", truncation=True)
        res.data.pop("attention_mask")
        data.append(res.data)
        from auto_round import AutoRound

        ar = AutoRound(tiny_qwen_moe_model_path, iters=1, dataset=data, seqlen=8, quant_lm_head=True)
        ar.quantize()

    def test_low_cpu_mem_usage(self, tiny_opt_model_path):
        bits, group_size = 4, 32
        model = AutoModelForCausalLM.from_pretrained(tiny_opt_model_path, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path, trust_remote_code=True)
        quantized_model_path = "./saved"
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            iters=2,
            seqlen=10,
            low_cpu_mem_usage=True,
        )
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        shutil.rmtree(quantized_model_path, ignore_errors=True)
