import copy
import re
import shutil
import sys
import unittest

sys.path.insert(0, "../..")
import torch
import transformers
from lm_eval.utils import make_table  # pylint: disable=E0401
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.versions import require_version

from auto_round import AutoRound, AutoRoundAdam
from auto_round.eval.evaluation import simple_evaluate
from auto_round.testing_utils import require_awq, require_gptqmodel, require_optimum, require_package_version_ut


def get_accuracy(data):
    match = re.search(r"\|acc\s+\|[↑↓]\s+\|\s+([\d.]+)\|", data)

    if match:
        accuracy = float(match.group(1))
        return accuracy
    else:
        return 0.0


class TestMainFunc(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.save_dir = "./saved"
        self.tasks = "lambada_openai"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @require_gptqmodel
    @require_optimum
    def test_backend(self):
        model_name = "/models/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        autoround = AutoRound(model, tokenizer, bits=4, group_size=128)
        autoround.quantize()

        ##test auto_round format
        autoround.save_quantized(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        res = simple_evaluate(model="hf", model_args=model_args, tasks=self.tasks, batch_size="auto")
        res = make_table(res)
        accuracy = get_accuracy(res)
        assert accuracy > 0.35
        shutil.rmtree("./saved", ignore_errors=True)

        ##test auto_gptq format
        autoround.save_quantized(self.save_dir, format="auto_gptq", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        res = simple_evaluate(model="hf", model_args=model_args, tasks=self.tasks, batch_size="auto")
        res = make_table(res)
        accuracy = get_accuracy(res)
        assert accuracy > 0.35
        shutil.rmtree("./saved", ignore_errors=True)

    @require_optimum
    @require_awq
    @require_package_version_ut("transformers", "<4.57.0")
    def test_backend_awq(self):
        model_name = "/models/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        autoround = AutoRound(model, tokenizer, bits=4, group_size=128)
        autoround.quantize()

        ##test auto_awq format
        autoround.save_quantized(self.save_dir, format="auto_awq", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        res = simple_evaluate(model="hf", model_args=model_args, tasks=self.tasks, batch_size="auto")
        res = make_table(res)
        accuracy = get_accuracy(res)
        assert accuracy > 0.35
        shutil.rmtree("./saved", ignore_errors=True)

    @unittest.skipIf(torch.cuda.is_available() is False, "Skipping because no cuda")
    @require_gptqmodel
    def test_fp_layers(self):
        model_name = "/models/opt-125m"
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
        res = simple_evaluate(model="hf", model_args=model_args, tasks=self.tasks, batch_size="auto")
        res = make_table(res)
        accuracy = get_accuracy(res)
        assert accuracy > 0.35
        shutil.rmtree("./saved", ignore_errors=True)

    @unittest.skipIf(torch.cuda.is_available() is False, "Skipping because no cuda")
    @require_awq
    @require_package_version_ut("transformers", "<4.57.0")
    def test_fp_layers_awq(self):
        model_name = "/models/opt-125m"
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
        res = simple_evaluate(model="hf", model_args=model_args, tasks=self.tasks, batch_size="auto")
        res = make_table(res)
        accuracy = get_accuracy(res)
        assert accuracy > 0.35
        shutil.rmtree("./saved", ignore_errors=True)

    @unittest.skipIf(torch.cuda.is_available() is False, "Skipping because no cuda")
    def test_undivided_group_size_tuning(self):
        model_name = "/models/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        autoround = AutoRound(model, tokenizer, bits=4, group_size=127, nsamples=2, iters=2)
        autoround.quantize()

    @require_gptqmodel
    def test_adam(self):
        model_name = "/models/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        autoround = AutoRoundAdam(model, tokenizer, bits=4, group_size=128)
        autoround.quantize()

        ##test auto_round format
        autoround.save_quantized(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        res = simple_evaluate(model="hf", model_args=model_args, tasks=self.tasks, batch_size="auto")
        res = make_table(res)
        accuracy = get_accuracy(res)
        assert accuracy > 0.34
        shutil.rmtree("./saved", ignore_errors=True)

    def test_autoround_asym(self):  ##need to install false
        try:
            from autoround_exllamav2_kernels import gemm_half_q_half, make_q_matrix
        except ImportError as e:
            print("skip autoround asym test, as autoround is not installed from source")
            return
        model_name = "/models/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        autoround = AutoRound(model, tokenizer, bits=4, group_size=128, sym=False)
        autoround.quantize()

        ##test auto_round format
        autoround.save_quantized(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        res = simple_evaluate(model="hf", model_args=model_args, tasks=self.tasks, batch_size="auto")
        res = make_table(res)
        accuracy = get_accuracy(res)
        assert accuracy > 0.35
        shutil.rmtree("./saved", ignore_errors=True)

    def test_attention_mask_lm_head(self):
        from transformers import AutoTokenizer

        model_name = "/models/Qwen3-8B"
        # model_name = "/models/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text = ["haha", "hello world"]
        res = tokenizer(text, return_tensors="pt", max_length=8, padding="max_length", truncation=True)
        res.data.pop("attention_mask")
        data = [res.data]

        text = ["qudd", "hfd"]
        res = tokenizer(text, return_tensors="pt", max_length=8, padding="max_length", truncation=True)
        res.data.pop("attention_mask")
        data.append(res.data)
        from auto_round import AutoRound

        ar = AutoRound(model_name, iters=1, dataset=data, seqlen=8, quant_lm_head=True)
        ar.quantize()

    def test_save_block_immediate(self):
        bits, group_size = 4, 32
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        quantized_model_path = "./saved"
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader,
            is_packing_immediate=True,
            save_block_immediate=True,
        )
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        shutil.rmtree(quantized_model_path, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
