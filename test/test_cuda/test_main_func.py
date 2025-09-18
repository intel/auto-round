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

from auto_round import AutoRound, AutoRoundAdam
from auto_round.eval.evaluation import simple_evaluate
from auto_round.testing_utils import require_awq, require_gptqmodel, require_optimum


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
    @require_awq
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
    @require_awq
    def test_fp_layers(self):
        model_name = "/models/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        from auto_round.utils import get_fp_layer_names

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


if __name__ == "__main__":
    unittest.main()
