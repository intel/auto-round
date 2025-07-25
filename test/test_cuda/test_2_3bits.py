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

from auto_round import AutoRound
from auto_round.eval.evaluation import simple_evaluate
from auto_round.testing_utils import require_autogptq, require_greater_than_050, require_greater_than_051


def get_accuracy(data):
    match = re.search(r"\|acc\s+\|[↑↓]\s+\|\s+([\d.]+)\|", data)

    if match:
        accuracy = float(match.group(1))
        return accuracy
    else:
        return 0.0


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.save_dir = "./saved"
        self.tasks = "lambada_openai"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @require_greater_than_051
    def test_3bits_autoround(self):
        model_name = "/models/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        autoround = AutoRound(model, tokenizer, bits=3)
        autoround.quantize_and_save(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        res = simple_evaluate(
            model="hf",
            model_args=model_args,
            #   tasks="arc_easy",
            tasks=self.tasks,
            batch_size="auto",
        )

        ## 0.3130
        accuracy = res["results"]["lambada_openai"]["acc,none"]
        assert accuracy > 0.3
        shutil.rmtree("./saved", ignore_errors=True)

    @require_greater_than_051
    def test_3bits_asym_autoround(self):
        model_name = "/models/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bits, sym = 3, False
        autoround = AutoRound(model, tokenizer, bits=bits, sym=sym)
        autoround.quantize_and_save(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        res = simple_evaluate(
            model="hf",
            model_args=model_args,
            #   tasks="arc_easy",
            tasks=self.tasks,
            batch_size="auto",
        )

        ## 0.3423
        accuracy = res["results"]["lambada_openai"]["acc,none"]
        assert accuracy > 0.32
        shutil.rmtree("./saved", ignore_errors=True)

    @require_greater_than_050
    def test_norm_bias_tuning(self):
        model_name = "/models/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        autoround = AutoRound(model, tokenizer, bits=2, group_size=64, enable_norm_bias_tuning=True)
        autoround.quantize()

        ##test auto_round format
        autoround.save_quantized(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        res = simple_evaluate(model="hf", model_args=model_args, tasks=self.tasks, batch_size="auto")
        res = make_table(res)  ##0.2212 0.1844
        accuracy = get_accuracy(res)
        assert accuracy > 0.18
        shutil.rmtree("./saved", ignore_errors=True)

    @require_greater_than_050
    def test_2bits_autoround(self):
        model_name = "/models/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        autoround = AutoRound(model, tokenizer, bits=2, group_size=64)
        autoround.quantize()

        ##test auto_round format
        autoround.save_quantized(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        res = simple_evaluate(model="hf", model_args=model_args, tasks=self.tasks, batch_size="auto")
        res = make_table(res)  ##0.1985
        accuracy = get_accuracy(res)
        assert accuracy > 0.18
        shutil.rmtree("./saved", ignore_errors=True)

        autoround.save_quantized(self.save_dir, format="auto_gptq", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        res = simple_evaluate(model="hf", model_args=model_args, tasks=self.tasks, batch_size="auto")
        res = make_table(res)  ##0.1985
        accuracy = get_accuracy(res)
        assert accuracy > 0.18
        shutil.rmtree("./saved", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
