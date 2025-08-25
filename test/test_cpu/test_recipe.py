import shutil
import sys
import unittest

sys.path.insert(0, "../..")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        model_name = "facebook/opt-125m"
        self.save_dir = "./saved"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_recipe_api(self):
        bits = 4
        act_bits = 4
        data_type = "nv_fp"
        act_data_type = "nv_fp4_with_static_gs"
        group_size = 16
        sym = True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            act_bits=act_bits,
            data_type=data_type,
            act_data_type=act_data_type,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        layer_config = autoround._generate_recipe(
            mp_dtype={
                "data_type": "mx_fp8",
                "act_data_type": "mx_fp8",
            },
            mp_config={
                "mp_ratio": 1 / 3,
                "loss_weight": 2.0,
                "numel_weight": 1.0,
            },
        )
        autoround.layer_config = layer_config
        autoround.quantize()
        # autoround.quantize_and_save()  # save is not supported for mix-precision
        print(autoround.model)


if __name__ == "__main__":
    unittest.main()
