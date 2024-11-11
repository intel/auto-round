import copy
import shutil
import sys
import unittest

sys.path.insert(0, "..")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)

# TODO: remove it before merge
import os
import pathlib
def test_ext():
    import auto_round
    AUTO_ROUND_PATH = pathlib.Path(auto_round.__path__[0])
    print(f"auto_round.__path__ = {auto_round.__path__}")
    # list all files in the auto_round's package directory
    files = os.listdir(AUTO_ROUND_PATH.parent)
    for f in files:
        if "auto" in f:
            print(f)
    
    if "auto_round_extension" in files:
        print("!!!!!!!!!!!! auto_round_extension exists")
        files_under_auto_round_extension = os.listdir(AUTO_ROUND_PATH.parent / "auto_round_extension")
        for f in files_under_auto_round_extension:
            print(f"file under ext: {f}")

class TestQuantizationConv1d(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "MBZUAI/LaMini-GPT-124M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)


    def test_quant(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,

        )

        autoround.quantize()
        try:
            import auto_gptq
        except:
            return
        autoround.save_quantized("./saved")


if __name__ == "__main__":
    unittest.main()
