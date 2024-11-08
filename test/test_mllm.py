import sys
import unittest

sys.path.insert(0, "..")

from auto_round import AutoRoundMLLM

from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

import shutil


class FakeDataLoader:
    def __init__(self):
        self.batch_size = 1

        self.data = {
            "text": [{'role': 'user',
                      'content': '<|vision_start|><|image_pad|><|vision_end|>\nWhat are the colors of the bus in the image?'},
                     {'role': 'assistant', 'content': 'The bus in the image is white and red.'}],
            "image": "http://images.cocodataset.org/train2017/000000033471.jpg"
        }

    def __iter__(self):
        for i in range(2):
            yield self.data


class TestAutoRoundMLLM(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "Qwen/Qwen2-VL-2B-Instruct"
        self.dataset = FakeDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

        return super().tearDownClass()

    def test_tune(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.processor = processor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto")
        bits, group_size = 4, 128
        autoround = AutoRoundMLLM(
            model, tokenizer, bits=bits, group_size=group_size,
            nsamples=2,
            batch_size=1, iters=2, dataset=self.dataset)
        autoround.quantize()
        autoround.save_quantized("./saved/", format="auto_gptq", inplace=False)
        autoround.save_quantized("./saved/", format="auto_round", inplace=False)

    def test_quant_vision(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.processor = processor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto")
        bits, group_size = 4, 128
        autoround = AutoRoundMLLM(
            model, tokenizer, bits=bits, group_size=group_size,
            nsamples=2,
            batch_size=1, iters=2, dataset=self.dataset, quant_nontext_module=True)
        autoround.quantize()
        autoround.save_quantized("./saved/", format="auto_round", inplace=False)


if __name__ == "__main__":
    unittest.main()
