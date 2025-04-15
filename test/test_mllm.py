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
            "text": [{
                'role': 'user',
                'content': '<image>\nWhat are the colors of the bus in the image?'
            }, {
                'role': 'assistant',
                'content': 'The bus in the image is white and red.'
            }],
            "image": "http://images.cocodataset.org/train2017/000000033471.jpg"
        }

    def __iter__(self):
        for i in range(4):
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
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto")
        bits, group_size = 4, 128
        autoround = AutoRoundMLLM(
            model, tokenizer, processor=processor,
            bits=bits, group_size=group_size,
            nsamples=1,
            batch_size=1, iters=2, dataset=self.dataset,seqlen=10)
        autoround.quantize()
        autoround.save_quantized("./saved/", format="auto_gptq", inplace=False)
        autoround.save_quantized("./saved/", format="auto_round", inplace=False)

    def test_quant_vision(self): ## bug need to fix
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto")
        bits, group_size = 4, 128
        autoround = AutoRoundMLLM(
            model, tokenizer, processor=processor,
            bits=bits, group_size=group_size,
            nsamples=5,
            batch_size=3, iters=2, dataset=self.dataset, quant_nontext_module=True, seqlen=10)
        autoround.quantize()
        autoround.save_quantized("./saved/", format="auto_round", inplace=True)

    def test_quant_block_names(self):
        from auto_round.utils import get_block_names,find_matching_blocks
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto")
        to_quant_block_names = 'visual.*12,layers.0,model.layers.*9'
        target_blocks = [['visual.blocks.12'], ['model.layers.0', 'model.layers.9', 'model.layers.19']]
        all_blocks = get_block_names(model, quant_vision=True)
        blocks = find_matching_blocks(model, all_blocks, to_quant_block_names)
        assert target_blocks == blocks

    def test_dataset_check(self):
        from auto_round.mllm.mllm_dataset import MLLM_DATASET
        class Myclass:
            model_type=None
        dataset = MLLM_DATASET['liuhaotian/llava'](template=Myclass(), model=None, tokenzier=None, dataset_path="liuhaotian/llava", seqlen=32, nsamples=32)
        self.assertEqual(len(dataset.questions), 32)
        dataset = MLLM_DATASET['liuhaotian/llava'](template=Myclass(), model=None, tokenzier=None, dataset_path="liuhaotian/llava", seqlen=2048, nsamples=512)
        self.assertEqual(len(dataset.questions), 512)

    def test_diff_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto")
        bits, group_size = 4, 128
        dataset = ["dataset test", "list test"]
        autoround = AutoRoundMLLM(
            model, tokenizer, processor=processor,
            bits=bits, group_size=group_size,
            nsamples=2,
            batch_size=1, iters=2, dataset=dataset, seqlen=1)
        autoround.quantize()
    
    def test_pure_text_model_check(self):
        from transformers import AutoModelForCausalLM
        from auto_round.utils import is_pure_text_model
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto")
        self.assertFalse(is_pure_text_model(model))
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", trust_remote_code=True)
        self.assertTrue(is_pure_text_model(model))

    def test_str_input(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto")
        bits, group_size = 4, 128
        dataset = ["test pure text", "input for mllm"]
        autoround = AutoRoundMLLM(
            model, tokenizer, processor=processor,
            bits=bits, group_size=group_size,
            nsamples=2,
            batch_size=1, iters=2, dataset=dataset, seqlen=1)
        autoround.quantize()
        quantized_model_path = "./saved"
        autoround.save_quantized(quantized_model_path, format="auto_round", inplace=False)

        from auto_round import AutoRoundConfig
        import requests
        from PIL import Image

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            quantized_model_path,
            torch_dtype="float16",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(quantized_model_path)
        image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_url,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs = Image.open(requests.get(image_url, stream=True).raw)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0])


if __name__ == "__main__":
    unittest.main()
