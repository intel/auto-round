import shutil

import pytest
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from auto_round import AutoRoundMLLM

from ...helpers import get_model_path, opt_name_or_path


class FakeDataLoader:
    def __init__(self):
        self.batch_size = 1

        self.data = {
            "text": [
                {"role": "user", "content": "<image>\nWhat are the colors of the bus in the image?"},
                {"role": "assistant", "content": "The bus in the image is white and red."},
            ],
            "image": "http://images.cocodataset.org/train2017/000000033471.jpg",
        }

    def __iter__(self):
        for i in range(4):
            yield self.data


class TestAutoRoundMLLM:
    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("Qwen/Qwen2-VL-2B-Instruct")
        self.dataset = FakeDataLoader()

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_tune(self, tiny_qwen_vl_model_path):
        bits, group_size = 4, 128
        autoround = AutoRoundMLLM(
            model=tiny_qwen_vl_model_path,
            bits=bits,
            group_size=group_size,
            nsamples=1,
            batch_size=1,
            iters=2,
            dataset=self.dataset,
            seqlen=10,
        )
        autoround.quantize()
        autoround.save_quantized("./saved/", format="auto_gptq", inplace=False)
        autoround.save_quantized("./saved/", format="auto_round", inplace=False)

    def test_quant_vision(self, tiny_qwen_vl_model_path):  ## bug need to fix
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_vl_model_path)
        processor = AutoProcessor.from_pretrained(tiny_qwen_vl_model_path, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            tiny_qwen_vl_model_path, trust_remote_code=True, device_map="auto"
        )
        bits, group_size = 4, 128
        autoround = AutoRoundMLLM(
            model,
            tokenizer,
            processor=processor,
            bits=bits,
            group_size=group_size,
            nsamples=5,
            batch_size=3,
            iters=2,
            dataset=self.dataset,
            quant_nontext_module=True,
            seqlen=10,
        )
        autoround.quantize()
        autoround.save_quantized("./saved/", format="auto_round", inplace=True)

    def test_quant_block_names(self):
        from auto_round.utils import find_matching_blocks, get_block_names

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto"
        )
        to_quant_block_names = "visual.*12,layers.0,model.layers.*9"
        target_blocks = [
            ["model.visual.blocks.12"],
            ["model.language_model.layers.0", "model.language_model.layers.9", "model.language_model.layers.19"],
        ]
        all_blocks = get_block_names(model, quant_vision=True)
        blocks = find_matching_blocks(model, all_blocks, to_quant_block_names)
        assert target_blocks == blocks

    def test_dataset_check(self):
        from auto_round.compressors.mllm.dataset import MLLM_DATASET

        class Myclass:
            model_type = None

        dataset = MLLM_DATASET["liuhaotian/llava"](
            template=Myclass(), model=None, tokenizer=None, dataset_path="liuhaotian/llava", seqlen=32, nsamples=32
        )
        assert len(dataset.questions) == 32
        dataset = MLLM_DATASET["liuhaotian/llava"](
            template=Myclass(), model=None, tokenizer=None, dataset_path="liuhaotian/llava", seqlen=2048, nsamples=512
        )
        assert len(dataset.questions) == 512

    def test_diff_dataset(self, tiny_qwen_vl_model_path):
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_vl_model_path)
        processor = AutoProcessor.from_pretrained(tiny_qwen_vl_model_path, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            tiny_qwen_vl_model_path, trust_remote_code=True, device_map="auto"
        )
        bits, group_size = 4, 128
        dataset = ["dataset test", "list test"]
        autoround = AutoRoundMLLM(
            model,
            tokenizer,
            processor=processor,
            bits=bits,
            group_size=group_size,
            nsamples=2,
            batch_size=1,
            iters=2,
            dataset=dataset,
            seqlen=1,
        )
        autoround.quantize()

    def test_pure_text_model_check(self, tiny_qwen_vl_model_path):
        from transformers import AutoModelForCausalLM

        from auto_round.utils import is_pure_text_model

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            tiny_qwen_vl_model_path, trust_remote_code=True, device_map="auto"
        )
        assert not is_pure_text_model(model)
        model = AutoModelForCausalLM.from_pretrained(opt_name_or_path, trust_remote_code=True)
        assert is_pure_text_model(model)

    def test_str_input(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto"
        )
        bits, group_size = 4, 128
        dataset = ["test pure text", "input for mllm"]
        autoround = AutoRoundMLLM(
            model,
            tokenizer,
            processor=processor,
            bits=bits,
            group_size=group_size,
            nsamples=2,
            batch_size=1,
            iters=2,
            dataset=dataset,
            seqlen=1,
        )
        autoround.quantize()
        quantized_model_path = "./saved"
        autoround.save_quantized(quantized_model_path, format="auto_round", inplace=False)
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
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = Image.open(requests.get(image_url, stream=True).raw)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0])

    def test_qwen2_5(self, tiny_qwen_2_5_vl_model_path):
        from auto_round.utils import mllm_load_model

        model_name = tiny_qwen_2_5_vl_model_path
        model, processor, tokenizer, image_processor = mllm_load_model(model_name)
        autoround = AutoRoundMLLM(
            model,
            tokenizer,
            iters=1,
            nsamples=1,
            seqlen=32,
            quant_nontext_module=True,
            processor=processor,
            image_processor=image_processor,
        )
        autoround.quantize_and_save("./saved/", format="auto_round")

        import requests
        from PIL import Image
        from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained("./saved", torch_dtype="auto", device_map="auto")
        image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        processor = AutoProcessor.from_pretrained("./saved")
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

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = Image.open(requests.get(image_url, stream=True).raw)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=5)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
