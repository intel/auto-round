import os
import shutil
import sys

import pytest
import requests
from packaging import version
from PIL import Image
from transformers import AutoRoundConfig  # # must import for auto-round format

from auto_round.testing_utils import require_gptqmodel, require_package_version_ut, require_vlm_env

from ...helpers import transformers_version

AUTO_ROUND_PATH = __file__.split("/")
AUTO_ROUND_PATH = "/".join(AUTO_ROUND_PATH[: AUTO_ROUND_PATH.index("test")])


class TestSupportVLMS:

    @classmethod
    def setup_class(self):
        self.save_dir = os.path.join(os.path.dirname(__file__), "ut_saved")
        self.python_path = sys.executable
        self.device = 0

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @require_gptqmodel
    def test_qwen2(self):
        model_path = "/models/Qwen2-VL-2B-Instruct/"
        # test tune
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {self.python_path} -m auto_round --mllm "
            f"--model {model_path} --iter 2 --output_dir {self.save_dir} --device {self.device}"
        )
        assert not (res > 0 or res == -1), "qwen2 tuning fail"

        # test infer
        quantized_model_path = os.path.join(self.save_dir, "Qwen2-VL-2B-Instruct-w4g128")

        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            quantized_model_path,
            torch_dtype="float16",
            device_map=f"cuda:{self.device}",
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
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @require_vlm_env
    @require_package_version_ut("transformers", "<4.54.0")
    def test_phi3(self):
        model_path = "/models/Phi-3.5-vision-instruct/"
        ## test tune
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {self.python_path} -m auto_round --mllm "
            f"--model {model_path} --iter 2 --output_dir {self.save_dir} --device {self.device}"
        )
        assert not (res > 0 or res == -1), "Phi-3.5 tuning fail"

        ## test infer
        from transformers import AutoModelForCausalLM, AutoProcessor

        quantized_model_path = os.path.join(self.save_dir, "Phi-3.5-vision-instruct-w4g128")
        res = os.system(f"cp /models/Phi-3.5-vision-instruct/*.py {quantized_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            device_map=f"cuda:{self.device}",
            trust_remote_code=True,
            torch_dtype="float16",
        )
        processor = AutoProcessor.from_pretrained(quantized_model_path, trust_remote_code=True, num_crops=4)

        image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        content = "Describe this image."
        messages = [
            {"role": "user", "content": "<|image_1|>\n" + content},
        ]

        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = Image.open(requests.get(image_url, stream=True).raw)
        inputs = processor(prompt, image_inputs, return_tensors="pt").to(model.device)

        generation_args = {
            "max_new_tokens": 1000,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

        # remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(response)
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @require_vlm_env
    @require_package_version_ut("transformers", "<4.57.0")
    def test_phi3_vision_awq(self):
        model_path = "/models/Phi-3.5-vision-instruct/"
        ## test tune
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {self.python_path} -m auto_round --mllm "
            f"--model {model_path} --iter 2 --quant_nontext_module "
            f"--nsample 64 --seqlen 32 "
            f"--format auto_awq --output_dir {self.save_dir} --device {self.device}"
        )
        assert not (res > 0 or res == -1), "Phi-3.5 tuning fail"

        ## test infer
        from transformers import AutoModelForCausalLM, AutoProcessor

        quantized_model_path = os.path.join(self.save_dir, "Phi-3.5-vision-instruct-w4g128")
        res = os.system(f"cp /models/Phi-3.5-vision-instruct/*.py {quantized_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map=f"cuda:{self.device}", trust_remote_code=True, torch_dtype="auto"
        )
        assert "WQLinear_GEMM" in str(
            type(model.model.vision_embed_tokens.img_processor.vision_model.encoder.layers[0].mlp.fc1)
        ), "model quantization failed."
        processor = AutoProcessor.from_pretrained(quantized_model_path, trust_remote_code=True, num_crops=4)

        image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        content = "Describe this image."
        messages = [
            {"role": "user", "content": "<|image_1|>\n" + content},
        ]

        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = Image.open(requests.get(image_url, stream=True).raw)
        inputs = processor(prompt, image_inputs, return_tensors="pt").to(model.device)

        generation_args = {
            "max_new_tokens": 1000,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

        # remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(response)
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @pytest.mark.skipif(
        transformers_version >= version.parse("4.57.0"),
        reason="transformers api changed",
    )
    def test_glm(self):
        model_path = "/models/glm-4v-9b/"
        ## test tune
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {self.python_path} -m auto_round "
            f"--model {model_path} --iter 1 --output_dir {self.save_dir} --device {self.device}"
        )
        assert not (res > 0 or res == -1), "glm-4v-9b tuning fail"

    def test_granite_vision(self):
        model_path = "/models/granite-vision-3.2-2b"
        ## test tune
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {self.python_path} -m auto_round "
            f"--model {model_path} --iter 1 --output_dir {self.save_dir} --device {self.device}"
        )
        assert not (res > 0 or res == -1), "granite-vision-3.2-2b tuning fail"
