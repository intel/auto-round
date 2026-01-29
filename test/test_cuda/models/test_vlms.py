import copy
import os
import re
import shutil

import pytest
import requests
from PIL import Image
from transformers import AutoRoundConfig

from auto_round.testing_utils import require_gptqmodel, require_optimum, require_vlm_env


class TestAutoRound:
    @classmethod
    def setup_class(self):
        self.save_dir = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.save_dir, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    # def test_vision_generation(self):
    #     quantized_model_path = "OPEA/Phi-3.5-vision-instruct-qvision-int4-sym-inc"
    #     from transformers import AutoRoundConfig
    #     device = "auto"  ##cpu, hpu, cuda
    #     quantization_config = AutoRoundConfig(
    #         backend=device
    #     )
    #     model = AutoModelForCausalLM.from_pretrained(quantized_model_path, trust_remote_code=True,
    #                                                  device_map=device, quantization_config=quantization_config)
    #     tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
    #     text = "There is a girl who likes adventure,"
    #     inputs = tokenizer(text, return_tensors="pt").to(model.device)
    #     res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
    #     print(res)
    #     assert (
    #             res == """<s> There is a girl who likes adventure, and she is looking for a partner to go on a treasure hunt. She has found a map that leads to a hidden treasure, but she needs a partner to help her decipher the clues and find the treasure. You""")

    def qwen_inference(self, quantized_model_dir):
        from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)
        processor = AutoProcessor.from_pretrained(quantized_model_dir, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            quantized_model_dir,
            torch_dtype="float16",
            device_map="auto",
            ##revision="df7f44c" ##AutoGPTQ format
        )

        image_url = "https://github.com/intel/auto-round/raw/main/docs/imgs/norm_bias_overview.png"
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

    @require_gptqmodel
    @require_optimum
    def test_vlm_tune(self):
        from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

        from auto_round import AutoRoundMLLM

        ## load the model
        model_name = "/models/Qwen2-VL-2B-Instruct"
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        ## quantize the model
        bits, group_size, sym = 4, 128, True
        autoround = AutoRoundMLLM(
            model, tokenizer, processor, bits=bits, group_size=group_size, sym=sym, iters=1, nsamples=1
        )
        autoround.quantize()

        quantized_model_path = self.save_dir
        autoround.save_quantized(quantized_model_path, format="auto_round", inplace=False)
        self.qwen_inference(quantized_model_path)
        shutil.rmtree(self.save_dir, ignore_errors=True)
        autoround.save_quantized(quantized_model_path, format="auto_gptq", inplace=False)
        self.qwen_inference(quantized_model_path)
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @require_vlm_env
    def test_mm_block_name(self):
        from auto_round.utils import get_block_names

        model_name = "/models/Llama-3.2-11B-Vision-Instruct/"
        from transformers import MllamaForConditionalGeneration

        model = MllamaForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        block_name = get_block_names(model, quant_vision=True)
        assert len(block_name) == 3
        assert any(["vision_model.global_transformer.layers.0" not in n for n in block_name])
        assert any(["vision_model.transformer.layers.0" not in n for n in block_name])
        block_name = get_block_names(model, quant_vision=False)
        assert len(block_name) == 1
        assert get_block_names(model) == block_name

    def test_mllm_detect(self):
        from auto_round.utils import is_mllm_model, llm_load_model, mllm_load_model

        for model_name in [
            "/models/Llama-3.2-11B-Vision-Instruct/",
            "/models/deepseek-vl2-tiny/",
            "/models/gemma-3-12b-it/",
            "/models/Phi-3.5-vision-instruct",
            "/models/Qwen2-VL-2B-Instruct",
            "/models/SmolVLM-256M-Instruct",
            "/models/Mistral-Small-3.1-24B-Instruct-2503",
            "/models/InternVL3-1B",
            "/models/pixtral-12b",
        ]:
            assert is_mllm_model(model_name)
            try:
                model, _, _, _ = mllm_load_model(model_name)
            except:
                continue
            assert is_mllm_model(model)

        for model_name in ["/models/Qwen2.5-1.5B-Instruct/"]:
            assert not is_mllm_model(model_name)
            model, _ = llm_load_model(model_name)
            assert not is_mllm_model(model)
