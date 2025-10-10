import os
import shutil
import sys
import unittest

sys.path.insert(0, "../..")

import requests
from PIL import Image

from auto_round import AutoRoundConfig  # # must import for auto-round format
from auto_round.testing_utils import require_gptqmodel, require_package_version_ut, require_vlm_env


class TestSupportVLMS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # self.save_dir = os.path.join(os.path.dirname(__file__), "ut_saved")
        self.save_dir = os.path.join(os.path.dirname("/data5/hengguo"), "ut_saved")
        self.python_path = sys.executable
        self.device = 0

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_dir, ignore_errors=True)

    # @require_gptqmodel
    # def test_qwen2(self):
    #     model_path = "/models/Qwen2-VL-2B-Instruct/"
    #     # test tune
    #     res = os.system(
    #         f"cd ../.. && {self.python_path} -m auto_round --mllm "
    #         f"--model {model_path} --iter 2 --output_dir {self.save_dir} --device {self.device}")
    #     self.assertFalse(res > 0 or res == -1, msg="qwen2 tuning fail")

    #     # test infer
    #     quantized_model_path = os.path.join(self.save_dir, "Qwen2-VL-2B-Instruct-w4g128")

    #     from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    #     model = Qwen2VLForConditionalGeneration.from_pretrained(
    #         quantized_model_path,
    #         torch_dtype="float16",
    #         device_map=f"cuda:{self.device}",
    #     )
    #     processor = AutoProcessor.from_pretrained(quantized_model_path)
    #     image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "image",
    #                     "image": image_url,
    #                 },
    #                 {"type": "text", "text": "Describe this image."},
    #             ],
    #         }
    #     ]

    #     # Preparation for inference
    #     text = processor.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     image_inputs = Image.open(requests.get(image_url, stream=True).raw)
    #     inputs = processor(
    #         text=[text],
    #         images=image_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #     )
    #     inputs = inputs.to(model.device)

    #     generated_ids = model.generate(**inputs, max_new_tokens=128)
    #     generated_ids_trimmed = [
    #         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    #     ]
    #     output_text = processor.batch_decode(
    #         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    #     )
    #     print(output_text[0])
    #     shutil.rmtree(quantized_model_path, ignore_errors=True)

    @require_vlm_env
    def test_phi3(self):
        model_path = "/models/Phi-3.5-vision-instruct/"
        ## test tune
        res = os.system(
            f"cd ../.. && {self.python_path} -m auto_round --mllm "
            f"--model {model_path} --iter 2 --output_dir {self.save_dir} --device {self.device}"
        )
        self.assertFalse(res > 0 or res == -1, msg="Phi-3.5 tuning fail")

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
            f"cd ../.. && {self.python_path} -m auto_round --mllm "
            f"--model {model_path} --iter 2 --quant_nontext_module "
            f"--nsample 64 --seqlen 32 "
            f"--format auto_awq --output_dir {self.save_dir} --device {self.device}"
        )
        self.assertFalse(res > 0 or res == -1, msg="Phi-3.5 tuning fail")

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

    @require_vlm_env
    def test_llava(self):
        model_path = "/models/llava-v1.5-7b/"
        ## test tune
        res = os.system(
            f"cd ../.. && {self.python_path} -m auto_round --mllm "
            f"--model {model_path} --iter 2 --output_dir {self.save_dir} --device {self.device}"
        )
        self.assertFalse(res > 0 or res == -1, msg="llava-v1.5-7b tuning fail")

        ## test infer
        from llava.model.builder import load_pretrained_model
        from llava.train.train import preprocess, preprocess_multimodal

        class DataArgs:
            is_multimodal = True
            mm_use_im_start_end = False

        quantized_model_path = os.path.join(self.save_dir, "llava-v1.5-7b-w4g128")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            quantized_model_path,
            model_base=None,
            model_name=quantized_model_path,
            torch_dtype="auto",
            device_map=f"cuda:{self.device}",
        )
        image_url = "http://images.cocodataset.org/train2017/000000116003.jpg"
        messages = [{"from": "human", "value": "What is the tennis player doing in the image?\n<image>"}]

        # Preparation for inference
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        image_input = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].to(model.device)
        input_data = preprocess_multimodal([messages], DataArgs())
        inputs = preprocess(input_data, tokenizer, has_image=(image_input is not None))

        output = model.generate(
            inputs["input_ids"].to(model.device), images=image_input.unsqueeze(0).half(), max_new_tokens=50
        )
        print(tokenizer.batch_decode(output))
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    # @require_gptqmodel
    # def test_llama(self):
    #     model_path = "/models/Llama-3.2-11B-Vision-Instruct/"
    #     ## test tune
    #     res = os.system(
    #         f"cd ../.. && {self.python_path} -m auto_round --mllm "
    #         f"--model {model_path} --iter 2 --output_dir {self.save_dir} --device {self.device}")
    #     self.assertFalse(res > 0 or res == -1, msg="llama-3.2 tuning fail")

    #     ## test infer
    #     from transformers import MllamaForConditionalGeneration, AutoProcessor
    #     quantized_model_path = os.path.join(self.save_dir, "Llama-3.2-11B-Vision-Instruct-w4g128")
    #     model = MllamaForConditionalGeneration.from_pretrained(
    #         quantized_model_path,
    #         torch_dtype="float16",
    #         device_map=f"cuda:{self.device}",
    #     )
    #     processor = AutoProcessor.from_pretrained(quantized_model_path)
    #     image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    #     messages = [
    #         {"role": "user", "content": [
    #             {"type": "image"},
    #             {"type": "text", "text": "Please write a haiku for this one, it would be: "}
    #         ]}
    #     ]

    #     # Preparation for inference
    #     image = Image.open(requests.get(image_url, stream=True).raw)
    #     input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    #     inputs = processor(
    #         image,
    #         input_text,
    #         add_special_tokens=False,
    #         return_tensors="pt"
    #     ).to(model.device)

    #     output = model.generate(**inputs, max_new_tokens=50)
    #     print(processor.decode(output[0]))
    #     shutil.rmtree(quantized_model_path, ignore_errors=True)

    @require_vlm_env
    def test_cogvlm(self):
        model_path = "/models/cogvlm2-llama3-chat-19B/"
        ## test tune
        res = os.system(
            f"cd ../.. && {self.python_path} -m auto_round --mllm "
            f"--model {model_path} --iter 2 --output_dir {self.save_dir} --device {self.device}"
        )
        self.assertFalse(res > 0 or res == -1, msg="cogvlm2 tuning fail")

        ## test infer
        DEVICE = f"cuda:{self.device}"
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quantized_model_path = os.path.join(self.save_dir, "cogvlm2-llama3-chat-19B-w4g128")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, trust_remote_code=True)
        model = (
            AutoModelForCausalLM.from_pretrained(
                quantized_model_path,
                torch_dtype="float16",
                trust_remote_code=True,
                device_map=DEVICE,
            )
            .to(DEVICE)
            .eval()
        )

        image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        content = "Describe this image."

        text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
        query = text_only_template.format(content)

        image = Image.open(requests.get(image_url, stream=True).raw)
        input_by_model = model.build_conversation_input_ids(
            tokenizer, query=query, images=[image], template_version="chat"
        )
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(DEVICE),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(DEVICE),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(DEVICE),
            "images": [[input_by_model["images"][0].to(DEVICE).to(model.dtype)]] if image is not None else None,
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
        }

        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(outputs[0])
        response = response.split("<|end_of_text|>")[0]
        print(response)
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    # def test_72b(self):
    #     model_path = "/models/Qwen2-VL-72B-Instruct/"
    #     res = os.system(
    #         f"cd ../.. && {self.python_path} -m auto_round --mllm "
    #         f"--model {model_path} --iter 1 --nsamples 1 --bs 1 --output_dir {self.save_dir} --device {self.device}"
    #         )
    #     self.assertFalse(res > 0 or res == -1, msg="qwen2-72b tuning fail")
    #     shutil.rmtree(self.save_dir, ignore_errors=True)

    @require_vlm_env
    def test_deepseek_vl2(self):
        model_path = "/models/deepseek-vl2-tiny"
        res = os.system(
            f"cd ../.. && {self.python_path} -m auto_round --mllm "
            f"--model {model_path} --iter 3 --nsamples 10 --bs 4 --output_dir {self.save_dir} --device auto --group_size 32 "
            f"--fp_layers language.model.layers.4,language.model.layers.6"
        )
        self.assertFalse(res > 0 or res == -1, msg="deepseek vl2 tuning fail")

        quantized_model_path = os.path.join(self.save_dir, "deepseek-vl2-tiny-w4g32")
        from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
        from transformers import AutoModelForCausalLM

        vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(quantized_model_path)
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            trust_remote_code=True,
            device_map=f"cuda:{self.device}",
            torch_dtype="float16",
        )
        vl_gpt = vl_gpt.eval()

        image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        content = "Describe this image."

        ## single image conversation example
        conversation = [
            {
                "role": "<|User|>",
                "content": content,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = Image.open(requests.get(image_url, stream=True).raw)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=[pil_images], force_batchify=True, system_prompt=""
        )
        prepare_inputs = prepare_inputs.to(vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = vl_gpt.language.generate(
            input_ids=prepare_inputs["input_ids"],
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"{prepare_inputs['sft_format'][0]}", answer)


if __name__ == "__main__":
    unittest.main()
