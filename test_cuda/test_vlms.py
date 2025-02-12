import copy
import shutil
import sys
import unittest
import re
import os

sys.path.insert(0, "..")

from PIL import Image
from auto_round import AutoRoundConfig
import requests


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.save_dir = "./saved"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_dir, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    #
    # def test_vision_generation(self):
    #     quantized_model_path = "OPEA/Phi-3.5-vision-instruct-qvision-int4-sym-inc"
    #     from auto_round import AutoRoundConfig
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
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)
        processor = AutoProcessor.from_pretrained(quantized_model_dir, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            quantized_model_dir,
            torch_dtype="float16",
            device_map="auto",
            ##revision="df7f44c" ##AutoGPTQ format
        )

        image_url = "../docs/imgs/norm_bias_overview.png"
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
        image_inputs = Image.open(image_url)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0])

    def test_vlm_tune(self):
        from auto_round import AutoRoundMLLM
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer

        ## load the model
        model_name = "/models/Qwen2-VL-2B-Instruct"
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, trust_remote_code=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        ## quantize the model
        bits, group_size, sym = 4, 128, True
        autoround = AutoRoundMLLM(model, tokenizer, processor,
                                  bits=bits, group_size=group_size, sym=sym, iters=1, nsamples=1)
        autoround.quantize()

        quantized_model_path = self.save_dir
        autoround.save_quantized(quantized_model_path, format='auto_round', inplace=False)
        self.qwen_inference(quantized_model_path)
        shutil.rmtree(self.save_dir, ignore_errors=True)
        autoround.save_quantized(quantized_model_path, format='auto_gptq', inplace=False)
        self.qwen_inference(quantized_model_path)
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def phi3_infernece(self, quantized_model_dir):
        from transformers import AutoModelForCausalLM, AutoProcessor
        quantized_model_path = os.path.join(quantized_model_dir, "Phi-3.5-vision-instruct-w4g128-auto_round")
        res = os.system(f"cp /models/Phi-3.5-vision-instruct/*.py {quantized_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, 
            device_map="auto", 
            trust_remote_code=True, 
            torch_dtype="float16",
            )
        processor = AutoProcessor.from_pretrained(quantized_model_path, 
        trust_remote_code=True, 
        num_crops=4
        )

        image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        content = "Describe this image."
        messages = [
            {"role": "user", 
            "content": "<|image_1|>\n"+content},
        ]

        prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
        )
        image_inputs = Image.open(requests.get(image_url, stream=True).raw)
        inputs = processor(prompt, image_inputs, return_tensors="pt").to(model.device) 

        generation_args = { 
            "max_new_tokens": 1000, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        generate_ids = model.generate(**inputs, 
        eos_token_id=processor.tokenizer.eos_token_id, 
        **generation_args
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False)[0] 

        print(response)

    def test_quant_not_text(self):
        from auto_round import AutoRoundMLLM
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        ## load the model
        model_name = "/models/Phi-3.5-vision-instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        ## quantize the model
        bits, group_size, sym = 4, 128, True
        autoround = AutoRoundMLLM(model, tokenizer, processor,
                                  bits=bits, group_size=group_size, sym=sym, iters=1, nsamples=1,quant_nontext_module=True)
        autoround.quantize()

        quantized_model_path = "./saved/Phi-3.5-vision-instruct-w4g128-auto_round"
        autoround.save_quantized(quantized_model_path, format='auto_round', inplace=False, safe_serialization=False)
        self.phi3_infernece("./saved")
        shutil.rmtree("./saved", ignore_errors=True)

    def test_quant_not_text_fp_layers(self):
        import  os
        python_path = sys.executable
        absolute_path = os.path.abspath(self.save_dir)
        res = os.system(
            f"cd .. && {python_path} -m auto_round --mllm --model /models/Phi-3.5-vision-instruct "
            f"--fp_layers model.layers.27,model.vision_embed_tokens.img_processor.vision_model.encoder.layers.16 "
            f"--quant_nontext_module --iters 1 --nsamples 1 --output_dir {absolute_path}")
        self.phi3_infernece(absolute_path)
        shutil.rmtree(absolute_path, ignore_errors=True)



if __name__ == "__main__":
    unittest.main()

