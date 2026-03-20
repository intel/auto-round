import copy
import shutil

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ..helpers import get_model_path


class TestAutoRoundXPU:
    @classmethod
    def setup_class(self):
        self.device = "xpu"
        pass

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
        pass

    def test_gptq_format(self, dataloader):
        model_name = get_model_path("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", trust_remote_code=True, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, True

        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path)

        quantization_config = AutoRoundConfig(backend="auto")
        # device_map="auto" doesn't work, must use "xpu"
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map=self.device, quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert "!!!" not in res

    def test_awq_format(self, dataloader):
        model_name = get_model_path("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", trust_remote_code=True, device_map=self.device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:auto_awq")

        quantization_config = AutoRoundConfig(backend="auto")
        # device_map="auto" doesn't work, must use "xpu"
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map=self.device, quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert "!!!" not in res

    @pytest.mark.parametrize(
        "scheme", ["W4A16", "W2A16", "W3A16", "W8A16", "MXFP4", "MXFP8", "NVFP4", "FPW8A16", "FP8_STATIC"]
    )
    def test_scheme(self, scheme, dataloader):
        model_name = get_model_path("facebook/opt-125m")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        ar = AutoRound(
            model=model_name,
            tokenizer=tokenizer,
            nsamples=32,
            seqlen=10,
            iters=1,
            device_map=self.device,
            scheme=scheme,
            dataset=dataloader,
        )
        quantized_model_path = "./saved"
        ar.quantize_and_save(output_dir=quantized_model_path, inplace=True, format="auto_round")

        # test loading
        if scheme not in ["FPW8A16"]:  # FPW8A16 group_size is 0
            # device_map="auto" doesn't work, must use "xpu"
            model = AutoModelForCausalLM.from_pretrained(
                quantized_model_path,
                device_map=self.device,
            )

        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_vlm_model(self, dataloader):
        scheme = "W4A16"
        model_name = get_model_path("Qwen/Qwen2-VL-2B-Instruct")
        from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

        fp32_model = Qwen2VLForConditionalGeneration.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        ar = AutoRound(
            model=model_name,
            nsamples=1,
            iters=0,
            seqlen=10,
            disable_opt_rtn=True,
            device_map=self.device,
            scheme=scheme,
            dataset=dataloader,
        )

        quantized_model_path = "./saved"
        ar.quantize_and_save(output_dir=quantized_model_path, inplace=True, format="auto_round")

        quantization_config = AutoRoundConfig(backend="auto")
        import requests
        from PIL import Image

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            quantized_model_path,
            torch_dtype="float16",
            device_map=self.device,
            quantization_config=quantization_config,
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

    def test_quant_lm_head(self, dataloader):
        bits, sym, group_size = 4, True, 128
        # Note that, to save UT tuning time, the local model is intentionally kept lightweight, using only 2 hidden layers.
        model_name = get_model_path("Qwen/Qwen3-8B")
        layer_config = {
            "lm_head": {"bits": 4},  # set lm_head quant
            "layer": {"bits": 16},
        }
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        ar = AutoRound(
            model=model_name,
            tokenizer=tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            nsamples=2,
            iters=0,
            seqlen=2,
            layer_config=layer_config,
            device_map=self.device,
            dataset=dataloader,
        )
        quantized_model_path = "./saved"
        ar.quantize_and_save(output_dir=quantized_model_path, inplace=True, format="auto_round")

        quantization_config = AutoRoundConfig(backend="auto")
        # device_map="auto" doesn't work, must use "xpu"
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map=self.device, quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
