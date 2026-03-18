import copy
import os
import re
import shutil

import pytest
import requests
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from auto_round import AutoRoundMLLM
from auto_round.utils import get_block_names

from ...envs import require_gptqmodel, require_optimum, require_vlm_env
from ...helpers import get_model_path


class VisionDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for _ in range(2):
            yield {
                "text": [
                    {
                        "role": "user",
                        "content": "<image>\nDescribe the image and list the main objects you see.",
                    },
                    {
                        "role": "assistant",
                        "content": "The image shows a bus and surrounding street scene.",
                    },
                ],
                "image": "http://images.cocodataset.org/train2017/000000033471.jpg",
            }


@pytest.mark.skip_ci(reason="Only tiny model is suggested")
class TestAutoRoundMLLM:
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
        from transformers import AutoProcessor, AutoTokenizer

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
        from auto_round import AutoRoundMLLM

        ## load the model
        model_name = get_model_path("Qwen/Qwen2-VL-2B-Instruct")
        ## quantize the model
        bits, group_size, sym = 4, 128, True
        autoround = AutoRoundMLLM(model_name, bits=bits, group_size=group_size, sym=sym, iters=1, nsamples=1)
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

        model_name = get_model_path("meta-llama/Llama-3.2-11B-Vision-Instruct")
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
            get_model_path("meta-llama/Llama-3.2-11B-Vision-Instruct"),
            get_model_path("deepseek-ai/deepseek-vl2-tiny"),
            get_model_path("google/gemma-3-12b-it"),
            get_model_path("microsoft/Phi-3.5-vision-instruct"),
            get_model_path("Qwen/Qwen2-VL-2B-Instruct"),
            get_model_path("HuggingFaceTB/SmolVLM-256M-Instruct"),
            get_model_path("mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
            get_model_path("OpenGVLab/InternVL3-1B"),
            get_model_path("mistralai/Pixtral-12B-2409"),
        ]:
            assert is_mllm_model(model_name)
            try:
                model, _, _, _ = mllm_load_model(model_name)
            except:
                continue
            assert is_mllm_model(model)

        for model_name in [get_model_path("Qwen/Qwen2.5-1.5B-Instruct")]:
            assert not is_mllm_model(model_name)
            model, _ = llm_load_model(model_name)
            assert not is_mllm_model(model)

    def test_llama32_vision_early_stop_tracking(self):
        """Test early-stop during calibration for Llama-3.2-11B-Vision-Instruct."""
        model_path = get_model_path("meta-llama/Llama-3.2-11B-Vision-Instruct")
        if not os.path.exists(model_path):
            pytest.skip(f"Llama-3.2-11B-Vision-Instruct not found in {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, trust_remote_code=True, device_map="auto", torch_dtype="auto"
        )

        autoround = AutoRoundMLLM(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            bits=4,
            group_size=128,
            nsamples=2,
            batch_size=1,
            iters=1,
            dataset=VisionDataLoader(),
            seqlen=8,
            quant_nontext_module=True,
        )

        call_log = []
        original_should_stop = autoround._should_stop_cache_forward

        def tracked_should_stop(name):
            result = original_should_stop(name)
            call_log.append(
                {
                    "name": name,
                    "result": result,
                    "last_cache_name": getattr(autoround, "last_cache_name", None),
                }
            )
            return result

        autoround._should_stop_cache_forward = tracked_should_stop

        try:
            all_blocks = get_block_names(model, quant_vision=True)
            if not all_blocks:
                pytest.skip("No blocks found in Llama-3.2-11B-Vision-Instruct")

            all_first_block_names = [block[0] for block in all_blocks if block]
            if len(all_first_block_names) < 3:
                pytest.skip("Need 3 block groups for Llama-3.2 vision test")

            inputs = autoround.cache_inter_data(all_first_block_names, nsamples=2)
            print(f"call_log: {call_log}")

            stop_calls = [c for c in call_log if c["result"] is True]
            assert len(stop_calls) > 0, "Should trigger early-stop during calibration"

            last_cache_values = [c["last_cache_name"] for c in call_log]
            assert last_cache_values[0] is None, "last_cache_name should start as None"
            assert (
                last_cache_values[-1] == "model.language_model.layers.0"
            ), "last_cache_name should update to model.language_model.layers.0"

            assert "model.language_model.layers.0" in inputs, "Should have cached language model block"
            assert len(inputs) >= 3, "Should have cached at least 3 input keys"
        finally:
            autoround._should_stop_cache_forward = original_should_stop
