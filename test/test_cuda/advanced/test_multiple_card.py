import os
import re
import shutil
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ...envs import multi_card, require_gptqmodel, require_greater_than_050
from ...helpers import evaluate_accuracy, get_model_path, get_tiny_model

AUTO_ROUND_PATH = __file__.split("/")
AUTO_ROUND_PATH = "/".join(AUTO_ROUND_PATH[: AUTO_ROUND_PATH.index("test")])


@pytest.mark.skip_ci(reason="multiple card test")
class TestAutoRoundCli:
    save_dir = "./saved"
    tasks = "lambada_openai"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @multi_card
    def test_multiple_card_calib(self, tiny_opt_model_path):
        python_path = sys.executable

        ##test llm script
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {tiny_opt_model_path} --devices '0,1' --quant_lm_head --iters 1 --nsamples 1 --output_dir None"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

    @multi_card
    def test_multiple_card_nvfp4(self, tiny_opt_model_path):
        python_path = sys.executable

        ##test llm script
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {tiny_opt_model_path}  --scheme NVFP4 --devices '0,1' --iters 1 --nsamples 1 --enable_torch_compile --low_gpu_mem_usage"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"


@pytest.mark.skip_ci(reason="multiple card test")
class TestAutoRound:
    save_dir = "./saved"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("./tmp_autoround", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @multi_card
    @require_gptqmodel
    def test_device_map_str(self):
        model_name = get_model_path("Qwen/Qwen2-0.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device_map = ".*q_proj:0,.*k_proj:cuda:0,v_proj:1,.*up_proj:1"
        autoround = AutoRound(model, tokenizer, device_map=device_map)
        autoround.quantize()
        autoround.save_quantized(self.save_dir, format="auto_round", inplace=False)
        evaluate_accuracy(self.save_dir, threshold=0.45, batch_size="auto")
        shutil.rmtree("./saved", ignore_errors=True)

    @multi_card
    def test_act_quantization(self, tiny_qwen_model_path):
        model = AutoModelForCausalLM.from_pretrained(tiny_qwen_model_path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_model_path)
        device_map = {".*q_proj": "0", ".*k_proj": "cuda:1", "v_proj": 1, ".*up_proj": "1"}
        autoround = AutoRound(
            model, tokenizer, iters=2, device_map=device_map, nsamples=7, seqlen=32, act_bits=4, act_dynamic=False
        )
        autoround.quantize_and_save()

    @multi_card
    def test_device_map_and_norm_bias_tuning(self, tiny_qwen_model_path):
        device_map = {
            ".*q_proj": "0",
            ".*k_proj": "cuda:1",
            "v_proj": 1,
            ".*up_proj": "cpu",
            "lm_head": 1,
            "norm": "cuda:1",
        }
        autoround = AutoRound(
            tiny_qwen_model_path, iters=2, device_map=device_map, nsamples=7, seqlen=32, enable_norm_bias_tuning=True
        )
        autoround.quantize_and_save()

    @multi_card
    @require_greater_than_050
    def test_device_map_for_triton(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc"

        device_map = {}
        for i in range(0, 32):
            key = f"model.layers.{str(i)}"
            device_map[key] = "cuda:0"
        device_map["model.layers.1"] = "cpu"
        device_map["model.layers.2"] = "cpu"
        device_map["model.layers.3"] = "cpu"
        device_map["model.layers.4"] = "cpu"
        device_map["model.layers.5"] = "cpu"
        device_map["model.layers.6"] = "cuda:1"
        device_map["model.layers.21"] = "cuda:1"
        device_map["lm_head"] = "cuda"
        device_map["model.norm"] = "cuda"
        device_map["model.rotary_emb"] = "cuda"
        device_map["model.embed_tokens"] = "cuda"

        from transformers import AutoRoundConfig

        quantization_config = AutoRoundConfig(backend="tritonv2")

        for tmp_device_map in [
            device_map,
            0,
            "balanced",
            "balanced_low_0",
            "sequential",
            "cuda:0",
            "cuda",
            "auto",
        ]:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map=tmp_device_map, quantization_config=quantization_config
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            prompts = [
                "The capital of France is",
            ]

            texts = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                texts.append(text)

            inputs = tokenizer(texts, return_tensors="pt", padding=False, truncation=True)

            outputs = model.generate(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                do_sample=False,  ## change this to follow official usage
                max_new_tokens=5,
            )
            generated_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], outputs)
            ]

            decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            assert "Paris" in decoded_outputs[0], f"Expected 'Paris' in the output, but got: {decoded_outputs[0]}"

            for i, prompt in enumerate(prompts):
                print(f"Prompt: {prompt}")
                print(f"Generated: {decoded_outputs[i]}")
                print("-" * 50)
            model = None
            del model
            torch.cuda.empty_cache()

    @multi_card
    def test_mllm_device_map(self, tiny_qwen_2_5_vl_model_path):
        from auto_round import AutoRoundMLLM

        device_map = "0,1"
        ar = AutoRoundMLLM(tiny_qwen_2_5_vl_model_path, device_map=device_map)
        assert ar.device == "cuda:0"
        assert ar.device_map == device_map

        device_map = 1
        ar = AutoRoundMLLM(ar.model, ar.tokenizer, processor=ar.processor, device_map=device_map)
        assert ar.device == "cuda:1"
        assert ar.device_map == device_map

        device_map = "auto"
        ar = AutoRoundMLLM(ar.model, ar.tokenizer, processor=ar.processor, device_map=device_map)
        assert ar.device == "cuda"
        assert ar.device_map == device_map

        device_map = {"model.language_model.layers": 0, "model.visual.blocks": 1}
        ar = AutoRoundMLLM(ar.model, ar.tokenizer, processor=ar.processor, device_map=device_map)
        assert ar.model.model.language_model.layers[0].self_attn.q_proj.tuning_device == "cuda:0"
        assert ar.model.model.visual.blocks[0].mlp.gate_proj.tuning_device == "cuda:1"
