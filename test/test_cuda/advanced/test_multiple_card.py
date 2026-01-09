import re
import shutil

import pytest
import torch
from lm_eval.utils import make_table  # pylint: disable=E0401
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.eval.evaluation import simple_evaluate
from auto_round.testing_utils import multi_card, require_gptqmodel, require_greater_than_050

from ...helpers import get_model_path, get_tiny_model


def get_accuracy(data):
    match = re.search(r"\|acc\s+\|[↑↓]\s+\|\s+([\d.]+)\|", data)

    if match:
        accuracy = float(match.group(1))
        return accuracy
    else:
        return 0.0


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
class TestAutoRound:
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
    @require_gptqmodel
    def test_device_map_str(self):
        model_name = "/models/Qwen2-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device_map = ".*q_proj:0,.*k_proj:cuda:0,v_proj:1,.*up_proj:1"
        autoround = AutoRound(model, tokenizer, device_map=device_map)
        autoround.quantize()
        autoround.save_quantized(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        res = simple_evaluate(model="hf", model_args=model_args, tasks=self.tasks, batch_size="auto")
        res = make_table(res)
        accuracy = get_accuracy(res)
        print(accuracy)
        assert accuracy > 0.45  ##0.4786
        shutil.rmtree("./saved", ignore_errors=True)

    @multi_card
    def test_layer_norm(self, tiny_opt_model_path):
        model = AutoModelForCausalLM.from_pretrained(tiny_opt_model_path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path)
        device_map = {"norm": "cuda:1"}
        autoround = AutoRound(
            model, tokenizer, iters=2, device_map=device_map, nsamples=7, seqlen=32, enable_norm_bias_tuning=True
        )
        autoround.quantize()

    @multi_card
    def test_rms_norm(self, tiny_qwen_model_path):
        model = AutoModelForCausalLM.from_pretrained(tiny_qwen_model_path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_model_path)
        device_map = {"norm": "cuda:1"}
        autoround = AutoRound(
            model, tokenizer, iters=2, device_map=device_map, nsamples=7, seqlen=32, enable_norm_bias_tuning=True
        )
        autoround.quantize()

    @multi_card
    def test_act_quantization(self, tiny_qwen_model_path):
        model = AutoModelForCausalLM.from_pretrained(tiny_qwen_model_path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_model_path)
        device_map = {".*q_proj": "0", ".*k_proj": "cuda:1", "v_proj": 1, ".*up_proj": "1"}
        autoround = AutoRound(
            model, tokenizer, iters=2, device_map=device_map, nsamples=7, seqlen=32, act_bits=4, act_dynamic=False
        )
        autoround.quantize()

    @multi_card
    def test_lm_head(self):
        model_path = get_model_path("qwen/Qwen2.5-7B-Instruct")
        model = get_tiny_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device_map = {".*q_proj": "0", ".*k_proj": "cuda:1", "v_proj": 1, ".*up_proj": "1", "lm_head": 1}
        layer_config = {"lm_head": {"bits": 4}}
        autoround = AutoRound(
            model,
            tokenizer,
            iters=2,
            device_map=device_map,
            nsamples=7,
            seqlen=32,
            enable_norm_bias_tuning=True,
            layer_config=layer_config,
        )
        autoround.quantize()

    @multi_card
    def test_device_map(self, tiny_qwen_model_path):
        model = AutoModelForCausalLM.from_pretrained(tiny_qwen_model_path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_model_path)
        device_map = {".*q_proj": "0", ".*k_proj": "cuda:1", "v_proj": 1, ".*up_proj": "cpu"}
        autoround = AutoRound(model, tokenizer, iters=2, device_map=device_map, nsamples=7, seqlen=32)
        autoround.quantize()

        model_name = "OPEA/Meta-Llama-3.1-8B-Instruct-int4-sym-inc"

        device_map = {}
        for i in range(0, 32):
            key = f"model.layers.{str(i)}"
            device_map[key] = "cuda"
        device_map["model.layers.1"] = "cpu"
        device_map["model.layers.2"] = "cpu"
        device_map["model.layers.3"] = "cpu"
        device_map["model.layers.4"] = "cpu"
        device_map["model.layers.5"] = "cpu"
        device_map["model.layers.6"] = "cpu"
        device_map["model.layers.27"] = "cpu"
        device_map["model.layers.28"] = "cpu"
        device_map["model.layers.30"] = "cpu"
        device_map["model.layers.31"] = "cpu"
        device_map["lm_head"] = "cuda"
        device_map["model.norm"] = "cuda"
        device_map["model.rotary_emb"] = "cuda"
        device_map["model.embed_tokens"] = "cuda"

        device_map1 = {}
        for i in range(0, 32):
            key = f"model.layers.{str(i)}"
            device_map1[key] = "cuda"
        device_map1["model.layers.1"] = "cuda:1"
        device_map1["model.layers.2"] = "cuda:1"
        device_map1["model.layers.3"] = "cuda:1"
        device_map1["model.layers.4"] = "cuda:1"
        device_map1["model.layers.5"] = "cuda:1"
        device_map1["model.layers.6"] = "cuda:1"
        device_map1["model.layers.27"] = "cuda:1"
        device_map1["model.layers.28"] = "cuda:1"
        device_map1["model.layers.30"] = "cuda:1"
        device_map1["model.layers.31"] = "cuda:1"
        device_map1["lm_head"] = "cuda:1"
        device_map1["model.norm"] = "cuda"
        device_map1["model.rotary_emb"] = "cuda"
        device_map1["model.embed_tokens"] = "cuda"

        for tmp_device_map in [
            device_map1,
            device_map,
            None,
            0,
            "balanced",
            "balanced_low_0",
            "sequential",
            "cpu",
            "cuda:0",
            "cuda",
            "auto",
        ]:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=tmp_device_map)

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            prompts = [
                "Hello,my name is",
                # "The president of the United States is",
                # "The capital of France is",
                # "The future of AI is",
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

            for i, prompt in enumerate(prompts):
                print(f"Prompt: {prompt}")
                print(f"Generated: {decoded_outputs[i]}")
                print("-" * 50)
            model = None
            del model
            torch.cuda.empty_cache()

    @multi_card
    def test_device_map_dict(self, tiny_opt_model_path):
        device_map = {".*q_proj": "0", ".*k_proj": "cuda:1", "v_proj": 1, ".*up_proj": "1"}
        bits, group_size, sym = 4, 128, False
        model = AutoModelForCausalLM.from_pretrained(tiny_opt_model_path, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path)
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            device_map=device_map,
        )
        autoround.quantize()

        # test model_name
        autoround = AutoRound(
            tiny_opt_model_path,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            device_map=device_map,
        )
        autoround.quantize()

        # test rtn
        autoround = AutoRound(
            tiny_opt_model_path,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=0,
            seqlen=2,
            device_map=device_map,
        )
        autoround.quantize()

    @multi_card
    @require_greater_than_050
    def test_device_map_for_triton(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "OPEA/Meta-Llama-3.1-8B-Instruct-int4-sym-inc"

        device_map = {}
        for i in range(0, 32):
            key = f"model.layers.{str(i)}"
            device_map[key] = "cuda"
        device_map["model.layers.1"] = "cpu"
        device_map["model.layers.2"] = "cpu"
        device_map["model.layers.3"] = "cpu"
        device_map["model.layers.4"] = "cpu"
        device_map["model.layers.5"] = "cpu"
        device_map["model.layers.6"] = "cpu"
        device_map["model.layers.27"] = "cpu"
        device_map["model.layers.28"] = "cpu"
        device_map["model.layers.30"] = "cpu"
        device_map["model.layers.31"] = "cpu"
        device_map["lm_head"] = "cuda"
        device_map["model.norm"] = "cuda"
        device_map["model.rotary_emb"] = "cuda"
        device_map["model.embed_tokens"] = "cuda"

        device_map1 = {}
        for i in range(0, 32):
            key = f"model.layers.{str(i)}"
            device_map1[key] = "cuda"
        device_map1["model.layers.1"] = "cuda:1"
        device_map1["model.layers.2"] = "cuda:1"
        device_map1["model.layers.3"] = "cuda:1"
        device_map1["model.layers.4"] = "cuda:1"
        device_map1["model.layers.5"] = "cuda:1"
        device_map1["model.layers.6"] = "cuda:1"
        device_map1["model.layers.27"] = "cuda:1"
        device_map1["model.layers.28"] = "cuda:1"
        device_map1["model.layers.30"] = "cuda:1"
        device_map1["model.layers.31"] = "cuda:1"
        device_map1["lm_head"] = "cuda:1"
        device_map1["model.norm"] = "cuda"
        device_map1["model.rotary_emb"] = "cuda"
        device_map1["model.embed_tokens"] = "cuda"
        from auto_round import AutoRoundConfig

        quantization_config = AutoRoundConfig(backend="tritonv2")

        for tmp_device_map in [
            device_map1,
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
                "Hello,my name is",
                # "The president of the United States is",
                # "The capital of France is",
                # "The future of AI is",
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

            for i, prompt in enumerate(prompts):
                print(f"Prompt: {prompt}")
                print(f"Generated: {decoded_outputs[i]}")
                print("-" * 50)
            model = None
            del model
            torch.cuda.empty_cache()

    @multi_card
    def test_mllm_device_map(self):
        model_name = get_model_path("qwen/Qwen2-VL-2B-Instruct/")
        from auto_round import AutoRoundMLLM

        device_map = "0,1"
        ar = AutoRoundMLLM(model_name, device_map=device_map)
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
        assert ar.model.model.visual.blocks[0].mlp.fc1.tuning_device == "cuda:1"
