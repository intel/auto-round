import os
import shutil
import sys

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.testing_utils import require_gguf

from ...helpers import get_model_path, get_tiny_model, save_tiny_model


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
        shutil.rmtree("runs", ignore_errors=True)

    @require_gguf
    def test_gguf_format(self, tiny_qwen_model_path, dataloader):
        bits, group_size, sym = 4, 32, False
        autoround = AutoRound(
            tiny_qwen_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            nsamples=2,
            dataset=dataloader,
        )
        autoround.quantize()
        quantized_model_path = "./saved"
        autoround.save_quantized(output_dir=quantized_model_path, format="gguf:q4_1")

        from llama_cpp import Llama

        gguf_file = os.listdir("saved")[0]
        llm = Llama(f"saved/{gguf_file}", n_gpu_layers=-1)
        output = llm("There is a girl who likes adventure,", max_tokens=32)
        print(output)
        shutil.rmtree("./saved", ignore_errors=True)

        save_dir = os.path.join(os.path.dirname(__file__), "saved")
        res = os.system(
            f"PYTHONPATH='../../..:$PYTHONPATH' {sys.executable} -m auto_round --model {tiny_qwen_model_path} --iter 2 "
            f"--output_dir {save_dir} --nsample 2 --format gguf:q4_0 --device 0"
        )
        print(save_dir)
        assert not (res > 0 or res == -1), "qwen2 tuning fail"

        from llama_cpp import Llama

        gguf_file = os.listdir(f"{save_dir}/tiny_qwen_model_path-gguf")[0]
        llm = Llama(f"{save_dir}/tiny_qwen_model_path-gguf/{gguf_file}", n_gpu_layers=-1)
        output = llm("There is a girl who likes adventure,", max_tokens=32)
        print(output)
        shutil.rmtree(save_dir, ignore_errors=True)

    @require_gguf
    def test_q2_k_export(self, dataloader):
        bits, group_size, sym = 2, 16, False
        model_path = get_model_path("Qwen/Qwen2.5-1.5B-Instruct")
        model = get_tiny_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=1,
            dataset=dataloader,
            data_type="int_asym_dq",
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="gguf:q2_k_s")
        gguf_file = os.listdir(quantized_model_path)[0]
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = autoround.tokenizer(text, return_tensors="pt").to(model.device)
        result = autoround.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0])
        print(result)
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @require_gguf
    def test_q4_0(self):
        model_name = get_model_path("Qwen/Qwen2.5-0.5B-Instruct")
        bits, group_size, sym = 4, 32, True
        autoround = AutoRound(model_name, bits=bits, group_size=group_size, sym=sym, iters=1, data_type="int")
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="gguf:q4_0")
        gguf_file = os.listdir(quantized_model_path)[0]
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = autoround.tokenizer(text, return_tensors="pt").to(model.device)
        print(autoround.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))

        from auto_round.eval.evaluation import simple_evaluate_user_model

        result = simple_evaluate_user_model(model, autoround.tokenizer, batch_size=16, tasks="piqa")
        assert result["results"]["piqa"]["acc,none"] > 0.54
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @require_gguf
    def test_all_format(self):
        for model_name in ["qwen/Qwen3-8B", "meta-llama/Llama-3.2-3B"]:
            for gguf_format in ["gguf:q5_0", "gguf:q5_1", "gguf:q3_k_m", "gguf:q5_k_m", "gguf:q6_k", "gguf:q8_0"]:
                model_path = get_model_path(model_name)
                tiny_model_path = "tmp_tiny_model"
                tiny_model_path = save_tiny_model(model_path, tiny_model_path, num_layers=2)
                ar = AutoRound(tiny_model_path, scheme=gguf_format, iters=0, nsamples=1, seqlen=16)
                ar.quantize_and_save(output_dir=self.save_dir, format=gguf_format)

                ar = AutoRound(tiny_model_path, scheme=gguf_format, iters=1, nsamples=1, seqlen=16)
                ar.quantize_and_save(output_dir=self.save_dir, format=gguf_format)

                shutil.rmtree(tiny_model_path, ignore_errors=True)
                shutil.rmtree(self.save_dir, ignore_errors=True)

    @require_gguf
    def test_vlm_gguf(self):
        model_name = "/models/Qwen2-VL-2B-Instruct"
        from auto_round import AutoRoundMLLM
        from auto_round.utils import mllm_load_model

        model, processor, tokenizer, image_processor = mllm_load_model(model_name)
        autoround = AutoRoundMLLM(
            model,
            tokenizer=tokenizer,
            processor=processor,
            image_processor=image_processor,
            device="auto",
            iters=0,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_0")
        assert "mmproj-model.gguf" in os.listdir("./saved")
        file_size = os.path.getsize("./saved/Qwen2-VL-2B-Instruct-Q4_0.gguf") / 1024**2
        assert abs(file_size - 894) < 5.0
        file_size = os.path.getsize("./saved/mmproj-model.gguf") / 1024**2
        assert abs(file_size - 2580) < 5.0
        shutil.rmtree("./saved", ignore_errors=True)

        model_name = "/models/gemma-3-12b-it"

        model, processor, tokenizer, image_processor = mllm_load_model(model_name)
        autoround = AutoRoundMLLM(
            model,
            tokenizer=tokenizer,
            processor=processor,
            image_processor=image_processor,
            device="auto",
            nsamples=32,
            iters=0,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_k_m")
        assert "mmproj-model.gguf" in os.listdir("./saved")
        file_size = os.path.getsize("./saved/gemma-3-12B-it-Q4_K_M.gguf") / 1024**2
        assert abs(file_size - 6568) < 5.0
        file_size = os.path.getsize("./saved/mmproj-model.gguf") / 1024**2
        assert abs(file_size - 1599) < 5.0
        shutil.rmtree(quantized_model_path, ignore_errors=True)
