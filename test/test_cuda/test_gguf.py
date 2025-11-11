import os
import shutil
import sys
import unittest

sys.path.insert(0, "../..")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.testing_utils import require_gguf


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @require_gguf
    def test_gguf_format(self):
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        bits, group_size, sym = 4, 32, False
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            nsamples=2,
            dataset=LLMDataLoader(),
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
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        res = os.system(
            f"cd ../.. && {sys.executable} -m auto_round --model {model_path} --iter 2 "
            f"--output_dir {save_dir} --nsample 2 --format gguf:q4_0 --device 0"
        )
        print(save_dir)
        self.assertFalse(res > 0 or res == -1, msg="qwen2 tuning fail")

        from llama_cpp import Llama

        gguf_file = os.listdir("saved/Qwen2.5-0.5B-Instruct-gguf")[0]
        llm = Llama(f"saved/Qwen2.5-0.5B-Instruct-gguf/{gguf_file}", n_gpu_layers=-1)
        output = llm("There is a girl who likes adventure,", max_tokens=32)
        print(output)
        shutil.rmtree("./saved", ignore_errors=True)

    @require_gguf
    def test_q2_k_export(self):
        bits, group_size, sym = 2, 16, False
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=1,
            dataset=LLMDataLoader(),
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

        from auto_round.eval.evaluation import simple_evaluate_user_model

        result = simple_evaluate_user_model(model, autoround.tokenizer, batch_size=16, tasks="piqa")
        self.assertGreater(result["results"]["piqa"]["acc,none"], 0.45)

        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @require_gguf
    def test_basic_usage(self):
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        python_path = sys.executable
        res = os.system(
            f"cd ../.. && {python_path} -m auto_round --model {model_name} --eval_task_by_task"
            f" --tasks piqa,openbookqa --bs 16 --iters 1 --nsamples 1 --format fake,gguf:q4_0 --eval_model_dtype bf16"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("./saved", ignore_errors=True)

    @require_gguf
    def test_q4_0(self):
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
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
        self.assertGreater(result["results"]["piqa"]["acc,none"], 0.54)
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @require_gguf
    def test_q4_1(self):
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        bits, group_size, sym = 4, 32, False
        autoround = AutoRound(model=model_name, bits=bits, group_size=group_size, sym=sym, iters=1, data_type="int")
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="gguf:q4_1")
        gguf_file = os.listdir(quantized_model_path)[0]
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = autoround.tokenizer(text, return_tensors="pt").to(model.device)
        print(autoround.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))

        from auto_round.eval.evaluation import simple_evaluate_user_model

        result = simple_evaluate_user_model(model, autoround.tokenizer, batch_size=16, tasks="piqa")
        self.assertGreater(result["results"]["piqa"]["acc,none"], 0.54)
        shutil.rmtree("./saved", ignore_errors=True)

    @require_gguf
    def test_all_format(self):
        from auto_round.export.export_to_gguf.config import GGUF_CONFIG

        python_path = sys.executable
        for model_name in ["/models/Qwen3-8B/", "/models/Llama-3.2-3B/", "/models/Meta-Llama-3.1-8B-Instruct"]:
            for gguf_format in GGUF_CONFIG.keys():
                print(model_name, gguf_format)
                res = os.system(
                    f"cd ../.. && {python_path} -m auto_round --model {model_name} "
                    f" --bs 16 --iters 1 --nsamples 1 --format fake,{gguf_format}"
                )
                if res > 0 or res == -1:
                    assert False, "cmd line test fail, please have a check"
                shutil.rmtree("../../tmp_autoround", ignore_errors=True)

                res = os.system(
                    f"cd ../.. && {python_path} -m auto_round --model {model_name} "
                    f" --bs 16 --iters 0 --nsamples 1 --format {gguf_format}"
                )
                if res > 0 or res == -1:
                    assert False, "cmd line test fail, please have a check"
                shutil.rmtree("../../tmp_autoround", ignore_errors=True)

    @require_gguf
    def test_vlm_gguf(self):
        model_name = "/models/Qwen2.5-VL-7B-Instruct"
        from auto_round import AutoRoundMLLM
        from auto_round.utils import mllm_load_model

        # model, processor, tokenizer, image_processor = mllm_load_model(model_name)
        # autoround = AutoRoundMLLM(
        #     model,
        #     tokenizer=tokenizer,
        #     processor=processor,
        #     image_processor=image_processor,
        #     device="auto",
        #     iters=0,
        # )
        # quantized_model_path = "./saved"
        # autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_0")
        # self.assertTrue("mmproj-model.gguf" in os.listdir("./saved"))
        # file_size = os.path.getsize("./saved/Qwen2.5-VL-7B-Instruct-Q4_0.gguf") / 1024**2
        # self.assertAlmostEqual(file_size, 4226, delta=5.0)
        # file_size = os.path.getsize("./saved/mmproj-model.gguf") / 1024**2
        # self.assertAlmostEqual(file_size, 2580, delta=5.0)
        # shutil.rmtree("./saved", ignore_errors=True)

        model_name = "/models/gemma-3-12b-it"

        model, processor, tokenizer, image_processor = mllm_load_model(model_name)
        autoround = AutoRoundMLLM(
            model,
            tokenizer=tokenizer,
            processor=processor,
            image_processor=image_processor,
            device="auto",
            nsamples=32,
            iters=1,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_k_m")
        self.assertTrue("mmproj-model.gguf" in os.listdir("./saved"))
        file_size = os.path.getsize("./saved/gemma-3-12B-it-Q4_K_M.gguf") / 1024**2
        self.assertAlmostEqual(file_size, 6568, delta=5.0)
        file_size = os.path.getsize("./saved/mmproj-model.gguf") / 1024**2
        self.assertAlmostEqual(file_size, 1599, delta=5.0)
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    # @require_gguf
    # def test_llama_4(self):
    #     model_name = "/dataset/Llama-4-Scout-17B-16E-Instruct/"
    #     from auto_round import AutoRoundMLLM
    #     from auto_round.utils import mllm_load_model

    #     model, processor, tokenizer, image_processor = mllm_load_model(model_name, use_auto_mapping=False)
    #     autoround = AutoRoundMLLM(
    #         model,
    #         tokenizer=tokenizer,
    #         processor=processor,
    #         image_processor=image_processor,
    #         device="auto",
    #         iters=0,
    #     )
    #     quantized_model_path = "/dataset/Llam-4-test"
    #     shutil.rmtree(quantized_model_path, ignore_errors=True)
    #     autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_0")
    #     self.assertTrue("mmproj-model.gguf" in os.listdir(quantized_model_path))
    #     file_size = (
    #         os.path.getsize(os.path.join(quantized_model_path, "Llama-4-Scout-17B-16E-Instruct-16x17B-Q4_0.gguf"))
    #         / 1024**2
    #     )
    #     self.assertAlmostEqual(file_size, 58093.62, delta=1.0)
    #     file_size = os.path.getsize(os.path.join(quantized_model_path, "mmproj-model.gguf")) / 1024**2
    #     self.assertAlmostEqual(file_size, 3326.18, delta=5.0)
    #     shutil.rmtree(quantized_model_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
