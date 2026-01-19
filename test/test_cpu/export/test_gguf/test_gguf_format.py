import os
import shutil
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ....helpers import get_model_path

AUTO_ROUND_PATH = __file__.split("/")
AUTO_ROUND_PATH = "/".join(AUTO_ROUND_PATH[: AUTO_ROUND_PATH.index("test")])


class TestGGUF:

    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("Qwen/Qwen2.5-0.5B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_basic_usage(self, tiny_gemma_model_path, tiny_qwen_model_path):
        python_path = sys.executable
        res = os.system(
            f"PYTHONPATH='AUTO_ROUND_PATH:$PYTHONPATH' {python_path} -m auto_round --model {tiny_gemma_model_path} "
            f" --bs 16 --iters 0 --nsamples 1 --format gguf:q4_k_m"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("./saved", ignore_errors=True)

        res = os.system(
            f"PYTHONPATH='AUTO_ROUND_PATH:$PYTHONPATH' {python_path} -m auto_round --model {tiny_qwen_model_path}"
            f" --bs 16 --iters 1 --nsamples 1 --format fake,gguf:q4_0"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("./saved", ignore_errors=True)

    def test_q4_0(self):
        bits, group_size, sym = 4, 32, True
        autoround = AutoRound(
            self.model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            data_type="int",
            nsamples=1,
            seqlen=8,
        )
        quantized_model_path = "./saved"

        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q4_0")
        gguf_file = os.listdir(quantized_model_path)[0]
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))

        shutil.rmtree("./saved", ignore_errors=True)

    def test_func(self):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model_name,
            iters=1,
            nsamples=1,
            seqlen=10,
            # data_type="int"
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q*_1")
        assert autoround.group_size == 32
        assert not autoround.sym
        gguf_file = os.listdir("saved")[0]
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
        shutil.rmtree("./saved", ignore_errors=True)

    def test_all_format(self, tiny_qwen_model_path):
        model_name = tiny_qwen_model_path
        python_path = sys.executable
        # for gguf_format in ["gguf:q4_0", "gguf:q4_1", "gguf:q4_k_m", "gguf:q6_k"]:
        for gguf_format in ["gguf:q4_k_m"]:
            res = os.system(
                f"PYTHONPATH='AUTO_ROUND_PATH:$PYTHONPATH' {python_path} -m auto_round --model {model_name} "
                f" --bs 16 --iters 1 --nsamples 1 --seqlen 16 --format {gguf_format}"
            )
            if res > 0 or res == -1:
                assert False, "cmd line test fail, please have a check"
            shutil.rmtree("../../tmp_autoround", ignore_errors=True)

            res = os.system(
                f"PYTHONPATH='AUTO_ROUND_PATH:$PYTHONPATH' {python_path} -m auto_round --model {model_name}"
                f" --bs 16 --iters 0 --nsamples 1 --seqlen 16 --format fake,{gguf_format}"
            )
            if res > 0 or res == -1:
                assert False, "cmd line test fail, please have a check"
            shutil.rmtree("../../tmp_autoround", ignore_errors=True)

        # test mixed q2_k_s
        res = os.system(
            f"PYTHONPATH='AUTO_ROUND_PATH:$PYTHONPATH' {python_path} -m auto_round --model {model_name}"
            f" --bs 16 --iters 0 --nsamples 1 --seqlen 16 --scheme GGUF:Q2_K_MIXED"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("../../tmp_autoround", ignore_errors=True)
