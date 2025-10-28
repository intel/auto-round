import os
import shutil
import sys
import unittest

sys.path.insert(0, "../..")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound


class LLMDataLoader:

    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestGGUF(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.model_name = "/tf_dataset/auto_round/models/Qwen/Qwen2.5-0.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_basic_usage(self):
        python_path = sys.executable
        res = os.system(
            f"cd ../.. && {python_path} -m auto_round --model /tf_dataset/auto_round/models/benzart/gemma-2b-it-fine-tuning-for-code-test "
            f" --bs 16 --iters 0 --nsamples 1 --format gguf:q4_k_m"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("./saved", ignore_errors=True)

        res = os.system(
            f"cd ../.. && {python_path} -m auto_round --model {self.model_name}"
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

        # from auto_round.eval.evaluation import simple_evaluate_user_model
        # result = simple_evaluate_user_model(model, self.tokenizer, batch_size=16, tasks="openbookqa", eval_model_dtype="bf16")
        # # 0.246
        # self.assertGreater(result['results']['openbookqa']['acc,none'], 0.23)
        shutil.rmtree("./saved", ignore_errors=True)

    # def test_q4_1(self):
    #     bits, group_size, sym = 4, 32, False
    #     autoround = AutoRound(
    #         self.model, self.tokenizer, bits=bits, group_size=group_size, sym=sym, iters=1, data_type="int", nsamples=1
    #     )
    #     quantized_model_path = "./saved"
    #
    #     autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q4_1")
    #     gguf_file = os.listdir(quantized_model_path)[0]
    #     model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
    #     text = "There is a girl who likes adventure,"
    #     inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
    #     print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
    #
    #     # from auto_round.eval.evaluation import simple_evaluate_user_model
    #     # result = simple_evaluate_user_model(model, self.tokenizer, batch_size=16, tasks="openbookqa", eval_model_dtype="bf16")
    #     # # 0.23
    #     # self.assertGreater(result['results']['openbookqa']['acc,none'], 0.22)
    #     shutil.rmtree("./saved", ignore_errors=True)

    def test_func(self):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model_name,
            # bits=bits,
            # group_size=group_size,
            # sym=sym,
            iters=1,
            nsamples=1,
            seqlen=10,
            # data_type="int"
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q*_1")
        self.assertTrue(autoround.group_size == 32)
        self.assertFalse(autoround.sym)
        gguf_file = os.listdir("saved")[0]
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
        shutil.rmtree("./saved", ignore_errors=True)

        # model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        # autoround = AutoRound(
        #     model,
        #     self.tokenizer,
        #     bits=3,
        #     group_size=16,
        #     sym=True,
        #     iters=1,
        #     nsamples=1,
        #     data_type="int_sym_dq",
        #     super_group_size=16,
        #     super_bits=6,
        # )
        quantized_model_path = "./saved"
        # autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q*_k_s")
        # from auto_round.eval.evaluation import simple_evaluate_user_model
        # gguf_file = os.listdir("saved")[0]
        # model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        # result = simple_evaluate_user_model(model, self.tokenizer, batch_size=16, tasks="lambada_openai", eval_model_dtype="bf16")
        # self.assertGreater(result['results']['lambada_openai']['acc,none'], 0.5)
        shutil.rmtree("./saved", ignore_errors=True)

    #
    # def test_q5_k(self):
    #     model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    #     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
    #     autoround = AutoRound(
    #         model,
    #         self.tokenizer,
    #         bits=5,
    #         group_size=32,
    #         sym=False,
    #         iters=1,
    #         nsamples=1,
    #         data_type="int_asym_dq",
    #         super_group_size=8,
    #         super_bits=6,
    #     )
    #     quantized_model_path = "./saved"
    #     autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q*_k_s")
    #     gguf_file = os.listdir("saved")[0]
    #     model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
    #     text = "There is a girl who likes adventure,"
    #     inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
    #     print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
    #     shutil.rmtree("./saved", ignore_errors=True)

    # def test_q6_k(self):
    #     model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    #     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
    #     autoround = AutoRound(
    #         model,
    #         self.tokenizer,
    #         bits=6,
    #         group_size=16,
    #         sym=True,
    #         iters=1,
    #         nsamples=1,
    #         data_type="int_sym_dq",
    #         super_group_size=16,
    #         super_bits=8,
    #     )
    #     quantized_model_path = "./saved"
    #     autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q*_k")
    #     gguf_file = os.listdir("saved")[0]
    #     model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
    #     text = "There is a girl who likes adventure,"
    #     inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
    #     print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
    #     shutil.rmtree("./saved", ignore_errors=True)

    def test_gguf_baseline(self):
        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen2.5-1.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=3,
            group_size=16,
            sym=True,
            iters=0,
            nsamples=8,
            seqlen=2,
            data_type="rtn_int_sym_dq",
            super_group_size=16,
            super_bits=6,
            disable_opt_rtn=True,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="fake")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
        shutil.rmtree("./saved", ignore_errors=True)
        #
        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        # autoround = AutoRound(
        #     model,
        #     self.tokenizer,
        #     bits=5,
        #     group_size=32,
        #     sym=True,
        #     iters=0,
        #     nsamples=8,
        #     data_type="int_asym_dq",
        #     super_group_size=8,
        #     super_bits=6,
        #     disable_opt_rtn=True,
        # )
        # quantized_model_path = "./saved"
        # autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q5_k_s,fake")
        # model = AutoModelForCausalLM.from_pretrained(quantized_model_path + "/fake", device_map="auto")
        # text = "There is a girl who likes adventure,"
        # inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        # print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
        # shutil.rmtree("./saved", ignore_errors=True)

    def test_q4_k_m(self):
        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen2.5-1.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        layer_config = {
            "lm_head": {
                "bits": 4,
                "group_size": 32,
                "sym": False,
                "data_type": "int_asym_dq",
                "super_bits": 6,
                "super_group_size": 8,
            },
            "model.embed_tokens": {"bits": 6, "group_size": 32, "super_bits": 6, "super_group_size": 8},
            "model.layers.12.mlp.gate_proj": {"bits": 3},
            "model.layers.10.mlp.gate_proj": {"bits": 8},
        }
        autoround = AutoRound(
            model,
            tokenizer,
            layer_config=layer_config,
            iters=0,
            seqlen=1,
            nsamples=8,
            dataset=self.llm_dataloader,
            disable_opt_rtn=True,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_k_m,fake")
        self.assertEqual(autoround.layer_config["model.layers.11.self_attn.v_proj"]["super_group_size"], 16)
        self.assertEqual(autoround.layer_config["model.layers.11.self_attn.v_proj"]["data_type"], "int_sym_dq")
        self.assertEqual(autoround.layer_config["model.layers.7.self_attn.v_proj"]["data_type"], "int_asym_dq")
        self.assertEqual(autoround.model.model.layers[0].self_attn.v_proj.bits, 6)
        self.assertEqual(autoround.model.model.layers[12].self_attn.v_proj.bits, 4)
        self.assertEqual(autoround.model.model.embed_tokens.bits, 6)
        self.assertEqual(autoround.model.model.embed_tokens.group_size, 16)
        self.assertEqual(autoround.model.model.layers[12].mlp.gate_proj.bits, 3)
        self.assertEqual(autoround.model.model.layers[10].mlp.gate_proj.bits, 8)
        self.assertEqual(autoround.layer_config["model.layers.10.mlp.gate_proj"]["mostly"], "gguf:q8_0")
        shutil.rmtree("./saved", ignore_errors=True)

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(model, tokenizer, iters=0, nsamples=1, seqlen=128, disable_opt_rtn=False)
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_k_m,fake")
        shutil.rmtree("./saved", ignore_errors=True)

    def test_all_format(self):
        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen2.5-1.5B-Instruct"
        python_path = sys.executable
        # for gguf_format in ["gguf:q4_0", "gguf:q4_1", "gguf:q4_k_m", "gguf:q6_k"]:
        for gguf_format in ["gguf:q4_k_m"]:
            res = os.system(
                f"cd ../.. && {python_path} -m auto_round --model {model_name} "
                f" --bs 16 --iters 1 --nsamples 1 --seqlen 16 --format {gguf_format}"
            )
            if res > 0 or res == -1:
                assert False, "cmd line test fail, please have a check"
            shutil.rmtree("../../tmp_autoround", ignore_errors=True)

            res = os.system(
                f"cd ../.. && {python_path} -m auto_round --model {model_name}"
                f" --bs 16 --iters 0 --nsamples 1 --seqlen 16 --format fake,{gguf_format}"
            )
            if res > 0 or res == -1:
                assert False, "cmd line test fail, please have a check"
            shutil.rmtree("../../tmp_autoround", ignore_errors=True)

    def test_vlm_gguf(self):
        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen2-VL-2B-Instruct"
        from auto_round import AutoRoundMLLM
        from auto_round.utils import mllm_load_model

        model, processor, tokenizer, image_processor = mllm_load_model(model_name)
        autoround = AutoRoundMLLM(
            model,
            tokenizer=tokenizer,
            processor=processor,
            image_processor=image_processor,
            iters=0,
            nsamples=8,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_0")
        self.assertTrue("mmproj-model.gguf" in os.listdir("./saved"))
        for file_name in os.listdir(quantized_model_path):
            file_size = os.path.getsize(os.path.join(quantized_model_path, file_name)) / 1024**2
            if file_name == "mmproj-model.gguf":
                self.assertAlmostEqual(file_size, 2535, delta=1.0)
            else:
                self.assertAlmostEqual(file_size, 892, delta=1.0)
        shutil.rmtree("./saved", ignore_errors=True)

    def test_qtype_setting(self):
        # Qwen2.5-0.5B-Instruct no output, token_embed q6_k fallbakc to q8_0 336M
        # Qwen3-0.6B output q6_k, token_embed q4_0  448M
        # Qwen3-8B output q6_k, token_embed q4_0 4.5G
        # Llama-3.2-1B-Instruct o output, token_embed q6_k 736M
        from auto_round.compressors import get_layer_config_by_gguf_format, set_layer_config
        from auto_round.export.export_to_gguf.config import ModelType

        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen2.5-0.5B-Instruct"
        ar = AutoRound(model=model_name, scheme="gguf:q4_0", iters=0)
        ar.formats = ["gguf:q4_0"]
        ar.layer_config, _, _ = set_layer_config(
            ar.model,
            ar.layer_config,
            ar.scheme,
            ar.scale_dtype,
            ar.supported_types,
            ar.inner_supported_types,
            ar.quant_block_list,
            ar.fp_layers,
            ar.quant_lm_head,
            enable_gguf_official_mixed=True,
            is_mllm=ar.mllm,
        )
        self.assertTrue(ar.layer_config["model.embed_tokens"]["bits"] == 8)
        self.assertTrue("lm_head" not in ar.layer_config)

        model_name = "Qwen/Qwen3-0.6B"
        ar = AutoRound(model=model_name, scheme="gguf:q4_0", iters=0)
        ar.formats = ["gguf:q4_0"]
        ar.layer_config, _, _ = set_layer_config(
            ar.model,
            ar.layer_config,
            ar.scheme,
            ar.scale_dtype,
            ar.supported_types,
            ar.inner_supported_types,
            ar.quant_block_list,
            ar.fp_layers,
            ar.quant_lm_head,
            enable_gguf_official_mixed=True,
            is_mllm=ar.mllm,
        )
        self.assertTrue(ar.layer_config["model.embed_tokens"]["bits"] == 4)
        self.assertTrue(ar.layer_config["lm_head"]["bits"] == 6 and ar.layer_config["lm_head"]["super_bits"] == 8)

        layer_config = {
            "model.embed_tokens": {"bits": 6, "super_bits": 8},
            "lm_head": {"bits": 4},
        }
        ar = AutoRound(model=model_name, scheme="gguf:q4_0", iters=0, layer_config=layer_config)
        ar.formats = ["gguf:q4_0"]
        ar.layer_config, _, _ = set_layer_config(
            ar.model,
            ar.layer_config,
            ar.scheme,
            ar.scale_dtype,
            ar.supported_types,
            ar.inner_supported_types,
            ar.quant_block_list,
            ar.fp_layers,
            ar.quant_lm_head,
            enable_gguf_official_mixed=True,
            is_mllm=ar.mllm,
        )
        self.assertTrue(ar.layer_config["lm_head"]["bits"] == 4)
        self.assertTrue(
            ar.layer_config["model.embed_tokens"]["bits"] == 6
            and ar.layer_config["model.embed_tokens"]["super_bits"] == 8
        )


if __name__ == "__main__":
    unittest.main()
