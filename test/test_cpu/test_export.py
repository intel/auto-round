import os
import shutil
import sys
import unittest

from parameterized import parameterized

sys.path.insert(0, "../..")
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound


def _get_folder_size(path: str) -> float:
    """Return folder size in GB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)  # convert to GB


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        self.save_dir = "./saved"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_autogptq_format(self):
        for group_size in [-1, 32, 128]:
            bits, sym = 4, False
            model_name = self.model_name
            autoround = AutoRound(
                model=model_name,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=2,
                seqlen=2,
                dataset=self.llm_dataloader,
            )

            quantized_model_path = "./saved"
            autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")

            if group_size == -1:
                shutil.rmtree("./saved", ignore_errors=True)
                continue
            quantization_config = AutoRoundConfig()
            model = AutoModelForCausalLM.from_pretrained(
                quantized_model_path, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
            )
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
            shutil.rmtree("./saved", ignore_errors=True)

    def test_autoround_format(self):
        for group_size in [-1, 32, 128]:
            bits, sym = 4, True
            model_name = self.model_name
            autoround = AutoRound(
                model=model_name,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=2,
                seqlen=2,
                dataset=self.llm_dataloader,
            )
            quantized_model_path = "./saved"
            autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")

            if group_size == -1:
                shutil.rmtree("./saved", ignore_errors=True)
                continue
            model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu")
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
            shutil.rmtree("./saved", ignore_errors=True)

    def test_autoround_awq_format(self):
        for group_size in [-1, 32, 128]:
            bits, sym = 4, False
            model_name = self.model_name
            autoround = AutoRound(
                model=model_name,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=2,
                seqlen=2,
                dataset=self.llm_dataloader,
            )
            quantized_model_path = "./saved"

            autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:auto_awq")

            # quantization_config = AutoRoundConfig(
            #     backend="cpu"
            # )
            if group_size == -1:
                continue

            model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu")
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
            shutil.rmtree("./saved", ignore_errors=True)

    def test_autoawq_format(self):
        for group_size in [-1, 32, 128]:
            bits, sym = 4, False
            autoround = AutoRound(
                self.model,
                self.tokenizer,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=2,
                seqlen=2,
                dataset=self.llm_dataloader,
            )
            autoround.quantize()
            quantized_model_path = "./saved"

            autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_awq")
            if group_size == -1:
                shutil.rmtree("./saved", ignore_errors=True)
                continue
            quantization_config = AutoRoundConfig()

            model = AutoModelForCausalLM.from_pretrained(
                quantized_model_path, device_map="cpu", quantization_config=quantization_config
            )
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
            shutil.rmtree("./saved", ignore_errors=True)

    def test_autoround_3bit_asym_format(self):
        bits, group_size, sym = 3, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = self.save_dir

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")
        device = "cpu"  ##cpu, hpu, cuda
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_autoround_3bit_sym_format(self):
        bits, group_size, sym = 3, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = self.save_dir

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")
        device = "cpu"  ##cpu, hpu, cuda
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @parameterized.expand([(None,), ("fp8",), ("float16")])
    def test_static_afp8_export(self, static_kv_dtype):
        import os

        from safetensors import safe_open

        model_name = self.model_name
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=8,
            group_size=-1,
            iters=0,
            act_bits=8,
            nsamples=2,
            seqlen=2,
            data_type="fp8",
            act_data_type="fp8",
            act_dynamic=False,
            act_group_size=0,
            static_kv_dtype=static_kv_dtype,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        f = safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt")
        self.assertIn("model.decoder.layers.8.self_attn.k_proj.input_scale", f.keys())
        self.assertIn("model.decoder.layers.8.self_attn.k_proj.weight_scale", f.keys())
        self.assertEqual(f.get_tensor("model.decoder.layers.5.self_attn.v_proj.input_scale").shape, torch.Size([1]))
        self.assertEqual(f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype, torch.float8_e4m3fn)
        if static_kv_dtype is None:
            with torch.no_grad():
                import transformers

                model = transformers.AutoModelForCausalLM.from_pretrained(
                    quantized_model_path,
                    torch_dtype="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                model.eval()
                assert (
                    model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__
                    == "WeightFP8ActFP8StaticQuantLinear"
                ), f"Expected WeightFP8ActFP8StaticQuantLinear, got {model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__}"
                tokenizer = transformers.AutoTokenizer.from_pretrained(quantized_model_path)
                prompt = "AI is "
                encode = tokenizer.encode(prompt, return_tensors="pt")
                with torch.no_grad():
                    output_tokens = model.generate(
                        encode,
                        max_length=10,
                    )
                    output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                    print(f"Prompt: {prompt}")
                    print(f"Output: {output}")
                    assert output is not None, "Output should not be None"

        if static_kv_dtype == "fp8":
            self.assertIn("model.decoder.layers.8.self_attn.k_scale", f.keys())
            self.assertIn("model.decoder.layers.8.self_attn.v_scale", f.keys())
            self.assertEqual(f.get_tensor("model.decoder.layers.5.self_attn.v_scale").shape, torch.Size([1]))
            self.assertEqual(f.get_tensor("model.decoder.layers.5.self_attn.k_scale").shape, torch.Size([1]))
            self.assertEqual(f.get_tensor("model.decoder.layers.5.self_attn.k_scale").dtype, torch.float32)
        shutil.rmtree(quantized_model_path, ignore_errors=True)

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=8,
            group_size=-1,
            iters=1,
            act_bits=8,
            nsamples=2,
            seqlen=2,
            data_type="fp8",
            act_data_type="fp8",
            act_dynamic=False,
            act_group_size=0,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")

        f = safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt")
        self.assertIn("model.decoder.layers.8.self_attn.k_proj.input_scale", f.keys())
        self.assertIn("model.decoder.layers.8.self_attn.k_proj.weight_scale", f.keys())
        self.assertEqual(f.get_tensor("model.decoder.layers.5.self_attn.v_proj.input_scale").shape, torch.Size([1]))
        self.assertEqual(f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype, torch.float8_e4m3fn)
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_static_fp8_attn(self):
        import os

        from safetensors import safe_open

        model_name = self.model_name
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(
            model,
            self.tokenizer,
            iters=0,
            nsamples=2,
            seqlen=2,
            scheme="FP8_STATIC",
            static_attention_dtype="fp8",
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        f = safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt")
        self.assertIn("model.decoder.layers.8.self_attn.k_proj.input_scale", f.keys())
        self.assertIn("model.decoder.layers.8.self_attn.k_proj.weight_scale", f.keys())
        self.assertEqual(f.get_tensor("model.decoder.layers.5.self_attn.v_proj.input_scale").shape, torch.Size([1]))
        self.assertEqual(f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype, torch.float8_e4m3fn)
        check_attrs = ["k_scale", "v_scale", "q_scale"]
        for attr in check_attrs:
            weight_name = f"model.decoder.layers.8.self_attn.{attr}"
            self.assertIn(weight_name, f.keys())
            self.assertEqual(f.get_tensor(weight_name).shape, torch.Size([1]))
            self.assertEqual(f.get_tensor(weight_name).dtype, torch.float32)

        shutil.rmtree(quantized_model_path, ignore_errors=True)


    def test_awq_lmhead_export(self):
        bits, sym, group_size = 4, False, 128
        model_name = "/tf_dataset/auto_round/models/microsoft/phi-2"
        layer_config = {
            "lm_head": {"bits": 4},  # set lm_head quant
        }
        autoround = AutoRound(
            model=model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            layer_config=layer_config,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = "./saved"
        compressed_model,_ = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_awq")
        lm_head = compressed_model.lm_head
        from auto_round.export.export_to_awq.utils import WQLinear_GEMM
        assert isinstance(lm_head, WQLinear_GEMM), "Illegal GPTQ quantization for lm_head layer"
        quantization_config = AutoRoundConfig()
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="cpu", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree(quantized_model_path, ignore_errors=True)


    def test_gptq_lmhead_export(self):
        bits, sym, group_size = 4, True, 128
        model_name = "/tf_dataset/auto_round/models/microsoft/phi-2"
        layer_config = {
            "lm_head": {"bits": 4},  # set lm_head quant
        }
        autoround = AutoRound(
            model=model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            layer_config=layer_config,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = "./saved"
        compressed_model,_ = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")
        lm_head = compressed_model.lm_head
        assert hasattr(lm_head, "bits") and lm_head.bits == 4, "Illegal GPTQ quantization for lm_head layer"
        quantization_config = AutoRoundConfig()
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="cpu", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=5)[0])
        print(res)
        shutil.rmtree(quantized_model_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

