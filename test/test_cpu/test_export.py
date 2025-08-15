import shutil
import sys
import unittest

sys.path.insert(0, "../..")
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        model_name = "facebook/opt-125m"
        self.save_dir = "./saved"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_autogptq_format(self):
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
            autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_gptq")

            if group_size == -1:
                shutil.rmtree("./saved", ignore_errors=True)
                continue
            quantization_config = AutoRoundConfig()
            model = AutoModelForCausalLM.from_pretrained(
                quantized_model_path, device_map="cpu", trust_remote_code=True, quantization_config=quantization_config
            )
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
            shutil.rmtree("./saved", ignore_errors=True)

    def test_autoround_format(self):
        for group_size in [-1, 32, 128]:
            bits, sym = 4, True
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
            autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")

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

            autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round:auto_awq")

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

    def test_static_afp8_export(self):
        import os

        from safetensors import safe_open

        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=8,
            group_size=-1,
            iters=0,
            act_bits=8,
            nsamples=2,
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
        self.assertEqual(f.get_tensor("model.decoder.layers.5.self_attn.v_proj.input_scale").shape, torch.Size([1, 1]))
        self.assertEqual(f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype, torch.float8_e4m3fn)
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
        self.assertEqual(f.get_tensor("model.decoder.layers.5.self_attn.v_proj.input_scale").shape, torch.Size([1, 1]))
        self.assertEqual(f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype, torch.float8_e4m3fn)
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_mxfp4_llmcompressor_format(self):
        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        from transformers import AutoConfig

        bits = 4
        data_type = "mx_fp4e2m1"
        group_size = 32
        sym = True
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            data_type=data_type,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = self.save_dir
        autoround.quantize()
        compressed_model = autoround.save_quantized(
            output_dir=quantized_model_path, inplace=True, format="llmcompressor"
        )
        tmp_layer = compressed_model.model.decoder.layers[3].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_packed")
            and tmp_layer.weight_scale.dtype is torch.uint8
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal MXFP4 packing name or data_type or shape"
        quantization_config = AutoConfig.from_pretrained(
            quantized_model_path, trust_remote_code=True
        ).quantization_config
        assert (
            quantization_config["format"] == "float-quantized"
            and quantization_config["config_groups"]["group_0"]["weights"]["is_mx"] is True
            and quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] == 4
        ), f"Invalid MXFP4 quantization configuration: {quantization_config}"
        shutil.rmtree("./saved", ignore_errors=True)

    def test_mxfp8_llmcompressor_format(self):
        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        from transformers import AutoConfig

        bits = 8
        data_type = "mx_fp8e4m3_rceil"
        group_size = 32
        sym = True
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            data_type=data_type,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="llmcompressor")
        tmp_layer = compressed_model.model.decoder.layers[3].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight")
            and tmp_layer.weight.dtype is torch.float8_e4m3fn
            and tmp_layer.weight_scale.dtype is torch.uint8
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal MXFP8 packing name or data_type or shape"
        quantization_config = AutoConfig.from_pretrained(
            quantized_model_path, trust_remote_code=True
        ).quantization_config
        assert (
            quantization_config["format"] == "float-quantized"
            and quantization_config["config_groups"]["group_0"]["weights"]["is_mx"] is True
            and quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] == 8
        ), f"Invalid MXFP8 quantization configuration: {quantization_config}"
        shutil.rmtree("./saved", ignore_errors=True)

    def test_nvfp4_llmcompressor_format(self):
        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        from transformers import AutoConfig

        bits = 4
        act_bits = 4
        data_type = "nv_fp"
        act_data_type = "nv_fp4_with_static_gs"
        group_size = 16
        sym = True
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=bits,
            act_bits=act_bits,
            data_type=data_type,
            act_data_type=act_data_type,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="llmcompressor")
        tmp_layer = compressed_model.model.decoder.layers[3].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_global_scale")
            and hasattr(tmp_layer, "input_global_scale")
            and tmp_layer.weight_packed.dtype is torch.uint8
            and tmp_layer.weight_scale.dtype is torch.float8_e4m3fn
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal NVFP4 packing name or data_type or shape"
        quantization_config = AutoConfig.from_pretrained(
            quantized_model_path, trust_remote_code=True
        ).quantization_config
        assert (
            quantization_config["format"] == "nvfp4-pack-quantized"
            and quantization_config["config_groups"]["group_0"]["input_activations"]["num_bits"] == 4
        ), f"Invalid NVFP4 quantization configuration: {quantization_config}"
        shutil.rmtree("./saved", ignore_errors=True)

    def test_nvfp4_autoround_format(self):
        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        from transformers import AutoConfig

        bits = 4
        act_bits = 4
        data_type = "nv_fp"
        act_data_type = "nv_fp4_with_static_gs"
        group_size = 16
        sym = True
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=bits,
            act_bits=act_bits,
            data_type=data_type,
            act_data_type=act_data_type,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        tmp_layer = compressed_model.model.decoder.layers[3].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_global_scale")
            and hasattr(tmp_layer, "input_global_scale")
            and tmp_layer.weight_packed.dtype is torch.uint8
            and tmp_layer.weight_scale.dtype is torch.float8_e4m3fn
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal NVFP4 packing name or data_type or shape"
        shutil.rmtree("./saved", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
