import json
import os
import shutil

import pytest
import torch
from packaging import version
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.export.export_to_autogptq import export as autogptq_export
from auto_round.export.export_to_autoround import export as autoround_export
from auto_round.export.export_to_autoround import export_to_fp8 as autoround_fp8_export
from auto_round.export.export_to_awq import export as awq_export

from ...helpers import forbid_threaded_packing, get_model_path, opt_name_or_path, transformers_version


def _get_folder_size(path: str) -> float:
    """Return folder size in GB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)  # convert to GB


class TestAutoRound:

    @classmethod
    def setup_class(self):
        self.model_name = opt_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_autogptq_format(self, dataloader):
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
                dataset=dataloader,
            )

            quantized_model_path = self.save_dir
            _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")

            if group_size == -1:
                continue
            quantization_config = AutoRoundConfig()
            model = AutoModelForCausalLM.from_pretrained(
                quantized_model_path, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
            )
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    def test_autoround_format(self, dataloader):
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
                dataset=dataloader,
            )
            quantized_model_path = self.save_dir
            _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")

            if group_size == -1:
                continue
            model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu")
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    def test_autoround_awq_format(self, dataloader):
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
                dataset=dataloader,
            )
            quantized_model_path = self.save_dir

            _, quantized_model_path = autoround.quantize_and_save(
                output_dir=quantized_model_path, format="auto_round:auto_awq"
            )

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

    def test_autoawq_format(self, dataloader):
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
                dataset=dataloader,
            )
            autoround.quantize()
            quantized_model_path = self.save_dir

            autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_awq")
            if group_size == -1:
                continue
            quantization_config = AutoRoundConfig()

            model = AutoModelForCausalLM.from_pretrained(
                quantized_model_path, device_map="cpu", quantization_config=quantization_config
            )
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    def test_autoround_3bit_asym_format(self, dataloader):
        bits, group_size, sym = 3, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
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

    def test_autoround_3bit_sym_format(self, dataloader):
        bits, group_size, sym = 3, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
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

    @pytest.mark.parametrize("static_kv_dtype", ["fp8", "float16"])
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
            scheme="fp8_static",
            nsamples=2,
            seqlen=2,
            static_kv_dtype=static_kv_dtype,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        f = safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt")
        assert "model.decoder.layers.8.self_attn.k_proj.input_scale" in f.keys()
        assert "model.decoder.layers.8.self_attn.k_proj.weight_scale" in f.keys()
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.input_scale").shape == torch.Size([1])
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn
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
            assert "model.decoder.layers.8.self_attn.k_scale" in f.keys()
            assert "model.decoder.layers.8.self_attn.v_scale" in f.keys()
            assert f.get_tensor("model.decoder.layers.5.self_attn.v_scale").shape == torch.Size([1])
            assert f.get_tensor("model.decoder.layers.5.self_attn.k_scale").shape == torch.Size([1])
            assert (
                f.get_tensor("model.decoder.layers.5.self_attn.k_scale").dtype == torch.float32
                or f.get_tensor("model.decoder.layers.5.self_attn.k_scale").dtype == torch.bfloat16
            )

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
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")

        f = safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt")
        assert "model.decoder.layers.8.self_attn.k_proj.input_scale" in f.keys()
        assert "model.decoder.layers.8.self_attn.k_proj.weight_scale" in f.keys()
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.input_scale").shape == torch.Size([1])
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn
