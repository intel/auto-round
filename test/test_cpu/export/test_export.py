import json
import os
import shutil

import pytest
import torch
from packaging import version
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ...helpers import get_model_path, opt_name_or_path, transformers_version


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
        self.save_dir = "./saved"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

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
        shutil.rmtree(quantized_model_path, ignore_errors=True)

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
        shutil.rmtree(quantized_model_path, ignore_errors=True)

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
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
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
        assert "model.decoder.layers.8.self_attn.k_proj.input_scale" in f.keys()
        assert "model.decoder.layers.8.self_attn.k_proj.weight_scale" in f.keys()
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.input_scale").shape == torch.Size([1])
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn
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
        assert "model.decoder.layers.8.self_attn.k_proj.input_scale" in f.keys()
        assert "model.decoder.layers.8.self_attn.k_proj.weight_scale" in f.keys()
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.input_scale").shape == torch.Size([1])
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn
        check_attrs = ["k_scale", "v_scale", "q_scale"]
        for attr in check_attrs:
            weight_name = f"model.decoder.layers.8.self_attn.{attr}"
            assert weight_name in f.keys()
            assert f.get_tensor(weight_name).shape == torch.Size([1])
            assert f.get_tensor(weight_name).dtype == torch.float32 or f.get_tensor(weight_name).dtype == torch.bfloat16

        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @pytest.mark.skipif(
        transformers_version >= version.parse("5.0"),
        reason="PhiConfig missing pad_token_id, https://github.com/huggingface/transformers/pull/43453",
    )
    def test_awq_lmhead_export(self, dataloader):
        bits, sym, group_size = 4, False, 128
        model_name = get_model_path("microsoft/phi-2")
        layer_config = {
            "lm_head": {"bits": 4},  # set lm_head quant
            "layer": {"bits": 16},
        }
        autoround = AutoRound(
            model=model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            nsamples=2,
            seqlen=2,
            layer_config=layer_config,
            dataset=dataloader,
        )
        quantized_model_path = "./saved"
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_awq")
        lm_head = compressed_model.lm_head
        from auto_round.export.export_to_awq.utils import WQLinear_GEMM

        assert isinstance(lm_head, WQLinear_GEMM), "Illegal AWQ quantization for lm_head layer"
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @pytest.mark.skipif(
        transformers_version >= version.parse("5.0"),
        reason="PhiConfig missing pad_token_id, https://github.com/huggingface/transformers/pull/43453",
    )
    def test_gptq_lmhead_export(self, dataloader):
        bits, sym, group_size = 4, True, 128
        # Note that, to save UT tuning time, the local model is intentionally kept lightweight, using only 2 hidden layers.
        model_name = get_model_path("microsoft/phi-2")
        layer_config = {
            "lm_head": {"bits": 4},  # set lm_head quant
            "layer": {"bits": 16},
        }
        autoround = AutoRound(
            model=model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            nsamples=2,
            iters=2,
            seqlen=2,
            layer_config=layer_config,
            dataset=dataloader,
        )
        quantized_model_path = "./saved"
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")
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

    def test_export_format(self):
        from auto_round.formats import get_formats

        autoround = AutoRound(
            self.model_name,
            scheme="FP8_STATIC",
        )
        format_list = get_formats("auto_round, llm_compressor, auto_round:llm_compressor", autoround)
        assert len(format_list) == 3
        assert format_list[0].output_format == "auto_round"
        assert format_list[0].get_backend_name() == "auto_round:fp8_static"
        assert format_list[1].output_format == "llm_compressor"
        assert format_list[1].get_backend_name() == "llm_compressor:fp8_static"
        assert format_list[2].output_format == "auto_round"
        assert format_list[2].get_backend_name() == "auto_round:llm_compressor:fp8_static"

        autoround = AutoRound(
            self.model_name,
            scheme="W4A16",
        )
        format_list = get_formats("auto_round:auto_awq, auto_gptq", autoround)
        assert format_list[0].output_format == "auto_round"
        assert format_list[0].get_backend_name() == "auto_round:auto_awq"
        assert format_list[1].output_format == "auto_gptq"
        assert format_list[1].get_backend_name() == "auto_gptq"

    def test_export_format_with_scheme(self, tiny_qwen_model_path):
        from auto_round.formats import get_formats

        ar = AutoRound(
            model=tiny_qwen_model_path,
            scheme="W4A16",
            bits=2,
            group_size=32,
            sym=True,
        )
        with pytest.raises(ValueError, match="auto_awq format support quantization scheme with W4A16 but got bits=2"):
            get_formats("auto_round:auto_awq", ar)

        with pytest.raises(ValueError, match="but got bits=2, data_type=int"):
            get_formats("auto_round:llm_compressor", ar)

        ar = AutoRound(
            model=tiny_qwen_model_path,
            scheme="FP8_STATIC",
            bits=4,
            group_size=32,
            sym=True,
        )
        with pytest.raises(ValueError, match="but got data_type=fp, bits=4"):
            get_formats("auto_round:llm_compressor", ar)

        ar = AutoRound(
            model=tiny_qwen_model_path,
            scheme="w2a16",
            bits=4,
            group_size=256,
            sym=True,
        )
        get_formats("auto_round:auto_awq", ar)

    def test_autoawq_qwen3_vl_infer(self, dataloader):
        model_path = get_model_path("Qwen/Qwen3-VL-2B-Instruct")
        autoround = AutoRound(
            model=model_path,
            scheme="W4A16",
            iters=0,
            seqlen=2,
            batch_size=1,
            dataset=dataloader,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="auto_awq")

        # Check items of modules_to_not_convert in quantization config
        quantization_config_path = f"{quantized_model_path}/quantization_config.json"
        with open(quantization_config_path, "r") as f:
            quantization_config = json.load(f)
        modules_to_not_convert = quantization_config.get("modules_to_not_convert", [])
        assert (
            "model.visual.merger.linear_fc2" in modules_to_not_convert
        ), f"'model.visual.merger.linear_fc2' should be in modules_to_not_convert. Got: {modules_to_not_convert}"
        assert (
            "model.visual.merger.linear_fc1" in modules_to_not_convert
        ), f"'model.visual.merger.linear_fc1' should be in modules_to_not_convert. Got: {modules_to_not_convert}"
        assert (
            "model.visual.blocks" in modules_to_not_convert
        ), f"'model.visual.blocks' should be in modules_to_not_convert. Got: {modules_to_not_convert}"
