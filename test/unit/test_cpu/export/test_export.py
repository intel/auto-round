import json
import os
import shutil
from test.helpers import forbid_threaded_packing, get_model_path, opt_name_or_path, transformers_version

import pytest
import torch
from packaging import version
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.compressors.config_resolution import ResolvedScheme
from auto_round.export.export_to_autogptq import export as autogptq_export
from auto_round.export.export_to_autoround import export as autoround_export
from auto_round.export.export_to_autoround import export_to_fp8 as autoround_fp8_export
from auto_round.export.export_to_awq import export as awq_export
from auto_round.export.formats import resolve_formats


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
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.fixture(autouse=True)
    def _test_setup(self, tiny_opt_model_path, tmp_path):
        self.model_name = tiny_opt_model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            tiny_opt_model_path, torch_dtype="auto", trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path, trust_remote_code=True)
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
        with safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt") as f:
            assert "model.decoder.layers.0.self_attn.k_proj.input_scale" in f.keys()
            assert "model.decoder.layers.0.self_attn.k_proj.weight_scale" in f.keys()
            assert f.get_tensor("model.decoder.layers.0.self_attn.v_proj.input_scale").shape == torch.Size([1])
            assert f.get_tensor("model.decoder.layers.0.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn
            if static_kv_dtype == "fp8":
                assert "model.decoder.layers.0.self_attn.k_scale" in f.keys()
                assert "model.decoder.layers.0.self_attn.v_scale" in f.keys()
                assert f.get_tensor("model.decoder.layers.0.self_attn.v_scale").shape == torch.Size([1])
                assert f.get_tensor("model.decoder.layers.0.self_attn.k_scale").shape == torch.Size([1])
                assert (
                    f.get_tensor("model.decoder.layers.0.self_attn.k_scale").dtype == torch.float32
                    or f.get_tensor("model.decoder.layers.0.self_attn.k_scale").dtype == torch.bfloat16
                )
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
                ), (
                    "Expected WeightFP8ActFP8StaticQuantLinear, "
                    f"got {model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__}"
                )
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

        with safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt") as f:
            assert "model.decoder.layers.0.self_attn.k_proj.input_scale" in f.keys()
            assert "model.decoder.layers.0.self_attn.k_proj.weight_scale" in f.keys()
            assert f.get_tensor("model.decoder.layers.0.self_attn.v_proj.input_scale").shape == torch.Size([1])
            assert f.get_tensor("model.decoder.layers.0.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn

    def test_static_afp8_per_head_export(self):
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
            static_kv_dtype="fp8",
            static_kv_granularity="head",
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")
        f = safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt")
        assert "model.decoder.layers.0.self_attn.k_proj.input_scale" in f.keys()
        assert "model.decoder.layers.0.self_attn.k_proj.weight_scale" in f.keys()
        assert f.get_tensor("model.decoder.layers.0.self_attn.v_proj.input_scale").shape == torch.Size([1])
        assert f.get_tensor("model.decoder.layers.0.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn
        assert f.get_tensor("model.decoder.layers.0.self_attn.k_scale").shape == torch.Size(
            [model.config.num_attention_heads]
        )
        assert f.get_tensor("model.decoder.layers.0.self_attn.v_scale").shape == torch.Size(
            [model.config.num_attention_heads]
        )

        with open(os.path.join(quantized_model_path, "config.json")) as config_file:
            config = json.load(config_file)
        quantization_config = config["quantization_config"]
        assert quantization_config["static_kv_dtype"] == "fp8"
        assert quantization_config["static_kv_granularity"] == "head"

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
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        with safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt") as f:
            assert "model.decoder.layers.0.self_attn.k_proj.input_scale" in f.keys()
            assert "model.decoder.layers.0.self_attn.k_proj.weight_scale" in f.keys()
            assert f.get_tensor("model.decoder.layers.0.self_attn.v_proj.input_scale").shape == torch.Size([1])
            assert f.get_tensor("model.decoder.layers.0.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn
            check_attrs = ["k_scale", "v_scale", "q_scale"]
            for attr in check_attrs:
                weight_name = f"model.decoder.layers.0.self_attn.{attr}"
                assert weight_name in f.keys()
                assert f.get_tensor(weight_name).shape == torch.Size([1])
                assert (
                    f.get_tensor(weight_name).dtype == torch.float32
                    or f.get_tensor(weight_name).dtype == torch.bfloat16
                )
            assert not any(key.endswith(".q_max") for key in f.keys())

    def test_static_fp8_per_head_attn(self):
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
            static_attention_granularity="head",
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")
        f = safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt")
        for attr in ("k_scale", "v_scale", "q_scale"):
            weight_name = f"model.decoder.layers.0.self_attn.{attr}"
            assert weight_name in f.keys()
            assert f.get_tensor(weight_name).shape == torch.Size([model.config.num_attention_heads])

        with open(os.path.join(quantized_model_path, "config.json")) as config_file:
            config = json.load(config_file)
        quantization_config = config["quantization_config"]
        assert quantization_config["static_attention_dtype"] == "fp8"
        assert quantization_config["static_attention_granularity"] == "head"

    def test_awq_lmhead_export(self, dataloader):
        bits, sym, group_size = 4, False, 128
        model_name = get_model_path("microsoft/phi-4")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if model.config.tie_word_embeddings:
            model.config.tie_word_embeddings = False
            model._tied_weights_keys = []
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

        layer_config = {
            "lm_head": {"bits": 4},  # set lm_head quant
            "layer": {"bits": 16},
        }

        autoround = AutoRound(
            model=model,
            tokenizer=tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            nsamples=2,
            seqlen=2,
            layer_config=layer_config,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_awq")
        lm_head = compressed_model.lm_head
        from auto_round.export.export_to_awq.utils import WQLinear_GEMM

        assert isinstance(lm_head, WQLinear_GEMM), "Illegal AWQ quantization for lm_head layer"

    def test_gptq_lmhead_export(self, dataloader):
        bits, sym, group_size = 4, True, 128
        # Note that, to save UT tuning time, the local model is intentionally kept lightweight, using only 2 hidden layers.
        model_name = get_model_path("microsoft/phi-4")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if model.config.tie_word_embeddings:
            model.config.tie_word_embeddings = False
            model._tied_weights_keys = []
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

        layer_config = {
            "lm_head": {"bits": 4},  # set lm_head quant
            "layer": {"bits": 16},
        }
        autoround = AutoRound(
            model=model,
            tokenizer=tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            nsamples=2,
            iters=2,
            seqlen=2,
            layer_config=layer_config,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir
        compressed_model, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="auto_gptq"
        )
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

    def test_export_format(self):
        autoround = AutoRound(
            self.model_name,
            scheme="FP8_STATIC",
        )
        autoround.post_init()
        resolution = resolve_formats(
            ResolvedScheme.from_scheme(autoround.scheme_context),
            format="auto_round, llm_compressor, auto_round:llm_compressor",
            model=autoround.model_context.model,
            scale_dtype=autoround.scale_dtype,
        )
        format_list = list(resolution.formats)
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
        autoround.post_init()
        resolution = resolve_formats(
            ResolvedScheme.from_scheme(autoround.scheme_context),
            format="auto_round:auto_awq, auto_gptq",
            model=autoround.model_context.model,
            scale_dtype=autoround.scale_dtype,
        )
        format_list = list(resolution.formats)
        assert format_list[0].output_format == "auto_round"
        assert format_list[0].get_backend_name() == "auto_round:auto_awq"
        assert format_list[1].output_format == "auto_gptq"
        assert format_list[1].get_backend_name() == "auto_gptq"

        autoround = AutoRound(
            model=self.model_name,
            scheme="INT8",
        )
        autoround.post_init()
        resolution = resolve_formats(
            ResolvedScheme.from_scheme(autoround.scheme_context),
            format="llm_compressor, auto_round:llm_compressor",
            model=autoround.model_context.model,
            scale_dtype=autoround.scale_dtype,
        )
        format_list = list(resolution.formats)
        assert format_list[0].output_format == "llm_compressor"
        assert format_list[0].get_backend_name() == "llm_compressor:int8_w8a8"
        assert format_list[1].output_format == "auto_round"
        assert format_list[1].get_backend_name() == "auto_round:llm_compressor:int8_w8a8"

        # Verify backward compatibility: INT8_W8A8 (old name) produces identical formats to INT8
        autoround_old = AutoRound(
            model=self.model_name,
            scheme="INT8_W8A8",
        )
        autoround_old.post_init()
        resolution_old = resolve_formats(
            ResolvedScheme.from_scheme(autoround_old.scheme_context),
            format="llm_compressor, auto_round:llm_compressor",
            model=autoround_old.model_context.model,
            scale_dtype=autoround_old.scale_dtype,
        )
        format_list_old = list(resolution_old.formats)
        assert format_list_old[0].output_format == "llm_compressor"
        assert format_list_old[0].get_backend_name() == "llm_compressor:int8_w8a8"
        assert format_list_old[1].output_format == "auto_round"
        assert format_list_old[1].get_backend_name() == "auto_round:llm_compressor:int8_w8a8"

    def test_export_format_with_scheme(self, tiny_qwen_model_path):
        ar = AutoRound(
            model=tiny_qwen_model_path,
            scheme="W4A16",
            bits=2,
            group_size=32,
            sym=True,
        )
        ar.post_init()
        with pytest.raises(
            ValueError,
            match="auto_awq format support quantization scheme with W4A16,W5A16,W6A16,W7A16 but got bits=2",
        ):
            resolve_formats(
                ResolvedScheme.from_scheme(ar.scheme_context),
                format="auto_round:auto_awq",
                model=ar.model_context.model,
                scale_dtype=ar.scale_dtype,
            )

        ar = AutoRound(
            model=tiny_qwen_model_path,
            scheme="FP8_STATIC",
            bits=4,
            group_size=32,
            sym=True,
        )
        ar.post_init()
        with pytest.raises(ValueError, match="but got data_type=fp, bits=4"):
            resolve_formats(
                ResolvedScheme.from_scheme(ar.scheme_context),
                format="auto_round:llm_compressor",
                model=ar.model_context.model,
                scale_dtype=ar.scale_dtype,
            )

        ar = AutoRound(
            model=tiny_qwen_model_path,
            scheme="w2a16",
            bits=4,
            group_size=256,
            sym=True,
        )
        ar.post_init()
        resolve_formats(
            ResolvedScheme.from_scheme(ar.scheme_context),
            format="auto_round:auto_awq",
            model=ar.model_context.model,
            scale_dtype=ar.scale_dtype,
        )

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
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, inplace=False, format="auto_awq"
        )

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

    @pytest.mark.parametrize(
        "iters,use_dataloader,scheme",
        [
            (0, False, "INT8"),  # RTN with new scheme name
            (1, True, "INT8"),  # tuning with new scheme name
            (0, False, "INT8_W8A8"),  # RTN with old scheme name (backward compat)
        ],
        ids=["rtn", "tuning", "rtn-old-scheme"],
    )
    def test_llmc_dynamic_wint8aint8_export(self, iters, use_dataloader, scheme, dataloader):
        from safetensors import safe_open

        dataset = dataloader if use_dataloader else None
        autoround = AutoRound(
            self.model_name,
            iters=iters,
            nsamples=2,
            seqlen=2,
            dataset=dataset,
            scheme=scheme,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="llm_compressor")
        with safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt") as f:
            assert "model.decoder.layers.0.self_attn.k_proj.weight_scale" in f.keys()
            assert f.get_tensor("model.decoder.layers.0.self_attn.v_proj.weight").dtype == torch.int8
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @pytest.mark.parametrize(
        "scheme,bits,group_size,sym",
        [
            ("W4A16", 4, 128, True),
            ("W4A16", 4, -1, True),
            ("W8A16", 8, -1, True),
        ],
    )
    def test_llmc_wint_a16_export(self, scheme, bits, group_size, sym):
        from safetensors import safe_open

        autoround = AutoRound(
            self.model_name,
            iters=2,
            nsamples=2,
            seqlen=2,
            scheme=scheme,
            bits=bits,
            group_size=group_size,
            sym=sym,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="llm_compressor")
        with safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt") as f:
            # weights must be packed as int32 (compressed-tensors stores both int4 and int8 as torch.int32)
            weight = f.get_tensor("model.decoder.layers.0.self_attn.v_proj.weight_packed")
            assert weight.dtype == torch.int32, f"Expected int32 weight for {scheme}, got {weight.dtype}"
            # weight_scale must be present and be a float tensor
            scale_key = "model.decoder.layers.0.self_attn.k_proj.weight_scale"
            assert scale_key in f.keys(), f"Missing {scale_key} for {scheme} export"
            scale = f.get_tensor(scale_key)
            assert scale.dtype in (
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ), f"Expected float weight_scale for {scheme}, got {scale.dtype}"
            # No input_scale should be present for weight-only quantization
            input_scale_keys = [k for k in f.keys() if k.endswith(".input_scale")]
            assert (
                len(input_scale_keys) == 0
            ), f"Expected no input_scale for weight-only {scheme}, but found: {input_scale_keys[:5]}"
        shutil.rmtree(quantized_model_path, ignore_errors=True)


@pytest.mark.parametrize(
    "format_name,export_module,sym",
    [
        ("auto_gptq", autogptq_export, False),
        ("auto_awq", awq_export, False),
        ("auto_round", autoround_export, True),
    ],
)
def test_weight_only_exports_pack_serially(tiny_opt_model_path, tmp_path, monkeypatch, format_name, export_module, sym):
    autoround = AutoRound(
        tiny_opt_model_path,
        bits=4,
        group_size=128,
        sym=sym,
        iters=0,
        disable_opt_rtn=True,
    )
    autoround.quantize()
    forbid_threaded_packing(monkeypatch, export_module)
    autoround.save_quantized(output_dir=tmp_path, inplace=False, format=format_name)
    assert os.path.exists(os.path.join(tmp_path, "config.json"))


def test_fp8_autoround_export_packs_serially(tiny_opt_model_path, tmp_path, monkeypatch):
    from safetensors import safe_open

    autoround = AutoRound(
        tiny_opt_model_path,
        bits=8,
        group_size=-1,
        iters=0,
        scheme="FP8_STATIC",
        nsamples=2,
        seqlen=2,
        static_kv_dtype="fp8",
    )
    autoround.quantize()
    forbid_threaded_packing(monkeypatch, autoround_fp8_export)
    autoround.save_quantized(output_dir=tmp_path, format="auto_round")
    with safe_open(os.path.join(tmp_path, "model.safetensors"), framework="pt") as f:
        assert "model.decoder.layers.0.self_attn.k_proj.weight_scale" in f.keys()


@pytest.mark.parametrize("low_cpu_mem_usage", [True, False])
def test_immediate_saving_mode(tiny_opt_model_path, tmp_path, low_cpu_mem_usage, caplog):
    """Verify that immediate_saving (triggered by low_cpu_mem_usage) produces a complete model output."""
    import logging

    output_dir = str(tmp_path / "output")
    with caplog.at_level(logging.DEBUG, logger="auto_round"):
        autoround = AutoRound(
            tiny_opt_model_path,
            scheme="MXFP4",
            iters=2,
            seqlen=2,
            nsamples=2,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=output_dir, format="llm_compressor")

    # No spurious "already exists" warning should be emitted
    conflict_messages = [r.message for r in caplog.records if "already exists" in r.message]
    assert len(conflict_messages) == 0, f"Unexpected conflict warnings: {conflict_messages}"

    # All essential files must exist regardless of immediate_saving mode
    assert os.path.exists(os.path.join(quantized_model_path, "config.json")), "config.json missing"
    assert os.path.exists(
        os.path.join(quantized_model_path, "quantization_config.json")
    ), "quantization_config.json missing"

    # Exactly 1 safetensors shard for this tiny model
    safetensor_files = [f for f in os.listdir(quantized_model_path) if f.endswith(".safetensors")]
    assert len(safetensor_files) == 1, f"Expected 1 safetensors file, got {len(safetensor_files)}: {safetensor_files}"

    # Tokenizer files must be present
    assert os.path.exists(os.path.join(quantized_model_path, "tokenizer_config.json")), "tokenizer_config.json missing"

    # Total file count: config.json, quantization_config.json, model.safetensors,
    # generation_config.json, tokenizer.json, tokenizer_config.json = 6
    all_files = os.listdir(quantized_model_path)
    assert len(all_files) == 6, f"Expected 6 files, got {len(all_files)}: {sorted(all_files)}"

    # Verify weights are loadable and non-empty
    from safetensors import safe_open

    with safe_open(os.path.join(quantized_model_path, safetensor_files[0]), framework="pt") as f:
        keys = f.keys()
        assert len(keys) > 0, "Safetensors file has no tensors"


def test_save_model_writes_diffusers_config(tmp_path):
    """A diffusers config is a FrozenDict with no save_pretrained; the export must still write it."""
    diffusers = pytest.importorskip("diffusers")

    from auto_round.export.utils import save_model

    model = diffusers.SD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=4,
        num_layers=1,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=64,
        caption_projection_dim=64,
        pooled_projection_dim=64,
        out_channels=4,
    )
    assert not hasattr(model.config, "save_pretrained")
    model.config.quantization_config = {"quant_method": "auto-round", "bits": 4}

    # immediate_saving: weights are already on disk, only the configs are written
    save_model(model, str(tmp_path), immediate_saving=True)

    with open(os.path.join(tmp_path, "config.json")) as f:
        config = json.load(f)
    assert config["_class_name"] == "SD3Transformer2DModel"
    assert config["quantization_config"] == {"quant_method": "auto-round", "bits": 4}


def test_awq_gemm_kernel_supported():
    """AWQ GEMM kernel divisibility contract (IC/OC % group_size, OC % 64)."""
    from auto_round.export.export_to_awq.utils import awq_gemm_kernel_supported

    # Canonical servable shapes
    assert awq_gemm_kernel_supported(2048, 2048, 4, 128)
    assert awq_gemm_kernel_supported(256, 128, 4, 128)
    # out_features < group_size, e.g. DeltaNet in_proj_ba (OC=32/64, gs=128)
    assert not awq_gemm_kernel_supported(256, 64, 4, 128)
    assert not awq_gemm_kernel_supported(256, 32, 4, 128)
    # in_features not a multiple of group_size
    assert not awq_gemm_kernel_supported(96, 256, 4, 128)
    # OC % 64 (cta_N tile) still binds when group_size is small
    assert not awq_gemm_kernel_supported(64, 96, 4, 32)
    assert awq_gemm_kernel_supported(64, 192, 4, 32)
    # Bit widths without a servable packing layout are rejected
    assert not awq_gemm_kernel_supported(256, 256, 8, 128)
    assert not awq_gemm_kernel_supported(256, 256, None, 128)
    # Per-channel group_size is left to the caller
    assert awq_gemm_kernel_supported(256, 256, 4, -1)
    assert awq_gemm_kernel_supported(256, 64, 4, -1)


def test_awq_format_excludes_unservable_layers():
    """Layers the AWQ GEMM kernel cannot serve are marked fp16 during format resolution."""
    from types import SimpleNamespace

    import torch.nn as nn

    from auto_round.export.formats.backends.auto_awq import AutoAWQFormat
    from auto_round.schemes import preset_name_to_scheme

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(256, 256)
            self.in_proj_ba = nn.Linear(256, 64)  # OC=64 < group_size -> unservable
            self.in_proj_b = nn.Linear(96, 256)  # IC=96 % 128 != 0 -> unservable
            self.mlp_up = nn.Linear(256, 96)  # OC=96 % 64 != 0 -> unservable

    model = ToyModel()
    scheme = preset_name_to_scheme("W4A16")
    # Fields normally filled by scheme resolution
    scheme.act_data_type = scheme.act_data_type or "fp16"
    scheme.act_dynamic = False if scheme.act_dynamic is None else scheme.act_dynamic
    ctx = SimpleNamespace(
        model=model,
        layer_config={},
        mllm=False,
        quant_block_list=None,
    )
    output_format = AutoAWQFormat("auto_awq", scheme, ctx)
    output_format.check_and_reset_format(scheme, ctx)

    # None of the shapes are caught by _check_divisible_by_32 (all % 32 == 0),
    # so any fp16 mark comes from the AWQ kernel constraint check.
    assert ctx.layer_config["in_proj_ba"]["bits"] == 16
    assert ctx.layer_config["in_proj_ba"]["data_type"] == "fp"
    assert ctx.layer_config["in_proj_b"]["bits"] == 16
    assert ctx.layer_config["mlp_up"]["bits"] == 16
    assert "q_proj" not in ctx.layer_config


def test_awq_format_honors_explicit_layer_config():
    """At format-resolution time a raw user entry (no fixed_by_user flag yet)
    counts as explicit configuration: the unservable layer is not marked fp16
    and is flagged for the packers."""
    from types import SimpleNamespace

    import torch.nn as nn

    from auto_round.export.export_to_awq.utils import AWQ_USER_FORCED_ATTR
    from auto_round.export.formats.backends.auto_awq import AutoAWQFormat
    from auto_round.schemes import preset_name_to_scheme
    from auto_round.utils import get_module

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_ba = nn.Linear(256, 64)  # OC=64 < group_size -> unservable
            self.mlp_bad = nn.Linear(96, 256)  # IC=96 % 128 != 0 -> unservable, no user entry

    model = ToyModel()
    scheme = preset_name_to_scheme("W4A16")
    scheme.act_data_type = scheme.act_data_type or "fp16"
    scheme.act_dynamic = False if scheme.act_dynamic is None else scheme.act_dynamic
    ctx = SimpleNamespace(
        model=model,
        layer_config={"in_proj_ba": {"bits": 4}},
        mllm=False,
        quant_block_list=None,
    )
    output_format = AutoAWQFormat("auto_awq", scheme, ctx)
    output_format.check_and_reset_format(scheme, ctx)

    # The explicit entry is honored
    assert ctx.layer_config["in_proj_ba"]["bits"] == 4
    assert getattr(get_module(model, "in_proj_ba"), AWQ_USER_FORCED_ATTR, False)
    # The layer without a user entry is still marked fp16
    assert ctx.layer_config["mlp_bad"]["bits"] == 16
    assert not getattr(get_module(model, "mlp_bad"), AWQ_USER_FORCED_ATTR, False)


def test_awq_pack_layer_skips_unservable_layer():
    """pack_layer must not AWQ-pack a layer whose shape the GEMM kernel cannot serve."""
    import torch.nn as nn

    from auto_round.export.export_to_awq.export import pack_layer
    from auto_round.utils import get_module

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_ba = nn.Linear(256, 64)

    model = ToyModel()
    layer = get_module(model, "in_proj_ba")
    # Simulate a layer that went through quantization (attrs set by the quantizer)
    layer.bits = 4
    layer.group_size = 128
    layer.sym = True
    layer.scale = torch.ones(2, 64)
    layer.zp = torch.zeros(2, 64)

    pack_layer("in_proj_ba", model, backend="auto_awq")

    assert type(get_module(model, "in_proj_ba")) is nn.Linear


def test_awq_export_lists_unservable_layers(tmp_path):
    """Layers the AWQ GEMM kernel cannot serve land in modules_to_not_convert.

    Covers the split quantize() -> save_quantized(format="auto_awq") flow where
    the layer was already quantized before the format check ran.
    """
    import torch.nn as nn

    from auto_round.export.export_to_awq.export import save_quantized_as_autoawq
    from auto_round.utils import get_module

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_ba = nn.Linear(256, 64)  # OC=64 < group_size -> unservable
            self.q_proj = nn.Linear(256, 256)
            self.dtype = torch.float32

        def save_pretrained(self, save_dir, **kwargs):
            os.makedirs(save_dir, exist_ok=True)

    model = ToyModel()
    # Simulate a layer that already went through quantization
    layer = get_module(model, "in_proj_ba")
    layer.bits = 4
    layer.group_size = 128
    layer.sym = True
    # fixed_by_user=False mirrors what resolve_layer_config writes for
    # default-filled entries (a user entry would carry True).
    layer_config = {
        "in_proj_ba": {"bits": 4, "group_size": 128, "sym": True, "data_type": "int", "fixed_by_user": False}
    }
    serialization_dict = {"bits": 4, "group_size": 128, "sym": True}

    save_quantized_as_autoawq(
        str(tmp_path),
        model=model,
        layer_config=layer_config,
        inplace=True,
        serialization_dict=serialization_dict,
    )

    assert "in_proj_ba" in serialization_dict["modules_to_not_convert"]
    # Not packed into an AWQ layer
    assert type(get_module(model, "in_proj_ba")) is nn.Linear


def test_awq_export_packs_user_forced_layer(tmp_path):
    """A layer_config entry the user explicitly set for quantization is honored:
    the unservable layer is packed anyway and does not land in
    modules_to_not_convert."""
    import torch.nn as nn

    from auto_round.export.export_to_awq.export import save_quantized_as_autoawq
    from auto_round.export.export_to_awq.utils import WQLinear_GEMM
    from auto_round.utils import get_module

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_ba = nn.Linear(256, 64)  # OC=64 < group_size -> unservable
            self.q_proj = nn.Linear(256, 256)
            self.dtype = torch.float32

        def save_pretrained(self, save_dir, **kwargs):
            os.makedirs(save_dir, exist_ok=True)

    model = ToyModel()
    layer = get_module(model, "in_proj_ba")
    layer.bits = 4
    layer.group_size = 128
    layer.sym = True
    # The quantizer stores scale/zp as (out_features, num_groups); pack_layer
    # transposes them before handing them to the packer.
    layer.scale = torch.ones(64, 2)
    layer.zp = torch.zeros(64, 2)
    layer_config = {
        "in_proj_ba": {"bits": 4, "group_size": 128, "sym": True, "data_type": "int", "fixed_by_user": True}
    }
    serialization_dict = {"bits": 4, "group_size": 128, "sym": True}

    save_quantized_as_autoawq(
        str(tmp_path),
        model=model,
        layer_config=layer_config,
        inplace=True,
        serialization_dict=serialization_dict,
    )

    assert "in_proj_ba" not in serialization_dict["modules_to_not_convert"]
    assert isinstance(get_module(model, "in_proj_ba"), WQLinear_GEMM)


def test_awq_explicit_layer_config_overrides_unservable_mark():
    """Explicit layer_config entries keep unservable layers quantized.

    A user-supplied entry (exact name or expanded regex) counts as user
    intent: the AWQ unservable marking must leave it untouched instead of
    forcing fp16, and flag the module for the packers. Layers without a user
    entry are still marked fp16."""
    import torch.nn as nn

    from auto_round.compressors.config_resolution import ResolvedScheme
    from auto_round.compressors.layer_config_resolver import resolve_layer_config
    from auto_round.export.export_to_awq.utils import AWQ_USER_FORCED_ATTR
    from auto_round.schemes import preset_name_to_scheme
    from auto_round.utils import get_module

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(256, 256)  # servable
            self.in_proj_ba = nn.Linear(256, 64)  # OC=64 < group_size -> unservable
            self.mlp_bad = nn.Linear(96, 256)  # IC=96 % 128 != 0 -> unservable, not user-configured

    model = ToyModel()
    scheme = ResolvedScheme.from_scheme(preset_name_to_scheme("W4A16"))
    resolved = resolve_layer_config(
        model=model,
        scheme=scheme,
        layer_config={"proj": {"bits": 4}},  # regex matches q_proj and in_proj_ba
        format="auto_awq",
    )
    # The user-configured unservable layer keeps its quantized setting
    assert resolved["in_proj_ba"]["bits"] == 4
    assert getattr(get_module(model, "in_proj_ba"), AWQ_USER_FORCED_ATTR, False)
    # The regex-matched sibling is servable and untouched as before
    assert resolved["q_proj"]["bits"] == 4
    # The unservable layer without a user entry is still marked fp16
    assert resolved["mlp_bad"]["bits"] == 16
    assert resolved["mlp_bad"]["data_type"] == "fp"
    assert not getattr(get_module(model, "mlp_bad"), AWQ_USER_FORCED_ATTR, False)


def test_awq_pack_layer_packs_user_forced_layer():
    """pack_layer must AWQ-pack an unservable layer flagged by an explicit
    layer_config entry (AWQ_USER_FORCED_ATTR)."""
    import torch.nn as nn

    from auto_round.export.export_to_awq.export import pack_layer
    from auto_round.export.export_to_awq.utils import AWQ_USER_FORCED_ATTR, WQLinear_GEMM
    from auto_round.utils import get_module

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_ba = nn.Linear(256, 64)

    model = ToyModel()
    layer = get_module(model, "in_proj_ba")
    # Simulate a layer that went through quantization (attrs set by the
    # quantizer); scale/zp are stored as (out_features, num_groups).
    layer.bits = 4
    layer.group_size = 128
    layer.sym = True
    layer.scale = torch.ones(64, 2)
    layer.zp = torch.zeros(64, 2)
    setattr(layer, AWQ_USER_FORCED_ATTR, True)

    pack_layer("in_proj_ba", model, backend="auto_awq")

    assert isinstance(get_module(model, "in_proj_ba"), WQLinear_GEMM)
