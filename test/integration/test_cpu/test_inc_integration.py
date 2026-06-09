import copy
import os
import shutil

import pytest
import torch
import transformers
from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    convert,
    prepare,
    quantize,
)
from neural_compressor.torch.utils import logger
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.__allow_nonbracketed_mutation_flag = True

try:
    import compressed_tensors

    ct_installed = True
except ImportError:
    ct_installed = False


target_modules = ["QuantLinear", "QuantLinearGPTQ", "QuantLinearAWQ", "WQLinear_GEMM", "AwqTorchQuantLinear"]


@torch.no_grad()
def run_fn(model, dataloader):
    for data in dataloader:
        if isinstance(data, tuple) or isinstance(data, list):
            model(*data)
        elif isinstance(data, dict):
            model(**data)
        else:
            model(data)


class TestAutoRoundCPU:
    @pytest.fixture(autouse=True, scope="class")
    def setup_class_fixture(self, tiny_opt_model_path, request):
        cls = request.cls
        cls.opt_model = transformers.AutoModelForCausalLM.from_pretrained(
            tiny_opt_model_path,
        ).to("cpu")
        cls.inp = torch.ones([1, 10], dtype=torch.long, device="cpu")
        cls.tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_opt_model_path, trust_remote_code=True)
        from neural_compressor.torch.algorithms.autoround import get_dataloader

        cls.dataloader = get_dataloader(
            cls.tokenizer, 32, dataset_name="NeelNanda/pile-10k", seed=42, bs=8, nsamples=10
        )
        yield
        shutil.rmtree("saved_results", ignore_errors=True)
        shutil.rmtree("tmp_auto_round", ignore_errors=True)

    def setup_method(self, method):
        logger.info(f"Running TestAutoRound test: {method.__name__}")

    def test_quant_lm_head(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            # "trl-internal-testing/tiny-Phi3ForCausalLM",
            "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM", trust_remote_code=True
        )

        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            iters=0,
            amp=False,
            disable_opt_rtn=True,
            scale_dtype="fp32",
            quant_lm_head=True,
            group_size=32,
        )
        model = prepare(model=model, quant_config=quant_config)
        q_model = convert(model)
        assert q_model.lm_head.__class__.__name__ in target_modules, "packing model failed."

    def test_int4_dtype(self):
        fp32_model = copy.deepcopy(self.opt_model)
        quant_config = AutoRoundConfig(dtype="int4", nsamples=32, seqlen=10, iters=1, amp=False, scale_dtype="fp32")
        logger.info(f"Test AutoRound with config {quant_config}")

        # prepare + convert API
        model = prepare(model=fp32_model, quant_config=quant_config)

        run_fn(model, self.dataloader)
        q_model = convert(model)
        _ = q_model(self.inp)  # inference
        assert (
            q_model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__ in target_modules
        ), "packing model failed."

    def test_autoround_with_quantize_API(self):
        fp32_model = copy.deepcopy(self.opt_model)

        quant_config = AutoRoundConfig(
            scheme="W4A16", seqlen=10, iters=1, use_sym=False, amp=False, scale_dtype="fp32", reloading=False
        )
        logger.info(f"Test AutoRound with config {quant_config}")

        # quantize API
        q_model = quantize(
            model=fp32_model,
            quant_config=quant_config,
            run_fn=run_fn,
            run_args=(self.dataloader,),
        )
        assert (
            q_model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__ in target_modules
        ), "packing model failed."

    def test_conv1d(self, tiny_lamini_model_path):
        model = AutoModelForCausalLM.from_pretrained(tiny_lamini_model_path, device_map="cpu", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(tiny_lamini_model_path, trust_remote_code=True)
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt")
        quant_config = AutoRoundConfig(
            nsamples=32,
            seqlen=10,
            iters=0,
            amp=False,
            tokenizer=tokenizer,
            export_format="auto_round",
            device_map="cpu",
        )
        model = prepare(model=model, quant_config=quant_config)
        q_model = convert(model)
        output = tokenizer.decode(q_model.generate(**encoded_input, max_new_tokens=10)[0])
        print(output)
        assert output is not None
        assert not isinstance(
            q_model.transformer.h[0].attn.c_attn, transformers.pytorch_utils.Conv1D
        ), "loading compressed model failed."

    def test_mllm(self, tiny_qwen_vl_model_path):
        input = torch.randn(1, 32)
        from neural_compressor.torch.algorithms.autoround import get_mllm_dataloader
        from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

        model_name = tiny_qwen_vl_model_path
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True, device_map="cpu")
        dataloader, template, truncation, batch_size, gradient_accumulate_steps, seqlen, nsamples = get_mllm_dataloader(
            template=None,
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            image_processor=None,
            dataset="NeelNanda/pile-10k",
            extra_data_dir=None,
            seqlen=32,
            batch_size=1,
            split=None,
            apply_template=None,
            truncation=False,
            seed=42,
            nsamples=1,
            gradient_accumulate_steps=1,
            quant_nontext_module=True,
        )
        quant_config = AutoRoundConfig(
            bits=4,
            group_size=128,
            nsamples=1,
            batch_size=batch_size,
            iters=1,
            seqlen=seqlen,
            quant_nontext_module=True,
            truncation=truncation,
            gradient_accumulate_steps=gradient_accumulate_steps,
            device_map="cpu",
            tokenizer=tokenizer,
            processor=processor,
        )

        model = prepare(model=model, quant_config=quant_config)
        run_fn(model, dataloader)
        q_model = convert(model)
        assert (
            q_model.model.language_model.layers[0].mlp.up_proj.__class__.__name__ in target_modules
        ), "model quantization failed."

    def test_set_local(self, tiny_opt_model_path, tmp_path):
        fp32_model = AutoModelForCausalLM.from_pretrained(
            tiny_opt_model_path,
            device_map="cpu",
        )
        output_dir = tmp_path
        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path, trust_remote_code=True)
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            output_dir=output_dir,
            dtype="int4",
            iters=0,
            amp=False,
            disable_opt_rtn=True,
            scale_dtype="fp32",
            export_format="auto_round",
            device_map="cpu",
        )
        logger.info(f"Test AutoRound with config {quant_config}")
        quant_config.set_local("self_attn", AutoRoundConfig(bits=16, data_type="float", act_bits=16))

        # prepare + convert API
        model = prepare(model=fp32_model, quant_config=quant_config)
        q_model = convert(model)

        # Autoround applied subfolder for formats during saving, such as, './saved_inc/opt-125m-w4g128'.
        output_dir = q_model.name_or_path
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype="auto",
            device_map="cpu",
        )
        out = model(self.inp)[0]
        assert isinstance(q_model.model.decoder.layers[0].self_attn.v_proj, torch.nn.Linear), "set_local failed."

    @pytest.mark.skipif(not ct_installed, reason="The compressed-tensors module is not installed.")
    @pytest.mark.parametrize("scheme", ["W3A16", "W8A16", "MXFP4", "NVFP4", "FP8_STATIC"])
    def test_scheme(self, scheme, tiny_opt_model_path, tmp_path):
        # INC API
        fp32_model = copy.deepcopy(self.opt_model)
        inp = torch.ones([1, 10], dtype=torch.long, device="cpu")
        tokenizer = copy.deepcopy(self.tokenizer)

        output_dir = tmp_path
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            nsamples=32,
            seqlen=10,
            iters=1,
            amp=False,
            scale_dtype="fp16",
            scheme=scheme,
            export_format="auto_round",
            output_dir=output_dir,  # default is "tmp_auto_round"
            device_map="cpu",
        )

        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        inc_model = convert(model)
        # Autoround applied subfolder for formats during saving, such as, './saved_inc/opt-125m-w4g128'.
        output_dir = inc_model.name_or_path
        inc_model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype="auto",
            device_map="cpu",
        )
        out = inc_model(inp)[0]

    @pytest.mark.skipif(not ct_installed, reason="The compressed-tensors module is not installed.")
    def test_target_bits(self, tiny_opt_model_path, tmp_path):
        fp32_model = AutoModelForCausalLM.from_pretrained(
            tiny_opt_model_path,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path, trust_remote_code=True)

        output_dir = tmp_path
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            nsamples=32,
            seqlen=10,
            iters=1,
            target_bits=5,
            options=("MXFP4", "MXFP8"),
            enable_torch_compile=True,
            low_gpu_mem_usage=True,
            export_format="auto_round",
            device_map="cpu",
        )
        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        model = convert(model)
        # mxfp4/8 model inference relies on autoround extension for vLLM.
        target_modules = ["MXFP4QuantLinear", "MXFP8QuantLinear"]
        assert (
            model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__ in target_modules
            and model.model.decoder.layers[1].fc1.__class__.__name__ in target_modules
        ), "model is not quantized correctly, please check."

    def test_static_attention_dtype(self, tiny_opt_model_path, tmp_path):
        fp32_model = AutoModelForCausalLM.from_pretrained(
            tiny_opt_model_path,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path, trust_remote_code=True)

        output_dir = tmp_path
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            iters=0,
            nsamples=2,
            seqlen=2,
            scheme="FP8_STATIC",
            static_attention_dtype="fp8",
            output_dir=output_dir,
            export_format="auto_round",
            device_map="cpu",
        )
        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        model = convert(model)
        output_dir = model.name_or_path

        from safetensors import safe_open

        f = safe_open(os.path.join(output_dir, "model.safetensors"), framework="pt")
        assert "model.decoder.layers.1.self_attn.k_proj.input_scale" in f.keys()
        assert "model.decoder.layers.1.self_attn.k_proj.weight_scale" in f.keys()
        assert f.get_tensor("model.decoder.layers.0.self_attn.v_proj.input_scale").shape == torch.Size([1])
        assert f.get_tensor("model.decoder.layers.0.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn
        check_attrs = ["k_scale", "v_scale", "q_scale"]

        for attr in check_attrs:
            weight_name = f"model.decoder.layers.1.self_attn.{attr}"
            assert weight_name in f.keys()
            assert f.get_tensor(weight_name).shape == torch.Size([1])
            assert f.get_tensor(weight_name).dtype == torch.float32

    @pytest.mark.parametrize("static_kv_dtype", [None, "fp8", "float16"])
    def test_static_afp8_export(self, static_kv_dtype, tiny_opt_model_path, tmp_path):
        fp32_model = AutoModelForCausalLM.from_pretrained(
            tiny_opt_model_path,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path, trust_remote_code=True)

        output_dir = tmp_path
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
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
            export_format="auto_round",
            output_dir=output_dir,
            device_map="cpu",
        )

        # quantizer execute
        model = prepare(model=fp32_model, quant_config=quant_config)
        model = convert(model)
        output_dir = model.name_or_path

        from safetensors import safe_open

        f = safe_open(os.path.join(output_dir, "model.safetensors"), framework="pt")
        assert "model.decoder.layers.1.self_attn.k_proj.input_scale" in f.keys()
        assert "model.decoder.layers.1.self_attn.k_proj.weight_scale" in f.keys()
        assert f.get_tensor("model.decoder.layers.0.self_attn.v_proj.input_scale").shape == torch.Size([1])
        assert f.get_tensor("model.decoder.layers.0.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn
        if static_kv_dtype is None:
            with torch.no_grad():
                import transformers

                model = transformers.AutoModelForCausalLM.from_pretrained(
                    output_dir,
                    torch_dtype="auto",
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                model.eval()
                assert (
                    model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__
                    == "WeightFP8ActFP8StaticQuantLinear"
                ), f"Expected WeightFP8ActFP8StaticQuantLinear, got {model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__}"
                tokenizer = transformers.AutoTokenizer.from_pretrained(output_dir)
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

    @pytest.mark.parametrize(
        "scheme,  static_kv_dtype, static_attention_dtype",
        [
            ("MXFP4", None, "fp8"),
            ("MXFP4", "fp8", None),
            ("MXFP8", None, "fp8"),
            ("MXFP8", "fp8", None),
            ("NVFP4", None, "fp8"),
            ("NVFP4", "fp8", None),
        ],
    )
    def test_fp8_kv_attn(self, scheme, static_kv_dtype, static_attention_dtype, tiny_opt_model_path, tmp_path):

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.models.opt.modeling_opt import OPTForCausalLM

        model_name = tiny_opt_model_path
        config = AutoConfig.from_pretrained(model_name)
        config.num_hidden_layers = 1
        model = OPTForCausalLM(config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        output_dir = tmp_path
        quant_config = AutoRoundConfig(
            tokenizer=tokenizer,
            scheme=scheme,
            iters=0,
            seqlen=2,
            static_kv_dtype=static_kv_dtype,
            static_attention_dtype=static_attention_dtype,
            export_format="auto_round",
            output_dir=output_dir,
            reloading=False,
            device_map="cpu",
        )

        # quantizer execute
        model = prepare(model=model, quant_config=quant_config)
        compressed_model = convert(model)

        attn = compressed_model.model.decoder.layers[0].self_attn
        q_proj = attn.q_proj

        # weight_scale should exist for all quantized schemes
        assert hasattr(q_proj, "weight_scale"), f"Missing weight_scale in q_proj for scheme={scheme}"
        if static_kv_dtype == "fp8":
            assert (
                compressed_model.config.quantization_config["static_kv_dtype"] == "fp8"
            ), f"Invalid static_kv_dtype in config for scheme={scheme}, static_kv_dtype={static_kv_dtype}"

        # Only when static_kv_dtype / static_attention_dtype are fp8 do we expect FP8 KV scales
        if static_kv_dtype == "fp8" or static_attention_dtype == "fp8":
            assert attn.k_scale is not None and attn.v_scale is not None, (
                f"Missing k_scale/v_scale in attention for scheme={scheme}, "
                f"static_kv_dtype={static_kv_dtype}, static_attention_dtype={static_attention_dtype}"
            )

        if static_attention_dtype == "fp8":
            assert (
                compressed_model.config.quantization_config["static_attention_dtype"] == "fp8"
            ), f"Invalid static_attention_dtype in config for scheme={scheme}, static_attention_dtype={static_attention_dtype}"
            assert (
                getattr(attn, "q_scale", None) is not None
            ), f"Missing q_scale in attention for scheme={scheme}, static_attention_dtype={static_attention_dtype}"
