import json
import os
import shutil
from test.helpers import forbid_threaded_packing, get_model_path, opt_name_or_path

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AutoScheme
from auto_round.export.export_to_llmcompressor import export_to_fp as llmc_fp_export
from auto_round.export.export_to_llmcompressor import export_to_static_fp as llmc_static_fp_export

from ...envs import is_compressed_tensors_available

pytestmark = pytest.mark.skipif(not is_compressed_tensors_available(), reason="test requires compressed-tensors")


class TestLLMCZpInt8:
    """pack_layer must hand compressed_tensors an int8 zero-point tensor:
    NeUQI stores zp as a float32 integral-valued tensor, and CT's packer
    (pack_to_int32) hard-requires torch.int8."""

    def _make_layer(self, zp_tensor, bits=4, sym=False):
        import torch.nn as nn

        layer = nn.Linear(16, 8)
        layer.bits = bits
        layer.sym = sym
        layer.group_size = 16
        layer.data_type = "int"
        layer.act_bits = 16
        layer.act_sym = True
        layer.act_data_type = None
        layer.scale = torch.full((8, 1), 0.01)
        layer.zp = zp_tensor
        return layer

    def test_tensor_zp_cast_to_int8(self):
        import torch

        from auto_round.export.export_to_llmcompressor.export import pack_layer

        layer = self._make_layer(torch.full((8, 1), 5.0))  # float32 integral-valued
        model = torch.nn.Sequential()
        model.add_module("q_proj", layer)
        pack_layer("q_proj", model, device="cpu")
        # after compression CT owns the params: zp arrives packed to int32 and
        # the layer is marked COMPRESSED (the pre-fix code raised in pack_to_int32)
        zp = getattr(layer, "weight_zero_point", None)
        assert zp is not None and zp.dtype == torch.int32
        assert getattr(layer, "weight_packed", None) is not None

    def test_out_of_range_zp_raises(self):
        import torch

        from auto_round.export.export_to_llmcompressor.export import pack_layer

        layer = self._make_layer(torch.full((8, 1), 200.0), bits=4)  # zp >> 2^b-1: invalid at any convention
        model = torch.nn.Sequential()
        model.add_module("q_proj", layer)
        try:
            pack_layer("q_proj", model, device="cpu")
        except ValueError as e:
            assert "zero-point" in str(e)
        else:
            raise AssertionError("out-of-range zp must raise")

    def test_asym_zp_signed_convention_round_trip(self):
        """AutoRound zp is unsigned (q in [0, 2^b-1], W = (q - zp) * s);
        compressed-tensors re-quantizes with the SIGNED range
        [-2^(b-1), 2^(b-1)-1] (clamp AFTER adding zp). The unsigned zp must be
        shifted into the signed convention before packing, or CT's clamp
        collapses every level above 2^(b-1)-1. Round-trip through CT's own
        compressor must reproduce AutoRound's fake-quantized weight exactly."""
        import torch
        import torch.nn as nn
        from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor

        from auto_round.export.export_to_llmcompressor.export import pack_layer

        torch.manual_seed(0)
        out_f, in_f, gs = 8, 32, 16
        w = torch.randn(out_f, in_f)
        scale = torch.rand(out_f, in_f // gs) * 0.02 + 0.005  # [out, in/g]
        zp = torch.randint(0, 16, (out_f, in_f // gs)).float()
        zp[:, 0] = 12  # exercise levels above 2^(b-1)-1 (the pre-fix clamp)

        layer = nn.Linear(in_f, out_f)
        with torch.no_grad():
            layer.weight.copy_(w)
        layer.bits, layer.sym, layer.group_size = 4, False, gs
        layer.data_type, layer.act_bits, layer.act_sym, layer.act_data_type = "int", 16, True, None
        layer.scale, layer.zp = scale, zp

        model = nn.Sequential()
        model.add_module("q_proj", layer)
        pack_layer("q_proj", model, device="cpu")

        # AutoRound's own fake-quantized weight (unsigned convention)
        s = scale.repeat_interleave(gs, dim=1)
        z = zp.repeat_interleave(gs, dim=1)
        q = (w / s + z).round().clamp(0, 15)
        w_ref = (q - z) * s

        state = {
            "weight_packed": layer.weight_packed.detach(),
            "weight_scale": layer.weight_scale.detach(),
            "weight_zero_point": layer.weight_zero_point.detach(),
            "weight_shape": layer.weight_shape.detach(),
        }
        out = PackedQuantizationCompressor.decompress(state, layer.quantization_scheme)
        assert torch.equal(out["weight"].to(torch.float32), w_ref)

    def test_w8_asym_pack_round_trip(self):
        """8-bit asym is the boundary case: zp spans [0, 255] and only fits int8
        via the signed-convention shift. The llm_compressor format accepts W8
        asym (vLLM serves it through the compressed-tensors kernel path), so
        the pack must round-trip."""
        import torch
        import torch.nn as nn
        from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor

        from auto_round.export.export_to_llmcompressor.export import pack_layer

        torch.manual_seed(0)
        out_f, in_f, gs = 8, 32, 16
        w = torch.randn(out_f, in_f)
        scale = torch.rand(out_f, in_f // gs) * 0.02 + 0.005
        zp = torch.randint(0, 256, (out_f, in_f // gs)).float()  # full 8-bit span
        zp[:, 0] = 250  # far above 127: unsigned convention, must survive the shift

        layer = nn.Linear(in_f, out_f)
        with torch.no_grad():
            layer.weight.copy_(w)
        layer.bits, layer.sym, layer.group_size = 8, False, gs
        layer.data_type, layer.act_bits, layer.act_sym, layer.act_data_type = "int", 16, True, None
        layer.scale, layer.zp = scale, zp

        model = nn.Sequential()
        model.add_module("q_proj", layer)
        pack_layer("q_proj", model, device="cpu")

        s = scale.repeat_interleave(gs, dim=1)
        z = zp.repeat_interleave(gs, dim=1)
        q = (w / s + z).round().clamp(0, 255)
        w_ref = (q - z) * s

        state = {
            "weight_packed": layer.weight_packed.detach(),
            "weight_scale": layer.weight_scale.detach(),
            "weight_zero_point": layer.weight_zero_point.detach(),
            "weight_shape": layer.weight_shape.detach(),
        }
        out = PackedQuantizationCompressor.decompress(state, layer.quantization_scheme)
        assert torch.equal(out["weight"].to(torch.float32), w_ref)


class TestLLMC:

    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("stas/tiny-random-llama-2")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    def test_mixed_int_bits_llmcompressor_format(self, tiny_opt_model_path, tmp_path):
        """Mixed int W5A16-W8A16 auto-scheme export with an unsupported-as-first-option scheme
        leading the options (W5A16) must resolve and export through the llm_compressor backend
        (compressed-tensors pack-quantized handles any int width in 2..8)."""
        scheme = AutoScheme(
            avg_bits=6.0,
            options=("W5A16", "W6A16", "W7A16", "W8A16"),
        )
        ar = AutoRound(
            model=tiny_opt_model_path,
            iters=0,
            disable_opt_rtn=True,
            scheme=scheme,
        )
        _, tmp_path = ar.quantize_and_save(output_dir=tmp_path, format="auto_round:llm_compressor")
        model = AutoModelForCausalLM.from_pretrained(tmp_path, torch_dtype="auto", trust_remote_code=True)
        quantization_config = model.config.quantization_config.to_dict()
        assert quantization_config["format"] == "pack-quantized"
        assert quantization_config["config_groups"], "no quantized config groups were exported"
        for group in quantization_config["config_groups"].values():
            weights = group["weights"]
            assert weights["num_bits"] in (5, 6, 7, 8), f"unexpected bit-width {weights['num_bits']}"
            assert weights["type"] == "int"
            assert weights["symmetric"] is True
            assert group["input_activations"] is None  # weight-only quantization

    # remove since w8a8 not in llmcompressor format supported schemes
    # def test_llmcompressor_w8a8(self):
    #     bits, group_size, sym, act_bits = 8, -1, True, 8
    #     ## quantize the model
    #     autoround = AutoRound(
    #         self.model,
    #         self.tokenizer,
    #         bits=bits,
    #         group_size=group_size,
    #         sym=sym,
    #         act_bits=act_bits,
    #         seqlen=8,
    #         nsamples=2,
    #         iters=0,
    #     )
    #     autoround.quantize()
    #     autoround.save_quantized("./saved", format="llm_compressor", inplace=True)

    @pytest.mark.timeout(120)
    def test_llmcompressor_fp8(self, tmp_path):
        ## quantize the model
        model_name = opt_name_or_path
        autoround = AutoRound(
            model_name,
            scheme="FP8_STATIC",
            seqlen=8,
            nsamples=2,
            iters=0,
        )
        _, quantized_model_path = autoround.quantize_and_save(tmp_path, format="llm_compressor")
        # from vllm import LLM
        # model = LLM(tmp_path)
        # result = model.generate("Hello my name is")
        # print(result)

        import json

        from safetensors import safe_open

        config = json.load(open(os.path.join(quantized_model_path, "config.json")))
        assert "group_0" in config["quantization_config"]["config_groups"]
        assert config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["num_bits"] == 8
        assert config["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"] == "channel"
        assert config["quantization_config"]["quant_method"] == "compressed-tensors"

        f = safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt")
        assert len(f.get_tensor("model.decoder.layers.0.fc1.weight_scale").shape) == 2

    def test_autoround_llmcompressor_fp8(self, tmp_path):
        ## quantize the model
        model_name = opt_name_or_path
        autoround = AutoRound(
            model_name,
            scheme="FP8_STATIC",
            seqlen=8,
            group_size=0,
            nsamples=2,
            iters=0,
        )
        _, quantized_model_path = autoround.quantize_and_save(tmp_path, format="auto_round:llm_compressor")

        import json

        config = json.load(open(os.path.join(quantized_model_path, "config.json")))
        assert "group_0" in config["quantization_config"]["config_groups"]
        assert config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["num_bits"] == 8
        assert config["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"] == "tensor"
        assert config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["strategy"] == "tensor"
        assert config["quantization_config"]["quant_method"] == "compressed-tensors"

    def test_mxfp8_llmcompressor_format(self, tiny_opt_model_path, tmp_path):
        scheme = "mxfp8"
        ar = AutoRound(
            model=tiny_opt_model_path,
            iters=0,
            disable_opt_rtn=True,
            scheme=scheme,
            disable_model_free=True,
        )
        compressed_model, tmp_path = ar.quantize_and_save(output_dir=tmp_path, format="llm_compressor")
        tmp_layer = compressed_model.model.decoder.layers[1].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight")
            and tmp_layer.weight_scale.dtype is torch.uint8
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal MXFP8 packing name or data_type or shape"
        quantization_config = transformers.AutoConfig.from_pretrained(
            tmp_path, trust_remote_code=True
        ).quantization_config
        assert (
            quantization_config["format"] == "mxfp8-quantized"
            and quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] == 8
            and quantization_config["config_groups"]["group_0"]["weights"]["group_size"] == 32
            and quantization_config["config_groups"]["group_0"]["weights"]["scale_dtype"] == "torch.uint8"
            and quantization_config["config_groups"]["group_0"]["input_activations"]["num_bits"] == 8
            and quantization_config["config_groups"]["group_0"]["input_activations"]["group_size"] == 32
            and quantization_config["config_groups"]["group_0"]["input_activations"]["scale_dtype"] == "torch.uint8"
            and quantization_config["ignore"] == ["lm_head"]
        ), f"Invalid MXFP8 quantization configuration: {quantization_config}"

    @pytest.mark.timeout(60)
    def test_mxfp8_llmcompressor_kv_config(self, tiny_opt_model_path, tmp_path):
        ar = AutoRound(
            model=tiny_opt_model_path,
            iters=0,
            disable_opt_rtn=True,
            scheme="mxfp8",
            static_kv_dtype="fp8",
        )
        _, quantized_model_path = ar.quantize_and_save(output_dir=tmp_path, format="llm_compressor")

        with open(os.path.join(quantized_model_path, "config.json")) as f:
            config = json.load(f)

        kv_cache_scheme = config["quantization_config"]["kv_cache_scheme"]
        assert kv_cache_scheme is not None
        assert kv_cache_scheme["num_bits"] == 8
        assert kv_cache_scheme["type"] == "float"
        assert kv_cache_scheme["strategy"] == "tensor"
        assert kv_cache_scheme["dynamic"] is False
        assert kv_cache_scheme["symmetric"] is True

    def test_mxfp8_llmcompressor_per_head_kv_config(self, tiny_opt_model_path, tmp_path):
        from safetensors import safe_open

        ar = AutoRound(
            model=tiny_opt_model_path,
            iters=0,
            disable_opt_rtn=True,
            scheme="mxfp8",
            static_kv_dtype="fp8",
            static_kv_granularity="head",
        )
        compressed_model, quantized_model_path = ar.quantize_and_save(output_dir=tmp_path, format="llm_compressor")

        with open(os.path.join(quantized_model_path, "config.json")) as f:
            config = json.load(f)

        kv_cache_scheme = config["quantization_config"]["kv_cache_scheme"]
        assert kv_cache_scheme is not None
        assert kv_cache_scheme["strategy"] == "attn_head"

        num_kv_heads = compressed_model.config.num_attention_heads
        if hasattr(compressed_model.config, "num_key_value_heads"):
            num_kv_heads = compressed_model.config.num_key_value_heads
        with safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt") as f:
            k_scale = f.get_tensor("model.decoder.layers.0.self_attn.k_scale")
            v_scale = f.get_tensor("model.decoder.layers.0.self_attn.v_scale")
        assert k_scale.shape == torch.Size([num_kv_heads])
        assert v_scale.shape == torch.Size([num_kv_heads])

    def test_mxfp8_llmcompressor_attention_config(self, tiny_opt_model_path, tmp_path):
        from safetensors import safe_open

        ar = AutoRound(
            model=tiny_opt_model_path,
            iters=0,
            disable_opt_rtn=True,
            scheme="mxfp8",
            static_attention_dtype="fp8",
        )
        compressed_model, quantized_model_path = ar.quantize_and_save(output_dir=tmp_path, format="llm_compressor")

        with open(os.path.join(quantized_model_path, "config.json")) as f:
            saved_config = json.load(f)

        # Attention config is stored in a dedicated top-level field (not config_groups).
        attention_config = saved_config["quantization_config"]["attention_input_activations"]
        assert attention_config is not None
        assert attention_config["targets"] == [compressed_model.model.decoder.layers[0].self_attn.__class__.__name__]
        assert attention_config["input_activations"]["num_bits"] == 8
        assert attention_config["input_activations"]["type"] == "float"
        assert attention_config["input_activations"]["strategy"] == "tensor"
        assert attention_config["input_activations"]["dynamic"] is False
        assert attention_config["input_activations"]["symmetric"] is True
        assert saved_config["quantization_config"]["kv_cache_scheme"] is not None

        quantization_config = transformers.AutoConfig.from_pretrained(
            quantized_model_path, trust_remote_code=True
        ).quantization_config
        config_groups = quantization_config["config_groups"]
        # Only Linear layers should be in config_groups.
        for group_targets in (g["targets"] for g in config_groups.values()):
            assert "Linear" in group_targets, f"Unexpected non-Linear targets in config_groups: {group_targets}"

        assert quantization_config["kv_cache_scheme"] is not None
        assert getattr(compressed_model.model.decoder.layers[0].self_attn, "q_scale", None) is not None

        with safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt") as checkpoint:
            keys = list(checkpoint.keys())
        assert any(key.endswith(".self_attn.q_scale") for key in keys), "q_scale not found in checkpoint"
        assert not any(key.endswith(".q_max") for key in keys), "q_max should not be exported"

    def test_mxfp8_llmcompressor_per_head_attention_config(self, tiny_opt_model_path, tmp_path):
        from safetensors import safe_open

        ar = AutoRound(
            model=tiny_opt_model_path,
            iters=0,
            disable_opt_rtn=True,
            scheme="mxfp8",
            static_attention_dtype="fp8",
            static_attention_granularity="head",
        )
        compressed_model, quantized_model_path = ar.quantize_and_save(output_dir=tmp_path, format="llm_compressor")

        with open(os.path.join(quantized_model_path, "config.json")) as f:
            saved_config = json.load(f)

        attention_config = saved_config["quantization_config"]["attention_input_activations"]
        assert attention_config is not None
        assert attention_config["targets"] == [compressed_model.model.decoder.layers[0].self_attn.__class__.__name__]
        assert attention_config["input_activations"]["strategy"] == "attn_head"
        assert saved_config["quantization_config"]["kv_cache_scheme"]["strategy"] == "attn_head"

        with safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt") as f:
            q_scale = f.get_tensor("model.decoder.layers.0.self_attn.q_scale")
            k_scale = f.get_tensor("model.decoder.layers.0.self_attn.k_scale")
        assert q_scale.shape == torch.Size([compressed_model.config.num_attention_heads])
        assert k_scale.ndim == 1

    @pytest.mark.timeout(60)
    def test_mixed_precision_llmcompressor_format(self, tiny_opt_model_path, tmp_path):
        scheme = AutoScheme(
            avg_bits=7,
            options=("MXFP4", "MXFP8"),
            shared_layers=["q_proj", "k_proj", "v_proj"],
        )
        ar = AutoRound(
            model=tiny_opt_model_path,
            iters=0,
            disable_opt_rtn=True,
            scheme=scheme,
        )
        _, tmp_path = ar.quantize_and_save(output_dir=tmp_path, format="llm_compressor")
        model = AutoModelForCausalLM.from_pretrained(tmp_path, torch_dtype="auto", trust_remote_code=True)
        op = model.model.decoder.layers[0].fc1
        if op.quantization_scheme.targets != ["Linear"]:
            assert (
                op.quantization_scheme.weights.num_bits == 8
                and op.quantization_scheme.input_activations.num_bits == 8
                and op.quantization_scheme.weights.group_size == 32
                and op.quantization_scheme.input_activations.group_size == 32
                and op.quantization_scheme.weights.scale_dtype == torch.uint8
                and op.quantization_scheme.input_activations.scale_dtype == torch.uint8
            ), "Illegal MXFP4 packing name or data_type or shape"
        quantization_config = model.config.quantization_config.to_dict()
        assert (
            quantization_config["format"] == "mixed-precision"
            and quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] == 8
            and quantization_config["config_groups"]["group_0"]["input_activations"]["num_bits"] == 8
            and quantization_config["config_groups"]["group_0"]["format"] == "mxfp8-quantized"
            and quantization_config["config_groups"]["group_1"]["weights"]["num_bits"] == 4
            and quantization_config["config_groups"]["group_1"]["input_activations"]["num_bits"] == 4
            and quantization_config["config_groups"]["group_1"]["format"] == "mxfp4-pack-quantized"
            and quantization_config["ignore"] == ["lm_head"]
        ), f"Invalid mixed precision quantization configuration: {quantization_config}"


def test_llmcompressor_static_fp_export_packs_serially(tiny_opt_model_path, dataloader, tmp_path, monkeypatch):
    autoround = AutoRound(
        tiny_opt_model_path,
        scheme="FP8_STATIC",
        seqlen=8,
        nsamples=2,
        iters=0,
        dataset=dataloader,
    )
    autoround.quantize()
    forbid_threaded_packing(monkeypatch, llmc_static_fp_export)
    autoround.save_quantized(tmp_path, format="llm_compressor")
    assert os.path.exists(os.path.join(tmp_path, "config.json"))


def test_llmcompressor_static_fp8_kv_config(tiny_opt_model_path, dataloader, tmp_path):
    autoround = AutoRound(
        tiny_opt_model_path,
        scheme="FP8_STATIC",
        seqlen=8,
        nsamples=2,
        iters=0,
        dataset=dataloader,
        static_kv_dtype="fp8",
    )
    _, quantized_model_path = autoround.quantize_and_save(tmp_path, format="llm_compressor")

    with open(os.path.join(quantized_model_path, "config.json")) as f:
        config = json.load(f)
    kv_cache_scheme = config["quantization_config"]["kv_cache_scheme"]
    assert kv_cache_scheme is not None
    assert kv_cache_scheme["num_bits"] == 8
    assert kv_cache_scheme["type"] == "float"
    assert kv_cache_scheme["strategy"] == "tensor"
    assert kv_cache_scheme["dynamic"] is False
    assert kv_cache_scheme["symmetric"] is True


def test_llmcompressor_static_fp8_attention_config(dataloader, tmp_path):
    model_name = get_model_path("stas/tiny-random-llama-2")
    autoround = AutoRound(
        model_name,
        scheme="FP8_STATIC",
        seqlen=8,
        nsamples=2,
        iters=0,
        dataset=dataloader,
        static_attention_dtype="fp8",
    )
    _, quantized_model_path = autoround.quantize_and_save(tmp_path, format="llm_compressor")

    with open(os.path.join(quantized_model_path, "config.json")) as f:
        saved_config = json.load(f)

    # Attention config is stored in a dedicated top-level field (not config_groups)
    # to prevent compressed_tensors from trying to compress attention modules at load time.
    attention_config = saved_config["quantization_config"]["attention_input_activations"]
    assert attention_config is not None
    assert "Linear" not in attention_config["targets"]
    assert attention_config["input_activations"]["num_bits"] == 8
    assert attention_config["input_activations"]["type"] == "float"
    assert attention_config["input_activations"]["strategy"] == "tensor"
    assert attention_config["input_activations"]["dynamic"] is False
    assert attention_config["input_activations"]["symmetric"] is True
    assert saved_config["quantization_config"]["kv_cache_scheme"] is not None

    # Verify the model can be loaded without crashing.
    model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", trust_remote_code=True)
    quantization_config = model.config.quantization_config
    config = quantization_config.to_dict() if hasattr(quantization_config, "to_dict") else quantization_config
    config_groups = config["config_groups"]

    # Only Linear layers should be in config_groups; attention modules use the
    # separate attention_input_activations field.
    assert "group_0" in config_groups
    for group_targets in (g["targets"] for g in config_groups.values()):
        assert "Linear" in group_targets, f"Unexpected non-Linear targets in config_groups: {group_targets}"

    assert config["kv_cache_scheme"] is not None

    # Verify KV cache quant parameters (q_scale, k_scale, v_scale) were saved
    # in the checkpoint. They show as UNEXPECTED on the loaded model because the
    # base HuggingFace architecture does not register them as standard parameters.
    import glob as _glob

    safetensors_files = _glob.glob(os.path.join(quantized_model_path, "*.safetensors"))
    assert len(safetensors_files) > 0, "No safetensors file found"
    from safetensors import safe_open as _safe_open

    has_q_scale = False
    for sf in safetensors_files:
        with _safe_open(sf, framework="pt") as f:
            for key in f.keys():
                if "self_attn.q_scale" in key:
                    has_q_scale = True
                    break
    assert has_q_scale, "q_scale not found in checkpoint"


def test_llmcompressor_mxfp8_export_packs_serially(tmp_path, monkeypatch):
    autoround = AutoRound(
        model=opt_name_or_path,
        iters=0,
        disable_opt_rtn=True,
        scheme="mxfp8",
    )
    autoround.quantize()
    forbid_threaded_packing(monkeypatch, llmc_fp_export)
    compressed_model = autoround.save_quantized(output_dir=tmp_path, format="llm_compressor")
    tmp_layer = compressed_model.model.decoder.layers[1].self_attn.q_proj
    assert hasattr(tmp_layer, "weight_scale")


class TestLLMCZpUnsignedDomain:
    """The packed-level validation must reject zero points outside the
    UNSIGNED level range [0, 2^bits-1] before the signed-convention shift; a
    shifted-only check would let e.g. a raw 4-bit zp of 100 pass (shifted 92
    fits int8)."""

    def _make_layer(self, zp, bits=4, sym=False):
        import torch.nn as nn

        layer = nn.Linear(16, 8)
        layer.bits = bits
        layer.sym = sym
        layer.group_size = 16
        layer.data_type = "int"
        layer.act_bits = 16
        layer.act_sym = True
        layer.act_data_type = None
        layer.scale = torch.full((8, 1), 0.01)
        layer.zp = zp
        return layer

    def test_tensor_zp_above_unsigned_range_rejected(self):
        import torch

        from auto_round.export.export_to_llmcompressor.export import pack_layer

        layer = self._make_layer(torch.full((8, 1), 100.0))  # raw 100 > 15
        m = torch.nn.Sequential()
        m.add_module("proj", layer)
        with pytest.raises(ValueError, match="unsigned level range"):
            pack_layer("proj", m, device="cpu")

    def test_tensor_zp_negative_rejected(self):
        import torch

        from auto_round.export.export_to_llmcompressor.export import pack_layer

        layer = self._make_layer(torch.full((8, 1), -3.0))  # shifted -11 fits int8
        m = torch.nn.Sequential()
        m.add_module("proj", layer)
        with pytest.raises(ValueError, match="unsigned level range"):
            pack_layer("proj", m, device="cpu")

    def test_scalar_zp_out_of_range_rejected(self):
        from auto_round.export.export_to_llmcompressor.export import pack_layer

        for bad in (17.0, -1.0):  # 4-bit unsigned range is [0, 15]
            layer = self._make_layer(bad)
            m = torch.nn.Sequential()
            m.add_module("proj", layer)
            with pytest.raises(ValueError, match="unsigned level range"):
                pack_layer("proj", m, device="cpu")

    def test_in_range_values_pass(self):
        import torch

        from auto_round.export.export_to_llmcompressor.export import pack_layer

        layer = self._make_layer(torch.randint(0, 16, (8, 1)).float())
        m = torch.nn.Sequential()
        m.add_module("proj", layer)
        pack_layer("proj", m, device="cpu")
        # CT's compressor consumes the int8 zp and stores its bit-packed form
        assert layer.weight_zero_point.dtype == torch.int32
        assert layer.weight_packed is not None


class TestLLMCZpFiniteIntegral:
    """NaN passes every min/max comparison and would be cast straight to
    int8; non-integral values must not be silently rounded into a level."""

    def _make_layer(self, zp, bits=4):
        import torch.nn as nn

        layer = nn.Linear(16, 8)
        layer.bits = bits
        layer.sym = False
        layer.group_size = 16
        layer.data_type = "int"
        layer.act_bits = 16
        layer.act_sym = True
        layer.act_data_type = None
        layer.scale = torch.full((8, 1), 0.01)
        layer.zp = zp
        return layer

    def _expect_refusal(self, zp, match):
        from auto_round.export.export_to_llmcompressor.export import pack_layer

        layer = self._make_layer(zp)
        m = torch.nn.Sequential()
        m.add_module("proj", layer)
        with pytest.raises(ValueError, match=match):
            pack_layer("proj", m, device="cpu")

    def test_nan_tensor_zp_rejected(self):
        zp = torch.zeros(8, 1)
        zp[3, 0] = float("nan")
        self._expect_refusal(zp, "non-finite")

    def test_non_integral_tensor_zp_rejected(self):
        self._expect_refusal(torch.full((8, 1), -0.4), "non-integral")

    def test_nan_scalar_zp_rejected(self):
        self._expect_refusal(float("nan"), "not a finite integer")

    def test_non_integral_scalar_zp_rejected(self):
        self._expect_refusal(7.3, "not a finite integer")
