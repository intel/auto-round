import os
import shutil
import sys

import pytest
import torch
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig

from ...helpers import eval_generated_prompt, get_model_path, get_tiny_model, save_tiny_model

AUTO_ROUND_PATH = __file__.split("/")
AUTO_ROUND_PATH = "/".join(AUTO_ROUND_PATH[: AUTO_ROUND_PATH.index("test")])


class TestGGUF:

    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("Qwen/Qwen2.5-0.5B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_q4_0(self, tiny_qwen_model_path):
        bits, group_size, sym = 4, 32, True
        autoround = AutoRound(
            tiny_qwen_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            data_type="int",
            nsamples=1,
            seqlen=8,
        )
        quantized_model_path = self.save_dir

        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, inplace=False, format="gguf:q4_0"
        )
        gguf_file = os.listdir(quantized_model_path)[0]
        assert gguf_file.endswith(".gguf"), "Saved file is not in gguf format"
        # Accuracy test is covered in test_cuda/export/test_gguf_format.py::TestAutoRound::test_q4_0_accuracy

    def test_q2_k_s_routes_calibrated_rtn(self, tiny_qwen_model_path):
        autoround = AutoRound(
            tiny_qwen_model_path,
            scheme="gguf:q2_k_s",
            iters=0,
            nsamples=1,
            seqlen=8,
        )

        assert type(autoround).__name__ == "CalibratedRTNCompressor"
        assert isinstance(autoround.quantize_config, OptimizedRTNConfig)

    def test_func(self):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model_name,
            iters=0,
            disable_opt_rtn=True,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, inplace=False, format="gguf:q*_1"
        )
        assert autoround.group_size == 32
        assert not autoround.sym
        gguf_file = os.listdir(quantized_model_path)[0]
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        eval_generated_prompt(model, self.tokenizer)

    def test_q4_k_m(self, dataloader, tiny_qwen_model_path):
        model_name = tiny_qwen_model_path
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
            "model.layers.1.mlp.gate_proj": {"bits": 3},
            "model.layers.0.mlp.gate_proj": {"bits": 8},
        }
        autoround = AutoRound(
            model_name,
            layer_config=layer_config,
            iters=0,
            seqlen=1,
            nsamples=8,
            dataset=dataloader,
            disable_opt_rtn=True,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="gguf:q4_k_m,fake"
        )
        assert autoround.layer_config["model.layers.1.self_attn.v_proj"]["super_group_size"] == 16
        assert autoround.layer_config["model.layers.1.self_attn.v_proj"]["data_type"] == "int_sym_dq"
        assert autoround.layer_config["model.layers.0.self_attn.v_proj"]["data_type"] == "int_asym_dq"
        assert autoround.model.model.layers[0].self_attn.v_proj.bits == 4
        assert autoround.model.model.layers[1].self_attn.v_proj.bits == 6
        assert autoround.model.model.embed_tokens.bits == 6
        assert autoround.model.model.embed_tokens.group_size == 16
        assert autoround.model.model.layers[1].mlp.gate_proj.bits == 3
        assert autoround.model.model.layers[0].mlp.gate_proj.bits == 8
        assert autoround.layer_config["model.layers.0.mlp.gate_proj"]["mostly"] == "gguf:q8_0"

    def test_all_format(self, tiny_qwen_model_path):
        model_name = tiny_qwen_model_path
        python_path = sys.executable
        # for gguf_format in ["gguf:q4_0", "gguf:q4_1", "gguf:q4_k_m", "gguf:q6_k"]:
        for gguf_format in ["gguf:q4_k_m"]:
            res = os.system(
                f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {model_name} "
                f" --bs 16 --iters 1 --nsamples 1 --seqlen 16 --format {gguf_format}"
            )
            if res > 0 or res == -1:
                assert False, "cmd line test fail, please have a check"
            shutil.rmtree("../../tmp_autoround", ignore_errors=True)

            res = os.system(
                f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {model_name}"
                f" --bs 16 --iters 0 --nsamples 1 --seqlen 16 --format fake,{gguf_format}"
            )
            if res > 0 or res == -1:
                assert False, "cmd line test fail, please have a check"
            shutil.rmtree("../../tmp_autoround", ignore_errors=True)

        # test q2_k_mixed with iters=0 (RTN) on non-MoE model — should still work
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {model_name}"
            f" --bs 16 --iters 0 --disable_opt_rtn --nsamples 1 --seqlen 16 --scheme GGUF:Q2_K_MIXED"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("../../tmp_autoround", ignore_errors=True)

        # test q2_k_mixed with iters=1 on non-MoE model — should fallback to q4_k_m
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {model_name}"
            f" --bs 16 --iters 1 --nsamples 1 --seqlen 16 --format gguf:q2_k_mixed"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("../../tmp_autoround", ignore_errors=True)

    def test_vlm_gguf(self, tiny_qwen_vl_model_path):
        from auto_round import AutoRound

        autoround = AutoRound(
            tiny_qwen_vl_model_path,
            iters=0,
            nsamples=8,
            disable_opt_rtn=True,
            quant_nontext_module=True,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_0")
        assert "mmproj-model.gguf" in os.listdir(quantized_model_path)
        for file_name in os.listdir(quantized_model_path):
            file_size = os.path.getsize(os.path.join(quantized_model_path, file_name)) / 1024**2
            if file_name == "mmproj-model.gguf":
                assert file_size < 60, f"file size {file_size} MB is too large for non-quantized mmproj-model.gguf"
            else:
                assert file_size < 270, f"file size {file_size} MB is too large for non-quantized mmproj-model.gguf"

    def test_vlm_gguf_wo_quant_nontext_module(self, tiny_qwen_vl_model_path):
        from auto_round import AutoRound

        autoround = AutoRound(
            tiny_qwen_vl_model_path,
            iters=0,
            nsamples=8,
            disable_opt_rtn=True,
            quant_nontext_module=False,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_0")
        assert "mmproj-model.gguf" in os.listdir(quantized_model_path)
        for file_name in os.listdir(quantized_model_path):
            file_size = os.path.getsize(os.path.join(quantized_model_path, file_name)) / 1024**2
            if file_name == "mmproj-model.gguf":
                assert abs(file_size - 361) < 5.0
            else:
                assert abs(file_size - 264) < 5.0

    def test_mmproj_uses_native_f32_export_when_nontext_is_not_quantized(self, tmp_path, monkeypatch):
        from auto_round.export.export_to_gguf import export
        from auto_round.export.export_to_gguf.config import ModelType

        ftypes = {
            "q4_0": object(),
            "f32": object(),
        }
        wrapper_calls = []

        class FakeConfig:
            model_type = "qwen2_vl"

        class FakeModel:
            name_or_path = str(tmp_path)
            config = FakeConfig()

        class FakeModelClass:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.model_arch = "MMPROJ"

        class FakeModelBase:
            @staticmethod
            def load_hparams(_dir_model, _is_mistral_format):
                return {"architectures": ["FakeArchitecture"], "quantization_config": {"bits": 4}}

        class FakeConversion:
            ModelBase = FakeModelBase

            @staticmethod
            def model_type(model_type):
                return model_type

            @staticmethod
            def get_model_architecture(_hparams, _model_type):
                return "FakeArchitecture"

            @staticmethod
            def get_model_class(_model_architecture, model_type):
                return FakeModelClass

        def fake_wrapper(model_instance, **kwargs):
            wrapper_calls.append(model_instance)
            model_instance.was_wrapped = True
            model_instance.model = kwargs["model"]
            return model_instance

        monkeypatch.setattr(export, "FTYPE_MAP", ftypes)
        monkeypatch.setattr(export, "get_conversion", lambda *_args, **_kwargs: FakeConversion)
        monkeypatch.setattr(export, "wrapper_model_instance", fake_wrapper)
        monkeypatch.setattr(export, "handle_special_model", lambda model_instance, _architecture: model_instance)

        model_instance = export.create_model_class(
            output_dir=str(tmp_path),
            model=FakeModel(),
            layer_config={},
            backend="gguf:q4_0",
            model_type=ModelType.MMPROJ,
            quant_nontext_module=False,
        )

        assert wrapper_calls == []
        assert not hasattr(model_instance, "model")
        assert model_instance.ftype is ftypes["f32"]
        assert model_instance.fname_out == tmp_path / "mmproj-model.gguf"

    def test_text_still_uses_autoround_wrapper_when_nontext_is_not_quantized(self, tmp_path, monkeypatch):
        from auto_round.export.export_to_gguf import export
        from auto_round.export.export_to_gguf.config import ModelType

        ftypes = {
            "q4_0": object(),
            "f32": object(),
        }
        wrapper_calls = []

        class FakeConfig:
            model_type = "qwen2_vl"

        class FakeModel:
            name_or_path = str(tmp_path)
            config = FakeConfig()

        class FakeModelClass:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.model_arch = "TEXT"

        class FakeModelBase:
            @staticmethod
            def load_hparams(_dir_model, _is_mistral_format):
                return {"architectures": ["FakeArchitecture"], "quantization_config": {"bits": 4}}

        class FakeConversion:
            ModelBase = FakeModelBase

            @staticmethod
            def model_type(model_type):
                return model_type

            @staticmethod
            def get_model_architecture(_hparams, _model_type):
                return "FakeArchitecture"

            @staticmethod
            def get_model_class(_model_architecture, model_type):
                return FakeModelClass

        def fake_wrapper(model_instance, **kwargs):
            wrapper_calls.append((model_instance, kwargs))
            model_instance.was_wrapped = True
            model_instance.model = kwargs["model"]
            return model_instance

        monkeypatch.setattr(export, "FTYPE_MAP", ftypes)
        monkeypatch.setattr(export, "get_conversion", lambda *_args, **_kwargs: FakeConversion)
        monkeypatch.setattr(export, "wrapper_model_instance", fake_wrapper)
        monkeypatch.setattr(export, "handle_special_model", lambda model_instance, _architecture: model_instance)

        model_instance = export.create_model_class(
            output_dir=str(tmp_path),
            model=FakeModel(),
            layer_config={},
            backend="gguf:q4_0",
            model_type=ModelType.TEXT,
            quant_nontext_module=False,
        )

        assert len(wrapper_calls) == 1
        assert wrapper_calls[0][0] is model_instance
        assert wrapper_calls[0][1]["quant_nontext_module"] is False
        assert model_instance.was_wrapped
        assert isinstance(model_instance.model, FakeModel)
        assert model_instance.ftype is ftypes["q4_0"]
        assert model_instance.fname_out == tmp_path

    def test_qtype_setting(self, tiny_qwen_vl_model_path):
        # Qwen2.5-0.5B-Instruct no output, token_embed q6_k fallbakc to q8_0 336M
        # Qwen3-0.6B output q6_k, token_embed q4_0  448M
        # Qwen3-8B output q6_k, token_embed q4_0 4.5G
        # Llama-3.2-1B-Instruct o output, token_embed q6_k 736M
        from auto_round.compressors.utils import set_layer_config
        from auto_round.export.export_to_gguf.config import ModelType

        model_name = get_model_path("Qwen/Qwen2.5-0.5B-Instruct")
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
            ar.ignore_layers,
            ar.quant_lm_head,
            enable_gguf_official_mixed=True,
            is_mllm=ar.mllm,
        )
        assert ar.layer_config["model.embed_tokens"]["bits"] == 8
        assert "lm_head" not in ar.layer_config

        model_name = get_model_path("Qwen/Qwen3-0.6B")
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
            ar.ignore_layers,
            ar.quant_lm_head,
            enable_gguf_official_mixed=True,
            is_mllm=ar.mllm,
        )
        assert ar.layer_config["model.embed_tokens"]["bits"] == 4
        assert ar.layer_config["lm_head"]["bits"] == 6 and ar.layer_config["lm_head"]["super_bits"] == 8

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
            ar.ignore_layers,
            ar.quant_lm_head,
            enable_gguf_official_mixed=True,
            is_mllm=ar.mllm,
        )
        assert (
            ar.layer_config["lm_head"]["bits"] == 4
            and ar.layer_config["model.embed_tokens"]["bits"] == 6
            and ar.layer_config["model.embed_tokens"]["super_bits"] == 8
        )

        ar = AutoRound(model=tiny_qwen_vl_model_path, scheme="gguf:q4_0", iters=0)
        ar.formats = ["gguf:q4_0"]
        ar.layer_config, _, _ = set_layer_config(
            ar.model,
            ar.layer_config,
            ar.scheme,
            ar.scale_dtype,
            ar.supported_types,
            ar.inner_supported_types,
            ar.quant_block_list,
            ar.ignore_layers,
            ar.quant_lm_head,
            enable_gguf_official_mixed=True,
            is_mllm=ar.mllm,
        )
        assert ar.layer_config["model.language_model.embed_tokens"]["bits"] == 6
        assert ar.layer_config["model.language_model.embed_tokens"]["super_bits"] == 8

        ar = AutoRound(
            model=tiny_qwen_vl_model_path,
            format="gguf:q4_0",
            iters=0,
            disable_opt_rtn=True,
            disable_model_free=True,
            quant_nontext_module=False,
        )
        ar.post_init()
        assert ar.quantizer.layer_config["model.language_model.embed_tokens"]["bits"] == 6
        assert ar.quantizer.layer_config["model.language_model.embed_tokens"]["super_bits"] == 8

    def test_q2k_mixed(self, tiny_qwen_moe_model_path):
        model_name = tiny_qwen_moe_model_path
        autoround = AutoRound(
            model_name,
            iters=0,
            nsamples=1,
            seqlen=16,
            disable_opt_rtn=True,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q2_k_mixed")
        gguf_file = os.listdir(quantized_model_path)[0]
        file_size = os.path.getsize(os.path.join(quantized_model_path, gguf_file)) / 1024**2
        assert file_size < 1150, f"file size {file_size} MB is too large for q2_k_mixed format"
        from gguf.gguf_reader import GGUFReader

        gguf_model = GGUFReader(os.path.join(quantized_model_path, gguf_file))
        assert gguf_model.get_tensor(2).name == "blk.0.attn_k.weight"
        assert gguf_model.get_tensor(2).tensor_type.name == "Q4_K"
        tensor_types = {tensor.name: tensor.tensor_type.name for tensor in gguf_model.tensors}
        assert tensor_types["blk.0.ffn_up_exps.weight"] == "Q2_K"

    def test_q2k_mixed_keeps_only_three_dim_expert_weights_at_q2k(self, tiny_qwen_moe_model_path):
        model_name = tiny_qwen_moe_model_path
        autoround = AutoRound(
            model_name,
            iters=0,
            nsamples=1,
            seqlen=16,
            disable_opt_rtn=True,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="gguf:q2_k_mixed")
        gguf_file = os.listdir(quantized_model_path)[0]
        from gguf.gguf_reader import GGUFReader

        gguf_model = GGUFReader(os.path.join(quantized_model_path, gguf_file))
        tensor_types = {tensor.name: tensor.tensor_type.name for tensor in gguf_model.tensors}

        assert tensor_types["token_embd.weight"] == "Q8_0"
        assert tensor_types["output.weight"] == "Q8_0"
        assert tensor_types["blk.0.attn_k.weight"] == "Q4_K"
        assert tensor_types["blk.0.ffn_down_exps.weight"] == "Q5_0"
        assert tensor_types["blk.0.ffn_up_exps.weight"] == "Q2_K"


class TestGGUFZeroBlock:
    """All-zero blocks (e.g. padded/unused vocab rows in an embedding tensor) must
    quantize with scale d=0.0, not NaN. A single NaN fp16 `d` in an exported GGUF
    makes llama.cpp return NaN logits for any batch touching those rows."""

    def test_q6_k_all_zero_block_no_nan(self):
        from auto_round.data_type.gguf import quant_tensor_gguf_sym_dq

        tensor = torch.zeros(2, 256)
        tensor[1] = torch.randn(256)
        qdq, scales, _ = quant_tensor_gguf_sym_dq(tensor, bits=6, scale_dtype=torch.float32)
        assert not torch.isnan(scales["scale"]).any(), "Q6_K sub-scales contain NaN for all-zero block"
        assert not torch.isnan(scales["d_scale"]).any(), "Q6_K d_scale contains NaN for all-zero block"
        assert not torch.isnan(qdq).any()
        assert torch.equal(qdq[0], torch.zeros(256)), "all-zero block must reconstruct exactly to zeros"

    def test_q6_k_all_zero_block_packs_zero_d(self):
        import numpy as np

        from auto_round.export.export_to_gguf.packing import ggml_quant

        tensor = torch.zeros(2, 256)
        tensor[1] = torch.randn(256)
        packed = ggml_quant(tensor, "q6_k", device="cpu")
        # Q6_K block layout: [ql(128), qh(64), scales(16), d(fp16)]
        d = np.ascontiguousarray(packed.reshape(2, 210)[:, -2:]).view(np.float16)
        assert not np.isnan(d).any(), f"packed Q6_K d scales contain NaN: {d}"
        assert d[0] == 0.0
