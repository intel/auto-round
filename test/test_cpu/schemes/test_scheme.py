import os
import shutil

import pytest
import torch
import torch.nn as nn
import transformers

from auto_round import AutoRound
from auto_round.schemes import QuantizationScheme, _handle_special_schemes

from ...helpers import get_model_path, get_tiny_model, opt_name_or_path, qwen_name_or_path, save_tiny_model


class TestAutoRound:

    @pytest.fixture(autouse=True)
    def setup_save_folder(self, tmp_path):
        self.save_folder = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_folder, ignore_errors=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    def test_gguf(self, tiny_qwen_model_path, dataloader):
        ar = AutoRound(
            tiny_qwen_model_path,
            scheme="W2A16",
            nsamples=1,
            iters=1,
            seqlen=2,
            dataset=dataloader,
        )
        ar.quantize_and_save(self.save_folder, format="gguf:q4_k_m")
        assert ar.bits == 4

    def test_w4a16(self, tiny_opt_model_path, dataloader):
        ar = AutoRound(tiny_opt_model_path, scheme="W4A16", nsamples=1, iters=1, seqlen=2, dataset=dataloader)
        ar.post_init()
        assert ar.bits == 4

    def test_w2a16_rtn(self, tiny_opt_model_path, dataloader):
        ar = AutoRound(tiny_opt_model_path, scheme="W2A16", nsamples=1, iters=0, seqlen=2, dataset=dataloader)
        ar.post_init()
        assert ar.bits == 2

    def test_w4a16_mixed(self, tiny_qwen_moe_model_path, dataloader):

        layer_config = {
            "model.layers.0.self_attn.k_proj": {"bits": 16},
        }
        ar = AutoRound(
            tiny_qwen_moe_model_path,
            scheme="W4A16_MIXED",
            nsamples=1,
            iters=0,
            seqlen=2,
            dataset=dataloader,
            low_cpu_mem_usage=False,
            layer_config=layer_config,
        )
        _, quantized_model_path = ar.quantize_and_save(self.save_folder)
        assert ar.bits == 4
        assert ar.model.model.layers[0].self_attn.q_proj.bits == 8
        assert ar.model.model.layers[0].self_attn.k_proj.bits == 16
        assert ar.model.model.layers[0].mlp.experts[0].up_proj.bits == 4
        # assert ar.model.model.layers[0].mlp.shared_expert.gate_proj.bits == 8 # gate has been added to ignore_layers
        model = transformers.AutoModelForCausalLM.from_pretrained(quantized_model_path, trust_remote_code=True)
        assert model is not None, "Model loading failed after quantization with W4A16_MIXED scheme on MoE"

    def test_w4a16_mixed_mllm(self, tiny_qwen_2_5_vl_model_path, dataloader):

        ar = AutoRound(
            tiny_qwen_2_5_vl_model_path,
            scheme="W4A16_MIXED",
            nsamples=1,
            batch_size=1,
            iters=0,
            seqlen=2,
            disable_opt_rtn=True,
            dataset=dataloader,
            low_cpu_mem_usage=False,
            disable_model_free=True,
        )
        _, quantized_model_path = ar.quantize_and_save(self.save_folder)
        model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(quantized_model_path)
        assert model is not None, "Model loading failed after quantization with W4A16_MIXED scheme on MLLM"
        assert ar.bits == 4
        assert ar.model.model.language_model.layers[0].self_attn.q_proj.bits == 16
        assert ar.model.model.visual.blocks[0].attn.qkv.bits == 16

    def test_mxfp4(self, tiny_opt_model_path, dataloader):
        ar = AutoRound(tiny_opt_model_path, scheme="MXFP4", nsamples=1, iters=1, seqlen=2, dataset=dataloader)
        ar.post_init()
        assert ar.bits == 4
        assert ar.act_bits == 4
        assert ar.data_type == "mx_fp"
        assert ar.act_data_type == "mx_fp"

    def test_mxfp4_rceil(self, tiny_opt_model_path):
        ar = AutoRound(tiny_opt_model_path, scheme="MXFP4_RCEIL", nsamples=1, iters=1)
        ar.post_init()
        assert ar.bits == 4
        assert ar.act_bits == 4
        assert ar.data_type == "mx_fp"
        assert ar.act_data_type == "mx_fp_rceil"
        _, quantized_model_path = ar.quantize_and_save()
        model = transformers.AutoModelForCausalLM.from_pretrained(quantized_model_path, trust_remote_code=True)
        assert model is not None, "Model loading failed after quantization with MXFP4 scheme"

    def test_vlm(self, tiny_qwen_vl_model_path):
        from auto_round import AutoRound

        ar = AutoRound(tiny_qwen_vl_model_path, scheme="W2A16", nsamples=1, iters=1, seqlen=2)
        ar.post_init()
        assert ar.bits == 2
        assert ar.act_bits == 16

    def test_nvfp4(self, tiny_opt_model_path, dataloader):
        ar = AutoRound(tiny_opt_model_path, scheme="NVFP4", nsamples=1, iters=1, seqlen=2, dataset=dataloader)
        ar.post_init()
        assert ar.bits == 4
        assert ar.act_bits == 4
        assert ar.data_type == "nv_fp"
        assert ar.act_data_type == "nv_fp4_with_static_gs"
        _, quantized_model_path = ar.quantize_and_save(self.save_folder)
        model = transformers.AutoModelForCausalLM.from_pretrained(quantized_model_path, trust_remote_code=True)
        assert model is not None, "Model loading failed after quantization with NVFP4 scheme"

    @pytest.mark.parametrize(
        "scheme", ["W8A16", "MXFP8", "FPW8A16", "FP8_BLOCK", "FP8_STATIC", "GGUF:Q2_K_S", "GGUF:Q4_K_M"]
    )
    def test_all_scheme(self, scheme, tiny_qwen_model_path, dataloader):
        model_name = tiny_qwen_model_path
        print(f"scheme={scheme}")
        ar = AutoRound(model_name, scheme=scheme, nsamples=1, iters=1, seqlen=2, dataset=dataloader)
        ar.quantize_and_save(self.save_folder)

    def test_scheme_in_layer_config(self, dataloader):
        model = get_tiny_model(opt_name_or_path, num_layers=5)
        tokenizer = transformers.AutoTokenizer.from_pretrained(opt_name_or_path, trust_remote_code=True)
        layer_config = {
            "model.decoder.layers.2.self_attn": {"bits": 2},
            "model.decoder.layers.3.self_attn.v_proj": "W8A16",
            "model.decoder.layers.4.self_attn.k_proj": QuantizationScheme.from_dict({"group_size": 64}),
        }
        ar = AutoRound(
            model,
            tokenizer,
            scheme="W3A16",
            nsamples=1,
            iters=1,
            layer_config=layer_config,
            seqlen=2,
            dataset=dataloader,
        )

        ar.quantize_and_save(self.save_folder)
        for n, m in ar.model.named_modules():
            if n == "model.decoder.layers.2.self_attn.q_proj":
                assert m.bits == 2
            if n == "model.decoder.layers.2.self_attn.k_proj":
                assert m.bits == 2
            if n == "model.decoder.layers.3.self_attn.v_proj":
                assert m.bits == 8
            if n == "model.decoder.layers.4.self_attn.k_proj":
                assert m.group_size == 64

    def test_parse_available_devices(self):
        from auto_round.utils.device import parse_available_devices

        device_list = parse_available_devices("auto")
        assert len(device_list) == 1 and "cpu" in device_list
        device_list = parse_available_devices("a:cuda:0,b:cuda:1,c:cpu")
        assert len(device_list) == 3
        assert device_list == ["cuda:0", "cuda:1", "cpu"]
        device_list = parse_available_devices("0,1")
        assert len(device_list) == 1 and "cpu" in device_list

    def test_set_scheme(self, tiny_qwen_model_path):
        ar = AutoRound(
            tiny_qwen_model_path,
            scheme="gguf:q2_k_s",
            data_type="fp",
            nsamples=1,
            disable_opt_rtn=True,
            iters=0,
            seqlen=2,
        )
        ar.quantize_and_save(self.save_folder)

        from auto_round.schemes import QuantizationScheme

        qs = QuantizationScheme.from_dict({"bits": 4, "group_size": 64})
        ar = AutoRound(
            tiny_qwen_model_path,
            scheme=qs,
            bits=2,
            data_type="int_asym_dq",
            nsamples=1,
            iters=0,
            disable_opt_rtn=True,
            seqlen=2,
        )
        ar.quantize_and_save(self.save_folder)

    def test_fp8_static(self, tiny_opt_model_path):
        ar = AutoRound(tiny_opt_model_path, scheme="FP8_STATIC", nsamples=1, iters=1)
        ar.post_init()
        assert ar.bits == 8
        assert ar.act_bits == 8
        assert ar.data_type == "fp"
        assert ar.act_data_type == "fp"
        assert ar.group_size == -1
        assert ar.act_dynamic is False
        _, quantized_model_path = ar.quantize_and_save()
        model = transformers.AutoModelForCausalLM.from_pretrained(quantized_model_path, trust_remote_code=True)
        assert model is not None, "Model loading failed after quantization with FP8_STATIC scheme"

    def test_fp8_static_rtn(self, tiny_opt_model_path):
        ar = AutoRound(tiny_opt_model_path, scheme="FP8_STATIC", nsamples=1, iters=0, disable_opt_rtn=True)
        ar.post_init()
        assert ar.bits == 8
        assert ar.act_bits == 8
        assert ar.data_type == "fp"
        assert ar.act_data_type == "fp"
        assert ar.group_size == -1
        assert ar.act_dynamic is False
        _, quantized_model_path = ar.quantize_and_save(self.save_folder)
        model = transformers.AutoModelForCausalLM.from_pretrained(quantized_model_path, trust_remote_code=True)
        assert model is not None, "Model loading failed after quantization with FP8_STATIC scheme"


class TestQ2KMixedRecipe:
    """Unit tests for the gguf:q2_k_mixed _handle_special_schemes recipe.

    Uses a small dummy model so no real checkpoint is required.
    """

    class _DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(8, 8)
            self.down_proj = nn.Linear(8, 8)
            self.gate_proj = nn.Linear(8, 8)

    class _DummyMoE(nn.Module):
        def __init__(self):
            super().__init__()
            # singular form used by Qwen2MoE / Qwen3.6 / Qwen3_5MoE
            self.shared_expert = TestQ2KMixedRecipe._DummyMLP()
            self.shared_expert_gate = nn.Linear(8, 1)
            # plural form used by DeepSeek / Mistral-Large
            self.shared_experts = TestQ2KMixedRecipe._DummyMLP()
            # regular routed experts (should NOT be promoted)
            self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])
            # fused expert weights stay 3D and should remain q2k in GGUF mapping
            self.gate_up_proj = nn.Linear(8, 16, bias=False)
            self.gate_up_proj.weight = nn.Parameter(torch.zeros(2, 16, 8))
            self.down_proj = nn.Linear(8, 8, bias=False)
            self.down_proj.weight = nn.Parameter(torch.zeros(2, 8, 8))

    class _DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = nn.Module()
            self.layer0.mlp = TestQ2KMixedRecipe._DummyMoE()
            # lm_head is Linear but should get Q8_0, not Q4_K_S
            self.lm_head = nn.Linear(8, 8)
            # embedding should get Q8_0
            self.embed_tokens = nn.Embedding(16, 8)

    def setup_method(self):
        self.model = self._DummyModel()
        self.cfg = _handle_special_schemes("gguf:q2_k_mixed", {}, self.model)

    def test_singular_shared_expert_linear_gets_q4ks(self):
        """2D shared expert linears should use Q4_K_S."""
        for sub in ("up_proj", "down_proj", "gate_proj"):
            key = f"layer0.mlp.shared_expert.{sub}"
            assert key in self.cfg, f"{key} missing from layer_config"
            assert self.cfg[key] == "GGUF:Q4_K_S", f"{key} expected Q4_K_S, got {self.cfg[key]}"

    def test_shared_expert_gate_gets_q4ks(self):
        """2D shared expert gate should use Q4_K_S."""
        key = "layer0.mlp.shared_expert_gate"
        assert key in self.cfg, f"{key} missing from layer_config"
        assert self.cfg[key] == "GGUF:Q4_K_S", f"{key} expected Q4_K_S, got {self.cfg[key]}"

    def test_plural_shared_experts_linear_gets_q4ks(self):
        """2D plural shared experts should also use Q4_K_S."""
        for sub in ("up_proj", "down_proj", "gate_proj"):
            key = f"layer0.mlp.shared_experts.{sub}"
            assert key in self.cfg, f"{key} missing from layer_config"
            assert self.cfg[key] == "GGUF:Q4_K_S", f"{key} expected Q4_K_S, got {self.cfg[key]}"

    def test_routed_experts_linears_not_in_recipe(self):
        """Routed expert linears are excluded from the recipe — falls through to dtype selector."""
        for i in range(2):
            key = f"layer0.mlp.experts.{i}"
            assert key not in self.cfg, f"{key} should not be in the recipe"

    def test_fused_three_dim_expert_weights_start_as_q2ks_recipe(self):
        """_handle_special_schemes sets 3D expert weights directly to Q2_K_S."""
        assert self.cfg.get("layer0.mlp.gate_up_proj") == "GGUF:Q2_K_S"
        assert self.cfg.get("layer0.mlp.down_proj") == "GGUF:Q2_K_S"

    def test_lm_head_gets_q8_0(self):
        """lm_head should be Q8_0."""
        assert self.cfg.get("lm_head") == "GGUF:Q8_0"

    def test_embedding_gets_q8_0(self):
        """embed_tokens embedding should be Q8_0."""
        assert self.cfg.get("embed_tokens") == "GGUF:Q8_0"
