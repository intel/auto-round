# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU CI tests for AWQ (Activation-Aware Weight Quantization) algorithm.

Covers:
- Normal LLM (OPT-125m): W4A16 quantization, inference, export args verification
- INT W8A8 export via llm_compressor format with config validation
- Tiny Qwen MoE: dynamic smoothing, quantized layer checks, quant config saving
"""

import json
import os
import shutil

import pytest
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ...helpers import generate_prompt, get_model_path, opt_name_or_path, save_tiny_model


class TestAWQNormalLLM:
    """AWQ quantization on a normal LLM (OPT-125m style tiny model).

    Tests W4A16 quantize → inference → export args validation.
    """

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_awq_w4a16_quantize_and_inference(self, tiny_opt_model_path):
        """W4A16 AWQ quantization produces valid layer_config and model can generate."""
        ar = AutoRound(
            tiny_opt_model_path,
            scheme="W4A16",
            algorithm="awq",
            nsamples=2,
            seqlen=32,
            batch_size=2,
        )
        model, layer_config = ar.quantize()

        # Verify quantization happened
        assert model is not None
        assert len(layer_config) > 0
        for name, cfg in layer_config.items():
            assert cfg["bits"] == 4, f"Layer {name} expected bits=4, got {cfg['bits']}"

        # Inference smoke test
        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path)
        output = generate_prompt(model, tokenizer, device="cpu")
        assert len(output) > 0, "Model should produce non-empty output"

    def test_awq_w4a16_export_auto_round_format(self, tiny_opt_model_path):
        """AWQ W4A16 export to auto_round format: verify quantization_config in saved config."""
        ar = AutoRound(
            tiny_opt_model_path,
            scheme="W4A16",
            algorithm="awq",
            nsamples=2,
            seqlen=32,
            batch_size=2,
        )
        _, save_path = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        # Verify config.json has quantization_config
        config = AutoConfig.from_pretrained(save_path)
        qconfig = config.quantization_config
        assert qconfig is not None, "quantization_config should be present"
        assert qconfig["bits"] == 4
        assert qconfig["group_size"] == 128
        assert "auto-round" in qconfig["quant_method"]

    def test_awq_w4a16_export_args_check(self, tiny_opt_model_path):
        """Verify key export args in saved quantization_config match input parameters."""
        bits, group_size, sym = 8, 64, True
        ar = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            algorithm="awq",
            nsamples=2,
            seqlen=8,
            batch_size=2,
        )
        _, save_path = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        config = AutoConfig.from_pretrained(save_path)
        qconfig = config.quantization_config
        assert qconfig["bits"] == bits
        assert qconfig["group_size"] == group_size
        assert qconfig["sym"] == sym


class TestAWQW8A8LLMCompressor:
    """AWQ INT W8A8 quantization with llm_compressor export format.

    Validates compressed-tensors config structure, num_bits, and key fields.
    """

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_awq_w8a8_llmc_export(self, tiny_opt_model_path):
        """W8A8 AWQ → llm_compressor: verify compressed-tensors metadata fields."""
        ar = AutoRound(
            tiny_opt_model_path,
            scheme="INT8",
            algorithm="awq",
            nsamples=2,
            seqlen=32,
            batch_size=2,
        )
        _, save_path = ar.quantize_and_save(output_dir=self.save_dir, format="llm_compressor")

        config = AutoConfig.from_pretrained(save_path, trust_remote_code=True)
        qconfig = config.quantization_config

        assert qconfig["quant_method"] == "compressed-tensors"

        group0 = qconfig["config_groups"]["group_0"]
        # Weight args
        assert group0["weights"]["num_bits"] == 8
        assert group0["weights"]["type"] == "int"
        assert group0["weights"]["symmetric"] is True
        # Activation args
        assert group0["input_activations"]["num_bits"] == 8
        # Targets
        targets = group0.get("targets")
        assert targets is not None and len(targets) > 0

        # QuantLinear check: verify saved weights are int8 with per-channel scales
        from safetensors import safe_open

        st_files = [f for f in os.listdir(save_path) if f.endswith(".safetensors")]
        assert len(st_files) > 0, f"No safetensors files in {save_path}"
        with safe_open(os.path.join(save_path, st_files[0]), framework="pt") as f:
            weight = f.get_tensor("model.decoder.layers.0.self_attn.k_proj.weight")
            assert weight.dtype == torch.int8, f"Expected int8 weight, got {weight.dtype}"
            scale = f.get_tensor("model.decoder.layers.0.self_attn.k_proj.weight_scale")
            assert scale.shape[1] == 1, f"Expected per-channel scale shape (out, 1), got {scale.shape}"


class TestAWQMoE:
    """AWQ on a tiny MoE model: dynamic smoothing, layer checks, config saving.

    Uses the session-scoped tiny_qwen_moe_model_path fixture.
    """

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_awq_moe_dynamic_smoothing(self, tiny_qwen_moe_model_path):
        """AWQ dynamic smoothing should resolve mappings on a MoE model without error."""
        from auto_round.algorithms.quantization.awq.mappings import resolve_mappings

        model = AutoModelForCausalLM.from_pretrained(
            tiny_qwen_moe_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        resolved = resolve_mappings(model, user_mappings=None)

        # Must resolve a non-trivial number of mappings
        assert len(resolved) > 0, "Expected non-empty resolved mappings"

        # Verify smooth layers are unique (no duplicate smoothing targets)
        smooth_names = [r.smooth_name for r in resolved]
        assert len(smooth_names) == len(
            set(smooth_names)
        ), f"Duplicate smooth names: {[n for n in smooth_names if smooth_names.count(n) > 1]}"

        # Must have attention-related mappings (input_layernorm→qkv, v_proj→o_proj)
        n_layers = model.config.num_hidden_layers
        attn_smooths = [n for n in smooth_names if "input_layernorm" in n or "self_attn.v_proj" in n]
        assert len(attn_smooths) == 2 * n_layers, (
            f"Expected {2 * n_layers} attn smooth layers, got {len(attn_smooths)}"
        )

        # Shared expert up→down should resolve at block level
        if hasattr(model.model.layers[0].mlp, "shared_expert"):
            shared_smooths = [n for n in smooth_names if "shared_expert" in n]
            assert len(shared_smooths) == n_layers, (
                f"Expected {n_layers} shared_expert smooth layers, got {len(shared_smooths)}"
            )

        del model

    def test_awq_moe_quantized_layers_check(self, tiny_qwen_moe_model_path):
        """AWQ on MoE: expert layers should be quantized, gates/routers stay FP."""
        ar = AutoRound(
            tiny_qwen_moe_model_path,
            scheme="W4A16",
            algorithm="awq",
            nsamples=2,
            seqlen=32,
            batch_size=2,
        )
        model, layer_config = ar.quantize()
        assert model is not None
        assert len(layer_config) > 0

        q4_layers = {k for k, v in layer_config.items() if v["bits"] == 4}
        fp_layers = {k for k, v in layer_config.items() if v["bits"] >= 16}
        other_layers = {k: v["bits"] for k, v in layer_config.items() if v["bits"] != 4 and v["bits"] < 16}

        # Tiny Qwen MoE: gate/router layers stay FP, all others are W4
        assert len(other_layers) == 0, f"Unexpected bit widths: {other_layers}"
        n_layers = model.config.num_hidden_layers
        assert len(fp_layers) == n_layers, (
            f"Expected {n_layers} FP shared_expert_gate layers, got {len(fp_layers)}: {sorted(fp_layers)}"
        )
        assert len(q4_layers) == len(layer_config) - len(fp_layers), (
            f"Expected {len(layer_config) - len(fp_layers)} W4 layers, got {len(q4_layers)}"
        )

        for name in fp_layers:
            assert "shared_expert_gate" in name, f"Unexpected FP layer: {name}"

    def test_awq_moe_save_quant_config(self, tiny_qwen_moe_model_path):
        """AWQ MoE: saved quantization_config should be consistent and loadable."""
        ar = AutoRound(
            tiny_qwen_moe_model_path,
            scheme="W4A16",
            algorithm="awq",
            nsamples=2,
            seqlen=32,
            batch_size=2,
        )
        _, save_path = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        # Load and verify config
        config_path = os.path.join(save_path, "config.json")
        assert os.path.exists(config_path), f"config.json not found at {save_path}"

        with open(config_path, "r") as f:
            config_data = json.load(f)

        qconfig = config_data.get("quantization_config")
        assert qconfig is not None, "quantization_config missing from saved config.json"
        assert qconfig["bits"] == 4
        assert qconfig["group_size"] == 128
        assert "auto-round" in qconfig["quant_method"]

    def test_awq_moe_save_compressed_size(self, tiny_qwen_moe_model_path):
        """AWQ MoE W4: quantized safetensors should be smaller than original."""
        ar = AutoRound(
            tiny_qwen_moe_model_path,
            scheme="W4A16",
            algorithm="awq",
            nsamples=2,
            seqlen=32,
            batch_size=2,
        )
        _, save_path = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        def _safetensors_size(path):
            return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".safetensors"))

        original_size = _safetensors_size(tiny_qwen_moe_model_path)
        quantized_size = _safetensors_size(save_path)
        assert quantized_size > 0, f"No safetensors files in {save_path}"

        ratio = quantized_size / original_size
        # W4G128 packing: (4-bit weights + scale/zp overhead)
        assert 0.35 < ratio < 0.65, (
            f"Compression ratio {ratio:.4f} outside expected range (0.35, 0.65): "
            f"original={original_size / (1024**2):.1f}MB, quantized={quantized_size / (1024**2):.1f}MB"
        )
