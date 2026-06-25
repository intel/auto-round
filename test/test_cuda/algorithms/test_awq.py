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

"""CUDA CI tests for AWQ (Activation-Aware Weight Quantization) algorithm.

Covers:
- Normal LLM (Like OPT-125m): W4A16 quantization, inference, export check
- INT W8A8: llm_compressor export args, vLLM inference
- Tiny Qwen MoE: dynamic smoothing, quantized layer checks, quant config saving
- Accuracy: lm_eval on OPT-125m AWQ W4A16 (lambada_openai, limit=50)
"""

import json
import os
import shutil

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ...helpers import eval_generated_prompt, evaluate_accuracy, generate_prompt, get_model_path, opt_name_or_path

# ---------------------------------------------------------------------------
# Section 1: Normal LLM (OPT-125m) – W4A16 quantize, inference, export args
# ---------------------------------------------------------------------------


class TestAWQLLM:
    """AWQ W4A16 on a tiny OPT model with CUDA acceleration."""

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_awq_w4a16_quantize_and_inference(self, tiny_opt_model_path):
        """W4A16 AWQ quantization and CUDA inference smoke test."""
        ar = AutoRound(
            tiny_opt_model_path,
            scheme="W4A16",
            algorithm="awq",
            n_grid=4,
            nsamples=2,
            seqlen=32,
            batch_size=2,
        )
        model, layer_config = ar.quantize()

        assert model is not None
        assert len(layer_config) > 0
        for name, cfg in layer_config.items():
            assert cfg["bits"] == 4, f"Layer {name} expected bits=4, got {cfg['bits']}"

        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path)
        output = generate_prompt(model, tokenizer)
        assert len(output) > 0 and "!!!" not in output, f"Unexpected generation output: {output}"

    def test_awq_w4a16_export_auto_round_args(self, tiny_opt_model_path):
        """Export to auto_round format: verify quantization_config fields."""
        bits, group_size, sym = 8, 64, True
        ar = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            algorithm="awq",
            n_grid=4,
            nsamples=2,
            seqlen=32,
            batch_size=2,
        )
        _, save_path = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        config = AutoConfig.from_pretrained(save_path)
        qconfig = config.quantization_config
        assert qconfig is not None
        assert qconfig["bits"] == bits
        assert qconfig["group_size"] == group_size
        assert qconfig["sym"] == sym
        assert "auto-round" in qconfig["quant_method"]

    def test_awq_w4a16_load_and_generate(self):
        """Quantize, save, reload, and generate on CUDA to verify round-trip."""
        model_name = get_model_path("facebook/opt-125m")
        ar = AutoRound(
            model_name,
            scheme="W4A16",
            algorithm="awq",
            n_grid=4,
            nsamples=2,
            seqlen=32,
        )
        _, quantized_model_path = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")
        eval_generated_prompt(quantized_model_path)


# ---------------------------------------------------------------------------
# Section 2: Tiny Qwen MoE – dynamic smoothing, quantized layers, config
# ---------------------------------------------------------------------------


class TestAWQMoE:
    """AWQ on a tiny MoE model with CUDA: smoothing, layers, config saving."""

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_awq_moe_dynamic_smoothing(self, tiny_qwen_moe_model_path):
        """AWQ mapping resolution works on MoE model."""
        from auto_round.algorithms.transforms.awq.mappings import resolve_mappings

        model = AutoModelForCausalLM.from_pretrained(
            tiny_qwen_moe_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        resolved = resolve_mappings(model, user_mappings=None)

        assert len(resolved) > 0, "Expected non-empty resolved mappings"

        # Smooth layers should be unique
        smooth_names = [r.smooth_name for r in resolved]
        assert len(smooth_names) == len(
            set(smooth_names)
        ), f"Duplicate smooth names: {[n for n in smooth_names if smooth_names.count(n) > 1]}"

        n_layers = model.config.num_hidden_layers
        attn_smooths = [n for n in smooth_names if "input_layernorm" in n or "self_attn.v_proj" in n]
        assert len(attn_smooths) == 2 * n_layers, f"Expected {2 * n_layers} attn smooth layers, got {len(attn_smooths)}"

        if hasattr(model.model.layers[0].mlp, "shared_expert"):
            shared_smooths = [n for n in smooth_names if "shared_expert" in n]
            assert (
                len(shared_smooths) == n_layers
            ), f"Expected {n_layers} shared_expert smooth layers, got {len(shared_smooths)}"

        del model

    def test_awq_moe_quantized_layers_check(self, tiny_qwen_moe_model_path):
        """Expert layers quantized to W4, gates/routers stay fp."""
        ar = AutoRound(
            tiny_qwen_moe_model_path,
            scheme="W4A16",
            algorithm="awq",
            nsamples=2,
            n_grid=4,
            seqlen=32,
            batch_size=2,
        )
        model, layer_config = ar.quantize()
        assert model is not None
        assert len(layer_config) > 0

        # Categorize layers
        q4_layers = {k for k, v in layer_config.items() if v["bits"] == 4}
        fp_layers = {k for k, v in layer_config.items() if v["bits"] >= 16}
        other_layers = {k: v["bits"] for k, v in layer_config.items() if v["bits"] != 4 and v["bits"] < 16}

        # Tiny Qwen MoE: mlp.gate is a TopKRouter (not Linear, excluded from
        # layer_config), mlp.shared_expert_gate is Linear but excluded from
        # quantization → 1 FP gate layer per block.
        assert len(other_layers) == 0, f"Unexpected bit widths: {other_layers}"
        n_layers = model.config.num_hidden_layers
        assert len(fp_layers) == n_layers, (
            f"Expected {n_layers} FP gate layers (mlp.shared_expert_gate per block), "
            f"got {len(fp_layers)}: {sorted(fp_layers)}"
        )
        assert len(q4_layers) == len(layer_config) - len(
            fp_layers
        ), f"Expected {len(layer_config) - len(fp_layers)} W4 layers, got {len(q4_layers)}"

        for name in fp_layers:
            assert name.endswith("gate"), f"Unexpected FP layer: {name}"

    def test_awq_moe_save_compressed_size(self, tiny_qwen_moe_model_path):
        """AWQ MoE W4: quantized safetensors should be smaller than original."""
        ar = AutoRound(
            tiny_qwen_moe_model_path,
            scheme="W4A16",
            algorithm="awq",
            n_grid=4,
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
        assert 0.35 < ratio < 0.65, (
            f"Compression ratio {ratio:.4f} outside expected range (0.35, 0.65): "
            f"original={original_size / (1024**2):.1f}MB, quantized={quantized_size / (1024**2):.1f}MB"
        )


class TestAWQEval:
    """AWQ accuracy evaluation on OPT-125m using lm_eval.

    Uses the full OPT-125m model (not tiny) for meaningful accuracy check.
    lambada_openai task, add limit for CI speed.
    """

    @classmethod
    def setup_class(cls):
        model_name = opt_name_or_path
        cls.model_name = model_name
        cls.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("runs", ignore_errors=True)

    def test_awq_w4a16_lmeval(self):
        """AWQ W4A16 on OPT-125m: lambada_openai accuracy check."""
        ar = AutoRound(
            self.model_name,
            scheme="W4A16",
            algorithm="awq",
            nsamples=32,
            seqlen=32,
            batch_size=8,
        )
        model, _ = ar.quantize()

        evaluate_accuracy(
            model,
            self.tokenizer,
            task="lambada_openai",
            threshold=0.3,
            batch_size="auto:8",
            limit=50,
        )


# ---------------------------------------------------------------------------
# Section 4: use_v2_scale_search detection and init-scale dispatch
# ---------------------------------------------------------------------------


class TestAWQUseV2ScaleSearch:
    """Unit tests for AWQ's ``use_v2_scale_search`` detection and dispatch.

    The flag is True whenever the terminal block quantizer resolves to
    ``SignRoundV2Quantizer``. The per-data-type init_scale injection (int/mx/nv,
    sym only) is then handled by ``search_optimized_init_scale`` inside the QDQ
    path. Detection lives on ``QDQTool`` (accessed via ``AWQTransform._qdq_tool``)
    and must go through the pipeline registry, not an ``_alg_cls`` string
    comparison (which always evaluated False before the fix).
    """

    @staticmethod
    def _make_compressor(block_config):
        import types

        return types.SimpleNamespace(quantize_config=block_config, alg_configs=[block_config])

    @staticmethod
    def _awq_quantizer():
        from auto_round.algorithms.transforms.awq.base import AWQTransform
        from auto_round.algorithms.transforms.awq.config import AWQConfig

        return AWQTransform(AWQConfig())

    @staticmethod
    def _signroundv2_config(data_type=None):
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
        from auto_round.algorithms.registry import normalize_algorithm_config

        cfg = normalize_algorithm_config(SignRoundConfig(iters=2, enable_alg_ext=True))
        assert type(cfg).__name__ == "SignRoundV2Config"
        if data_type is not None:
            cfg.data_type = data_type
        return cfg

    def test_rtn_block_is_not_v2(self):
        """An RTN block quantizer must NOT be detected as V2."""
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        q = self._awq_quantizer()
        compressor = self._make_compressor(RTNConfig(disable_opt_rtn=True))
        assert q._qdq_tool._block_quantizer_is_signroundv2(compressor) is False

    def test_use_v2_true_for_v2_block(self):
        """Gate is True for any SignRoundV2 block (data-type gating is per-layer)."""
        q = self._awq_quantizer()
        compressor = self._make_compressor(self._signroundv2_config(data_type="mx_fp"))
        assert q._qdq_tool._block_quantizer_is_signroundv2(compressor) is True

    def test_use_v2_false_for_non_v2_block(self):
        """Gate is False when the block is not SignRoundV2, regardless of dtype."""
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        q = self._awq_quantizer()
        block = SignRoundConfig(iters=2)
        block.data_type = "mx_fp"
        compressor = self._make_compressor(block)
        assert q._qdq_tool._block_quantizer_is_signroundv2(compressor) is False

    def test_init_scale_dispatch_by_data_type(self):
        """``search_optimized_init_scale`` injects only for sym int/mx/nv."""
        import torch

        from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, search_optimized_init_scale

        for dt, gs in (("int_sym", 128), ("mx_fp4", 32), ("nv_fp4", 16)):
            weight = torch.randn(32, 128)
            weight_reshape, _, _ = reshape_pad_tensor_by_group_size(weight, gs)
            init_scale = search_optimized_init_scale(weight_reshape, dt, 4, None)
            assert init_scale is not None, dt
            assert init_scale.shape[0] == weight_reshape.shape[0]

        # asym int and *_dq are not part of the optimized init-scale path.
        assert search_optimized_init_scale(torch.randn(4, 128), "int_asym", 4, None) is None
        assert search_optimized_init_scale(torch.randn(4, 128), "int_sym_dq", 4, None) is None
