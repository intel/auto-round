# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

"""Unified MLX format tests for AutoRound on Qwen3-0.6B (RTN mode).

Two export formats are covered:

  * ``mlx``        — native MLX checkpoint, loaded and run through ``mlx_lm``
                     on Apple Silicon.
  * ``auto_round`` — GPTQ-style packing. On Darwin the loader post-init
                     re-packs each layer into :class:`QuantLinearMLX` via
                     :meth:`QuantLinearMLX.from_gptq`; inference uses the
                     HuggingFace ``transformers`` + AutoRound pipeline.

Each format is exercised over different bit-widths (2/3/4/8) and a mixed-bit
configuration via ``layer_config``. All runs use RTN (``iters=0``) for speed.
Platform/runtime-specific inference tests are skipped automatically on
non-Darwin platforms or when the required packages are missing.
"""

import importlib.util
import json
import os
import platform
import shutil
from pathlib import Path

import pytest
import torch

from auto_round import AutoRound

from ...helpers import qwen_name_or_path


MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None
MLX_LM_AVAILABLE = importlib.util.find_spec("mlx_lm") is not None
IS_DARWIN = platform.system() == "Darwin"

# Native mlx checkpoint needs Darwin + mlx_lm to actually load & run.
requires_mlx_lm_runtime = pytest.mark.skipif(
    not (IS_DARWIN and MLX_LM_AVAILABLE),
    reason="Native MLX inference requires macOS (Darwin) with 'mlx_lm' installed.",
)
# auto_round GPTQ -> QuantLinearMLX post-init needs Darwin + mlx.
requires_mlx_runtime = pytest.mark.skipif(
    not (IS_DARWIN and MLX_AVAILABLE),
    reason="QuantLinearMLX post-init requires macOS (Darwin) with 'mlx' installed.",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _quantize_and_save(
    output_dir: str,
    scheme: str,
    bits: int,
    fmt: str,
    sym: bool = True,
    group_size: int = 128,
    layer_config: dict = None,
):
    """Run AutoRound RTN quantization on Qwen3-0.6B and export to ``fmt``."""
    ar = AutoRound(
        model=qwen_name_or_path,
        scheme=scheme,
        bits=bits,
        group_size=group_size,
        sym=sym,
        iters=0,  # RTN
        nsamples=1,
        seqlen=32,
        disable_opt_rtn=True,
        layer_config=layer_config,
    )
    ar.quantize_and_save(output_dir=output_dir, format=fmt)


def _read_quant_config(output_dir: str) -> dict:
    """Return the quantization_config dict from the exported model."""
    config_path = Path(output_dir) / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "quantization_config" in cfg:
            return cfg["quantization_config"]
    standalone = Path(output_dir) / "quantization_config.json"
    if standalone.exists():
        with open(standalone, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _count_mlx_layers(model: torch.nn.Module) -> int:
    """Count QuantLinearMLX instances in a loaded HF model."""
    from auto_round_extension.mlx.qlinear_mlx import QuantLinearMLX

    return sum(1 for m in model.modules() if isinstance(m, QuantLinearMLX))


def _mlx_lm_generate(output_dir: str, prompt: str = "The capital of France is", max_tokens: int = 8) -> str:
    """Load a native MLX checkpoint through mlx_lm and generate text."""
    from mlx_lm import generate, load

    model, tokenizer = load(output_dir)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


def _hf_generate(output_dir: str, prompt: str = "The capital of France is", max_new_tokens: int = 5):
    """Load an auto_round checkpoint through HF transformers and generate text.

    Returns:
        Tuple of (model, decoded_text).
    """
    from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
        quantization_config=AutoRoundConfig(),
    )
    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return model, tokenizer.decode(out[0], skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestMLXFormat:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "mlx_saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("runs", ignore_errors=True)

    # ---- 1) Native MLX export: sweep bits ------------------------------- #
    @pytest.mark.parametrize(
        "bits, scheme",
        [
            (2, "W2A16"),
            (3, "W3A16"),
            (4, "W4A16"),
            (8, "W8A16"),
        ],
    )
    def test_mlx_native_export_bits(self, bits, scheme):
        """Export Qwen3-0.6B as a native MLX checkpoint at 2/3/4/8 bits."""
        _quantize_and_save(
            output_dir=self.save_dir,
            scheme=scheme,
            bits=bits,
            fmt="mlx",
            sym=True,
            group_size=128,
        )
        assert os.path.exists(os.path.join(self.save_dir, "config.json"))
        qcfg = _read_quant_config(self.save_dir)
        assert qcfg, f"No quantization_config found for scheme={scheme}"
        assert qcfg.get("bits") == bits, f"expected bits={bits}, got {qcfg.get('bits')}"

    @requires_mlx_lm_runtime
    @pytest.mark.parametrize(
        "bits, scheme",
        [
            (2, "W2A16"),
            (3, "W3A16"),
            (4, "W4A16"),
            (8, "W8A16"),
        ],
    )
    def test_mlx_native_inference_with_mlx_lm(self, bits, scheme):
        """Native MLX checkpoint loads and runs end-to-end through mlx_lm."""
        _quantize_and_save(
            output_dir=self.save_dir,
            scheme=scheme,
            bits=bits,
            fmt="mlx",
            sym=True,
            group_size=128,
        )
        text = _mlx_lm_generate(self.save_dir)
        assert isinstance(text, str) and len(text) > 0, f"mlx_lm generated empty output for bits={bits}"

    # ---- 2) auto_round export: sweep bits + sym/asym -------------------- #
    @pytest.mark.parametrize(
        "bits, scheme, sym",
        [
            (2, "W2A16", True),
            (3, "W3A16", True),
            (4, "W4A16", True),
            (4, "W4A16", False),  # asymmetric -> exercises qzeros path
            (8, "W8A16", True),
        ],
    )
    def test_auto_round_export_bits(self, bits, scheme, sym):
        """Export Qwen3-0.6B as auto_round (GPTQ-style) across bits/sym."""
        _quantize_and_save(
            output_dir=self.save_dir,
            scheme=scheme,
            bits=bits,
            fmt="auto_round",
            sym=sym,
            group_size=128,
        )
        assert os.path.exists(os.path.join(self.save_dir, "config.json"))
        qcfg = _read_quant_config(self.save_dir)
        assert qcfg.get("bits") == bits
        assert qcfg.get("sym") == sym

    @requires_mlx_runtime
    @pytest.mark.parametrize(
        "bits, scheme, sym",
        [
            (4, "W4A16", True),
            (4, "W4A16", False),
            (8, "W8A16", True),
        ],
    )
    def test_auto_round_post_init_to_mlx(self, bits, scheme, sym):
        """On Darwin, auto_round layers must be post-init converted to QuantLinearMLX."""
        _quantize_and_save(
            output_dir=self.save_dir,
            scheme=scheme,
            bits=bits,
            fmt="auto_round",
            sym=sym,
            group_size=128,
        )
        model, text = _hf_generate(self.save_dir)
        assert _count_mlx_layers(model) > 0, (
            f"Expected auto_round->MLX post-init conversion to produce QuantLinearMLX "
            f"layers on Darwin, but found none (bits={bits}, sym={sym})."
        )
        assert isinstance(text, str) and len(text) > 0

    # ---- 3) Mixed-bit via layer_config ---------------------------------- #
    def test_mixed_bits_mlx_export(self):
        """Mixed-bit layer_config exported in native MLX format."""
        layer_config = {
            "lm_head": {"bits": 16},
            "model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 128, "sym": True},
            "model.layers.0.self_attn.k_proj": {"bits": 8, "group_size": 128, "sym": True},
            "model.layers.0.mlp.down_proj": {"bits": 2, "group_size": 128, "sym": True},
        }
        _quantize_and_save(
            output_dir=self.save_dir,
            scheme="W4A16",
            bits=4,
            fmt="mlx",
            sym=True,
            group_size=128,
            layer_config=layer_config,
        )
        assert os.path.exists(os.path.join(self.save_dir, "config.json"))
        qcfg = _read_quant_config(self.save_dir)
        assert qcfg.get("bits") == 4  # global default

    def test_mixed_bits_auto_round_export(self):
        """Mixed-bit layer_config exported as auto_round."""
        layer_config = {
            "lm_head": {"bits": 16},
            "model.layers.0.self_attn.v_proj": {"bits": 8, "group_size": 128, "sym": True},
            "model.layers.0.mlp.gate_proj": {"bits": 2, "group_size": 128, "sym": True},
        }
        _quantize_and_save(
            output_dir=self.save_dir,
            scheme="W4A16",
            bits=4,
            fmt="auto_round",
            sym=True,
            group_size=128,
            layer_config=layer_config,
        )
        qcfg = _read_quant_config(self.save_dir)
        assert qcfg.get("bits") == 4

    @requires_mlx_lm_runtime
    def test_mixed_bits_mlx_inference_with_mlx_lm(self):
        """Mixed-bit native MLX checkpoint runs end-to-end under mlx_lm."""
        layer_config = {
            "lm_head": {"bits": 16},
            "model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 128, "sym": True},
            "model.layers.0.mlp.down_proj": {"bits": 2, "group_size": 128, "sym": True},
        }
        _quantize_and_save(
            output_dir=self.save_dir,
            scheme="W4A16",
            bits=4,
            fmt="mlx",
            sym=True,
            group_size=128,
            layer_config=layer_config,
        )
        text = _mlx_lm_generate(self.save_dir)
        assert isinstance(text, str) and len(text) > 0

    @requires_mlx_runtime
    def test_mixed_bits_auto_round_inference_on_darwin(self):
        """Mixed-bit auto_round export, loaded via HF + post-init MLX repack."""
        layer_config = {
            "lm_head": {"bits": 16},
            "model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 128, "sym": True},
        }
        _quantize_and_save(
            output_dir=self.save_dir,
            scheme="W4A16",
            bits=4,
            fmt="auto_round",
            sym=True,
            group_size=128,
            layer_config=layer_config,
        )
        model, text = _hf_generate(self.save_dir)
        assert _count_mlx_layers(model) > 0
        assert isinstance(text, str) and len(text) > 0

