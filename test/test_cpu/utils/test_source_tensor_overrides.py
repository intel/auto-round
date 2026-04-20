"""Unit tests for ``auto_round.utils.source_tensor_overrides``.

A synthetic ``nn.Module`` with one Parameter and one buffer is paired
with a hand-built single-shard safetensors checkpoint that simulates the
on-disk layout AutoRound expects (here, the Nemotron-H ``backbone.→model.``
rename rule).  We then verify that
:func:`restore_tensors_from_source` re-binds both the parameter and the
buffer to the requested dtype while preserving values.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from auto_round.utils.source_tensor_overrides import (
    NEMOTRON_H_ROUTER_BIAS_PATTERNS,
    NEMOTRON_H_SSM_CORE_PATTERNS,
    _nemotron_h_source_to_module,
    restore_tensors_from_source,
)


class _Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        # Mimic Mamba2 SSM-core families.
        self.A_log = nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
        self.D = nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
        self.dt_bias = nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
        self.register_buffer("e_score_correction_bias", torch.zeros(4, dtype=torch.bfloat16))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = _Mixer()


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(2)])


class _Root(nn.Module):
    """Mimics ``NemotronHForCausalLM`` post-load: in-memory uses ``model.*``,
    on-disk uses ``backbone.*``."""

    def __init__(self):
        super().__init__()
        self.model = _Backbone()


def _make_source_checkpoint(tmpdir: str) -> dict[str, torch.Tensor]:
    """Write a ``model.safetensors`` shard with backbone-prefixed keys."""
    from safetensors.torch import save_file

    src_tensors: dict[str, torch.Tensor] = {}
    for layer_idx in range(2):
        prefix = f"backbone.layers.{layer_idx}.mixer"
        # Distinct, finite values that are NOT representable exactly in BF16
        # so the dtype change is observable.
        src_tensors[f"{prefix}.A_log"] = torch.tensor(
            [-0.123456789, -0.987654321, -1.111111, -2.222222, -3.333333, -4.444444, -5.555555, -6.666666],
            dtype=torch.float32,
        )
        src_tensors[f"{prefix}.D"] = torch.full((8,), 0.1234567, dtype=torch.float32)
        src_tensors[f"{prefix}.dt_bias"] = torch.full((8,), 0.7654321, dtype=torch.float32)
        src_tensors[f"{prefix}.gate.e_score_correction_bias"] = torch.tensor(
            [65.5470001, -32.1234567, 0.0001234, -0.0007654],
            dtype=torch.float32,
        )

    save_file(src_tensors, os.path.join(tmpdir, "model.safetensors"))
    return src_tensors


def test_restore_ssm_core_to_fp32():
    with tempfile.TemporaryDirectory() as tmpdir:
        src_tensors = _make_source_checkpoint(tmpdir)
        model = _Root()

        # Sanity: pre-restore everything is BF16 zeros.
        for layer in model.model.layers:
            assert layer.mixer.A_log.dtype is torch.bfloat16
            assert torch.all(layer.mixer.A_log == 0)

        restored = restore_tensors_from_source(
            model,
            tmpdir,
            NEMOTRON_H_SSM_CORE_PATTERNS,
            torch.float32,
            source_to_module=_nemotron_h_source_to_module,
        )

        # Three parameter families (A_log, D, dt_bias) × 2 layers = 6.
        # conv1d.weight is intentionally excluded from SSM-core patterns —
        # see source_tensor_overrides.NEMOTRON_H_SSM_CORE_PATTERNS docstring.
        assert len(restored) == 6, f"unexpected restored set: {restored}"

        for layer_idx, layer in enumerate(model.model.layers):
            for attr in ("A_log", "D", "dt_bias"):
                got = getattr(layer.mixer, attr)
                src_key = f"backbone.layers.{layer_idx}.mixer.{attr}"
                expected = src_tensors[src_key]
                assert got.dtype is torch.float32, f"{attr} not upcast"
                assert torch.equal(got, expected), f"{attr} value mismatch"
                # Old dtype recorded as BF16, new as FP32.
                key = f"model.layers.{layer_idx}.mixer.{attr}"
                old, new = restored[key]
                assert old is torch.bfloat16
                assert new is torch.float32


def test_restore_skips_when_layout_diverges():
    """When the source key points to a parent module that doesn't exist
    in-memory, the restore must skip gracefully (not raise, not corrupt).

    The synthetic ``_Mixer`` puts ``e_score_correction_bias`` directly on
    the mixer, but the Nemotron-H source layout puts it under
    ``mixer.gate.*``.  Verify the resolver bails cleanly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_source_checkpoint(tmpdir)
        model = _Root()
        original_buffer = model.model.layers[0].mixer.e_score_correction_bias.clone()

        restored = restore_tensors_from_source(
            model,
            tmpdir,
            NEMOTRON_H_ROUTER_BIAS_PATTERNS,
            torch.float32,
            source_to_module=_nemotron_h_source_to_module,
        )
        assert restored == {}
        # In-memory buffer untouched, still BF16 zeros.
        assert torch.equal(model.model.layers[0].mixer.e_score_correction_bias, original_buffer)
        assert model.model.layers[0].mixer.e_score_correction_bias.dtype is torch.bfloat16


def test_restore_router_bias_with_gate_submodule():
    """End-to-end buffer restore using a layout that DOES have ``mixer.gate``."""
    from safetensors.torch import save_file

    class GatedMixer(nn.Module):
        def __init__(self):
            super().__init__()

            class _Gate(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("e_score_correction_bias", torch.zeros(4, dtype=torch.bfloat16))

            self.gate = _Gate()

    class GatedRoot(nn.Module):
        def __init__(self):
            super().__init__()

            class _Backbone(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.ModuleList([type("L", (nn.Module,), {})() for _ in range(2)])
                    for layer in self.layers:
                        layer.mixer = GatedMixer()

            self.model = _Backbone()

    with tempfile.TemporaryDirectory() as tmpdir:
        src = {
            f"backbone.layers.{i}.mixer.gate.e_score_correction_bias": torch.tensor(
                [65.547, -32.123, 0.001, -0.0007], dtype=torch.float32
            )
            for i in range(2)
        }
        save_file(src, os.path.join(tmpdir, "model.safetensors"))

        model = GatedRoot()
        restored = restore_tensors_from_source(
            model,
            tmpdir,
            NEMOTRON_H_ROUTER_BIAS_PATTERNS,
            torch.float32,
            source_to_module=_nemotron_h_source_to_module,
        )
        assert len(restored) == 2
        for i, layer in enumerate(model.model.layers):
            buf = layer.mixer.gate.e_score_correction_bias
            assert buf.dtype is torch.float32
            assert torch.equal(buf, src[f"backbone.layers.{i}.mixer.gate.e_score_correction_bias"])
            # Buffer must still be tracked as a buffer.
            assert "e_score_correction_bias" in dict(layer.mixer.gate.named_buffers())


def test_unknown_source_dir_returns_empty():
    model = _Root()
    out = restore_tensors_from_source(
        model,
        "/nonexistent/path",
        [r"\.A_log$"],
        torch.float32,
    )
    assert out == {}


def test_no_pattern_match_logs_and_returns_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_source_checkpoint(tmpdir)
        model = _Root()
        out = restore_tensors_from_source(
            model,
            tmpdir,
            [r"\.does_not_exist$"],
            torch.float32,
        )
        assert out == {}


def test_shape_mismatch_skipped():
    """If the in-memory tensor shape disagrees with the source tensor, skip
    rather than corrupt the model parameter."""
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmpdir:
        # Source tensor with WRONG shape for layer 0 (size 16 vs model's 8).
        src = {"backbone.layers.0.mixer.A_log": torch.zeros(16, dtype=torch.float32)}
        save_file(src, os.path.join(tmpdir, "model.safetensors"))

        model = _Root()
        original = model.model.layers[0].mixer.A_log.clone()
        out = restore_tensors_from_source(
            model,
            tmpdir,
            NEMOTRON_H_SSM_CORE_PATTERNS,
            torch.float32,
            source_to_module=_nemotron_h_source_to_module,
        )
        assert out == {}
        # Parameter untouched.
        assert torch.equal(model.model.layers[0].mixer.A_log, original)
        assert model.model.layers[0].mixer.A_log.dtype is torch.bfloat16


def test_nemotron_h_source_to_module_renames():
    assert _nemotron_h_source_to_module("backbone.layers.0.mixer.A_log") == "model.layers.0.mixer.A_log"
    assert _nemotron_h_source_to_module("backbone.embedding.weight") == "model.embeddings.weight"
    # Non-matching keys pass through untouched.
    assert _nemotron_h_source_to_module("lm_head.weight") == "lm_head.weight"
