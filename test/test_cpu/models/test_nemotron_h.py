"""Tests for Nemotron-H handler: registration, post-load fix-ups,
default layer_config patterns, and source-tensor high-precision restore."""

from __future__ import annotations

import os
import tempfile
import types

import pytest
import torch
import torch.nn as nn


class _FakeZamba2RMSNormGated(nn.Module):
    pass


_FakeZamba2RMSNormGated.__name__ = "Zamba2RMSNormGated"


class _FakeNemotronHConfig(types.SimpleNamespace):
    pass


def _build_fake_nh_model(
    mamba_num_heads: int = 64,
    mamba_head_dim: int = 64,
    n_groups: int = 8,
    n_norms: int = 3,
) -> nn.Module:
    model = nn.Module()
    model.config = _FakeNemotronHConfig(
        model_type="nemotron_h",
        mamba_num_heads=mamba_num_heads,
        mamba_head_dim=mamba_head_dim,
        n_groups=n_groups,
        _name_or_path="",
    )
    for i in range(n_norms):
        setattr(model, f"norm_{i}", _FakeZamba2RMSNormGated())
    return model


# Registration tests


def test_nemotron_h_registered_in_model_config():
    from auto_round.modeling.unfused_moe import MODEL_CONFIG

    assert "nemotron_h" in MODEL_CONFIG
    cfg = MODEL_CONFIG["nemotron_h"]
    assert cfg.get("block_patch")
    assert cfg.get("dispatch_dict_patch")


def test_nemotron_h_class_is_importable():
    pytest.importorskip("transformers")
    from auto_round.modeling.unfused_moe.nemotron_h import LinearNemotronHMoE

    assert LinearNemotronHMoE is not None


def test_nemotron_h_preserves_prefix_rename_drops_expert_bundles_and_embedding_alias():
    pytest.importorskip("transformers.models.nemotron_h")
    from transformers import conversion_mapping

    from auto_round.modeling.unfused_moe import (
        MODEL_CONFIG,
        get_checkpoint_conversion_mapping_ar,
    )

    cfg = MODEL_CONFIG["nemotron_h"]
    assert cfg.get("preserve_upstream_conversion_mapping") is True
    drop_targets = cfg.get("drop_conversion_target_patterns", [])
    assert "mixer.experts.up_proj" in drop_targets
    assert "mixer.experts.down_proj" in drop_targets
    assert "embeddings.weight" in drop_targets

    upstream = conversion_mapping.get_checkpoint_conversion_mapping("nemotron_h")
    upstream_targets = {tuple(getattr(rule, "target_patterns", []) or []) for rule in upstream}
    assert ("model.",) in upstream_targets

    ar_mapping = get_checkpoint_conversion_mapping_ar("nemotron_h")
    ar_targets = {tuple(getattr(rule, "target_patterns", []) or []) for rule in ar_mapping}
    assert ("model.",) in ar_targets
    assert ("mixer.experts.up_proj",) not in ar_targets
    assert ("mixer.experts.down_proj",) not in ar_targets
    assert ("embeddings.weight",) not in ar_targets


def test_apply_model_monkey_patches_rewires_nemotron_h():
    pytest.importorskip("transformers.models.nemotron_h")
    import types

    from transformers import conversion_mapping
    from transformers.models.nemotron_h import modeling_nemotron_h

    from auto_round.modeling.unfused_moe import (
        apply_model_monkey_patches,
        get_checkpoint_conversion_mapping_ar,
    )
    from auto_round.modeling.unfused_moe.nemotron_h import LinearNemotronHMoE

    orig_class = modeling_nemotron_h.NemotronHMoE
    orig_mixer = modeling_nemotron_h.MIXER_TYPES.get("moe")
    orig_cm = conversion_mapping.get_checkpoint_conversion_mapping
    try:
        applied = apply_model_monkey_patches("nvidia/Nemotron-Cascade-2-30B-A3B", trust_remote_code=False)
        assert applied is True
        assert modeling_nemotron_h.NemotronHMoE is LinearNemotronHMoE
        assert modeling_nemotron_h.MIXER_TYPES["moe"] is LinearNemotronHMoE

        # Regression guard: transformers.from_pretrained resolves weight
        # renames via conversion_mapping.get_model_conversion_mapping, which
        # in turn looks up the module-global
        # conversion_mapping.get_checkpoint_conversion_mapping. Our patch
        # replaces exactly that symbol. If a future refactor breaks the
        # monkey-patch binding, the backbone.→model. rename is silently
        # dropped and gate.weight / e_score_correction_bias stay at
        # _init_weights defaults — the exact router-broken checkpoint that
        # motivated this guard.
        assert conversion_mapping.get_checkpoint_conversion_mapping is get_checkpoint_conversion_mapping_ar

        fake_model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type="nemotron_h"),
            __class__=type("Fake", (), {"__mro__": (object,)}),
        )
        resolved = conversion_mapping.get_model_conversion_mapping(fake_model, add_legacy=False)
        targets = {tuple(getattr(rule, "target_patterns", []) or []) for rule in resolved}
        assert ("model.",) in targets, f"backbone.→model. rename missing from resolved mapping: {targets}"
    finally:
        modeling_nemotron_h.NemotronHMoE = orig_class
        if orig_mixer is not None:
            modeling_nemotron_h.MIXER_TYPES["moe"] = orig_mixer
        conversion_mapping.get_checkpoint_conversion_mapping = orig_cm


# Post-load fix-ups + default layer_config patterns


def test_default_layer_config_patterns_returns_out_proj_overlay():
    from auto_round.modeling.unfused_moe.nemotron_h import (
        nemotron_h_default_layer_config_patterns,
    )

    patterns = nemotron_h_default_layer_config_patterns()
    assert any("mixer" in k and "out_proj" in k for k in patterns)
    out_proj_overlay = next(v for k, v in patterns.items() if "mixer" in k and "out_proj" in k)
    assert out_proj_overlay["scale_dtype"] is torch.bfloat16


def test_apply_nemotron_h_post_load_patches_zamba2_instances_and_class():
    from auto_round.modeling.unfused_moe.nemotron_h import (
        apply_nemotron_h_post_load,
    )

    model = _build_fake_nh_model(mamba_num_heads=64, mamba_head_dim=64, n_groups=8)
    summary = apply_nemotron_h_post_load(model, enable_high_precision_overrides=False)
    assert summary["zamba2_patched"] == 3
    for i in range(3):
        assert getattr(model, f"norm_{i}").group_size == 512
    assert _FakeZamba2RMSNormGated.group_size == 512


def test_apply_nemotron_h_post_load_noop_for_other_model_types():
    from auto_round.modeling.unfused_moe.nemotron_h import (
        apply_nemotron_h_post_load,
    )

    model = _build_fake_nh_model()
    model.config.model_type = "llama"
    summary = apply_nemotron_h_post_load(model, enable_high_precision_overrides=False)
    assert summary == {"zamba2_patched": 0, "high_precision_restored": 0}
    for i in range(3):
        assert "group_size" not in getattr(model, f"norm_{i}").__dict__


def test_apply_post_load_fixups_dispatches_to_model_config():
    from auto_round.modeling.unfused_moe import apply_post_load_fixups

    model = _build_fake_nh_model(mamba_num_heads=32, mamba_head_dim=128, n_groups=8)
    summary = apply_post_load_fixups(model, enable_high_precision_overrides=False)
    assert summary.get("zamba2_patched") == 3
    assert model.norm_0.group_size == 512


def test_apply_post_load_fixups_no_op_without_config():
    from auto_round.modeling.unfused_moe import apply_post_load_fixups

    assert apply_post_load_fixups(nn.Linear(4, 4)) == {}


def test_apply_post_load_fixups_unregistered_model_type_noop():
    from auto_round.modeling.unfused_moe import apply_post_load_fixups

    model = nn.Module()
    model.config = types.SimpleNamespace(model_type="not_a_real_model")
    assert apply_post_load_fixups(model) == {}


def test_get_default_layer_config_patterns_returns_empty_for_non_nh():
    from auto_round.modeling.unfused_moe import get_default_layer_config_patterns

    model = nn.Module()
    model.config = types.SimpleNamespace(model_type="llama")
    assert get_default_layer_config_patterns(model) == {}


def test_get_default_layer_config_patterns_returns_nh_patterns_for_nh():
    from auto_round.modeling.unfused_moe import get_default_layer_config_patterns

    patterns = get_default_layer_config_patterns(_build_fake_nh_model())
    assert patterns
    assert any("mixer" in k and "out_proj" in k for k in patterns)


def test_set_layer_config_injects_nh_defaults_into_fresh_config():
    from auto_round.compressors.utils import set_layer_config

    root = nn.Module()
    root.config = _FakeNemotronHConfig(model_type="nemotron_h")
    root.layers = nn.ModuleList([nn.Module()])
    root.layers[0].mixer = nn.Module()
    root.layers[0].mixer.out_proj = nn.Linear(64, 64, bias=False)
    root.layers[0].mixer.other_proj = nn.Linear(64, 64, bias=False)

    final_config, _, _ = set_layer_config(
        root,
        {},
        default_scheme="W4A16",
        default_scale_dtype=torch.float16,
        supported_types=(nn.Linear,),
        inner_supported_types=(),
    )

    assert final_config["layers.0.mixer.out_proj"]["scale_dtype"] == torch.bfloat16
    assert final_config["layers.0.mixer.other_proj"]["scale_dtype"] == torch.float16


def test_set_layer_config_user_override_wins_over_nh_default():
    from auto_round.compressors.utils import set_layer_config
    from auto_round.modeling.unfused_moe.nemotron_h import (
        nemotron_h_default_layer_config_patterns,
    )

    root = nn.Module()
    root.config = _FakeNemotronHConfig(model_type="nemotron_h")
    root.layers = nn.ModuleList([nn.Module()])
    root.layers[0].mixer = nn.Module()
    root.layers[0].mixer.out_proj = nn.Linear(64, 64, bias=False)

    shared_key = next(iter(nemotron_h_default_layer_config_patterns()))
    final_config, _, _ = set_layer_config(
        root,
        {shared_key: {"scale_dtype": torch.float32, "bits": 8}},
        default_scheme="W4A16",
        default_scale_dtype=torch.float16,
        supported_types=(nn.Linear,),
        inner_supported_types=(),
    )

    cfg = final_config["layers.0.mixer.out_proj"]
    assert cfg["scale_dtype"] is torch.float32
    assert cfg["bits"] == 8


def test_set_layer_config_untouched_for_non_nh_models():
    from auto_round.compressors.utils import set_layer_config

    root = nn.Module()
    root.config = types.SimpleNamespace(model_type="llama")
    root.layers = nn.ModuleList([nn.Module()])
    root.layers[0].mixer = nn.Module()
    root.layers[0].mixer.out_proj = nn.Linear(64, 64, bias=False)

    final_config, _, _ = set_layer_config(
        root,
        {},
        default_scheme="W4A16",
        default_scale_dtype=torch.float16,
        supported_types=(nn.Linear,),
        inner_supported_types=(),
    )
    assert final_config["layers.0.mixer.out_proj"]["scale_dtype"] is torch.float16


# High-precision source-tensor restore


class _Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
        self.D = nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
        self.dt_bias = nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = _Mixer()


class _Root(nn.Module):
    def __init__(self):
        super().__init__()

        class _Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_Layer() for _ in range(2)])

        self.model = _Backbone()


def _make_source_checkpoint(tmpdir: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import save_file

    src_tensors: dict[str, torch.Tensor] = {}
    for i in range(2):
        prefix = f"backbone.layers.{i}.mixer"
        src_tensors[f"{prefix}.A_log"] = torch.tensor(
            [-0.123456789, -0.987654321, -1.111111, -2.222222, -3.333333, -4.444444, -5.555555, -6.666666],
            dtype=torch.float32,
        )
        src_tensors[f"{prefix}.D"] = torch.full((8,), 0.1234567, dtype=torch.float32)
        src_tensors[f"{prefix}.dt_bias"] = torch.full((8,), 0.7654321, dtype=torch.float32)
    save_file(src_tensors, os.path.join(tmpdir, "model.safetensors"))
    return src_tensors


def test_restore_ssm_core_to_fp32():
    from auto_round.modeling.unfused_moe.nemotron_h_setup import (
        _NH_SSM_CORE_PATTERNS,
        _restore_tensors_from_source,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        src_tensors = _make_source_checkpoint(tmpdir)
        model = _Root()

        restored = _restore_tensors_from_source(model, tmpdir, _NH_SSM_CORE_PATTERNS, torch.float32)
        assert len(restored) == 6
        for layer_idx, layer in enumerate(model.model.layers):
            for attr in ("A_log", "D", "dt_bias"):
                got = getattr(layer.mixer, attr)
                expected = src_tensors[f"backbone.layers.{layer_idx}.mixer.{attr}"]
                assert got.dtype is torch.float32
                assert torch.equal(got, expected)


def test_restore_router_bias_with_gate_submodule():
    from safetensors.torch import save_file

    from auto_round.modeling.unfused_moe.nemotron_h_setup import (
        _NH_ROUTER_BIAS_PATTERNS,
        _restore_tensors_from_source,
    )

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
        restored = _restore_tensors_from_source(model, tmpdir, _NH_ROUTER_BIAS_PATTERNS, torch.float32)
        assert len(restored) == 2
        for i, layer in enumerate(model.model.layers):
            buf = layer.mixer.gate.e_score_correction_bias
            assert buf.dtype is torch.float32
            assert torch.equal(buf, src[f"backbone.layers.{i}.mixer.gate.e_score_correction_bias"])


def test_restore_unknown_source_dir_returns_empty():
    from auto_round.modeling.unfused_moe.nemotron_h_setup import _restore_tensors_from_source

    assert _restore_tensors_from_source(_Root(), "/nonexistent/path", [r"\.A_log$"], torch.float32) == {}


def test_restore_shape_mismatch_skipped():
    from safetensors.torch import save_file

    from auto_round.modeling.unfused_moe.nemotron_h_setup import (
        _NH_SSM_CORE_PATTERNS,
        _restore_tensors_from_source,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        src = {"backbone.layers.0.mixer.A_log": torch.zeros(16, dtype=torch.float32)}
        save_file(src, os.path.join(tmpdir, "model.safetensors"))

        model = _Root()
        original = model.model.layers[0].mixer.A_log.clone()
        out = _restore_tensors_from_source(model, tmpdir, _NH_SSM_CORE_PATTERNS, torch.float32)
        assert out == {}
        assert torch.equal(model.model.layers[0].mixer.A_log, original)


def test_nh_source_to_module_renames():
    from auto_round.modeling.unfused_moe.nemotron_h_setup import _nh_source_to_module

    assert _nh_source_to_module("backbone.layers.0.mixer.A_log") == "model.layers.0.mixer.A_log"
    assert _nh_source_to_module("backbone.embedding.weight") == "model.embeddings.weight"
    assert _nh_source_to_module("lm_head.weight") == "lm_head.weight"
