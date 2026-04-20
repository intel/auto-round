"""Tests for Nemotron-H-specific post-load patches and default
layer_config patterns.

These tests mock the architecture-level dependencies (no real
Nemotron-H checkpoint is downloaded) and focus on:

- ``apply_post_load_fixups`` is a no-op for non-NH model types;
- ``apply_nemotron_h_post_load`` patches ``Zamba2RMSNormGated``
  ``group_size`` both per-instance and on the class;
- ``nemotron_h_default_layer_config_patterns`` returns the expected
  regex → overlay map;
- ``set_layer_config`` merges NH default patterns when the user didn't
  set the same key, and leaves non-NH models alone.
"""

from __future__ import annotations

import types
from unittest import mock

import pytest
import torch
import torch.nn as nn


class _FakeZamba2RMSNormGated(nn.Module):
    """Stand-in for ``transformers.models.nemotron_h.modeling_nemotron_h.Zamba2RMSNormGated``.

    Crucially the real class is itself named ``Zamba2RMSNormGated`` — the
    post-load patcher matches on ``module.__class__.__name__``, not on a
    transformers import. We reproduce that here so the test runs without
    transformers nemotron_h support installed.
    """


# Mimic the real class's __name__ for the matcher.
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
        _name_or_path="",  # disable high-precision source reload in tests
    )
    for i in range(n_norms):
        setattr(model, f"norm_{i}", _FakeZamba2RMSNormGated())
    return model


def test_default_layer_config_patterns_returns_out_proj_overlay():
    from auto_round.modeling.unfused_moe.nemotron_h_setup import (
        nemotron_h_default_layer_config_patterns,
    )

    patterns = nemotron_h_default_layer_config_patterns()
    # Exactly one pattern key matching ``mixer.out_proj`` is expected; the
    # content may evolve but BF16 scale_dtype must remain.
    assert any("mixer" in k and "out_proj" in k for k in patterns), patterns
    out_proj_overlay = next(v for k, v in patterns.items() if "mixer" in k and "out_proj" in k)
    assert out_proj_overlay["scale_dtype"] is torch.bfloat16


def test_apply_nemotron_h_post_load_patches_zamba2_instances_and_class():
    from auto_round.modeling.unfused_moe.nemotron_h_setup import (
        apply_nemotron_h_post_load,
    )

    model = _build_fake_nh_model(mamba_num_heads=64, mamba_head_dim=64, n_groups=8)
    # 64 * 64 / 8 = 512
    expected_group_size = 512

    summary = apply_nemotron_h_post_load(model, enable_high_precision_overrides=False)
    assert summary["zamba2_patched"] == 3

    for i in range(3):
        mod = getattr(model, f"norm_{i}")
        assert mod.group_size == expected_group_size
    # Class attribute set as fallback.
    assert _FakeZamba2RMSNormGated.group_size == expected_group_size


def test_apply_nemotron_h_post_load_noop_for_other_model_types():
    from auto_round.modeling.unfused_moe.nemotron_h_setup import (
        apply_nemotron_h_post_load,
    )

    model = _build_fake_nh_model()
    model.config.model_type = "llama"  # change to non-NH

    summary = apply_nemotron_h_post_load(model, enable_high_precision_overrides=False)
    assert summary == {"zamba2_patched": 0, "high_precision_restored": 0}
    # And no instance should have been patched.
    for i in range(3):
        mod = getattr(model, f"norm_{i}")
        assert not hasattr(mod, "group_size") or mod.group_size == 512
        # If 512 (class attribute from the earlier test), that's still
        # class-level, not our doing here — no new instance attr was set.
        assert "group_size" not in mod.__dict__


def test_apply_post_load_fixups_dispatches_to_model_config():
    from auto_round.modeling.unfused_moe import apply_post_load_fixups

    model = _build_fake_nh_model(mamba_num_heads=32, mamba_head_dim=128, n_groups=8)
    # 32 * 128 / 8 = 512
    expected_group_size = 512

    summary = apply_post_load_fixups(model, enable_high_precision_overrides=False)
    assert summary.get("zamba2_patched") == 3
    assert getattr(model.norm_0, "group_size") == expected_group_size


def test_apply_post_load_fixups_no_op_without_config():
    from auto_round.modeling.unfused_moe import apply_post_load_fixups

    model = nn.Linear(4, 4)  # has no ``config``
    summary = apply_post_load_fixups(model)
    assert summary == {}


def test_apply_post_load_fixups_unregistered_model_type_noop():
    from auto_round.modeling.unfused_moe import apply_post_load_fixups

    model = nn.Module()
    model.config = types.SimpleNamespace(model_type="not_a_real_model")
    summary = apply_post_load_fixups(model)
    assert summary == {}


def test_get_default_layer_config_patterns_returns_empty_for_non_nh():
    from auto_round.modeling.unfused_moe import get_default_layer_config_patterns

    model = nn.Module()
    model.config = types.SimpleNamespace(model_type="llama")
    assert get_default_layer_config_patterns(model) == {}


def test_get_default_layer_config_patterns_returns_nh_patterns_for_nh():
    from auto_round.modeling.unfused_moe import get_default_layer_config_patterns

    model = _build_fake_nh_model()
    patterns = get_default_layer_config_patterns(model)
    assert patterns
    assert any("mixer" in k and "out_proj" in k for k in patterns)


def test_set_layer_config_injects_nh_defaults_into_fresh_config():
    """When the caller passes an empty ``layer_config`` for a
    Nemotron-H model, ``set_layer_config`` must merge the NH default
    patterns so downstream layer resolution picks up BF16 scales for
    ``mixer.out_proj`` layers."""

    from auto_round.compressors.utils import set_layer_config

    # Build a minimal NH-like module tree: one Linear named
    # ``layers.0.mixer.out_proj`` that should match the NH default.
    root = nn.Module()
    root.config = _FakeNemotronHConfig(model_type="nemotron_h")

    layers = nn.Module()
    root.layers = nn.ModuleList([layers])
    layer0 = root.layers[0]
    layer0.mixer = nn.Module()
    layer0.mixer.out_proj = nn.Linear(64, 64, bias=False)
    layer0.mixer.other_proj = nn.Linear(64, 64, bias=False)

    layer_config: dict = {}
    final_config, _, _ = set_layer_config(
        root,
        layer_config,
        default_scheme="W4A16",
        default_scale_dtype=torch.float16,
        supported_types=(nn.Linear,),
        inner_supported_types=(),
    )

    out_proj_cfg = final_config.get("layers.0.mixer.out_proj")
    other_proj_cfg = final_config.get("layers.0.mixer.other_proj")
    assert out_proj_cfg is not None
    assert other_proj_cfg is not None

    # out_proj got BF16 scales from NH default.
    assert out_proj_cfg.get("scale_dtype") == torch.bfloat16
    # other_proj got the global default (FP16).
    assert other_proj_cfg.get("scale_dtype") == torch.float16


def test_set_layer_config_user_override_wins_over_nh_default():
    """If the user specifies ``scale_dtype`` for the same regex the NH
    defaults target, the user's value must win."""

    from auto_round.compressors.utils import set_layer_config

    root = nn.Module()
    root.config = _FakeNemotronHConfig(model_type="nemotron_h")
    root.layers = nn.ModuleList([nn.Module()])
    root.layers[0].mixer = nn.Module()
    root.layers[0].mixer.out_proj = nn.Linear(64, 64, bias=False)

    # Use the exact same regex key the NH default uses so we exercise
    # the "user-entry-wins-on-the-same-pattern" merge branch.
    from auto_round.modeling.unfused_moe.nemotron_h_setup import (
        nemotron_h_default_layer_config_patterns,
    )

    nh_patterns = nemotron_h_default_layer_config_patterns()
    shared_key = next(iter(nh_patterns))
    user_layer_config = {
        shared_key: {"scale_dtype": torch.float32, "bits": 8},
    }
    final_config, _, _ = set_layer_config(
        root,
        user_layer_config,
        default_scheme="W4A16",
        default_scale_dtype=torch.float16,
        supported_types=(nn.Linear,),
        inner_supported_types=(),
    )

    cfg = final_config.get("layers.0.mixer.out_proj")
    assert cfg is not None
    assert cfg["scale_dtype"] is torch.float32  # user wins
    assert cfg["bits"] == 8


def test_set_layer_config_untouched_for_non_nh_models():
    """``set_layer_config`` must not inject any NH defaults when the
    model is not Nemotron-H."""

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

    cfg = final_config.get("layers.0.mixer.out_proj")
    assert cfg is not None
    # Must use the global default, not BF16.
    assert cfg["scale_dtype"] is torch.float16
