import os

from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.auto_scheme.utils import _build_layer_config_header_rows, _short_summary_name


def test_env_ar_auto_scheme_nsamples_overrides_default(monkeypatch):
    """AR_AUTO_SCHEME_NSAMPLES env var should override the built-in nsamples heuristic."""
    import auto_round.envs as envs

    monkeypatch.setenv("AR_AUTO_SCHEME_NSAMPLES", "7")
    assert envs.AR_AUTO_SCHEME_NSAMPLES == 7


def test_env_ar_auto_scheme_batch_size_overrides_default(monkeypatch):
    """AR_AUTO_SCHEME_BATCH_SIZE env var should override the built-in batch_size default."""
    import auto_round.envs as envs

    monkeypatch.setenv("AR_AUTO_SCHEME_BATCH_SIZE", "4")
    assert envs.AR_AUTO_SCHEME_BATCH_SIZE == 4


def test_env_ar_auto_scheme_batch_size_zero_raises(monkeypatch):
    """Zero value for AR_AUTO_SCHEME_BATCH_SIZE should raise ValueError."""
    import pytest

    import auto_round.envs as envs

    monkeypatch.setenv("AR_AUTO_SCHEME_BATCH_SIZE", "0")
    with pytest.raises(ValueError):
        _ = envs.AR_AUTO_SCHEME_BATCH_SIZE


def test_build_layer_config_header_rows_merges_adjacent_prefixes():
    """Adjacent columns with the same prefix should be merged into one compact header cell."""
    columns = ["mlp.down_proj", "mlp.gate_proj", "self_attn.q_proj", "self_attn.v_proj"]
    assert _build_layer_config_header_rows(columns) == [
        ["block", "mlp", "", "self_attn", ""],
        ["", "down_proj", "gate_proj", "q_proj", "v_proj"],
    ]


def test_short_summary_name_keeps_one_field_before_numeric_suffix():
    """Numeric block suffixes should be shortened to keep the preceding field."""
    assert _short_summary_name("model.layers.0") == "layers.0"


def test_build_expert_groups_groups_experts_per_block():
    """Expert layers in the same block should be grouped together."""
    import torch
    from torch import nn

    from auto_round.auto_scheme.utils import build_expert_groups

    # Build a minimal MoE-like model with 2 blocks, each with 2 experts having 2 projections
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList()
            for i in range(2):
                block = nn.Module()
                block.mlp = nn.Module()
                block.mlp.experts = nn.ModuleList()
                for j in range(2):
                    expert = nn.Module()
                    expert.gate_proj = nn.Linear(8, 8, bias=False)
                    expert.up_proj = nn.Linear(8, 8, bias=False)
                    expert.down_proj = nn.Linear(8, 8, bias=False)
                    block.mlp.experts.append(expert)
                block.self_attn = nn.Module()
                block.self_attn.q_proj = nn.Linear(8, 8, bias=False)
                self.model.layers.append(block)

    model = FakeModel()
    quant_layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    fixed_layer_scheme = {}

    groups = build_expert_groups(model, quant_layer_names, fixed_layer_scheme)
    # Should have 2 groups (one per block), each containing all 6 expert projections
    assert len(groups) == 2
    for group in groups:
        expert_layers = [n for n in group if "experts" in n]
        assert len(expert_layers) == 6  # 2 experts * 3 projections
        # Non-expert layers (q_proj) should NOT be in the group
        assert all("self_attn" not in n for n in group)


def test_build_expert_groups_skips_fixed_layers():
    """Expert layers already in fixed_layer_scheme should not be grouped."""
    import torch
    from torch import nn

    from auto_round.auto_scheme.utils import build_expert_groups

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList()
            block = nn.Module()
            block.mlp = nn.Module()
            block.mlp.experts = nn.ModuleList()
            for j in range(2):
                expert = nn.Module()
                expert.gate_proj = nn.Linear(8, 8, bias=False)
                block.mlp.experts.append(expert)
            self.model.layers.append(block)

    model = FakeModel()
    quant_layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    # Fix all expert layers
    fixed_layer_scheme = {n: {} for n in quant_layer_names if "experts" in n}

    groups = build_expert_groups(model, quant_layer_names, fixed_layer_scheme)
    assert len(groups) == 0


def test_autoscheme_cache_key_different_for_different_schemes():
    """Per-scheme cache: different schemes should produce different cache keys."""
    from auto_round.auto_scheme.delta_loss import _autoscheme_cache_key

    key_w4 = _autoscheme_cache_key(
        model_name="test-model",
        dataset="pile-10k",
        nsamples=16,
        seqlen=256,
        batch_size=8,
        quant_layer_names=["layer.0"],
        fixed_layer_scheme={},
        scheme="W4A16",
        force_mllm=False,
    )
    key_w8 = _autoscheme_cache_key(
        model_name="test-model",
        dataset="pile-10k",
        nsamples=16,
        seqlen=256,
        batch_size=8,
        quant_layer_names=["layer.0"],
        fixed_layer_scheme={},
        scheme="W8A16",
        force_mllm=False,
    )
    assert key_w4 != key_w8
    assert len(key_w4) == 16  # sha256 truncated to 16 chars


def test_autoscheme_cache_key_insensitive_to_layer_order():
    """Per-scheme cache: layer order should not affect the key (internally sorted)."""
    from auto_round.auto_scheme.delta_loss import _autoscheme_cache_key

    key1 = _autoscheme_cache_key(
        model_name="test-model",
        dataset="pile-10k",
        nsamples=16,
        seqlen=256,
        batch_size=8,
        quant_layer_names=["layer.0", "layer.1"],
        fixed_layer_scheme={},
        scheme="W4A16",
        force_mllm=False,
    )
    key2 = _autoscheme_cache_key(
        model_name="test-model",
        dataset="pile-10k",
        nsamples=16,
        seqlen=256,
        batch_size=8,
        quant_layer_names=["layer.1", "layer.0"],  # Different order
        fixed_layer_scheme={},
        scheme="W4A16",
        force_mllm=False,
    )
    assert key1 == key2  # Should match after internal sorting


def test_autoscheme_cache_save_and_load():
    """Per-scheme cache: scores can be saved and loaded correctly."""
    import tempfile

    from auto_round.auto_scheme.delta_loss import (
        _load_autoscheme_scores,
        _save_autoscheme_scores,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Manually create cache path in temp dir
        cache_key = "test_key_123"
        cache_path = os.path.join(tmpdir, f"scheme_00_{cache_key}.json")

        scheme_dict = {"bits": 4, "act_bits": 16}
        layer_scores = {
            "layer.0": [4, 1.2],
            "layer.1": [4, 0.9],
        }
        total_loss = 2.1
        total_params = 1000000

        # Save
        _save_autoscheme_scores(
            cache_path,
            cache_key,
            0,
            scheme_dict,
            layer_scores,
            total_loss,
            total_params,
        )

        # Load and verify
        loaded = _load_autoscheme_scores(cache_path)
        assert loaded is not None
        assert loaded["layer_scores"] == layer_scores
        assert loaded["total_loss_for_scheme"] == total_loss
        assert loaded["total_params"] == total_params
