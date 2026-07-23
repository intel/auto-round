import os
import shutil

import pytest

from auto_round import AutoRound, AutoScheme
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


def test_build_layer_config_header_rows_includes_experts_under_mlp():
    columns = ["mlp.down_proj", "self_attn.q_proj"]
    assert _build_layer_config_header_rows(columns, has_expert_layers=True) == [
        ["block", "mlp", "self_attn", "mlp"],
        ["", "down_proj", "q_proj", "experts"],
    ]


def test_short_summary_name_keeps_one_field_before_numeric_suffix():
    """Numeric block suffixes should be shortened to keep the preceding field."""
    assert _short_summary_name("model.layers.0") == "layers.0"


def test_choose_bits_per_layer_reconstructs_optimal_path():
    """DP parent pointers should preserve the optimal choices in layer order."""
    from auto_round.auto_scheme.delta_loss import choose_bits_per_layer_with_path

    layers = {
        "layer.0": [(0, 2, 4.0, ["layer.0"]), (1, 4, 1.0, ["layer.0"])],
        "layer.1": [(0, 2, 3.0, ["layer.1"]), (1, 4, 0.5, ["layer.1"])],
    }

    loss, path = choose_bits_per_layer_with_path(layers, P=6)

    assert loss == 4.0
    assert path == [(["layer.0"], 1), (["layer.1"], 0)]


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


class TestAutoScheme:
    @pytest.fixture(autouse=True)
    def setup_save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    def test_auto_scheme_export(self, tiny_opt_model_path):
        model_name = tiny_opt_model_path
        scheme = AutoScheme(avg_bits=2, options=("W2A16"), nsamples=1, ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        ar.quantize_and_save(self.save_dir)

        scheme = AutoScheme(avg_bits=4, options=("mxfp4"), nsamples=1, ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        ar.quantize_and_save(self.save_dir)

    def test_layer_config(self, tiny_opt_model_path):
        from auto_round.auto_scheme.utils import compute_avg_bits_for_model
        from auto_round.utils import get_module

        target_bits = 3.5
        model_name = tiny_opt_model_path
        scheme = AutoScheme(avg_bits=target_bits, options=("W2A16", "W4A16", "BF16"))
        user_layer_config = {"model.decoder.layers.1.fc1": {"bits": 8, "group_size": 32, "sym": False}}
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1, layer_config=user_layer_config)
        model, layer_config = ar.quantize()
        assert layer_config["model.decoder.layers.1.fc1"]["bits"] == 8
        assert layer_config["model.decoder.layers.1.fc1"]["sym"] is False
        assert layer_config["model.decoder.layers.1.fc1"]["group_size"] == 32
        layer = get_module(model, "model.decoder.layers.1.fc1")
        assert layer.bits == 8
        assert layer.sym is False
        assert layer.group_size == 32
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    def test_cache_files_saved_with_correct_format(self, tiny_opt_model_path, tmp_path, monkeypatch):
        """After AutoScheme runs, per-scheme JSON cache files must exist with individual layer scores."""
        import glob
        import json

        from auto_round.auto_scheme.delta_loss import _load_autoscheme_scores

        cache_dir = str(tmp_path / "ar_work_space")
        monkeypatch.setenv("AR_WORK_SPACE", cache_dir)

        scheme = AutoScheme(
            avg_bits=3,
            options=("W2A16", "W4A16"),
            nsamples=1,
            ignore_scale_zp_bits=True,
        )
        ar = AutoRound(model=tiny_opt_model_path, scheme=scheme, iters=0, nsamples=1)
        _, layer_config = ar.quantize()

        # Cache files must exist — one per scheme (2 schemes here)
        cache_files = glob.glob(f"{cache_dir}/auto_scheme_cache/scheme_*.json")
        assert (
            len(cache_files) == 2
        ), f"Expected 2 cache files (one per scheme), found {len(cache_files)}: {cache_files}"

        for path in cache_files:
            data = _load_autoscheme_scores(path)
            assert data is not None, f"Cache file {path} could not be loaded"
            assert data["version"] == 3
            assert data["score_granularity"] == "per_op"
            assert "layer_scores" in data
            assert "total_loss_for_scheme" in data

            # Every layer in layer_scores must have a [bits, loss] pair (individual, not merged)
            for layer_name, score_pair in data["layer_scores"].items():
                assert (
                    isinstance(score_pair, list) and len(score_pair) == 2
                ), f"layer_scores[{layer_name!r}] should be [bits, loss], got {score_pair}"
                bits, loss = score_pair
                assert bits > 0, f"bits must be positive for {layer_name}"
                assert loss >= 0, f"loss must be non-negative for {layer_name}"

            # All non-fixed linear layers in layer_config should appear individually in cache
            for layer_name in layer_config:
                assert layer_name in data["layer_scores"], (
                    f"Layer {layer_name!r} missing from cache {path} — "
                    f"cache may have stored merged group scores instead of individual scores"
                )

    def test_different_avg_bits_produces_different_layer_config(self, tiny_opt_model_path):
        """Changing avg_bits should change the resulting layer_config."""
        scheme_low = AutoScheme(
            avg_bits=2.5,
            options=("W2A16", "W4A16"),
            nsamples=1,
            ignore_scale_zp_bits=True,
        )
        ar_low = AutoRound(model=tiny_opt_model_path, scheme=scheme_low, iters=0, nsamples=1)
        _, config_low = ar_low.quantize()

        scheme_high = AutoScheme(
            avg_bits=3.5,
            options=("W2A16", "W4A16"),
            nsamples=1,
            ignore_scale_zp_bits=True,
        )
        ar_high = AutoRound(model=tiny_opt_model_path, scheme=scheme_high, iters=0, nsamples=1)
        _, config_high = ar_high.quantize()

        low_avg = sum(v["bits"] for v in config_low.values() if "bits" in v) / max(
            len([v for v in config_low.values() if "bits" in v]), 1
        )
        high_avg = sum(v["bits"] for v in config_high.values() if "bits" in v) / max(
            len([v for v in config_high.values() if "bits" in v]), 1
        )
        assert high_avg > low_avg, (
            f"avg_bits=4 should produce higher average bits than avg_bits=2, "
            f"got low={low_avg:.2f} high={high_avg:.2f}"
        )

    def test_shared_layers_assigns_same_bits(self, tiny_opt_model_path):
        """With shared_layers=[q_proj,k_proj,v_proj], all three must get the same bits per block."""
        scheme = AutoScheme(
            avg_bits=5,
            options=("MXFP4", "MXFP8"),
            nsamples=1,
            ignore_scale_zp_bits=True,
            shared_layers=[["fc1", "fc2"]],
        )
        ar = AutoRound(model=tiny_opt_model_path, scheme=scheme, iters=0, nsamples=1)
        _, layer_config = ar.quantize()

        scheme = AutoScheme(
            avg_bits=5,
            options=("MXFP4", "MXFP8"),
            nsamples=1,
            ignore_scale_zp_bits=True,
            shared_layers=[["q_proj", "k_proj", "v_proj"]],
        )
        ar = AutoRound(model=tiny_opt_model_path, scheme=scheme, iters=0, nsamples=1)
        _, layer_config = ar.quantize()

        # Collect per-block q/k/v bits
        block_qkv: dict[str, dict[str, int]] = {}
        for name, cfg in layer_config.items():
            if "bits" not in cfg:
                continue
            for proj in ("q_proj", "k_proj", "v_proj"):
                if name.endswith("." + proj):
                    prefix = name[: -len(proj) - 1]
                    block_qkv.setdefault(prefix, {})[proj] = cfg["bits"]

        assert block_qkv, "No q/k/v layers found in layer_config"
        for prefix, proj_bits in block_qkv.items():
            present = list(proj_bits.keys())
            bits_values = list(proj_bits.values())
            assert len(set(bits_values)) == 1, (
                f"Block {prefix!r}: q/k/v should all have the same bits with shared_layers, "
                f"got {dict(zip(present, bits_values))}"
            )


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


def test_autoscheme_cache_save_and_load(tmp_path):
    """Per-scheme cache: scores can be saved and loaded correctly."""
    from auto_round.auto_scheme.delta_loss import (
        _load_autoscheme_scores,
        _save_autoscheme_scores,
    )

    cache_key = "test_key_123"
    cache_path = os.path.join(str(tmp_path), f"scheme_00_{cache_key}.json")

    scheme_dict = {"bits": 4, "act_bits": 16}
    layer_scores = {
        "layer.0": [4, 1.2],
        "layer.1": [4, 0.9],
    }
    total_loss = 2.1
    total_params = 1000000

    _save_autoscheme_scores(
        cache_path,
        cache_key,
        0,
        scheme_dict,
        layer_scores,
        total_loss,
        total_params,
    )

    loaded = _load_autoscheme_scores(cache_path)
    assert loaded is not None
    assert loaded["layer_scores"] == layer_scores
    assert loaded["total_loss_for_scheme"] == total_loss
    assert loaded["total_params"] == total_params


def test_parallel_progress_events_are_applied_in_parent():
    """Worker progress events should update only the parent-owned progress bar."""
    import queue

    from auto_round.auto_scheme.delta_loss import _drain_progress_queue, _ProgressQueueProxy

    class FakeProgressBar:
        def __init__(self):
            self.steps = 0
            self.messages = []

        def update(self, steps):
            self.steps += steps

        def write(self, message):
            self.messages.append(message)

    progress_queue = queue.Queue()
    worker_progress = _ProgressQueueProxy(progress_queue)
    parent_progress = FakeProgressBar()

    worker_progress.update(2)
    worker_progress.write("scheme progress")
    worker_progress.update()
    _drain_progress_queue(progress_queue, parent_progress)

    assert parent_progress.steps == 3
    assert parent_progress.messages == ["scheme progress"]
    assert progress_queue.empty()


def test_assign_scheme_worker_devices_round_robin():
    from auto_round.auto_scheme.delta_loss import _assign_scheme_worker_devices

    assert _assign_scheme_worker_devices(5, 2) == ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0"]


def test_assign_scheme_worker_devices_shares_single_gpu():
    from auto_round.auto_scheme.delta_loss import _assign_scheme_worker_devices

    assert _assign_scheme_worker_devices(3, 1) == ["cuda:0", "cuda:0", "cuda:0"]


def test_assign_scheme_worker_devices_rejects_no_gpu():
    from auto_round.auto_scheme.delta_loss import _assign_scheme_worker_devices

    with pytest.raises(ValueError, match="at least 1"):
        _assign_scheme_worker_devices(1, 0)


def test_per_op_cache_compatibility_rejects_grouped_scores():
    from auto_round.auto_scheme.delta_loss import _is_per_op_cache_compatible

    quant_layers = ["layer.0", "layer.1", "fixed"]
    fixed_layers = {"fixed": {"bits": 8}}

    assert _is_per_op_cache_compatible(
        {"layer_scores": {"layer.0": [4, 1.0], "layer.1": [4, 2.0]}},
        quant_layers,
        fixed_layers,
    )
    assert not _is_per_op_cache_compatible(
        {"layer_scores": {"layer.0": [8, 3.0]}},
        quant_layers,
        fixed_layers,
    )


def test_version_two_cache_is_rejected(tmp_path):
    import json

    from auto_round.auto_scheme.delta_loss import _load_autoscheme_scores

    cache_path = tmp_path / "scheme_00_old.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": 2,
                "layer_scores": {"layer.0": [4, 1.0]},
                "total_loss_for_scheme": 1.0,
                "total_params": 4,
            }
        ),
        encoding="utf-8",
    )

    assert _load_autoscheme_scores(cache_path) is None


def test_worker_memory_reports_cover_all_processes_and_devices():
    from auto_round.auto_scheme.delta_loss import _merge_worker_memory_reports

    class FakeMemoryMonitor:
        peak_ram = 1.0
        peak_vram = {"0": 1.0}

        @staticmethod
        def _process_tree_rss():
            return 1.25

    monitor = FakeMemoryMonitor()
    _merge_worker_memory_reports(
        monitor,
        [
            {"device": "0", "peak_ram": 0.5, "peak_vram": 2.0},
            {"device": "0", "peak_ram": 0.75, "peak_vram": 3.0},
            {"device": "1", "peak_ram": 0.25, "peak_vram": 4.0},
        ],
    )

    assert monitor.peak_ram == 2.75
    assert monitor.peak_vram == {"0": 5.0, "1": 4.0}


def test_replacement_wrapper_without_tuning_device_uses_major_device():
    import torch

    from auto_round.auto_scheme.delta_loss import move_module_to_tuning_device

    class ReplacementWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.orig_layer = torch.nn.Linear(2, 2)

    wrapper = ReplacementWrapper()
    move_module_to_tuning_device(wrapper, major_device="cpu")

    assert wrapper.orig_layer.weight.device.type == "cpu"
