import json
import os
import shutil

import pytest

from auto_round import AutoRound, AutoScheme
from auto_round.auto_scheme.utils import _build_layer_config_header_rows, _short_summary_name


@pytest.mark.parametrize(
    "model_type, expected",
    [
        ("mllm", ("model", "tokenizer", "processor", "image_processor", None, True, False)),
        ("diffusion", ("model", None, None, None, "pipe", False, True)),
        ("llm", ("model", "tokenizer", None, None, None, False, False)),
    ],
)
def test_load_model_returns_named_values_for_each_model_type(monkeypatch, model_type, expected):
    import auto_round.utils.model as model_utils

    monkeypatch.setattr(model_utils, "is_mllm_model", lambda *_args, **_kwargs: model_type == "mllm")
    monkeypatch.setattr(model_utils, "is_diffusion_model", lambda *_args, **_kwargs: model_type == "diffusion")
    monkeypatch.setattr(
        model_utils,
        "mllm_load_model",
        lambda *_args, **_kwargs: ("model", "processor", "tokenizer", "image_processor"),
    )
    monkeypatch.setattr(model_utils, "diffusion_load_model", lambda *_args, **_kwargs: ("pipe", "model"))
    monkeypatch.setattr(model_utils, "llm_load_model", lambda *_args, **_kwargs: ("model", "tokenizer"))
    monkeypatch.setattr("auto_round.modeling.unfused_moe.apply_model_monkey_patches", lambda *_args, **_kwargs: None)

    assert model_utils.load_model("model-id") == expected


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


def test_env_ar_auto_scheme_cache(monkeypatch, tmp_path):
    """AR_AUTO_SCHEME_CACHE should expose an independent cache directory."""
    import auto_round.envs as envs

    cache_dir = str(tmp_path / "auto_scheme_cache")
    monkeypatch.setenv("AR_AUTO_SCHEME_CACHE", cache_dir)
    assert envs.AR_AUTO_SCHEME_CACHE == cache_dir


def test_env_ar_enable_auto_scheme_parallel(monkeypatch):
    import auto_round.envs as envs

    monkeypatch.delenv("AR_ENABLE_AUTO_SCHEME_PARALLEL", raising=False)
    assert envs.AR_ENABLE_AUTO_SCHEME_PARALLEL is True
    monkeypatch.setenv("AR_ENABLE_AUTO_SCHEME_PARALLEL", "0")
    assert envs.AR_ENABLE_AUTO_SCHEME_PARALLEL is False


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


def test_get_layer_config_supports_avg_bits_list(monkeypatch):
    from auto_round import auto_scheme
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme, GenScheme

    scheme = AutoScheme(avg_bits=[2.5, 3.5], options=("W2A16", "W4A16"))
    generator = GenScheme.__new__(GenScheme)
    generator.auto_scheme = scheme
    generator.model = object()
    generator.quant_layer_names = []
    generator.fixed_layer_scheme = {}
    generator.dataset = "pile-10k"
    generator.tokenizer = object()
    generator.device_map = None
    generator.enable_torch_compile = False
    generator.min_avg_bit_scheme = "W2A16"
    generator.processor = None

    def method_func(auto_scheme, *args, **kwargs):
        return {"layer": {"bits": auto_scheme.avg_bits}}

    monkeypatch.setitem(auto_scheme.AUTO_SCHEME_METHODS, "default", method_func)

    assert generator.get_layer_config() == {
        2.5: {"layer": {"bits": 2.5}},
        3.5: {"layer": {"bits": 3.5}},
    }
    assert scheme.avg_bits == [2.5, 3.5]


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


def test_activation_scoring_handles_reused_wrapper():
    """Each forward call should keep its own activation error until backward."""
    import types

    import torch

    from auto_round.auto_scheme.delta_loss import AutoSchemeWrapperLinear

    wrapper = AutoSchemeWrapperLinear.__new__(AutoSchemeWrapperLinear)
    torch.nn.Module.__init__(wrapper)
    wrapper.orig_layer = types.SimpleNamespace(act_bits=8)
    wrapper.act_qdq_func = lambda x, *_args, **_kwargs: (x * 0.5, 1.0, None)
    wrapper.grad_mode = True
    wrapper.act_cnt = 0
    wrapper.act_score = 0.0
    wrapper.weight_score = 0.0
    wrapper.mix_score = 0.0
    wrapper.max_act_value = 0

    first = torch.tensor([1.0, -2.0], requires_grad=True)
    second = torch.tensor([3.0, -4.0], requires_grad=True)
    qdq_first, _, _ = wrapper._qdq_act(first)
    qdq_second, _, _ = wrapper._qdq_act(second)

    (qdq_first.sum() + qdq_second.sum()).backward()

    assert wrapper.act_cnt == 2
    assert wrapper.act_score == pytest.approx(5.0)
    assert wrapper.mix_score == pytest.approx(5.0)


def test_prepare_replay_input_supports_keyword_hidden_states():
    import torch

    from auto_round.auto_scheme.delta_loss import _prepare_replay_input

    hidden_states = torch.randn(2, 4)
    attention_mask = torch.ones(2, 4, dtype=torch.int64)

    replay_input = _prepare_replay_input([], {"attention_mask": attention_mask, "hidden_states": hidden_states}, "0")

    assert replay_input is hidden_states
    assert replay_input.requires_grad


def test_prepare_replay_input_rejects_missing_floating_tensor():
    import torch

    from auto_round.auto_scheme.delta_loss import _prepare_replay_input

    with pytest.raises(RuntimeError, match="No floating replay input found for block 0"):
        _prepare_replay_input([], {"attention_mask": torch.ones(2, dtype=torch.int64)}, "0")


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

    @pytest.mark.timeout(60)
    def test_auto_scheme_export(self, tiny_opt_model_path):
        model_name = tiny_opt_model_path
        int_save_dir = os.path.join(self.save_dir, "int")
        scheme = AutoScheme(avg_bits=2, options=("W2A16"), nsamples=1, ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        _, int_model_path = ar.quantize_and_save(int_save_dir)

        with open(os.path.join(int_model_path, "config.json")) as f:
            int_config = json.load(f)["quantization_config"]
        assert int_config["quant_method"] == "auto-round"
        assert int_config["bits"] == 2
        assert int_config["data_type"] == "int"

        mxfp_save_dir = os.path.join(self.save_dir, "mxfp")
        scheme = AutoScheme(avg_bits=4, options=("mxfp4"), nsamples=1, ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        _, mxfp_model_path = ar.quantize_and_save(mxfp_save_dir)

        with open(os.path.join(mxfp_model_path, "config.json")) as f:
            mxfp_config = json.load(f)["quantization_config"]
        assert mxfp_config["quant_method"] == "auto-round"
        assert mxfp_config["bits"] == 4
        assert mxfp_config["data_type"] == "mx_fp"
        assert os.path.exists(os.path.join(int_model_path, "config.json"))

    @pytest.mark.timeout(120)
    def test_gguf_user_fixed_embedding_budget(self, tiny_qwen_model_path):
        """Regression test: a user-fixed embedding must be budget-priced at its fixed bits.

        apply_quant_scheme only covers quant_layer_names (embeddings are carved out),
        so the fixed-layer budget subtraction read bits=16 off the bare module and
        priced the embedding at 16 bits, making low targets infeasible (DP returned
        None -> TypeError, or 'Avg bits is too small').
        """
        target_bits = 3.0
        scheme = AutoScheme(
            avg_bits=target_bits,
            options=("gguf:q2_k_s", "gguf:q4_k_s", "gguf:q6_k"),
            nsamples=1,
            ignore_scale_zp_bits=True,
        )
        user_layer_config = {
            "model.embed_tokens": {
                "bits": 3,
                "super_bits": 6,
                "super_group_size": 16,
                "group_size": 16,
                "sym": True,
                "data_type": "int_sym_dq",
            }
        }
        ar = AutoRound(
            model=tiny_qwen_model_path,
            scheme=scheme,
            format="gguf:q2_k_s",
            iters=0,
            nsamples=1,
            seqlen=32,
            layer_config=user_layer_config,
        )
        weight_numels = {
            n: m.weight.numel()
            for n, m in ar.model.named_modules()
            if getattr(m, "weight", None) is not None and len(list(m.children())) == 0
        }
        model, layer_config = ar.quantize()
        assert layer_config["model.embed_tokens"]["bits"] == 3
        quant_layers = [n for n in layer_config if n in weight_numels]
        total_params = sum(weight_numels[n] for n in quant_layers)
        total_bits = sum(layer_config[n].get("bits", 16) * weight_numels[n] for n in quant_layers)
        avg_bits = total_bits / total_params
        assert avg_bits <= target_bits + 0.05

    def test_gguf_embedding_in_budget(self, tiny_qwen_model_path):
        """Regression test: the (tied) embedding must be charged against the avg_bits budget.

        On tiny Qwen the embedding holds >90% of the params. Before the fix it was
        silently dropped from the AutoScheme budget (dead `in quant_layer_names`
        check) and later filled with the gguf lm_head default (q6_k for tied
        embeddings), so the effective avg_bits landed near 6 instead of the target.
        """
        target_bits = 3.0
        scheme = AutoScheme(
            avg_bits=target_bits,
            options=("gguf:q2_k_s", "gguf:q4_k_s", "gguf:q6_k"),
            nsamples=1,
            ignore_scale_zp_bits=True,
        )
        ar = AutoRound(model=tiny_qwen_model_path, scheme=scheme, format="gguf:q2_k_s", iters=0, nsamples=1, seqlen=32)
        # Snapshot parameter counts before quantization: gguf packing releases
        # weights (module.weight = None) as blocks are packed, so numel is not
        # available on the model afterwards.
        weight_numels = {
            n: m.weight.numel()
            for n, m in ar.model.named_modules()
            if getattr(m, "weight", None) is not None and len(list(m.children())) == 0
        }
        model, layer_config = ar.quantize()
        # Only q2_k_s fits the budget for the embedding (q4_k_s/q6_k alone would exceed it).
        assert layer_config["model.embed_tokens"]["bits"] == 2
        quant_layers = [n for n in layer_config if n in weight_numels]
        total_params = sum(weight_numels[n] for n in quant_layers)
        total_bits = sum(layer_config[n].get("bits", 16) * weight_numels[n] for n in quant_layers)
        avg_bits = total_bits / total_params
        assert avg_bits <= target_bits + 0.05

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

        cache_dir = str(tmp_path / "auto_scheme_cache")
        monkeypatch.setenv("AR_AUTO_SCHEME_CACHE", cache_dir)

        scheme = AutoScheme(
            avg_bits=3,
            options=("W2A16", "W4A16"),
            nsamples=1,
            ignore_scale_zp_bits=True,
        )
        ar = AutoRound(model=tiny_opt_model_path, scheme=scheme, iters=0, nsamples=1)
        _, layer_config = ar.quantize()

        # Cache files must exist — one per scheme (2 schemes here)
        cache_files = glob.glob(f"{cache_dir}/scheme_*.json")
        assert (
            len(cache_files) == 2
        ), f"Expected 2 cache files (one per scheme), found {len(cache_files)}: {cache_files}"

        for path in cache_files:
            data = _load_autoscheme_scores(path)
            assert data is not None, f"Cache file {path} could not be loaded"
            assert data["version"] == 1
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
        low_gpu_mem_usage=True,
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
        low_gpu_mem_usage=True,
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
        low_gpu_mem_usage=True,
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
        low_gpu_mem_usage=True,
    )
    assert key1 == key2  # Should match after internal sorting


def test_autoscheme_cache_key_is_portable_across_model_paths():
    """A locally downloaded model should keep the same key when its parent directory changes."""
    from auto_round.auto_scheme.delta_loss import _autoscheme_cache_key

    kwargs = {
        "dataset": "pile-10k",
        "nsamples": 16,
        "seqlen": 256,
        "batch_size": 8,
        "quant_layer_names": ["layer.0"],
        "fixed_layer_scheme": {},
        "scheme": "W4A16",
        "force_mllm": False,
        "low_gpu_mem_usage": True,
    }

    assert _autoscheme_cache_key(model_name="/models/org/test-model", **kwargs) == _autoscheme_cache_key(
        model_name="/tmp/downloads/test-model", **kwargs
    )


def test_autoscheme_cache_key_normalizes_preset_scheme():
    """A preset name and its resolved scheme should identify the same scoring run."""
    from auto_round.auto_scheme.delta_loss import _autoscheme_cache_key
    from auto_round.schemes import preset_name_to_scheme

    kwargs = {
        "model_name": "test-model",
        "dataset": "pile-10k",
        "nsamples": 16,
        "seqlen": 256,
        "batch_size": 8,
        "quant_layer_names": ["layer.0"],
        "fixed_layer_scheme": {},
        "force_mllm": False,
        "low_gpu_mem_usage": True,
    }

    assert _autoscheme_cache_key(scheme="W4A16", **kwargs) == _autoscheme_cache_key(
        scheme=preset_name_to_scheme("W4A16"), **kwargs
    )


def test_autoscheme_cache_key_changes_only_with_scoring_config():
    """Execution settings that affect scoring invalidate the cache; bit accounting does not."""
    from auto_round.auto_scheme.delta_loss import _autoscheme_cache_key

    kwargs = {
        "model_name": "test-model",
        "dataset": "pile-10k",
        "nsamples": 16,
        "seqlen": 256,
        "batch_size": 8,
        "quant_layer_names": ["layer.0"],
        "fixed_layer_scheme": {},
        "scheme": "W4A16",
        "force_mllm": False,
        "low_gpu_mem_usage": True,
    }
    baseline = _autoscheme_cache_key(**kwargs)

    assert baseline != _autoscheme_cache_key(**{**kwargs, "low_gpu_mem_usage": False})
    assert baseline != _autoscheme_cache_key(**{**kwargs, "need_weight_grad": True})


def test_get_next_scheme_bits_is_order_independent():
    from auto_round.auto_scheme.delta_loss import _get_next_scheme_bits

    schemes = [{"bits": 8}, {"bits": 4}, {"bits": 6}]

    assert _get_next_scheme_bits(schemes, [0, 1, 2], 5) == 6
    assert _get_next_scheme_bits(schemes, [2, 1, 0], 5) == 6
    assert _get_next_scheme_bits(schemes, [0, 1, 2], 8) is None


def test_refresh_cached_layer_bits_preserves_loss():
    import torch
    from torch import nn

    from auto_round.auto_scheme.delta_loss import _refresh_cached_layer_bits

    model = nn.Module()
    model.layer = nn.Linear(8, 4, bias=False)
    scheme = {"bits": 4, "group_size": 4, "sym": True, "data_type": "int", "act_bits": 16}
    cached_scores = {"layer": [999, 1.25]}

    with_overhead = _refresh_cached_layer_bits(model, ["layer"], {}, scheme, cached_scores, False)
    without_overhead = _refresh_cached_layer_bits(model, ["layer"], {}, scheme, cached_scores, True)

    assert with_overhead["layer"][0] > without_overhead["layer"][0]
    assert without_overhead["layer"] == [model.layer.weight.numel() * 4, 1.25]
    assert cached_scores == {"layer": [999, 1.25]}


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
    cache_config = {"model_id": "test-model", "scheme": scheme_dict}

    _save_autoscheme_scores(
        cache_path,
        cache_key,
        0,
        scheme_dict,
        layer_scores,
        total_loss,
        total_params,
        cache_config,
    )

    loaded = _load_autoscheme_scores(cache_path)
    assert loaded is not None
    assert loaded["layer_scores"] == layer_scores
    assert loaded["total_loss_for_scheme"] == total_loss
    assert loaded["total_params"] == total_params


def test_find_compatible_downloaded_autoscheme_cache(tmp_path):
    """A compatible downloaded cache should work without renaming it to the locally computed key."""
    from auto_round.auto_scheme.delta_loss import (
        _autoscheme_cache_config,
        _find_compatible_autoscheme_cache,
        _save_autoscheme_scores,
    )
    from auto_round.schemes import preset_name_to_scheme

    quant_layer_names = ["layer.0", "layer.1"]
    cache_config = _autoscheme_cache_config(
        model_name="/models/test-model",
        dataset="pile-10k",
        nsamples=16,
        seqlen=256,
        batch_size=8,
        quant_layer_names=quant_layer_names,
        fixed_layer_scheme={},
        scheme="W4A16",
        force_mllm=False,
        low_gpu_mem_usage=True,
    )
    downloaded_path = tmp_path / "downloaded-from-release.json"
    layer_scores = {"layer.0": [4, 1.2], "layer.1": [4, 0.9]}
    _save_autoscheme_scores(
        downloaded_path,
        "key-from-another-commit",
        0,
        preset_name_to_scheme("W4A16").to_dict(),
        layer_scores,
        2.1,
        1000,
        cache_config=cache_config,
    )

    loaded = _find_compatible_autoscheme_cache(
        str(tmp_path / "scheme_00_current-key.json"),
        cache_config,
        quant_layer_names,
        {},
        1000,
    )

    assert loaded is not None
    assert loaded["layer_scores"] == layer_scores
    assert loaded["_cache_path"] == str(downloaded_path)
    assert (
        _find_compatible_autoscheme_cache(
            str(tmp_path / "scheme_00_current-key.json"),
            cache_config,
            quant_layer_names,
            {},
            999,
        )
        is None
    )


def test_autoscheme_cache_path_is_independent_from_workspace(tmp_path, monkeypatch):
    """AutoScheme caches default to the user cache and ignore AR_WORK_SPACE."""
    from auto_round.auto_scheme.delta_loss import _autoscheme_cache_path

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AR_WORK_SPACE", str(tmp_path / "workspace"))
    monkeypatch.delenv("AR_AUTO_SCHEME_CACHE", raising=False)

    cache_path = _autoscheme_cache_path("cache_key", 1)

    assert cache_path == str(tmp_path / "home" / ".cache" / "auto_round" / "scheme_01_cache_key.json")


def test_autoscheme_cache_path_can_be_overridden(tmp_path, monkeypatch):
    """AR_AUTO_SCHEME_CACHE overrides the default user cache directory."""
    from auto_round.auto_scheme.delta_loss import _autoscheme_cache_path

    cache_dir = tmp_path / "custom_cache"
    monkeypatch.setenv("AR_AUTO_SCHEME_CACHE", str(cache_dir))

    cache_path = _autoscheme_cache_path("cache_key", 2)

    assert cache_path == str(cache_dir / "scheme_02_cache_key.json")


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

    assert _assign_scheme_worker_devices(5, ["cuda:2", "cuda:5"]) == [
        "cuda:2",
        "cuda:5",
        "cuda:2",
        "cuda:5",
        "cuda:2",
    ]


def test_assign_scheme_worker_devices_shares_single_gpu():
    from auto_round.auto_scheme.delta_loss import _assign_scheme_worker_devices

    assert _assign_scheme_worker_devices(3, ["cuda:3"]) == ["cuda:3", "cuda:3", "cuda:3"]


def test_assign_scheme_worker_devices_rejects_no_gpu():
    from auto_round.auto_scheme.delta_loss import _assign_scheme_worker_devices

    with pytest.raises(ValueError, match="at least one"):
        _assign_scheme_worker_devices(1, [])


def test_scheme_worker_count_allows_workers_to_share_gpus():
    from auto_round.auto_scheme.delta_loss import _get_scheme_worker_count

    assert _get_scheme_worker_count(5, 1) == 5
    assert _get_scheme_worker_count(5, 2) == 5
    assert _get_scheme_worker_count(1, 4) == 1


def test_scheme_worker_count_rejects_no_gpu():
    from auto_round.auto_scheme.delta_loss import _get_scheme_worker_count

    with pytest.raises(ValueError, match="at least one GPU"):
        _get_scheme_worker_count(2, 0)


def test_parallel_scheme_scoring_supports_disk_stream_model():
    from auto_round.auto_scheme.delta_loss import _can_parallel_scheme_scoring

    assert _can_parallel_scheme_scoring(True, "local-model", 1, 2, False, True, False)


def test_parallel_scheme_scoring_rejects_disk_stream_vlm():
    from auto_round.auto_scheme.delta_loss import _can_parallel_scheme_scoring

    assert not _can_parallel_scheme_scoring(True, "local-model", 1, 2, False, True, True)


@pytest.mark.parametrize(
    "model_id,is_vlm,low_gpu_mem_usage,expected",
    [
        ("local-model", False, True, True),
        ("local-model", True, True, False),
        ("local-model", False, False, False),
        (None, False, True, False),
    ],
)
def test_prefer_disk_stream_scheme_worker(model_id, is_vlm, low_gpu_mem_usage, expected):
    from auto_round.auto_scheme.delta_loss import _prefer_disk_stream_scheme_worker

    assert _prefer_disk_stream_scheme_worker(model_id, is_vlm, low_gpu_mem_usage) is expected


def test_opt_scheme_worker_uses_low_cpu_memory_loading(monkeypatch):
    from test.helpers import opt_name_or_path

    from auto_round.auto_scheme import delta_loss

    captured = {}

    def fake_load_model(model_name, **kwargs):
        captured["model_name"] = model_name
        captured.update(kwargs)
        return ("model", "tokenizer", None, None, None, False, False)

    monkeypatch.setattr(delta_loss, "load_model", fake_load_model)

    result = delta_loss._load_scheme_worker_model(opt_name_or_path, True, True)

    assert result[0] == "model"
    assert captured == {
        "model_name": opt_name_or_path,
        "device": "cpu",
        "use_auto_mapping": False,
        "use_model_replacements": True,
        "low_cpu_mem_usage": True,
    }


def test_disk_stream_scheme_worker_builds_meta_model(monkeypatch):
    from auto_round.auto_scheme import delta_loss
    from auto_round.utils import disk_stream_util

    expected = (object(), object(), object())
    monkeypatch.setattr(disk_stream_util, "build_meta_model", lambda model_name: expected)

    assert delta_loss._load_disk_stream_scheme_worker_model("local-model") == expected


def test_disk_stream_scheme_worker_applies_model_replacements(monkeypatch):
    from auto_round import special_model_handler
    from auto_round.auto_scheme import delta_loss
    from auto_round.utils import disk_stream_util

    model = object()
    updated_model = object()
    handled_model = object()
    tokenizer = object()
    disk_index = object()
    calls = []

    monkeypatch.setattr(disk_stream_util, "build_meta_model", lambda model_name: (model, tokenizer, disk_index))

    def fake_update_module(input_model, formats, cleanup_original):
        calls.append(("update", input_model, formats, cleanup_original))
        return updated_model

    def fake_handle_special_model(input_model):
        calls.append(("handle", input_model))
        return handled_model

    monkeypatch.setattr(special_model_handler, "update_module", fake_update_module)
    monkeypatch.setattr(special_model_handler, "_handle_special_model", fake_handle_special_model)

    result = delta_loss._load_disk_stream_scheme_worker_model("local-model", use_model_replacements=True)

    assert result == (handled_model, tokenizer, disk_index)
    assert calls == [
        ("update", model, None, False),
        ("handle", updated_model),
    ]


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


def test_non_version_one_cache_is_rejected(tmp_path):
    import json

    from auto_round.auto_scheme.delta_loss import _load_autoscheme_scores

    cache_path = tmp_path / "scheme_00_old.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": 3,
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
