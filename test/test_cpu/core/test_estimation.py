import pytest

from auto_round.estimation import (
    _format_bytes,
    _format_time,
    discover_num_layers,
    dry_run_estimate,
    estimate_block_vram,
    estimate_output_size,
    estimate_parameter_count,
    estimate_time,
)
from auto_round.cli.parser import build_quantize_parser


class FakeConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _dense_config(**kwargs):
    values = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        "tie_word_embeddings": True,
    }
    values.update(kwargs)
    return FakeConfig(**values)


def test_estimate_block_vram_reuses_device_memory_helpers(monkeypatch):
    from auto_round.utils import device

    calls = {}

    def fake_estimate_tuning_block_mem(block, input_ids, batch_size):
        calls["block"] = block
        calls["input_shape"] = tuple(input_ids[0].shape)
        calls["batch_size"] = batch_size
        return {"self_attn.q_proj": {"param_memory": 0.25}}, 1.5, 2.0, 3.0

    def fake_get_moe_memory_ratio(block):
        calls["moe_block"] = block
        return 0.25, True

    monkeypatch.setattr(device, "estimate_tuning_block_mem", fake_estimate_tuning_block_mem)
    monkeypatch.setattr(device, "get_moe_memory_ratio", fake_get_moe_memory_ratio)

    estimate = estimate_block_vram(
        _dense_config(),
        4,
        model_dtype="float16",
        batch_size=2,
        seqlen=8,
        nsamples=4,
        low_gpu_mem_usage=False,
    )

    assert calls["input_shape"] == (4, 8, 16)
    assert calls["batch_size"] == 2
    assert calls["moe_block"] is calls["block"]
    assert estimate.card_0_used_gb == pytest.approx(6.5)
    assert estimate.block_param_gb == pytest.approx(0.25)
    assert estimate.effective_block_input_output_gb == pytest.approx(2.0)
    assert estimate.has_moe is True
    assert estimate.moe_memory_ratio == pytest.approx(0.25)


def test_low_gpu_mem_usage_excludes_block_input_output_cache():
    config = _dense_config()

    normal = estimate_block_vram(
        config,
        4,
        model_dtype="float16",
        batch_size=2,
        seqlen=8,
        nsamples=4,
        low_gpu_mem_usage=False,
    )
    low_mem = estimate_block_vram(
        config,
        4,
        model_dtype="float16",
        batch_size=2,
        seqlen=8,
        nsamples=4,
        low_gpu_mem_usage=True,
    )

    assert normal.block_input_output_gb > 0
    assert normal.card_0_used_gb == pytest.approx(
        normal.block_input_output_gb + normal.layer_activation_gb + normal.additional_gb
    )
    assert low_mem.effective_block_input_output_gb == 0
    assert low_mem.card_0_used_gb == pytest.approx(normal.card_0_used_gb - normal.block_input_output_gb)


def test_moe_estimate_uses_device_moe_memory_ratio():
    config = _dense_config(
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
    )

    estimate = estimate_block_vram(
        config,
        4,
        model_dtype="float16",
        batch_size=2,
        seqlen=8,
        nsamples=4,
    )

    assert estimate.has_moe is True
    assert estimate.moe_memory_ratio == pytest.approx(0.5)
    assert estimate.block_param_gb > 0


def test_layer_count_discovery_uses_fallback_fields_and_nested_configs():
    assert discover_num_layers(FakeConfig(n_layer=12)) == 12
    assert discover_num_layers(FakeConfig(text_config=FakeConfig(num_layers=7))) == 7
    assert discover_num_layers(FakeConfig(layer_types=["full", "sliding"], mtp_num_hidden_layers=1)) == 3


def test_parameter_count_uses_synthetic_block_not_required_num_hidden_layers():
    config = FakeConfig(
        n_embd=16,
        n_layer=3,
        n_head=4,
        n_vocab=64,
        n_inner=32,
    )

    param_count = estimate_parameter_count(config, scheme_bits=4, model_dtype="float16")

    assert param_count is not None
    assert param_count > 64 * 16


def test_dry_run_estimate_loads_config_only_and_reports_block_memory(monkeypatch):
    config = _dense_config()
    loaded = {}

    def fake_load_model_config(model_name, opts):
        loaded["model_name"] = model_name
        loaded["opts"] = opts
        return config

    monkeypatch.setattr("auto_round.estimation._load_model_config", fake_load_model_config)

    estimates = dry_run_estimate(
        "local-model",
        scheme_bits=4,
        group_size=128,
        model_dtype="float16",
        batch_size=2,
        seqlen=8,
        nsamples=4,
        iters=10,
        low_gpu_mem_usage=False,
    )

    assert loaded["model_name"] == "local-model"
    assert loaded["opts"]["trust_remote_code"] is True
    assert estimates["num_layers"] == 4
    assert estimates["peak_vram_bytes"] > 0
    assert estimates["block_input_output_cache_bytes"] > 0
    assert estimates["effective_block_input_output_cache_bytes"] == estimates["block_input_output_cache_bytes"]


def test_estimate_output_size_and_time_helpers():
    assert estimate_output_size(1_000, target_bits=4, group_size=128) > 500
    assert estimate_output_size(1_000, target_bits=4, group_size=0) == 500
    assert estimate_time(num_layers=4, iters=10, nsamples=8, batch_size=2) > 0
    assert "GB" in _format_bytes(2_000_000_000)
    assert "minutes" in _format_time(120)


def test_quantize_parser_accepts_dry_run_aliases():
    parser = build_quantize_parser()

    assert parser.parse_args(["local-model", "--dry-run"]).dry_run is True
    assert parser.parse_args(["local-model", "--dry_run"]).dry_run is True
