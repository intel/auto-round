import re

from auto_round.utils.regex import compress_autoround_extra_config, compress_regex_patterns


def test_autoround_extra_config_keeps_exact_layer_keys_literal():
    layer_names = [f"model.layers.{layer}.self_attn.q_proj" for layer in range(60)]
    overrides = {name: {"bits": 4, "group_size": 32} for name in layer_names}
    overrides[r".*model\.layers\.1\..*"] = {"bits": 8}
    overrides[r".*model\.layers\.2\..*"] = {"bits": 8}

    compacted = compress_autoround_extra_config(overrides, set(layer_names))

    assert list(compacted)[: len(layer_names)] == layer_names
    regex_keys = [key for key in compacted if key not in layer_names]
    assert len(regex_keys) == 1
    assert re.search(regex_keys[0], "model.layers.1.mlp.gate")
    assert re.search(regex_keys[0], "model.layers.2.mlp.gate")
    assert not re.search(regex_keys[0], "model.layers.3.mlp.gate")


def test_autoround_extra_config_collapses_uniform_experts_into_container_key():
    experts = [
        f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
        for layer in range(2)
        for expert in range(4)
        for proj in ("gate_proj", "down_proj")
    ]
    layer_names = set(experts) | {"model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj"}
    overrides = {name: {"bits": 4} for name in experts}
    overrides["model.layers.1.mlp.experts.3.down_proj"] = {"bits": 8}
    overrides["model.layers.0.self_attn.q_proj"] = {"bits": 8}

    compacted = compress_autoround_extra_config(overrides, layer_names)

    assert compacted["model.layers.0.mlp.experts"] == {"bits": 4}
    assert not any(key.startswith("model.layers.0.mlp.experts.") for key in compacted)
    assert "model.layers.1.mlp.experts" not in compacted
    assert compacted["model.layers.1.mlp.experts.3.down_proj"] == {"bits": 8}
    assert compacted["model.layers.0.self_attn.q_proj"] == {"bits": 8}


def test_autoround_extra_config_skips_container_whose_search_matches_other_layers():
    experts = [f"model.layers.1.mlp.experts.{expert}.gate_proj" for expert in range(2)]
    other = "model.layers.1.mlp.experts_extra.0.gate_proj"
    overrides = {name: {"bits": 4} for name in experts}

    compacted = compress_autoround_extra_config(overrides, set(experts) | {other})

    assert list(compacted) == experts


def test_autoround_extra_config_merges_layer_index_regexes_with_same_config():
    overrides = {rf".*model\.layers\.{layer}\.mlp\.gate.*": {"bits": 4, "group_size": 32} for layer in range(24)}

    compacted = compress_autoround_extra_config(overrides, set())

    assert len(compacted) == 1
    pattern, config = next(iter(compacted.items()))
    assert config == {"bits": 4, "group_size": 32}
    assert all(re.search(pattern, f"model.layers.{layer}.mlp.gate") for layer in range(24))
    assert not re.search(pattern, "model.layers.24.mlp.gate")


def test_shared_regex_compressor_merges_prefixed_layer_index_patterns():
    patterns = [rf"re:.*model\.layers\.{layer}\.mlp\.gate.*" for layer in range(24)]

    compacted = compress_regex_patterns(patterns)

    assert len(compacted) == 1
    pattern = compacted[0][3:]
    assert all(re.match(pattern, f"model.layers.{layer}.mlp.gate") for layer in range(24))
    assert not re.match(pattern, "model.layers.24.mlp.gate")


def test_shared_regex_compressor_merges_autoround_gate_rules_without_changing_matches():
    patterns = [rf".*model\.layers\.{layer}\.mlp\.gate.*" for layer in range(24)]

    compacted = compress_regex_patterns(patterns)

    assert len(compacted) == 1
    pattern = compacted[0]
    assert all(re.search(pattern, f"model.layers.{layer}.mlp.gate") for layer in range(24))
    assert not re.search(pattern, "model.layers.24.mlp.gate")
    assert not re.search(pattern, "model.layers.2.self_attn.gate")


def test_shared_regex_compressor_does_not_merge_numbers_outside_path_segments():
    patterns = [rf"re:^module\.layers\.{layer}\.name[0-9]$" for layer in range(2)]

    compacted = compress_regex_patterns(patterns)

    assert len(compacted) == 1
    pattern = compacted[0][3:]
    assert re.match(pattern, "module.layers.0.name3")
    assert re.match(pattern, "module.layers.1.name8")
    assert not re.match(pattern, "module.layers.2.name3")


def test_autoround_extra_config_does_not_merge_regexes_with_different_quantization():
    overrides = {}
    for layer in range(12):
        overrides[rf".*model\.layers\.{layer}\.mlp\.gate.*"] = {"bits": 4}
    for layer in range(12, 24):
        overrides[rf".*model\.layers\.{layer}\.mlp\.gate.*"] = {"bits": 8}

    compacted = compress_autoround_extra_config(overrides, set())
    entries = list(compacted.items())

    assert len(entries) == 2
    for layer in range(24):
        matches = [
            (config, pattern) for pattern, config in entries if re.search(pattern, f"model.layers.{layer}.mlp.gate")
        ]
        assert len(matches) == 1
        assert matches[0][0] == {"bits": 4 if layer < 12 else 8}
