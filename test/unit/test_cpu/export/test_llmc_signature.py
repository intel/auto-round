import re
from enum import Enum
from types import SimpleNamespace

import torch

from auto_round.export.formats.compressed_tensors.integer import (
    _quant_args_signature,
    _rewrite_config_group_targets,
)
from auto_round.export.formats.compressed_tensors.utils import (
    _compact_llmcompressor_regex_config,
    assign_config_group_targets,
    collect_fused_expert_modules,
    compact_expert_targets,
    ignore_fused_experts_for_vllm,
    revert_quantization_config_names,
)


class _FakeQuantizationArgs:
    def __init__(self, values):
        self.values = values

    def model_dump(self, mode):
        assert mode == "json"
        return self.values


class _DynamicType(Enum):
    LOCAL = "local"


def test_quant_args_signature_uses_all_serialized_fields():
    shared = {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "group_size": None,
        "strategy": "block",
        "block_structure": [128, 128],
        "dynamic": _DynamicType.LOCAL,
        "actorder": None,
        "scale_dtype": "float8_e4m3fn",
        "zp_dtype": "float8_e4m3fn",
        "observer": "minmax",
        "observer_kwargs": {"eps": 1e-6, "reduce_range": False},
    }
    reordered = dict(reversed(list(shared.items())))
    reordered["observer_kwargs"] = {"reduce_range": False, "eps": 1e-6}

    assert _quant_args_signature(shared) == _quant_args_signature(reordered)
    assert _quant_args_signature(shared) == _quant_args_signature(_FakeQuantizationArgs(shared))

    changed_fields = {
        "block_structure": [64, 128],
        "dynamic": False,
        "actorder": "weight",
        "scale_dtype": "float16",
        "zp_dtype": "int8",
        "observer": "memoryless",
        "observer_kwargs": {"eps": 1e-5},
    }
    for field, value in changed_fields.items():
        changed = shared | {field: value}
        assert _quant_args_signature(shared) != _quant_args_signature(changed), field


def test_rewrite_config_group_targets_matches_complete_quant_args_signature():
    first_args = {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "block",
        "block_structure": [128, 128],
        "dynamic": False,
    }
    second_args = first_args | {"block_structure": [64, 128]}

    model = torch.nn.Module()
    model.first = torch.nn.Linear(1, 1)
    model.first.quantization_scheme = SimpleNamespace(weights=first_args, input_activations=None)
    model.second = torch.nn.Linear(1, 1)
    model.second.quantization_scheme = SimpleNamespace(weights=second_args, input_activations=None)

    config = {
        "config_groups": {
            "group_0": {"weights": first_args, "input_activations": None, "targets": ["Linear"]},
            "group_1": {"weights": second_args, "input_activations": None, "targets": ["Linear"]},
        }
    }

    rewritten = _rewrite_config_group_targets(model, config)

    assert rewritten["config_groups"]["group_0"]["targets"] == ["Linear"]
    assert rewritten["config_groups"]["group_1"]["targets"] == ["second"]


def test_rewrite_config_group_targets_keeps_class_target_for_largest_group():
    small_args = {"num_bits": 8, "type": "int"}
    large_args = {"num_bits": 4, "type": "int"}
    model = torch.nn.Module()
    model.small = torch.nn.Linear(1, 1)
    model.small.quantization_scheme = SimpleNamespace(weights=small_args, input_activations=None)
    for idx in range(3):
        layer = torch.nn.Linear(1, 1)
        layer.quantization_scheme = SimpleNamespace(weights=large_args, input_activations=None)
        setattr(model, f"large{idx}", layer)
    config = {
        "config_groups": {
            "group_0": {"weights": small_args, "input_activations": None, "targets": ["Linear"]},
            "group_1": {"weights": large_args, "input_activations": None, "targets": ["Linear"]},
        }
    }

    rewritten = _rewrite_config_group_targets(model, config)

    assert rewritten["config_groups"]["group_0"]["targets"] == ["small"]
    assert rewritten["config_groups"]["group_1"]["targets"] == ["Linear"]


def test_rewrite_config_group_targets_adds_routed_experts_to_moe_fallback():
    moe_args = {"num_bits": 4, "type": "int"}
    dense_args = {"num_bits": 8, "type": "int"}
    model = torch.nn.Module()
    model.moe = torch.nn.Module()
    model.moe.experts = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
    for expert in model.moe.experts:
        expert.gate_proj = torch.nn.Linear(1, 1)
        expert.gate_proj.quantization_scheme = SimpleNamespace(weights=moe_args, input_activations=None)
    model.dense = torch.nn.Linear(1, 1)
    model.dense.quantization_scheme = SimpleNamespace(weights=dense_args, input_activations=None)
    config = {
        "config_groups": {
            "group_0": {"weights": moe_args, "input_activations": None, "targets": ["Linear"]},
            "group_1": {"weights": dense_args, "input_activations": None, "targets": ["Linear"]},
        }
    }

    rewritten = _rewrite_config_group_targets(model, config)

    assert rewritten["config_groups"]["group_0"]["targets"] == ["Linear", "RoutedExperts"]
    assert rewritten["config_groups"]["group_1"]["targets"] == ["dense"]


def test_rewrite_config_group_targets_preserves_uniform_targets():
    config = {"config_groups": {"group_0": {"targets": ["Linear"]}}}

    assert _rewrite_config_group_targets(torch.nn.Module(), config) == config


def test_compact_llmcompressor_config_keeps_exact_paths_and_merges_regex_ignores():
    gate_ignores = [rf"re:.*model\.layers\.{layer}\.mlp\.gate.*" for layer in range(24)]
    targets = ["model.layers.1.a", "model.layers.1.b", "Linear"]
    config = {
        "config_groups": {"group_0": {"targets": list(targets)}},
        "ignore": gate_ignores + ["model.layers.3.a", "lm_head"],
    }

    compacted = _compact_llmcompressor_regex_config(config)

    assert compacted["config_groups"]["group_0"]["targets"] == targets
    assert compacted["ignore"][1:] == ["model.layers.3.a", "lm_head"]
    pattern = compacted["ignore"][0][3:]
    assert all(re.match(pattern, f"model.layers.{layer}.mlp.gate") for layer in range(24))
    assert not re.match(pattern, "model.layers.24.mlp.gate")


def test_compact_expert_targets_keeps_vllm_probe_names_and_exact_match_set():
    projections = ("gate_proj", "up_proj", "down_proj")
    experts = [
        f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
        for layer in range(8)
        for expert in range(60)
        for proj in projections
    ]
    partial = [f"model.layers.9.mlp.experts.{expert}.gate_proj" for expert in (1, 2, 3)]
    config = {"config_groups": {"group_0": {"targets": experts + partial + ["model.layers.0.self_attn.q_proj"]}}}

    targets = compact_expert_targets(config)["config_groups"]["group_0"]["targets"]

    probes = [f"model.layers.{layer}.mlp.experts.0.{proj}" for layer in range(8) for proj in projections]
    assert targets[: len(probes) + 1] == probes + ["model.layers.0.self_attn.q_proj"]
    regexes = [target[3:] for target in targets if target.startswith("re:")]
    assert len(regexes) == 2
    listed = set(experts + partial)
    candidates = listed | {
        f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
        for layer in range(12)
        for expert in range(64)
        for proj in projections + ("w1",)
    }
    assert {name for name in candidates if any(re.match(regex, name) for regex in regexes)} == listed


def test_rewrite_config_group_targets_adds_routed_experts_for_moe():
    first_args = {"num_bits": 4, "type": "int"}
    second_args = {"num_bits": 8, "type": "int"}
    model = torch.nn.Module()
    model.moe = torch.nn.Module()
    model.moe.experts = torch.nn.ModuleList([torch.nn.Module()])
    model.moe.experts[0].w1 = torch.nn.Linear(1, 1)
    model.moe.experts[0].w1.quantization_scheme = SimpleNamespace(weights=first_args, input_activations=None)
    model.dense = torch.nn.Linear(1, 1)
    model.dense.quantization_scheme = SimpleNamespace(weights=second_args, input_activations=None)
    model.dense2 = torch.nn.Linear(1, 1)
    model.dense2.quantization_scheme = SimpleNamespace(weights=second_args, input_activations=None)
    config = {
        "config_groups": {
            "group_0": {"weights": first_args, "input_activations": None, "targets": ["Linear"]},
            "group_1": {"weights": second_args, "input_activations": None, "targets": ["Linear"]},
        }
    }

    rewritten = _rewrite_config_group_targets(model, config)

    assert rewritten["config_groups"]["group_0"]["targets"] == ["moe.experts.0.w1", "RoutedExperts"]
    assert rewritten["config_groups"]["group_1"]["targets"] == ["Linear"]


def test_ignored_fused_experts_get_vllm_ignore_entry():
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.Module()
    model.mlp.experts.gate_up_proj = torch.nn.Parameter(torch.zeros(2, 4, 2))
    model.mlp.shared_experts = torch.nn.Linear(2, 2)
    fused = collect_fused_expert_modules(model)
    assert fused == ["mlp.experts"]

    config = {"config_groups": {"group_0": {"targets": ["dense", "RoutedExperts"]}}, "ignore": ["re:.*lm_head"]}
    config = ignore_fused_experts_for_vllm(config, config["ignore"] + fused, ["dense"])

    assert config["ignore"] == [
        "re:.*lm_head",
        "mlp.experts.0.gate_proj",
        "mlp.experts.0.up_proj",
        "mlp.experts.0.down_proj",
    ]


def test_assign_config_group_targets_uses_class_fallback_only_for_largest_group():
    group_layers = {
        "moe": [f"model.layers.0.mlp.experts.{idx}.gate_proj" for idx in range(3)],
        "attn": ["model.layers.0.self_attn.q_proj"],
    }

    assert assign_config_group_targets(group_layers, ["Linear"]) == {
        "moe": ["Linear", "RoutedExperts"],
        "attn": ["model.layers.0.self_attn.q_proj"],
    }
    assert assign_config_group_targets(group_layers) == group_layers
    assert assign_config_group_targets(group_layers, ["Linear"], fallback_key="attn") == {
        "moe": group_layers["moe"],
        "attn": ["Linear"],
    }


def test_revert_quantization_config_names_uses_checkpoint_namespace():
    def revert(name):
        return name.replace(".mlp.", ".block_sparse_moe.")

    config = {
        "config_groups": {"group_0": {"targets": ["Linear", "model.layers.0.mlp.experts.0.down_proj"]}},
        "ignore": [
            "model.layers.0.mlp.experts.0.gate_proj",
            r"re:.*model\.layers\.0\.mlp\.gate.*",
            r"re:.*mlp\.(gate|up).*",
        ],
    }

    config = revert_quantization_config_names(config, revert)

    assert config["config_groups"]["group_0"]["targets"] == [
        "Linear",
        "model.layers.0.block_sparse_moe.experts.0.down_proj",
    ]
    assert config["ignore"] == [
        "model.layers.0.block_sparse_moe.experts.0.gate_proj",
        r"re:.*model\.layers\.0\.block_sparse_moe\.gate.*",
        r"re:.*mlp\.(gate|up).*",
    ]
