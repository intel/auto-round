from enum import Enum
from types import SimpleNamespace

import torch

from auto_round.export.export_to_llmcompressor.export import (
    _quant_args_signature,
    _rewrite_config_group_targets,
)
from auto_round.export.export_to_llmcompressor.utils import (
    collect_fused_expert_modules,
    ignore_fused_experts_for_vllm,
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

    assert rewritten["config_groups"]["group_0"]["targets"] == ["first"]
    assert rewritten["config_groups"]["group_1"]["targets"] == ["second"]


def test_rewrite_config_group_targets_preserves_uniform_targets():
    config = {"config_groups": {"group_0": {"targets": ["Linear"]}}}

    assert _rewrite_config_group_targets(torch.nn.Module(), config) == config


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
    config = {
        "config_groups": {
            "group_0": {"weights": first_args, "input_activations": None, "targets": ["Linear"]},
            "group_1": {"weights": second_args, "input_activations": None, "targets": ["Linear"]},
        }
    }

    rewritten = _rewrite_config_group_targets(model, config)

    assert rewritten["config_groups"]["group_0"]["targets"] == ["moe.experts.0.w1", "RoutedExperts"]
    assert rewritten["config_groups"]["group_1"]["targets"] == ["dense"]


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

    assert config["ignore"] == ["re:.*lm_head", r"re:^mlp\.experts\.\d+\.(gate|up|down)_proj$"]
