# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Characterization for the `layer-config-resolution` capability.

Locks the per-layer output of ``set_layer_config`` (the public entry point that
the ``_resolve_*`` / ``_apply_*`` helpers were split out of) against verbatim
pre-refactor baselines for the W4A16, mixed-precision, and GGUF paths, plus the
ignore-layer regex-export behavior. Baselines were harvested from HEAD=33b7df0e.
"""

import shutil
import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import save_file

from auto_round.compressors.utils import set_layer_config
from auto_round.schemes import PRESET_SCHEMES


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module() for _ in range(2)])
        for i, layer in enumerate(self.model.layers):
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(32, 32)
            layer.mlp = nn.Module()
            layer.mlp.up_proj = nn.Linear(32, 64)
        self.lm_head = nn.Linear(32, 100)


class _Router(nn.Module):
    def __init__(self, size: int = 32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size, size))


class _UnsupportedGateModel(nn.Module):
    def __init__(self, size: int = 32):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate = _Router(size)
        self.mlp.up_proj = nn.Linear(size, size)


def _layer_config_for(scheme_name, layer_config=None, name_or_path=None):
    model = _TinyModel()
    if name_or_path is not None:
        # GGUF path calls is_separate_lm_head(model), which reads model.name_or_path
        # and inspects a real on-disk safetensors file — a local dir is required to
        # avoid a network download.
        model.name_or_path = name_or_path
    scheme = PRESET_SCHEMES[scheme_name]
    # set_layer_config returns (layer_config, has_qlayer_outside_block, regex_config).
    return set_layer_config(
        model=model,
        layer_config=layer_config,
        default_scheme=scheme,
        default_scale_dtype=torch.float16,
        supported_types=(nn.Linear,),
        inner_supported_types=(),
        quant_block_list=None,
        quant_lm_head=False,
        enable_gguf_official_mixed=False,
        is_mllm=False,
    )


_COMMON_NON_LM_HEAD = {
    "act_bits": 16,
    "act_data_type": None,
    "act_dynamic": None,
    "act_group_size": None,
    "act_sym": None,
    "bits": 4,
    "data_type": "int",
    "group_size": 128,
    "in_blocks": True,
    "rotation_config": None,
    "scale_dtype": torch.float16,
    "super_bits": None,
    "super_group_size": None,
    "sym": True,
}

_LM_HEAD_FP16 = {
    "act_bits": 16,
    "act_data_type": None,
    "act_dynamic": None,
    "act_group_size": None,
    "act_sym": None,
    "bits": 16,
    "data_type": "fp",
    "fixed_by_user": True,
    "group_size": 128,
    "in_blocks": False,
    "rotation_config": None,
    "scale_dtype": torch.float16,
    "super_bits": None,
    "super_group_size": None,
    "sym": True,
}

EXPECTED_W4A16_BASELINE = (
    {
        "lm_head": dict(_LM_HEAD_FP16),
        "model.layers.0.self_attn.q_proj": {**_COMMON_NON_LM_HEAD, "fixed_by_user": False},
        "model.layers.0.mlp.up_proj": {**_COMMON_NON_LM_HEAD, "fixed_by_user": False},
        "model.layers.1.self_attn.q_proj": {**_COMMON_NON_LM_HEAD, "fixed_by_user": False},
        "model.layers.1.mlp.up_proj": {**_COMMON_NON_LM_HEAD, "fixed_by_user": False},
    },
    False,
    {},
)

EXPECTED_MIXED_BASELINE = (
    {
        "lm_head": dict(_LM_HEAD_FP16),
        "model.layers.0.self_attn.q_proj": {**_COMMON_NON_LM_HEAD, "bits": 8, "fixed_by_user": True},
        "model.layers.0.mlp.up_proj": {**_COMMON_NON_LM_HEAD, "fixed_by_user": False},
        "model.layers.1.self_attn.q_proj": {**_COMMON_NON_LM_HEAD, "fixed_by_user": False},
        "model.layers.1.mlp.up_proj": {**_COMMON_NON_LM_HEAD, "fixed_by_user": False},
    },
    False,
    {},
)

_GGUF_NON_LM_HEAD = {
    "act_bits": 16,
    "act_data_type": None,
    "act_dynamic": None,
    "act_group_size": None,
    "act_sym": None,
    "bits": 4,
    "data_type": "int_asym_dq",
    "fixed_by_user": False,
    "group_size": 32,
    "in_blocks": True,
    "rotation_config": None,
    "scale_dtype": torch.float16,
    "super_bits": 6,
    "super_group_size": 8,
    "sym": False,
}

EXPECTED_GGUF_BASELINE = (
    {
        "model.layers.0.self_attn.q_proj": dict(_GGUF_NON_LM_HEAD),
        "model.layers.0.mlp.up_proj": dict(_GGUF_NON_LM_HEAD),
        "model.layers.1.self_attn.q_proj": dict(_GGUF_NON_LM_HEAD),
        "model.layers.1.mlp.up_proj": dict(_GGUF_NON_LM_HEAD),
        "lm_head": {
            "act_bits": 16,
            "bits": 6,
            "data_type": "int_sym_dq",
            "embedding": "gguf:q6_k",
            "fixed_by_user": False,
            "group_size": 16,
            "lm_head": "gguf:q6_k",
            "mostly": "gguf:q6_k",
            "scale_dtype": torch.float16,
            "super_bits": 8,
            "super_group_size": 16,
            "sym": True,
        },
    },
    True,
    {},
)


def test_w4a16_plain_baseline():
    assert _layer_config_for("W4A16") == EXPECTED_W4A16_BASELINE


def test_mixed_precision_override_baseline():
    override = {"model.layers.0.self_attn.q_proj": {"bits": 8, "data_type": "int"}}
    assert _layer_config_for("W4A16", layer_config=override) == EXPECTED_MIXED_BASELINE


def test_gguf_q4_k_m_baseline():
    tmp_dir = tempfile.mkdtemp(prefix="ar_characterization_gguf_")
    try:
        # Empty safetensors file so is_separate_lm_head() resolves locally (tied head).
        save_file({}, f"{tmp_dir}/model.safetensors")
        result = _layer_config_for("GGUF:Q4_K_M", name_or_path=tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    assert result == EXPECTED_GGUF_BASELINE


def test_ignore_layer_unsupported_module_kept_as_regex_export_entry():
    # An ignore target on an unsupported module type must not land in layer_config
    # but must be preserved verbatim in regex_config for export.
    layer_config, _, regex_config = _layer_config_for("W4A16")  # warm-up sanity
    model = _UnsupportedGateModel()
    layer_config, _, regex_config = set_layer_config(
        model=model,
        layer_config={},
        default_scheme="W4A16",
        default_scale_dtype=torch.float16,
        supported_types=(nn.Linear,),
        inner_supported_types=(),
        ignore_layers="mlp.gate",
        quant_lm_head=False,
        enable_gguf_official_mixed=False,
        is_mllm=False,
    )
    assert "mlp.gate" not in layer_config
    assert regex_config["mlp.gate"]["bits"] == 16
    assert regex_config["mlp.gate"]["act_bits"] == 16
    assert regex_config["mlp.gate"]["data_type"] == "float"
    assert regex_config["mlp.gate"]["act_data_type"] == "float"
    assert regex_config["mlp.gate"]["fixed_by_user"] is True
