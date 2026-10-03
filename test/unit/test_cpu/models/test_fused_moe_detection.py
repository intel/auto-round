# Copyright (c) 2026 Intel Corporation
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
"""The meta-skeleton load path must only trigger for fused-3D MoE checkpoints.

Building on meta pays off exactly when ``transformers`` would stack the checkpoint's
per-expert 2D tensors into a fused 3D ``nn.Parameter`` that AutoRound then has to split
back apart. Everything else (dense models, MoE families with a dedicated AutoRound
replacement) must keep the ordinary load path, so the blast radius stays small.
"""

import pytest
import torch
import transformers
from packaging import version
from safetensors.torch import save_file

from auto_round import envs
from auto_round.context.model import ModelContext
from auto_round.modeling.fused_moe.moe_experts_interface import (
    _config_model_types,
    config_has_fused_moe_experts,
)
from auto_round.modeling.fused_moe.replace_modules import BUILTIN_MODULES

pytestmark = pytest.mark.skipif(
    version.parse(transformers.__version__) < version.parse("5.0.0"),
    reason="fused 3D MoE experts only exist in transformers>=5",
)


class _FakeConfig:
    """Duck-typed stand-in so the test does not depend on every config class existing."""

    def __init__(self, model_type, **sub_configs):
        self.model_type = model_type
        for name, value in sub_configs.items():
            setattr(self, name, value)


def _native_fused_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    save_file(
        {
            "model.layers.0.experts.gate_up_proj": torch.empty(4, 8, 4),
            "model.layers.0.experts.down_proj": torch.empty(4, 4, 4),
        },
        checkpoint_dir / "model.safetensors",
    )
    return checkpoint_dir


def _model_context_for_detection(checkpoint_dir):
    context = object.__new__(ModelContext)
    context.model = str(checkpoint_dir)
    context.disk_stream_model_dir = str(checkpoint_dir)
    return context


# Fused 3D experts: `from_pretrained` merges `experts.<i>.<proj>` into one parameter.
FUSED_MOE_TYPES = ["qwen3_moe", "qwen2_moe", "deepseek_v3", "glm4_moe", "phimoe", "mixtral"]
# Dense models have no expert conversion at all.
DENSE_TYPES = ["llama", "qwen2", "gemma2", "mistral"]
# The checkpoint already stores the fused 3D tensor, so there is no per-expert merge.
ALREADY_FUSED_CHECKPOINT_TYPES = ["qwen3_vl_moe", "gpt_oss"]


@pytest.mark.parametrize("model_type", FUSED_MOE_TYPES)
def test_fused_moe_families_are_detected(model_type):
    assert config_has_fused_moe_experts(_FakeConfig(model_type)) is True


@pytest.mark.parametrize("model_type", DENSE_TYPES)
def test_dense_models_are_not_detected(model_type):
    assert config_has_fused_moe_experts(_FakeConfig(model_type)) is False


@pytest.mark.parametrize("model_type", ALREADY_FUSED_CHECKPOINT_TYPES)
def test_checkpoints_without_a_per_expert_merge_are_not_detected(model_type):
    assert config_has_fused_moe_experts(_FakeConfig(model_type)) is False


def test_detection_reaches_nested_text_config():
    """A VLM registers the MoE rules on its text sub-config, not the top level."""
    vlm = _FakeConfig("qwen3_5_moe", text_config=_FakeConfig("qwen3_5_moe_text"))

    assert config_has_fused_moe_experts(_FakeConfig("qwen3_5_moe")) is False, "precondition: top level alone is inert"
    assert config_has_fused_moe_experts(vlm) is True
    assert "qwen3_5_moe_text" in _config_model_types(vlm)


def test_detection_uses_real_transformers_configs():
    """Same behaviour with genuine config objects, not just duck-typed ones."""
    from transformers import LlamaConfig, Qwen3MoeConfig

    assert config_has_fused_moe_experts(Qwen3MoeConfig()) is True
    assert config_has_fused_moe_experts(LlamaConfig()) is False


def test_families_with_a_dedicated_replacement_are_left_alone():
    """`ModelContext` excludes these; they drive their own memory-aware materialization."""
    expected = {
        "llama4",
        "deepseek_v2",
        "step3p5",
        "qwen3_omni_moe",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
    if version.parse(transformers.__version__) < version.parse("5.0.0"):
        # These two only get a dedicated replacement on the pre-5.0 linear_loop path.
        expected |= {"qwen3_vl_moe", "gpt_oss"}
    assert set(BUILTIN_MODULES) == expected


def test_native_fused_checkpoint_enables_meta_skeleton_without_merge_converter(tmp_path, monkeypatch):
    checkpoint_dir = _native_fused_checkpoint(tmp_path)
    context = _model_context_for_detection(checkpoint_dir)
    config = _FakeConfig("gemma4", text_config=_FakeConfig("gemma4_text"))
    monkeypatch.setattr(envs, "AR_DISK_STREAM_MODEL", False)
    monkeypatch.setattr(envs, "AR_DISABLE_META_LOAD", False)
    monkeypatch.setattr(envs, "AR_DEBUG_LAYER_NUM", None)

    assert config_has_fused_moe_experts(config) is False, "precondition: Gemma4 has no merge converter"
    assert context._should_use_meta_skeleton(config) is True
    assert context.disk_stream_model_dir == str(checkpoint_dir)


def test_nested_dedicated_replacement_prevents_native_fused_auto_meta_load(tmp_path, monkeypatch):
    checkpoint_dir = _native_fused_checkpoint(tmp_path)
    context = _model_context_for_detection(checkpoint_dir)
    config = _FakeConfig("wrapper", text_config=_FakeConfig("llama4"))
    monkeypatch.setattr(envs, "AR_DISK_STREAM_MODEL", False)
    monkeypatch.setattr(envs, "AR_DISABLE_META_LOAD", False)
    monkeypatch.setattr(envs, "AR_DEBUG_LAYER_NUM", None)

    assert context._should_use_meta_skeleton(config) is False


@pytest.mark.parametrize(
    "force_disk_stream,disable_meta_load,debug_layer_num,expected",
    [
        (True, True, 1, True),
        (False, True, None, False),
        (False, False, 1, False),
    ],
)
def test_meta_skeleton_override_precedence(
    tmp_path, monkeypatch, force_disk_stream, disable_meta_load, debug_layer_num, expected
):
    checkpoint_dir = _native_fused_checkpoint(tmp_path)
    context = _model_context_for_detection(checkpoint_dir)
    config = _FakeConfig("gemma4", text_config=_FakeConfig("gemma4_text"))
    monkeypatch.setattr(envs, "AR_DISK_STREAM_MODEL", force_disk_stream)
    monkeypatch.setattr(envs, "AR_DISABLE_META_LOAD", disable_meta_load)
    monkeypatch.setattr(envs, "AR_DEBUG_LAYER_NUM", debug_layer_num)

    assert context._should_use_meta_skeleton(config) is expected
