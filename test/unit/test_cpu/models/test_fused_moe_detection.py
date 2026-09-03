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
import transformers
from packaging import version

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
    assert set(BUILTIN_MODULES) == {"llama4", "deepseek_v2", "step3p5", "qwen3_omni_moe"}

