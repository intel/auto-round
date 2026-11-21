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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from auto_round.utils.model import _is_fp8_model, llm_load_model


@pytest.mark.parametrize("expert_dtype", ["fp4", "fp8"])
def test_native_deepseek_v4_requests_dequantization(expert_dtype):
    config = SimpleNamespace(
        model_type="deepseek_v4", expert_dtype=expert_dtype, quantization_config={"quant_method": "fp8"}
    )
    with patch("transformers.AutoConfig.from_pretrained", return_value=config):
        assert _is_fp8_model("local-native-checkpoint")
        with (
            patch("transformers.AutoModelForCausalLM.from_pretrained") as load,
            patch("transformers.AutoTokenizer.from_pretrained"),
        ):
            llm_load_model("local-native-checkpoint")
        assert load.call_args.kwargs["quantization_config"].dequantize is True


def test_packed_fp4_values_survive_dequantize_and_unfuse():
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    from auto_round.modeling.fused_moe.moe_experts_interface import _unfuse_experts_weights_inplace

    if not hasattr(Fp8Dequantize, "_unpack_fp4"):
        pytest.skip("Transformers does not support native packed FP4 dequantization")
    # Every nibble, including negative values; non-unit E8M0 scales per 32 values.
    packed = torch.arange(256, dtype=torch.int16).to(torch.int8).reshape(1, 8, 32).repeat(2, 1, 1)
    scales = torch.tensor([126, 128], dtype=torch.uint8).expand(2, 8, 2).clone()
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    unsigned = packed.to(torch.long) & 255
    decoded = torch.stack((lut[unsigned & 15], lut[unsigned >> 4]), dim=-1).reshape(2, 8, 64)
    expected = decoded * torch.tensor([0.5, 2.0]).repeat_interleave(32)
    dequantizer = Fp8Dequantize(None)
    weight = dequantizer._dequantize_one(packed, scales, output_dtype=torch.bfloat16)
    torch.testing.assert_close(weight.float(), expected, rtol=0, atol=0)

    # Unfusion adds methods to the experts' class. Never mutate nn.Module
    # itself: that leaks expert-only __len__ into unrelated models/tests.
    class PackedExperts(nn.Module):
        pass

    experts = PackedExperts()
    experts.gate_up_proj = nn.Parameter(weight)
    experts.down_proj = nn.Parameter(torch.ones(2, 64, 4, dtype=torch.bfloat16))
    experts.is_transposed = False
    experts.num_experts = 2
    experts.has_bias = False
    experts.act_fn = nn.SiLU()
    assert _unfuse_experts_weights_inplace(experts, check_decorator=False)
    for i in range(2):
        expert = getattr(experts, str(i))
        assert expert.gate_proj.weight.shape == (4, 64)
        torch.testing.assert_close(expert.gate_proj.weight.float(), expected[i, :4], rtol=0, atol=0)
        torch.testing.assert_close(expert.up_proj.weight.float(), expected[i, 4:], rtol=0, atol=0)
