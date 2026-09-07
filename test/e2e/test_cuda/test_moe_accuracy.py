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

from __future__ import annotations

from test.helpers import get_model_path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skip_ci(reason="Real MoE accuracy checks are large and GPU-hungry"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for e2e MoE accuracy checks"),
]


def test_qwen_moe_nvfp4_accuracy(dataloader):
    free_gib = torch.cuda.mem_get_info()[0] / 1024**3
    if free_gib < 24:
        pytest.skip(f"only {free_gib:.1f} GiB free CUDA memory, need at least 24 GiB for Qwen1.5-MoE-A2.7B")

    model_name = get_model_path("Qwen/Qwen1.5-MoE-A2.7B")
    layer_config = {
        r"layers\.(?:[3-9]|1[0-9]|2[0-3])": {"bits": 16, "act_bits": 16},
    }

    autoround = AutoRound(
        model_name,
        scheme="nvfp4",
        iters=1,
        seqlen=3,
        nsamples=2,
        dataset=dataloader,
        layer_config=layer_config,
    )
    _, quantized_model_path = autoround.quantize_and_save(
        output_dir="tmp_qwen_moe_nvfp4", inplace=False, format="auto_round"
    )

    model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)

    from test.helpers import evaluate_accuracy

    evaluate_accuracy(model, tokenizer, threshold=0.49, batch_size=16, task="piqa", limit=10)
