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
"""Calibration memory failures must not change the requested dataset."""

import pytest
import torch

from auto_round.auto_scheme import delta_loss
from auto_round.compressors.mllm import dataset as mllm_dataset


@pytest.mark.parametrize("low_gpu_mem_usage", [False, True])
@pytest.mark.parametrize(
    "error_type, message, should_fallback",
    [
        (torch.OutOfMemoryError, "calibration failed", False),
        (MemoryError, "calibration failed", False),
        (RuntimeError, "CUDA out of memory", False),
        (RuntimeError, "Backend OUT OF MEMORY", False),
        (RuntimeError, "unsupported text inputs", True),
        (ValueError, "unsupported text inputs", True),
    ],
)
def test_calibration_memory_errors_do_not_change_dataset(
    monkeypatch, low_gpu_mem_usage, error_type, message, should_fallback
):
    model = torch.nn.Module()
    text_loader = [torch.ones(1, 2, dtype=torch.long)]
    fallback_calls = []
    error = error_type(message)

    monkeypatch.setattr(delta_loss, "get_block_names", lambda *args, **kwargs: [[]])
    monkeypatch.setattr(delta_loss, "get_dataloader", lambda *args, **kwargs: text_loader)

    def fail_forward(*args, **kwargs):
        raise error

    def multimodal_loader(**kwargs):
        fallback_calls.append(kwargs["dataset"])
        return [], None, None, None

    # Exercise both real calibration branches without allocating GPU memory.
    def low_gpu_forward(model, loader, **kwargs):
        if loader is text_loader:
            raise error

    monkeypatch.setattr(delta_loss, "model_forward", fail_forward)
    monkeypatch.setattr(delta_loss, "model_forward_low_gpu", low_gpu_forward)
    monkeypatch.setattr(mllm_dataset, "get_mllm_dataloader", multimodal_loader)
    kwargs = dict(
        model=model,
        tokenizer=None,
        quant_layer_names=[],
        fixed_layer_scheme={},
        dataset="NeelNanda/pile-10k",
        processor=object(),
        is_vlm=True,
        low_gpu_mem_usage=low_gpu_mem_usage,
    )
    if should_fallback:
        # Preserve the fallback for models that reject text-only inputs.
        assert delta_loss.get_score_for_scheme(**kwargs) == {}
        assert fallback_calls == ["liuhaotian/llava_conv_58k"]
    else:
        with pytest.raises(error_type) as caught:
            delta_loss.get_score_for_scheme(**kwargs)
        assert caught.value is error
        assert fallback_calls == []
