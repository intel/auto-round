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

"""CUDA regressions for QAD low-precision gradients and AMP master weights."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from auto_round.algorithms.composer import AlgorithmComposer
from auto_round.algorithms.quantization.qad import quantizer as qad_module
from auto_round.algorithms.quantization.qad.config import QADConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("use_triton", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("full_mode", [False, True])
def test_cuda_low_precision_gradient_matches_fp32(monkeypatch, use_triton, dtype, full_mode):
    if use_triton and not qad_module._HAS_TRITON:
        pytest.skip("Triton unavailable")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unavailable")
    monkeypatch.setattr(qad_module, "_HAS_TRITON", use_triton)
    logits = torch.zeros(2, 2, 2, dtype=dtype, device="cuda", requires_grad=True)
    target = torch.tensor([0.5001, 0.4999], device="cuda").expand(2, 2, 2).clone()
    indices = None if full_mode else torch.tensor([0, 1], device="cuda").expand(2, 2, 2)
    labels = torch.tensor([[0, 1], [-100, -100]], device="cuda")
    loss = qad_module._QADDistillLoss.apply(logits, indices, target, labels, 1.0, 1e-6, 1)
    (loss * 1024).backward()
    reference = logits.detach().float().requires_grad_()
    loss_ref = F.kl_div(F.log_softmax(reference[0, 0], -1), target[0, 0], reduction="sum")
    (loss_ref * 1024).backward()
    torch.testing.assert_close(logits.grad, reference.grad.to(dtype), atol=1e-5, rtol=0.01)
    assert logits.grad[0, 0].abs().min() > 0


class _HalfLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=8)
        self.embedding = torch.nn.Embedding(8, 8)
        self.proj = torch.nn.Linear(8, 8, bias=False)
        for key, value in dict(bits=4, group_size=8, sym=True, data_type="int", act_bits=16).items():
            setattr(self.proj, key, value)
        self.proj.scale_dtype = torch.float32

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.proj(self.embedding(input_ids)))


@pytest.mark.parametrize("use_teacher", [False, True])
def test_fp16_adamw_amp_whole_model_tuning(tmp_path, monkeypatch, use_teacher):
    torch.manual_seed(42)
    monkeypatch.setattr(qad_module, "device_manager", SimpleNamespace(device="cuda"))
    model = _HalfLM().to(device="cuda", dtype=torch.float16)
    original = model.embedding.weight.detach().clone()
    quantizer = qad_module.QADQuantizer(
        QADConfig(
            iters=2,
            lr=0.01,
            qad_optimizer="adamw",
            tune_non_quant_params=True,
            qad_kd_weight=int(use_teacher),
            qad_ce_weight=1,
            qad_workspace=str(tmp_path),
            topk=3,
            gradient_checkpointing=False,
            not_use_best_mse=True,
        )
    )
    quantizer.bind(
        SimpleNamespace(
            model_context=SimpleNamespace(amp=True),
            compress_context=SimpleNamespace(enable_torch_compile=False),
            calibration_context=SimpleNamespace(batch_size=1),
        )
    )
    samples = [torch.tensor([1, 2, 3]), torch.tensor([3, 2, 1])]
    composer = object.__new__(AlgorithmComposer)
    composer._run_tuning(model, lambda: quantizer.quantize_model(model, samples, device="cuda"))
    assert model.embedding.weight.dtype == torch.float16
    assert not torch.equal(model.embedding.weight, original)
    assert torch.isfinite(model.embedding.weight).all()
