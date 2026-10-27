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
"""CUDA parity test for the NeUQI Triton sweep kernel against the eager reference.

The kernel rounds with round-half-to-even and keeps the first-minimum zero-point
rule, so selections must match the eager per-candidate sweep up to the usual
fp32 summation ties (fp64-oracle checked). Skipped on hosts without CUDA+triton.
"""

import pytest
import torch

from auto_round.data_type.neuqi import neuqi_search_scale_zero

triton_sweep = pytest.importorskip("auto_round_extension.triton.neuqi_sweep").neuqi_sweep_triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _oracle_loss_fp64(data, qw, scale, zp, maxq):
    r = torch.round(data.double() / scale.double()).clamp_(min=-maxq - 1, max=2 * maxq + 1)
    q = (r + zp.double()).clamp_(0, maxq)
    deq = scale.double() * (q - zp.double())
    loss = (deq - data.double()) ** 2
    if qw is not None:
        loss = loss * qw.double()
    return loss.sum(-1)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("g", [32, 64, 100])  # 100: non-power-of-two group size (masked)
def test_triton_sweep_matches_eager(bits, weighted, g, monkeypatch):
    gen = torch.Generator().manual_seed(bits * 13 + int(weighted) * 7 + g)
    n = 512
    data = torch.randn(n, g, generator=gen)
    data[:, :5] *= 90.0
    qw = torch.rand(n, g, generator=gen) + 0.1 if weighted else None
    data_c, qw_c = data.cuda(), (qw.cuda() if qw is not None else None)

    scale_ref, zp_ref = neuqi_search_scale_zero(data.clone(), bits, qw=qw, coarse_n=8, fine_n=3)

    import auto_round.data_type.neuqi as N
    from auto_round import envs

    monkeypatch.setenv("AR_NEUQI_BACKEND", "triton")  # envs reads os.getenv per access; setenv auto-restores
    saved_batch = N._zp_batch_wanted
    N._zp_batch_wanted = lambda dev: True  # batching is device-automatic now; force it for the parity run
    try:
        N._triton_checked, N._triton_sweep, N._triton_broken = True, triton_sweep, False
        N._fused_zp_fns, N._fused_zp_broken, N._batched_broken = {}, False, False
        scale_t, zp_t = neuqi_search_scale_zero(data_c, bits, qw=qw_c, coarse_n=8, fine_n=3)
    finally:
        N._zp_batch_wanted = saved_batch

    assert not N._triton_broken, "kernel raised; fallback engaged instead"
    scale_ref, zp_ref = scale_ref.cpu(), zp_ref.cpu()
    scale_t, zp_t = scale_t.cpu(), zp_t.cpu()

    mismatch = (zp_ref != zp_t).logical_or(scale_ref != scale_t).nonzero().flatten()
    maxq = 2**bits - 1
    for i in mismatch.tolist():
        loss_ref = _oracle_loss_fp64(data[i], qw[i] if qw is not None else None, scale_ref[i], zp_ref[i], maxq)
        loss_t = _oracle_loss_fp64(data[i], qw[i] if qw is not None else None, scale_t[i], zp_t[i], maxq)
        assert (loss_t - loss_ref).abs().item() <= 1e-4 * loss_ref.abs().item()
