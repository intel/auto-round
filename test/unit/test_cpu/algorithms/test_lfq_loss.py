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

"""CPU tests for the LFQ memory-efficient soft cross-entropy.

The custom ``torch.autograd.Function`` behind ``soft_cross_entropy`` relies on
an analytic backward, row chunking, and valid-row normalization, so these tests
pin both the forward value and the gradient against a straightforward reference
implementation across the tricky cases:

* ``chunk_elems < vocab_size`` (forces the chunk loop to iterate),
* masked rows (some positions ignored),
* the all-masked case (no valid positions),
* 3-D ``[batch, seq, vocab]`` inputs that get flattened internally.
"""

import pytest
import torch

from auto_round.algorithms.quantization.sign_round.lfq_loss import soft_cross_entropy


def _reference_soft_ce(q_logits, fp_logits, valid_mask=None):
    """Naive, readable soft cross-entropy used as the ground truth."""
    vocab = q_logits.shape[-1]
    q2d = q_logits.reshape(-1, vocab)
    fp2d = fp_logits.reshape(-1, vocab).detach()

    log_q = torch.log_softmax(q2d.double(), dim=-1)
    p_fp = torch.softmax(fp2d.double(), dim=-1)
    ce = -(p_fp * log_q).sum(dim=-1)  # [N]

    if valid_mask is None:
        return ce.mean()
    m = valid_mask.reshape(-1).to(ce.dtype)
    return (ce * m).sum() / m.sum().clamp_min(1.0)


def _assert_matches(q, fp, mask, chunk_elems):
    """Compare loss + grad of the chunked impl against the reference."""
    q_test = q.detach().clone().requires_grad_(True)
    loss = soft_cross_entropy(q_test, fp, mask, chunk_elems=chunk_elems)
    loss.backward()

    q_ref = q.detach().clone().requires_grad_(True)
    ref = _reference_soft_ce(q_ref, fp, mask)
    ref.backward()

    assert torch.allclose(loss.double(), ref, atol=1e-6, rtol=1e-5), f"loss mismatch: {loss.item()} vs {ref.item()}"
    assert torch.allclose(
        q_test.grad.double(), q_ref.grad, atol=1e-6, rtol=1e-5
    ), f"grad max diff: {(q_test.grad.double() - q_ref.grad).abs().max().item()}"


@pytest.mark.parametrize("chunk_elems", [1 << 23, 60, 20])  # full, 3-row, 1-row chunks
def test_soft_ce_unmasked_chunking(chunk_elems):
    torch.manual_seed(0)
    n, v = 11, 20
    q = torch.randn(n, v, dtype=torch.float64)
    fp = torch.randn(n, v, dtype=torch.float64)
    _assert_matches(q, fp, None, chunk_elems)


@pytest.mark.parametrize("chunk_elems", [1 << 23, 20])
def test_soft_ce_masked_rows(chunk_elems):
    torch.manual_seed(1)
    n, v = 9, 20
    q = torch.randn(n, v, dtype=torch.float64)
    fp = torch.randn(n, v, dtype=torch.float64)
    mask = torch.tensor([1, 1, 0, 1, 0, 0, 1, 1, 0], dtype=torch.bool)
    _assert_matches(q, fp, mask, chunk_elems)


def test_soft_ce_all_masked():
    """All positions ignored → finite (zero) loss and zero gradients, no NaN."""
    torch.manual_seed(2)
    n, v = 5, 16
    q = torch.randn(n, v, dtype=torch.float64, requires_grad=True)
    fp = torch.randn(n, v, dtype=torch.float64)
    mask = torch.zeros(n, dtype=torch.bool)

    loss = soft_cross_entropy(q, fp, mask, chunk_elems=40)
    loss.backward()

    assert torch.isfinite(loss), "loss must be finite when all rows are masked"
    assert float(loss) == pytest.approx(0.0, abs=1e-9)
    assert torch.allclose(q.grad, torch.zeros_like(q.grad))


def test_soft_ce_3d_inputs_flattened():
    """[batch, seq, vocab] inputs and a [batch, seq] mask are handled."""
    torch.manual_seed(3)
    b, s, v = 2, 4, 18
    q = torch.randn(b, s, v, dtype=torch.float64)
    fp = torch.randn(b, s, v, dtype=torch.float64)
    mask = torch.ones(b, s, dtype=torch.bool)
    mask[0, -1] = False  # ignore one position
    _assert_matches(q, fp, mask, chunk_elems=v * 2)


def test_soft_ce_fp_target_is_detached():
    """Gradients must not flow into the FP soft target."""
    torch.manual_seed(4)
    n, v = 6, 20
    q = torch.randn(n, v, dtype=torch.float64, requires_grad=True)
    fp = torch.randn(n, v, dtype=torch.float64, requires_grad=True)

    loss = soft_cross_entropy(q, fp, None, chunk_elems=40)
    loss.backward()

    assert q.grad is not None
    assert fp.grad is None, "FP target must be detached (no gradient)"


def test_soft_ce_matches_hard_onehot_reference():
    """With one-hot FP targets, soft CE collapses to standard CE vs those labels."""
    torch.manual_seed(5)
    n, v = 7, 20
    q = torch.randn(n, v, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, v, (n,))
    # Sharp FP logits → near one-hot soft target at ``labels``.
    fp = torch.full((n, v), -30.0, dtype=torch.float64)
    fp[torch.arange(n), labels] = 30.0

    loss = soft_cross_entropy(q, fp, None, chunk_elems=1 << 23)
    ref = torch.nn.functional.cross_entropy(q.double(), labels, reduction="mean")
    assert torch.allclose(loss.double(), ref.double(), atol=1e-5, rtol=1e-4)
