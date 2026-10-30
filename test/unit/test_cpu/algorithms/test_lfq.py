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

"""CPU tests for the LFQ mixin and memory-efficient soft cross-entropy.

The custom ``torch.autograd.Function`` behind ``soft_cross_entropy`` relies on
an analytic backward, row chunking, and valid-row normalization, so these tests
pin both the forward value and the gradient against a straightforward reference
implementation across the tricky cases:

* ``chunk_elems < vocab_size`` (forces the chunk loop to iterate),
* masked rows (some positions ignored),
* the all-masked case (no valid positions),
* 3-D ``[batch, seq, vocab]`` inputs that get flattened internally.
"""

from types import SimpleNamespace

import pytest
import torch

from auto_round.algorithms.quantization.sign_round.lfq import LFQMixin, soft_cross_entropy
from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer


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


@pytest.fixture
def lfq():
    """Tiny standalone LFQ host, without model downloads or quantizer setup."""
    torch.manual_seed(6)
    host = LFQMixin()
    host.model = SimpleNamespace(
        model=SimpleNamespace(norm=torch.nn.LayerNorm(4)),
        lm_head=torch.nn.Linear(4, 8, bias=False),
    )
    for module in (host.model.model.norm, host.model.lm_head):
        module.requires_grad_(False)
    host.enable_lfq = True
    host.enable_qad = False
    host.lfq_loss_type = "ce"
    host.lfq_top_k = 3
    host.lfq_temperature = 1.5
    host.lfq_num_blocks = 2
    return host


def test_quantizer_inherits_lfq_methods():
    assert issubclass(SignRoundQuantizer, LFQMixin)
    for name in (
        "_init_lm_components",
        "_is_text_decoder_block",
        "_lfq_is_kd_block",
        "_get_lfq_downstream_blocks",
        "_forward_lfq_downstream",
        "_lfq_logits",
        "lfq_loss",
        "_decoupled_topk_kd_loss",
    ):
        assert name not in SignRoundQuantizer.__dict__
        assert getattr(SignRoundQuantizer, name) is getattr(LFQMixin, name)


def test_lfq_logits_and_component_cache(lfq):
    hidden = torch.randn(2, 4, 4)
    expected = lfq.model.lm_head(lfq.model.model.norm(hidden))
    torch.testing.assert_close(lfq._lfq_logits(hidden), expected)
    assert lfq._post_block_modules == [lfq.model.model.norm]
    head = lfq._lm_head
    lfq.model.lm_head = torch.nn.Identity()
    lfq._init_lm_components()
    assert lfq._lm_head is head


def test_lfq_opt_post_block_order(lfq):
    norm = torch.nn.LayerNorm(4)
    projection = torch.nn.Linear(4, 6, bias=False)
    head = torch.nn.Linear(6, 8, bias=False)
    lfq.model = SimpleNamespace(
        model=SimpleNamespace(decoder=SimpleNamespace(final_layer_norm=norm, project_out=projection)),
        lm_head=head,
    )
    hidden = torch.randn(2, 3, 4)
    torch.testing.assert_close(lfq._lfq_logits(hidden), head(projection(norm(hidden))))
    assert lfq._post_block_modules == [norm, projection]


@pytest.mark.parametrize("loss_type", ["ce", "dtk", "rkl", "tk"])
def test_lfq_ce_and_missing_teacher_fallback(lfq, loss_type):
    lfq.lfq_loss_type = loss_type
    hidden = torch.randn(2, 4, 4, requires_grad=True)
    labels = torch.tensor([[1, 2, -100, 4], [3, 4, 5, 6]])
    logits = lfq._lfq_logits(hidden)
    expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1))
    loss = lfq.lfq_loss(hidden, labels)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("loss_type", ["dtk", "rkl", "tk"])
@pytest.mark.parametrize("all_masked", [False, True])
def test_lfq_kd_gradient_and_mask(lfq, loss_type, all_masked):
    lfq.lfq_loss_type = loss_type
    student = torch.randn(2, 4, 4, requires_grad=True)
    teacher = torch.randn(2, 4, 4, requires_grad=True)
    labels = torch.full((2, 4), -100) if all_masked else torch.tensor([[1, 2, -100, 4], [3, 4, 5, 6]])
    loss = lfq.lfq_loss(student, labels, teacher_hidden_state=teacher, sample_indices=[0, 1])
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None
    if all_masked:
        assert loss.item() == 0.0
        assert torch.count_nonzero(student.grad) == 0
    else:
        assert torch.count_nonzero(student.grad) > 0


@pytest.mark.parametrize(
    "enabled,qad,name,index,expected",
    [
        (False, False, "model.layers.3", 3, False),
        (True, False, "model.layers.1", 1, False),
        (True, False, "model.layers.2", 2, True),
        (True, False, "model.layers.3", 3, True),
        (True, False, "vision.layers.3", 3, False),
        (True, True, "model.layers.0", 0, True),
        (True, True, "audio.layers.3", 3, False),
    ],
)
def test_lfq_block_selection(lfq, enabled, qad, name, index, expected):
    lfq.enable_lfq = enabled
    lfq.enable_qad = qad
    block_ctx = SimpleNamespace(block_name=name, block_index=index, block_cnt=4)
    assert lfq._lfq_is_kd_block(block_ctx) is expected


def test_lfq_downstream_empty_passthrough(lfq):
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    assert lfq._forward_lfq_downstream(hidden, [], None, {}, [0], "cpu") is hidden


def test_quantizer_finalize_clears_lfq_cache():
    quantizer = SignRoundQuantizer.__new__(SignRoundQuantizer)
    for name in ("_cached_valid_token_mask", "_lm_head", "_post_block_modules"):
        setattr(quantizer, name, object())
    quantizer.finalize_run()
    for name in ("_cached_valid_token_mask", "_lm_head", "_post_block_modules"):
        assert not hasattr(quantizer, name)
    quantizer.finalize_run()
