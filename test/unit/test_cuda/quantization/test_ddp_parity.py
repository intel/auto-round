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
"""CUDA-tier serial-vs-DDP result parity: RTN iters=0 and the default
SignRound tune at iters=2.

Both arms quantize the SAME tiny model (same fixture path -> identical
weights) with the SAME calibration samples. The DDP arm engages the
parallel lane via a world=2 ParallelPolicy on the compress context and receives the SAME samples per
replica through pinned shard draws: each iteration the serial lane
processes batch [a, b] while replica 0 draws [a] and replica 1 draws [b]
-- mean-of-shard-means equals the full-batch mean, so SignSGD consumes
identical gradients and both lanes evolve identically.

The final quantized weights are asserted BIT-EXACT (torch.equal). For
iters=0 this is guaranteed (the same deterministic per-module quantize on
identical weights). For iters=2 the two lanes are different floating-point
programs (batched reduction vs shard+average, bf16 gradient transport), so
bitwise equality is expected via the discrete sign steps + the final
round-to-grid; a mismatch is a real signal to investigate (transport
precision or reduction order), matched exactly.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2+ CUDA devices")

_DS = [
    "CUDA-tier DDP parity sample with enough tokens for quantization",
    "Another parity sample with a different length padding the batch",
    "Third sample keeps the calibration pool at four entries",
    "Fourth sample completes the pinned draw schedule",
]


class _FixedSampler:
    """Yields a fixed global-index schedule (list of lists) in order."""

    def __init__(self, draws):
        self._draws = list(draws)
        self._i = 0

    def next_batch(self):
        out = self._draws[self._i]
        self._i += 1
        return out


def _autoround(model_path, iters, disable_opt_rtn=False):
    from auto_round import AutoRound

    return AutoRound(
        model_path,
        bits=4,
        group_size=32,
        sym=False,
        iters=iters,
        nsamples=4,
        seqlen=8,
        batch_size=2,
        dataset=list(_DS),
        device_map="0",
        enable_torch_compile=False,
        disable_opt_rtn=disable_opt_rtn,
        disable_model_free=True,
    )


def _run(model_path, iters, disable_opt_rtn, monkeypatch, pin_draws, world=1):
    import auto_round.algorithms.parallel.tune_parallel as tp
    import auto_round.algorithms.quantization.sign_round.quantizer as v1
    from auto_round.algorithms.parallel.data_parallel import ParallelPolicy

    if pin_draws:
        # serial lane: batches [0, 1] then [2, 3]; DDP replicas draw the
        # SAME global indices per iteration, one shard sample each
        monkeypatch.setattr(
            tp, "shard_samplers", lambda ns, world, bpr: [_FixedSampler([[0], [2]]), _FixedSampler([[1], [3]])]
        )
        monkeypatch.setattr(v1, "IndexSampler", lambda nsamples, batch: _FixedSampler([[0, 1], [2, 3]]))
    ar = _autoround(model_path, iters, disable_opt_rtn)
    if world > 1:
        # the policy is set on the compress context (the CLI does it the same way)
        ar.compress_context.parallel_policy = ParallelPolicy(world=world)
    model, layer_config = ar.quantize()
    sd = model.state_dict()
    state = {}
    for name, cfg in layer_config.items():
        if cfg.get("data_type") != "int":
            continue
        key = name + ".weight"
        assert key in sd, f"quantized layer {name} has no weight in state_dict"
        state[name] = sd[key].detach().cpu().clone()
    assert state, "no quantized layers found"
    return state


def _assert_same_weights(serial, ddp):
    assert set(serial) == set(ddp), "quantized layer sets differ"
    for name in serial:
        assert torch.equal(serial[name], ddp[name]), (
            f"layer {name}: serial vs DDP final weights differ "
            f"(max abs diff {float((serial[name] - ddp[name]).abs().max())})"
        )


def test_rtn_iters0_serial_vs_ddp_bitexact(tiny_opt_model_path, monkeypatch):
    """Plain RTN (iters=0, disable_opt_rtn): the sharded per-module lane on
    resident mirrors must reproduce the serial lane bit-exactly."""
    serial = _run(tiny_opt_model_path, 0, True, monkeypatch, pin_draws=False)
    ddp = _run(tiny_opt_model_path, 0, True, monkeypatch, pin_draws=False, world=2)
    _assert_same_weights(serial, ddp)


def test_signround_iters2_serial_vs_ddp_bitexact(tiny_opt_model_path, monkeypatch):
    """Default SignRound tune at iters=2: pinned shard draws give every
    replica the serial lane's samples, so both lanes must land on identical
    final quantized weights."""
    serial = _run(tiny_opt_model_path, 2, False, monkeypatch, pin_draws=True)
    ddp = _run(tiny_opt_model_path, 2, False, monkeypatch, pin_draws=True, world=2)
    _assert_same_weights(serial, ddp)
