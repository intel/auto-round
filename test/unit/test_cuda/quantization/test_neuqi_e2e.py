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
"""End-to-end NeUQI runs through the public AutoRound API on CUDA.

The CPU tier covers the search math and dispatch tables; these runs verify
the full wiring (config routing, imatrix collection, frozen-init anchor,
layer_config output) with the REAL backend ladder engaged -- on CUDA the
Triton/compile sweep paths serve the search instead of the eager fallback,
so default grids (256/64) run in seconds. Hosts without CUDA skip."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_DS = ["local NeUQI calibration sample with enough tokens for quantization"] * 2


def _autoround(model_path, sym, iters):
    from auto_round import AutoRound

    return AutoRound(
        model_path,
        bits=4,
        group_size=32,
        sym=sym,
        enable_neuqi=True,
        iters=iters,
        nsamples=2,
        seqlen=8,
        batch_size=2,
        dataset=list(_DS),
        device_map="cuda:0",
        enable_torch_compile=False,
    )


def test_zero_shot_neuqi_e2e(tiny_opt_model_path, tmp_path):
    ar = _autoround(tiny_opt_model_path, sym=False, iters=0)
    model, layer_config = ar.quantize()
    assert model is not None and len(layer_config) > 0
    for name, cfg in layer_config.items():
        assert cfg["bits"] == 4, f"Layer {name} expected bits=4, got {cfg['bits']}"
        assert cfg["data_type"] == "int", f"Layer {name} expected int, got {cfg.get('data_type')}"


@pytest.mark.parametrize("sym", [False, True])
def test_frozen_init_e2e(tiny_opt_model_path, monkeypatch, sym):
    """Both symmetry classes share the tuning wiring; only the anchor's
    internals (joint vs two-stage search) differ."""
    import auto_round.wrapper as wrapper_mod

    calls = []
    orig_anchor = wrapper_mod._neuqi_init_anchor

    def spy(weight, *a, **kw):
        calls.append(tuple(weight.shape))
        return orig_anchor(weight, *a, **kw)

    monkeypatch.setattr(wrapper_mod, "_neuqi_init_anchor", spy)
    ar = _autoround(tiny_opt_model_path, sym=sym, iters=1)
    model, layer_config = ar.quantize()
    assert model is not None and len(layer_config) > 0
    assert calls, f"the NeUQI frozen-init anchor never fired on the iters>0 path (sym={sym})"
