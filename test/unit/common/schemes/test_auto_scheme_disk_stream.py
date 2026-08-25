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

"""
Unit tests for AutoScheme's disk-streaming mode (AR_DISK_STREAM_MODEL).

Verifies that streaming per-block sensitivity scoring from disk (instead of
fully materializing the checkpoint on CPU RAM up front) produces the same
mixed-bit layer_config as the non-streaming baseline, and that the underlying
materialize/free primitives round-trip correctly.
"""

import os
import shutil

import pytest
import torch

from auto_round import AutoRound, AutoScheme


@pytest.fixture(autouse=True)
def _clean_disk_stream_env():
    # AR_DISK_STREAM_MODEL is read lazily by auto_round.envs; make sure a test
    # that sets it can't leak into whichever test runs next.
    previous = os.environ.get("AR_DISK_STREAM_MODEL")
    yield
    if previous is None:
        os.environ.pop("AR_DISK_STREAM_MODEL", None)
    else:
        os.environ["AR_DISK_STREAM_MODEL"] = previous

class TestAutoSchemeDiskStream:
    @pytest.fixture(autouse=True)
    def setup_save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def _gen_layer_config(self, model_name, target_bits=3.5):
        # iters=1 (the standard tuning loop) rather than iters=0 (RTN): RTN's
        # separate block-materialization path doesn't support disk streaming yet
        # and is unrelated to this PR, which only streams AutoScheme's own
        # sensitivity-scoring pass.
        scheme = AutoScheme(avg_bits=target_bits, options=("W2A16", "W4A16", "BF16"), nsamples=1)
        ar = AutoRound(model=model_name, scheme=scheme, iters=1, nsamples=1)
        _, layer_config = ar.quantize()
        return {name: cfg["bits"] for name, cfg in layer_config.items() if "bits" in cfg}

    def test_disk_stream_matches_baseline_layer_config(self, tiny_opt_model_path):
        """AR_DISK_STREAM_MODEL=1 must select the exact same per-layer bits as the
        non-streaming baseline -- streaming changes *how* weights are loaded during
        scoring, not the scores themselves."""
        os.environ.pop("AR_DISK_STREAM_MODEL", None)
        baseline_bits = self._gen_layer_config(tiny_opt_model_path)

        os.environ["AR_DISK_STREAM_MODEL"] = "1"
        streamed_bits = self._gen_layer_config(tiny_opt_model_path)

        assert streamed_bits == baseline_bits

    def test_disk_stream_default_off(self):
        """With AR_DISK_STREAM_MODEL unset, behavior must be the unstreamed default."""
        os.environ.pop("AR_DISK_STREAM_MODEL", None)
        from auto_round import envs

        assert envs.AR_DISK_STREAM_MODEL is False

class TestDiskStreamWorkerModelPrep:
    def test_disk_stream_build_unfuses_moe_modules(self, monkeypatch):
        """The disk-stream worker build must structurally unfuse fused MoE experts
        (the regular load pipeline does this via custom replacements; the
        meta-skeleton build skips them, leaving per-expert quant layers
        unresolvable)."""
        import auto_round.auto_scheme.delta_loss as dl

        seen = {}
        sentinel_model, sentinel_tokenizer, sentinel_index = object(), object(), object()

        def _fake_build_meta_model(model_name):
            return sentinel_model, sentinel_tokenizer, sentinel_index

        def _fake_handle_moe(model):
            seen["model"] = model
            return []

        monkeypatch.setattr("auto_round.utils.disk_stream_util.build_meta_model", _fake_build_meta_model)
        monkeypatch.setattr("auto_round.modeling.fused_moe.replace_modules._handle_moe_modules", _fake_handle_moe)

        model, _tokenizer, _index = dl._load_disk_stream_scheme_worker_model("dummy-model")

        assert model is sentinel_model
        assert seen["model"] is sentinel_model
