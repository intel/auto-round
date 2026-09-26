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
from auto_round.utils.disk_stream_util import build_meta_model, free_module, materialize_module, total_resident_bytes


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


class TestDiskStreamUtilRoundTrip:
    """Tests for the materialize/free primitives directly, independent of AutoScheme."""

    def test_materialize_then_free_round_trip(self, tiny_opt_model_path):
        model, _tokenizer, index = build_meta_model(tiny_opt_model_path)
        block = model.model.decoder.layers[0]

        for _, tensor in list(block.named_parameters()):
            assert str(tensor.device) == "meta"

        materialize_module(block, "model.decoder.layers.0", index, device="cpu")
        for _, tensor in list(block.named_parameters()):
            assert str(tensor.device) != "meta"
        assert total_resident_bytes(block) > 0

        free_module(block)
        for _, tensor in list(block.named_parameters()):
            assert str(tensor.device) == "meta"
        assert total_resident_bytes(block) == 0
