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
Integration test for targeted block re-quantization (``to_quant_block_names``)
combined with disk streaming (``AR_DISK_STREAM_MODEL``).

Regression coverage for: restricting tuning to a block that isn't the first
one left every block *before* the target on the meta device, and the
"cache block inputs" calibration forward silently propagated that meta-ness
until it collided with a genuinely-materialized module -- "Tensor on device
meta is not on the expected device cpu!". A full (unrestricted) run never
hits this, since every block is already covered by quant_block_list in that
case.
"""

import os

import pytest

from auto_round import AutoRound


@pytest.fixture(autouse=True)
def _clean_disk_stream_env():
    previous = os.environ.get("AR_DISK_STREAM_MODEL")
    yield
    if previous is None:
        os.environ.pop("AR_DISK_STREAM_MODEL", None)
    else:
        os.environ["AR_DISK_STREAM_MODEL"] = previous


@pytest.fixture(scope="module")
def tiny_opt_3layer_model_path():
    from test.helpers import save_tiny_model

    path = save_tiny_model("facebook/opt-125m", "./tmp/tiny_opt_3layer_model_path", num_layers=3)
    yield path
    import shutil

    shutil.rmtree(path, ignore_errors=True)


class TestTargetedBlockCalibStream:
    def test_targeted_block_with_disk_streaming_does_not_crash(self, tiny_opt_3layer_model_path):
        """Restricting to_quant_block_names to the LAST of 3 blocks leaves
        blocks 0 and 1 meta-only (never in quant_block_list) while the
        calibration forward pass still needs to run through them to reach
        block 2 -- exactly the scenario that used to crash."""
        os.environ["AR_DISK_STREAM_MODEL"] = "1"

        ar = AutoRound(
            model=tiny_opt_3layer_model_path,
            scheme="W4A16",
            iters=1,
            nsamples=1,
            to_quant_block_names="model.decoder.layers.2",
        )
        _, layer_config = ar.quantize()

        quantized_layers = {name for name, cfg in layer_config.items() if "bits" in cfg}
        assert quantized_layers, "expected the target block's layers to be quantized"
        assert all(name.startswith("model.decoder.layers.2.") for name in quantized_layers)
        assert not any(
            name.startswith(("model.decoder.layers.0.", "model.decoder.layers.1.")) for name in quantized_layers
        )

    def test_targeted_block_without_disk_streaming_still_works(self, tiny_opt_3layer_model_path):
        """Baseline: the same targeted re-quantization without disk streaming
        never hit this bug (nothing is meta), so it must keep working too."""
        os.environ.pop("AR_DISK_STREAM_MODEL", None)

        ar = AutoRound(
            model=tiny_opt_3layer_model_path,
            scheme="W4A16",
            iters=1,
            nsamples=1,
            to_quant_block_names="model.decoder.layers.2",
        )
        _, layer_config = ar.quantize()

        quantized_layers = {name for name, cfg in layer_config.items() if "bits" in cfg}
        assert quantized_layers
        assert all(name.startswith("model.decoder.layers.2.") for name in quantized_layers)
