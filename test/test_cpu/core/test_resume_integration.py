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
Integration test for AR_RESUME_DIR: simulates a crash partway through the
tuning loop and verifies a fresh AutoRound run against the same resume
directory picks up from the first not-yet-completed block instead of
restarting from block 0.
"""

import os
from unittest import mock

import pytest

from auto_round import AutoRound
from auto_round.utils.resume import ResumeState


@pytest.fixture(autouse=True)
def _clean_resume_env():
    previous_resume_dir = os.environ.get("AR_RESUME_DIR")
    previous_disk_stream = os.environ.get("AR_DISK_STREAM_MODEL")
    yield
    for key, previous in (("AR_RESUME_DIR", previous_resume_dir), ("AR_DISK_STREAM_MODEL", previous_disk_stream)):
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


class TestResumeIntegration:
    def test_resume_skips_already_completed_blocks(self, tiny_opt_model_path, tmp_path):
        resume_dir = str(tmp_path / "resume")
        # AR_DISK_STREAM_MODEL keeps low_cpu_mem_usage active for this plain
        # dense (non-MoE) model -- without it, DataDrivenCompressor.quantize()
        # silently disables low_cpu_mem_usage, and a resumed process's
        # in-memory model would have meta/empty weights for blocks completed
        # by the prior (crashed) process.
        os.environ["AR_RESUME_DIR"] = resume_dir
        os.environ["AR_DISK_STREAM_MODEL"] = "1"

        original_mark_block_done = ResumeState.mark_block_done
        crashed_after = []

        def crash_after_first_block(self, block_name, q_input, input_ids):
            original_mark_block_done(self, block_name, q_input, input_ids)
            crashed_after.append(block_name)
            if len(crashed_after) == 1:
                raise RuntimeError("simulated crash")

        with mock.patch.object(ResumeState, "mark_block_done", crash_after_first_block):
            with pytest.raises(RuntimeError, match="simulated crash"):
                ar = AutoRound(model=tiny_opt_model_path, scheme="W4A16", iters=1, nsamples=1)
                ar.quantize()

        assert crashed_after == ["model.decoder.layers.0"]

        processed_on_resume = []

        def track_block(self, block_name, q_input, input_ids):
            original_mark_block_done(self, block_name, q_input, input_ids)
            processed_on_resume.append(block_name)

        with mock.patch.object(ResumeState, "mark_block_done", track_block):
            ar = AutoRound(model=tiny_opt_model_path, scheme="W4A16", iters=1, nsamples=1)
            _, layer_config = ar.quantize()

        # Only the block that never completed before the crash should be
        # (re-)processed -- block 0's already-durable result must be reused,
        # not redone.
        assert processed_on_resume == ["model.decoder.layers.1"]

        # The final layer_config still covers every quantized layer in both
        # blocks, proving the resumed run produced a complete result.
        quantized_layers = {name for name, cfg in layer_config.items() if "bits" in cfg}
        assert any(name.startswith("model.decoder.layers.0.") for name in quantized_layers)
        assert any(name.startswith("model.decoder.layers.1.") for name in quantized_layers)
