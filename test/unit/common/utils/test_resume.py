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

"""Unit tests for auto_round.utils.resume (crash/resume checkpointing, AR_RESUME_DIR)."""

import pytest
import torch

from auto_round.utils.resume import (
    ResumeState,
    compute_run_signature,
    layer_config_fingerprint,
)


class TestResumeState:
    def test_fresh_state_has_no_completed_blocks(self, tmp_path):
        state = ResumeState(str(tmp_path), "sig", ["b0", "b1", "b2"])
        assert state.resume_index == 0
        assert state.load_q_input() is None
        assert state.load_input_ids() is None

    def test_mark_block_done_advances_resume_index(self, tmp_path):
        state = ResumeState(str(tmp_path), "sig", ["b0", "b1", "b2"])
        state.mark_block_done("b0", q_input=torch.ones(2), input_ids=torch.zeros(2))
        assert state.resume_index == 1
        assert state.completed_blocks == ["b0"]

    def test_mark_block_done_out_of_order_raises(self, tmp_path):
        state = ResumeState(str(tmp_path), "sig", ["b0", "b1", "b2"])
        with pytest.raises(AssertionError):
            state.mark_block_done("b1", q_input=None, input_ids=torch.zeros(2))

    def test_q_input_and_input_ids_round_trip(self, tmp_path):
        state = ResumeState(str(tmp_path), "sig", ["b0", "b1"])
        q_input = {"a": torch.arange(4).reshape(2, 2)}
        input_ids = torch.arange(6).reshape(2, 3)
        state.mark_block_done("b0", q_input=q_input, input_ids=input_ids)

        # A fresh ResumeState pointed at the same dir with the same signature
        # picks the saved tensors back up, exactly like a resumed process would.
        resumed = ResumeState(str(tmp_path), "sig", ["b0", "b1"])
        assert resumed.resume_index == 1
        loaded_q_input = resumed.load_q_input()
        assert torch.equal(loaded_q_input["a"], q_input["a"])
        assert torch.equal(resumed.load_input_ids(), input_ids)

    def test_q_input_none_clears_any_previous_q_input_file(self, tmp_path):
        state = ResumeState(str(tmp_path), "sig", ["b0", "b1"])
        state.mark_block_done("b0", q_input=torch.ones(2), input_ids=torch.zeros(2))
        assert state.q_input_path.exists()

        # Simulate a second run reusing the dir where enable_quanted_input is
        # off (q_input legitimately None) -- the stale file must not linger
        # and be mistaken for this block's q_input on a later resume.
        state2 = ResumeState(str(tmp_path), "sig", ["b0", "b1"])
        state2.completed_blocks = []
        state2.mark_block_done("b0", q_input=None, input_ids=torch.zeros(2))
        assert not state2.q_input_path.exists()

    def test_mismatched_signature_starts_fresh(self, tmp_path):
        state = ResumeState(str(tmp_path), "sig-a", ["b0", "b1"])
        state.mark_block_done("b0", q_input=None, input_ids=torch.zeros(2))

        other = ResumeState(str(tmp_path), "sig-b", ["b0", "b1"])
        assert other.resume_index == 0
        assert other.completed_blocks == []

    def test_completed_blocks_not_a_prefix_starts_fresh(self, tmp_path):
        state = ResumeState(str(tmp_path), "sig", ["b0", "b1", "b2"])
        state.mark_block_done("b0", q_input=None, input_ids=torch.zeros(2))

        # Different block order for the same signature (e.g. block list changed
        # some other way the signature didn't capture) -- "b0" is no longer a
        # valid prefix of this new order, so the manifest must be distrusted.
        other = ResumeState(str(tmp_path), "sig", ["b1", "b0", "b2"])
        assert other.resume_index == 0
        assert other.completed_blocks == []

    def test_clear_removes_all_state(self, tmp_path):
        state = ResumeState(str(tmp_path), "sig", ["b0", "b1"])
        state.mark_block_done("b0", q_input=torch.ones(2), input_ids=torch.zeros(2))
        assert state.manifest_path.exists()

        state.clear()
        assert not state.manifest_path.exists()
        assert not state.q_input_path.exists()
        assert not state.input_ids_path.exists()

        fresh = ResumeState(str(tmp_path), "sig", ["b0", "b1"])
        assert fresh.resume_index == 0

    def test_full_completion_round_trip(self, tmp_path):
        block_names = ["b0", "b1", "b2"]
        state = ResumeState(str(tmp_path), "sig", block_names)
        for name in block_names:
            state.mark_block_done(name, q_input=None, input_ids=torch.zeros(2))
        assert state.resume_index == len(block_names)
        assert state.completed_blocks == block_names


class TestComputeRunSignature:
    def test_identical_inputs_produce_identical_signature(self):
        sig1 = compute_run_signature("m", "scheme", "dataset", 8, 2048, ["b0", "b1"])
        sig2 = compute_run_signature("m", "scheme", "dataset", 8, 2048, ["b0", "b1"])
        assert sig1 == sig2

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"model_dir": "other"},
            {"scheme_desc": "other"},
            {"dataset_desc": "other"},
            {"nsamples": 16},
            {"seqlen": 4096},
            {"block_names": ["b0", "b1", "b2"]},
        ],
    )
    def test_any_changed_field_changes_signature(self, kwargs):
        base = dict(
            model_dir="m",
            scheme_desc="scheme",
            dataset_desc="dataset",
            nsamples=8,
            seqlen=2048,
            block_names=["b0", "b1"],
        )
        sig1 = compute_run_signature(**base)
        sig2 = compute_run_signature(**{**base, **kwargs})
        assert sig1 != sig2

    def test_none_model_dir_does_not_crash(self):
        sig = compute_run_signature(None, "scheme", "dataset", 8, 2048, ["b0"])
        assert isinstance(sig, str) and len(sig) == 64  # sha256 hexdigest


class TestLayerConfigFingerprint:
    def test_empty_layer_config(self):
        assert layer_config_fingerprint(None) == "<no-layer-config>"
        assert layer_config_fingerprint({}) == "<no-layer-config>"

    def test_deterministic_regardless_of_dict_order(self):
        cfg_a = {"layer.1": {"bits": 4, "sym": True}, "layer.0": {"bits": 8, "sym": False}}
        cfg_b = {"layer.0": {"bits": 8, "sym": False}, "layer.1": {"bits": 4, "sym": True}}
        assert layer_config_fingerprint(cfg_a) == layer_config_fingerprint(cfg_b)

    def test_different_bits_produce_different_fingerprint(self):
        cfg_4bit = {"layer.0": {"bits": 4}}
        cfg_8bit = {"layer.0": {"bits": 8}}
        assert layer_config_fingerprint(cfg_4bit) != layer_config_fingerprint(cfg_8bit)

    def test_non_scalar_values_are_ignored(self):
        cfg_with_tensor = {"layer.0": {"bits": 4, "weight": torch.ones(2, 2)}}
        cfg_without_tensor = {"layer.0": {"bits": 4}}
        assert layer_config_fingerprint(cfg_with_tensor) == layer_config_fingerprint(cfg_without_tensor)
