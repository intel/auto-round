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
"""Tests for the small pure helpers in ``auto_round/calib_dataset.py``."""

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# register_dataset decorator
# ---------------------------------------------------------------------------
class TestRegisterDataset:
    def test_register_single_name(self):
        from auto_round.calib_dataset import CALIB_DATASETS, register_dataset

        @register_dataset("_test_ds_zzz_")
        class _StubDataset:
            pass

        try:
            assert "_test_ds_zzz_" in CALIB_DATASETS
            assert CALIB_DATASETS["_test_ds_zzz_"] is _StubDataset
        finally:
            CALIB_DATASETS.pop("_test_ds_zzz_", None)

    def test_register_multiple_names(self):
        from auto_round.calib_dataset import CALIB_DATASETS, register_dataset

        @register_dataset(["_a_", "_b_"])
        class _Dual:
            pass

        try:
            assert "_a_" in CALIB_DATASETS
            assert "_b_" in CALIB_DATASETS
        finally:
            CALIB_DATASETS.pop("_a_", None)
            CALIB_DATASETS.pop("_b_", None)

    def test_register_returns_class_unchanged(self):
        from auto_round.calib_dataset import CALIB_DATASETS, register_dataset

        @register_dataset("_return_check_")
        class _Returned:
            pass

        try:
            assert CALIB_DATASETS["_return_check_"] is _Returned
        finally:
            CALIB_DATASETS.pop("_return_check_", None)


# ---------------------------------------------------------------------------
# _make_map_fingerprint
# ---------------------------------------------------------------------------
class TestMakeMapFingerprint:
    def test_deterministic_for_same_inputs(self):
        from auto_round.calib_dataset import _make_map_fingerprint

        ds = MagicMock()
        ds._fingerprint = "fp_abc"
        tok = MagicMock()
        tok.name_or_path = "tok_xyz"

        fp1 = _make_map_fingerprint(ds, tok, 128, False, None)
        fp2 = _make_map_fingerprint(ds, tok, 128, False, None)
        assert fp1 == fp2

    def test_different_seqlen_changes_fingerprint(self):
        from auto_round.calib_dataset import _make_map_fingerprint

        ds = MagicMock()
        ds._fingerprint = "fp_abc"
        tok = MagicMock()
        tok.name_or_path = "tok_xyz"

        fp1 = _make_map_fingerprint(ds, tok, 128, False, None)
        fp2 = _make_map_fingerprint(ds, tok, 256, False, None)
        assert fp1 != fp2

    def test_different_system_prompt_changes_fingerprint(self):
        from auto_round.calib_dataset import _make_map_fingerprint

        ds = MagicMock()
        ds._fingerprint = "fp_abc"
        tok = MagicMock()
        tok.name_or_path = "tok_xyz"

        fp1 = _make_map_fingerprint(ds, tok, 128, False, None)
        fp2 = _make_map_fingerprint(ds, tok, 128, False, "you are a bot")
        assert fp1 != fp2

    def test_different_apply_chat_template_changes_fingerprint(self):
        from auto_round.calib_dataset import _make_map_fingerprint

        ds = MagicMock()
        ds._fingerprint = "fp_abc"
        tok = MagicMock()
        tok.name_or_path = "tok_xyz"

        fp1 = _make_map_fingerprint(ds, tok, 128, False, None)
        fp2 = _make_map_fingerprint(ds, tok, 128, True, None)
        assert fp1 != fp2

    def test_missing_dataset_fingerprint_falls_back(self):
        from auto_round.calib_dataset import _make_map_fingerprint

        ds = object()  # no _fingerprint
        tok = MagicMock()
        tok.name_or_path = "tok_xyz"

        # Should still produce a valid hash without raising
        fp = _make_map_fingerprint(ds, tok, 128, False, None)
        assert isinstance(fp, str)
        assert len(fp) == 64  # sha256 hex digest length

    def test_returns_sha256_hex(self):
        from auto_round.calib_dataset import _make_map_fingerprint

        ds = MagicMock()
        ds._fingerprint = "x"
        tok = MagicMock()
        tok.name_or_path = "y"

        fp = _make_map_fingerprint(ds, tok, 64, True, "sys")
        assert isinstance(fp, str)
        assert all(c in "0123456789abcdef" for c in fp)


# ---------------------------------------------------------------------------
# get_dataset_len
# ---------------------------------------------------------------------------
class TestGetDatasetLen:
    def test_supports_len_protocol(self):
        from auto_round.calib_dataset import get_dataset_len

        class _Sized:
            def __len__(self):
                return 7

        assert get_dataset_len(_Sized()) == 7

    def test_falls_back_to_iteration(self):
        from auto_round.calib_dataset import get_dataset_len

        class _Iterable:
            def __iter__(self):
                return iter([1, 2, 3, 4, 5])

        assert get_dataset_len(_Iterable()) == 5

    def test_empty_iterable(self):
        from auto_round.calib_dataset import get_dataset_len

        class _Empty:
            def __iter__(self):
                return iter([])

        assert get_dataset_len(_Empty()) == 0

    def test_list_input(self):
        from auto_round.calib_dataset import get_dataset_len

        assert get_dataset_len([1, 2, 3]) == 3


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------
class TestSelect:
    def test_yields_requested_indices(self):
        from auto_round.calib_dataset import select

        data = ["a", "b", "c", "d", "e"]
        result = list(select(data, [1, 3]))
        assert result == ["b", "d"]

    def test_stops_at_max_index(self):
        """Iteration should stop once max requested index is reached."""
        from auto_round.calib_dataset import select

        def _gen():
            yield "a"
            yield "b"
            yield "c"
            yield "d"

        # select only [0] -> should stop after first element
        result = list(select(_gen(), [0]))
        assert result == ["a"]

    def test_empty_indices_raises(self):
        """``select`` requires at least one index (max() on empty raises)."""
        from auto_round.calib_dataset import select

        data = ["a", "b", "c"]
        with pytest.raises(ValueError):
            list(select(data, []))

    def test_out_of_range_indices_ignored(self):
        from auto_round.calib_dataset import select

        data = ["a", "b"]
        # 5 is out of range and greater than max([1, 5])=5; the loop will stop
        # because idx > max(indices) hits when idx=3 > 5? No, 3 < 5.
        # So actually iteration continues until idx > 5.
        result = list(select(data, [5]))
        # The function stops once idx > max; for [5] that's after idx=5,
        # but data only has 2 elements, so the generator simply exhausts.
        assert result == []


# ---------------------------------------------------------------------------
# apply_chat_template_to_samples
# ---------------------------------------------------------------------------
class TestApplyChatTemplateToSamples:
    def test_with_string_samples(self):
        from auto_round.calib_dataset import apply_chat_template_to_samples

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<rendered text>"
        tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

        result = apply_chat_template_to_samples(
            ["hello", "world"],
            tokenizer,
            seqlen=8,
        )
        assert "input_ids" in result
        # apply_chat_template should have been called for each sample
        assert tokenizer.apply_chat_template.call_count == 2

    def test_with_dict_messages(self):
        from auto_round.calib_dataset import apply_chat_template_to_samples

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<rendered>"
        tokenizer.return_value = {"input_ids": [[1]]}

        # Each sample is a list of message dicts (multi-turn)
        samples = [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]]
        apply_chat_template_to_samples(samples, tokenizer, seqlen=4)
        # The messages should be passed as-is (not wrapped in a new dict)
        args, _ = tokenizer.apply_chat_template.call_args
        msgs_arg = args[0]
        assert len(msgs_arg) == 2
        assert msgs_arg[0]["role"] == "user"
        assert msgs_arg[1]["role"] == "assistant"

    def test_with_system_prompt(self):
        from auto_round.calib_dataset import apply_chat_template_to_samples

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<rendered>"
        tokenizer.return_value = {"input_ids": [[1]]}

        apply_chat_template_to_samples(["hello"], tokenizer, seqlen=4, system_prompt="you are helpful")
        args, _ = tokenizer.apply_chat_template.call_args
        msgs_arg = args[0]
        assert msgs_arg[0]["role"] == "system"
        assert msgs_arg[0]["content"] == "you are helpful"

    def test_fallback_when_template_fails(self):
        from auto_round.calib_dataset import apply_chat_template_to_samples

        tokenizer = MagicMock()
        # First call raises (the one with system prompt), second succeeds (fallback)
        tokenizer.apply_chat_template.side_effect = [
            Exception("template failed"),
            "<rendered without system>",
        ]
        tokenizer.return_value = {"input_ids": [[1]]}

        apply_chat_template_to_samples(["hello"], tokenizer, seqlen=4, system_prompt="system prompt")
        # Fallback call should have stripped the system role
        assert tokenizer.apply_chat_template.call_count == 2
        second_call_args = tokenizer.apply_chat_template.call_args_list[1]
        msgs_arg = second_call_args[0][0]
        assert all(m["role"] != "system" for m in msgs_arg)
