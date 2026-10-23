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
"""Mocked-dataset tests for the generic calibration loader.

These tests exercise the main feature path of ``_get_generic_dataset`` and its
integration through ``get_dataset`` without any network access.  They cover:

* explicit and default splits forwarded to ``load_dataset``
* automatic text-field detection (non-streaming and streaming)
* message-record extraction via templates / fields
* short / empty-sample filtering
* streaming timeout + partial collection (and the empty-result error path)

All HuggingFace I/O is mocked, so the tests are deterministic and offline.
"""

import time

import datasets
import pytest
from datasets import Dataset

from auto_round import calib_dataset as cd

# Writing map/filter cache artifacts to disk is pointless for these tiny,
# fully in-memory fixtures and would risk cross-test cache collisions.
datasets.disable_caching()


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class FakeTokenizer:
    """Minimal tokenizer mimicking the HF ``__call__`` batch interface.

    Each character maps to its ordinal, so ``len(input_ids) == len(text)``
    (capped at ``max_length``).  This makes the downstream ``seqlen`` length
    filter predictable from the raw string length.
    """

    name_or_path = "fake-tokenizer"
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, texts, truncation=True, max_length=None, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        input_ids = []
        attention_mask = []
        for t in texts:
            ids = [ord(c) for c in t]
            if max_length is not None:
                ids = ids[:max_length]
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeIterableDataset:
    """Stand-in for a streaming ``IterableDataset``.

    Deliberately lacks ``column_names`` so the loader falls back to inspecting
    the first sample for field auto-detection (the streaming code path).
    """

    def __init__(self, samples):
        self._samples = list(samples)

    def shuffle(self, seed=None):  # order-preserving for deterministic asserts
        return self

    def __iter__(self):
        return iter(self._samples)


class SlowIterableDataset:
    """Streaming dataset that stalls partway through to trigger a timeout."""

    def __init__(self, quick, slow, delay):
        self._quick = list(quick)
        self._slow = list(slow)
        self._delay = delay

    def shuffle(self, seed=None):
        return self

    def __iter__(self):
        yield from self._quick
        time.sleep(self._delay)
        yield from self._slow


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def _make_load_dataset(return_value, recorded_calls=None):
    """Return a ``load_dataset`` replacement capturing its call kwargs."""

    def _fake_load_dataset(*args, **kwargs):
        if recorded_calls is not None:
            recorded_calls.append({"args": args, "kwargs": kwargs})
        return return_value

    return _fake_load_dataset


# ---------------------------------------------------------------------------
# Splits forwarded to load_dataset
# ---------------------------------------------------------------------------
class TestSplits:
    def test_explicit_split_forwarded(self, tokenizer, monkeypatch):
        calls = []
        table = Dataset.from_dict({"text": ["hello world foo", "bar baz qux text"]})
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(table, calls))

        cd._get_generic_dataset(
            tokenizer,
            seqlen=4,
            dataset_name="my/ds",
            split=["validation"],
            streaming=False,
        )

        assert calls, "load_dataset was never called"
        # A list split is normalized to a single HF split string before loading.
        assert calls[0]["kwargs"]["split"] == "validation"

    def test_combined_splits_forwarded(self, tokenizer, monkeypatch):
        calls = []
        table = Dataset.from_dict({"text": ["hello world foo", "bar baz qux text"]})
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(table, calls))

        cd._get_generic_dataset(
            tokenizer,
            seqlen=4,
            dataset_name="my/ds",
            split=["train", "validation"],
            streaming=False,
        )

        # Multiple requested splits are combined into HF's "train+validation" form.
        assert calls[0]["kwargs"]["split"] == "train+validation"

    def test_default_split_selects_train(self, tokenizer, monkeypatch):
        calls = []
        # No split requested: load_dataset receives split=None and returns a
        # split dictionary, from which the loader selects the "train" split.
        split_dict = datasets.DatasetDict(
            {
                "train": Dataset.from_dict({"text": ["hello world foo", "bar baz qux text"]}),
                "validation": Dataset.from_dict({"text": ["ignored sample here"]}),
            }
        )
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(split_dict, calls))

        cd._get_generic_dataset(
            tokenizer,
            seqlen=4,
            dataset_name="my/ds",
            split=None,
            streaming=False,
        )

        assert calls[0]["kwargs"]["split"] is None


# ---------------------------------------------------------------------------
# Split normalization / reduction helpers
# ---------------------------------------------------------------------------
class TestSplitHelpers:
    def test_normalize_split_none(self):
        assert cd._normalize_split(None) is None

    def test_normalize_split_string_passthrough(self):
        assert cd._normalize_split("train") == "train"

    def test_normalize_split_list_combined(self):
        assert cd._normalize_split(["train", "test"]) == "train+test"

    def test_normalize_split_empty_list(self):
        assert cd._normalize_split([]) is None

    def test_reduce_single_dataset_passthrough(self):
        table = Dataset.from_dict({"text": ["a"]})
        assert cd._reduce_to_single_dataset(table) is table

    def test_reduce_dict_prefers_train(self):
        train = Dataset.from_dict({"text": ["train row"]})
        split_dict = datasets.DatasetDict({"validation": Dataset.from_dict({"text": ["v"]}), "train": train})
        assert cd._reduce_to_single_dataset(split_dict) is train

    def test_reduce_dict_first_when_no_train(self):
        val = Dataset.from_dict({"text": ["v"]})
        split_dict = datasets.DatasetDict({"validation": val, "test": Dataset.from_dict({"text": ["t"]})})
        assert cd._reduce_to_single_dataset(split_dict) is val

    def test_reduce_list_combines(self):
        a = Dataset.from_dict({"text": ["a"]})
        b = Dataset.from_dict({"text": ["b"]})
        combined = cd._reduce_to_single_dataset([a, b])
        assert len(combined) == 2


# ---------------------------------------------------------------------------
# trust_remote_code is passed through to load_dataset (default False)
# ---------------------------------------------------------------------------
class TestTrustRemoteCode:
    def test_defaults_to_false(self, tokenizer, monkeypatch):
        calls = []
        table = Dataset.from_dict({"text": ["hello world foo", "bar baz qux text"]})
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(table, calls))

        cd._get_generic_dataset(tokenizer, seqlen=4, dataset_name="my/ds", split="train", streaming=False)

        assert calls[0]["kwargs"]["trust_remote_code"] is False

    def test_forwarded_when_enabled(self, tokenizer, monkeypatch):
        calls = []
        table = Dataset.from_dict({"text": ["hello world foo", "bar baz qux text"]})
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(table, calls))

        cd._get_generic_dataset(
            tokenizer,
            seqlen=4,
            dataset_name="my/ds",
            split="train",
            streaming=False,
            trust_remote_code=True,
        )

        assert calls[0]["kwargs"]["trust_remote_code"] is True


# ---------------------------------------------------------------------------
# Auto-detection of the text field
# ---------------------------------------------------------------------------
class TestAutoDetection:
    def test_non_streaming_detects_priority_field(self, tokenizer, monkeypatch):
        # "content" wins over "title" per TEXT_FIELD_PRIORITY and is renamed
        # to "text" internally.
        table = Dataset.from_dict(
            {
                "title": ["t1", "t2"],
                "content": ["the quick brown fox", "lorem ipsum dolor sit"],
            }
        )
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(table))

        ds = cd._get_generic_dataset(
            tokenizer,
            seqlen=4,
            dataset_name="my/ds",
            streaming=False,
        )

        assert "text" in ds.column_names
        assert set(ds["text"]) == {"the quick brown fox", "lorem ipsum dolor sit"}

    def test_streaming_detects_and_collects(self, tokenizer, monkeypatch):
        # Keep the candidate pool == nsamples so collection stops promptly.
        monkeypatch.setenv("AR_CALIB_DATA_MULTIPLIER", "1")
        samples = [{"text": f"streaming sample number {i}"} for i in range(5)]
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(FakeIterableDataset(samples)))

        ds = cd._get_generic_dataset(
            tokenizer,
            seqlen=4,
            dataset_name="my/ds",
            streaming=True,
            nsamples=3,
        )

        assert len(ds) == 3
        for text in ds["text"]:
            assert text.startswith("streaming sample number")


# ---------------------------------------------------------------------------
# Message / chat records
# ---------------------------------------------------------------------------
class TestMessageRecords:
    def test_template_extracts_message_contents(self, tokenizer, monkeypatch):
        table = Dataset.from_dict(
            {
                "messages": [
                    [
                        {"role": "user", "content": "hello there"},
                        {"role": "assistant", "content": "general kenobi"},
                    ],
                    [
                        {"role": "user", "content": "what is two plus two"},
                        {"role": "assistant", "content": "the answer is four"},
                    ],
                ]
            }
        )
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(table))

        ds = cd._get_generic_dataset(
            tokenizer,
            seqlen=4,
            dataset_name="my/chat-ds",
            template="{messages}",
            separator=" | ",
            streaming=False,
        )

        assert set(ds["text"]) == {
            "hello there | general kenobi",
            "what is two plus two | the answer is four",
        }

    def test_streaming_message_records(self, tokenizer, monkeypatch):
        monkeypatch.setenv("AR_CALIB_DATA_MULTIPLIER", "1")
        samples = [
            {
                "messages": [
                    {"role": "user", "content": f"question {i} goes here"},
                    {"role": "assistant", "content": f"response {i} goes here"},
                ]
            }
            for i in range(3)
        ]
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(FakeIterableDataset(samples)))

        ds = cd._get_generic_dataset(
            tokenizer,
            seqlen=4,
            dataset_name="my/chat-ds",
            template="{messages}",
            separator=" ",
            streaming=True,
            nsamples=3,
        )

        assert len(ds) == 3
        for text in ds["text"]:
            assert "goes here" in text


# ---------------------------------------------------------------------------
# Empty / short sample filtering
# ---------------------------------------------------------------------------
class TestSampleFiltering:
    def test_empty_samples_filtered_non_streaming(self, tokenizer, monkeypatch):
        table = Dataset.from_dict({"text": ["a valid sentence here", "   ", "", "another valid line"]})
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(table))

        ds = cd._get_generic_dataset(
            tokenizer,
            seqlen=4,
            dataset_name="my/ds",
            streaming=False,
        )

        assert len(ds) == 2
        assert set(ds["text"]) == {"a valid sentence here", "another valid line"}

    def test_empty_samples_skipped_streaming(self, tokenizer, monkeypatch):
        monkeypatch.setenv("AR_CALIB_DATA_MULTIPLIER", "1")
        samples = [
            {"text": "keep this one"},
            {"text": "   "},
            {"text": ""},
            {"text": "keep this too"},
        ]
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(FakeIterableDataset(samples)))

        ds = cd._get_generic_dataset(
            tokenizer,
            seqlen=4,
            dataset_name="my/ds",
            streaming=True,
            nsamples=10,
        )

        assert set(ds["text"]) == {"keep this one", "keep this too"}


# ---------------------------------------------------------------------------
# Streaming timeout / partial collection
# ---------------------------------------------------------------------------
class TestStreamingTimeout:
    def test_timeout_yields_partial_collection(self, tokenizer, monkeypatch):
        # Two samples arrive instantly, then the stream stalls past the
        # timeout, so only the first two should be collected.
        quick = [{"text": "first good sample"}, {"text": "second good sample"}]
        slow = [{"text": "late sample one"}, {"text": "late sample two"}]
        ds_stream = SlowIterableDataset(quick, slow, delay=1.5)
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(ds_stream))

        ds = cd._get_generic_dataset(
            tokenizer,
            seqlen=4,
            dataset_name="my/ds",
            streaming=True,
            nsamples=100,
            timeout=0.5,
        )

        assert len(ds) == 2
        assert set(ds["text"]) == {"first good sample", "second good sample"}

    def test_no_valid_samples_raises(self, tokenizer, monkeypatch):
        samples = [{"text": ""}, {"text": "   "}, {"text": "\n\t"}]
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(FakeIterableDataset(samples)))

        with pytest.raises(ValueError, match="no valid samples"):
            cd._get_generic_dataset(
                tokenizer,
                seqlen=4,
                dataset_name="my/ds",
                streaming=True,
                nsamples=10,
                timeout=30,
            )


# ---------------------------------------------------------------------------
# Integration through get_dataset (in-process, generic-loader fallback)
# ---------------------------------------------------------------------------
class TestGetDatasetIntegration:
    @pytest.fixture(autouse=True)
    def _in_process(self, monkeypatch):
        # Run preprocessing in-process (no fork/pickle of the mock) and keep
        # the streaming candidate pool small.
        monkeypatch.setenv("AR_DISABLE_DATASET_SUBPROCESS", "1")
        monkeypatch.setenv("AR_CALIB_DATA_MULTIPLIER", "1")
        # Surface any error instead of silently falling back to a network fetch.
        monkeypatch.setattr(cd, "_fallback_to_fineweb_edu", lambda error, *a, **k: (_ for _ in ()).throw(error))

    def test_seqlen_short_samples_dropped(self, tokenizer, monkeypatch):
        # input_ids length == len(text) (truncated to seqlen); samples shorter
        # than seqlen are dropped by the integration-level length filter.
        samples = [
            {"text": "abcdefghij"},  # len 10 -> kept
            {"text": "klmnopqrst"},  # len 10 -> kept
            {"text": "ab"},  # len 2  -> dropped (< seqlen)
            {"text": "cd"},  # len 2  -> dropped (< seqlen)
        ]
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(FakeIterableDataset(samples)))

        ds = cd.get_dataset(
            tokenizer,
            seqlen=5,
            dataset_name="pytest/generic-ds",
            nsamples=10,
        )

        assert len(ds) == 2

    def test_template_spec_routes_to_generic_loader(self, tokenizer, monkeypatch):
        samples = [
            {
                "messages": [
                    {"role": "user", "content": "abcdefghij"},
                    {"role": "assistant", "content": "klmnopqrst"},
                ]
            }
            for _ in range(3)
        ]
        monkeypatch.setattr(cd, "load_dataset", _make_load_dataset(FakeIterableDataset(samples)))

        ds = cd.get_dataset(
            tokenizer,
            seqlen=5,
            dataset_name="pytest/chat-ds:template={messages}",
            nsamples=10,
        )

        assert len(ds) >= 1
        assert "input_ids" in ds.column_names
