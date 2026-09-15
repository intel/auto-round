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
"""Tests for auto_round.utils.dataset_utils."""

import pytest

from auto_round.utils.dataset_utils import (
    CalibDataset,
    auto_detect_text_field,
    auto_detect_text_field_from_sample,
    build_dataset_spec,
    extract_text_from_sample,
    normalize_dataset_spec,
    parse_dataset_spec,
)


# ---------------------------------------------------------------------------
# auto_detect_text_field
# ---------------------------------------------------------------------------
class TestAutoDetectTextField:
    def test_single_column(self):
        """Single-column dataset should return that column."""

        class FakeDS:
            column_names = ["text"]

        assert auto_detect_text_field(FakeDS()) == "text"

    def test_priority_order(self):
        """'text' should be preferred over 'content'."""

        class FakeDS:
            column_names = ["content", "text", "id"]

        assert auto_detect_text_field(FakeDS()) == "text"

    def test_content_when_no_text(self):
        """'content' should be picked when 'text' is absent."""

        class FakeDS:
            column_names = ["content", "id"]

        assert auto_detect_text_field(FakeDS()) == "content"

    def test_prompt_field(self):
        """'prompt' should be detected when no higher-priority field exists."""

        class FakeDS:
            column_names = ["prompt", "id"]

        assert auto_detect_text_field(FakeDS()) == "prompt"

    def test_fallback_to_longest_string(self):
        """When no known field exists, pick the longest string column."""

        class FakeDS:
            column_names = ["short_col", "long_col", "id"]

            def __iter__(self):
                yield {"short_col": "hi", "long_col": "a much longer text here", "id": 1}
                yield {"short_col": "yo", "long_col": "another long text", "id": 2}

        assert auto_detect_text_field(FakeDS()) == "long_col"

    def test_no_text_field_raises(self):
        """Should raise ValueError when no text field is found."""

        class FakeDS:
            column_names = ["id", "score"]

            def __iter__(self):
                yield {"id": 1, "score": 0.5}

        with pytest.raises(ValueError, match="Cannot auto-detect"):
            auto_detect_text_field(FakeDS())

    def test_features_attribute(self):
        """Should work with features attribute when column_names is absent."""

        class FakeDS:
            features = {"text": "string", "id": "int"}

            def __iter__(self):
                yield {"text": "hello", "id": 1}

        assert auto_detect_text_field(FakeDS()) == "text"


# ---------------------------------------------------------------------------
# auto_detect_text_field_from_sample
# ---------------------------------------------------------------------------
class TestAutoDetectTextFieldFromSample:
    def test_single_key(self):
        sample = {"text": "hello world"}
        assert auto_detect_text_field_from_sample(sample) == "text"

    def test_priority(self):
        sample = {"content": "hello", "text": "world", "id": 1}
        assert auto_detect_text_field_from_sample(sample) == "text"

    def test_longest_string_fallback(self):
        sample = {"short": "hi", "long": "a much longer text here", "id": 1}
        assert auto_detect_text_field_from_sample(sample) == "long"

    def test_no_text_raises(self):
        sample = {"id": 1, "score": 0.5}
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            auto_detect_text_field_from_sample(sample)


# ---------------------------------------------------------------------------
# extract_text_from_sample
# ---------------------------------------------------------------------------
class TestExtractTextFromSample:
    def test_single_field_string(self):
        """fields as a single string."""
        sample = {"text": "hello world", "id": 1}
        assert extract_text_from_sample(sample, fields="text") == "hello world"

    def test_single_field_list(self):
        """fields as a single-element list."""
        sample = {"text": "hello world", "id": 1}
        assert extract_text_from_sample(sample, fields=["text"]) == "hello world"

    def test_multiple_fields(self):
        sample = {"question": "What is 2+2?", "answer": "4"}
        result = extract_text_from_sample(sample, fields=["question", "answer"], separator=" ")
        assert result == "What is 2+2? 4"

    def test_template(self):
        sample = {"question": "What is 2+2?", "answer": "4"}
        result = extract_text_from_sample(sample, template="{question} {answer}")
        assert result == "What is 2+2? 4"

    def test_template_with_missing_field(self):
        sample = {"question": "What is 2+2?"}
        with pytest.raises(KeyError, match="answer"):
            extract_text_from_sample(sample, template="{question} {answer}")

    def test_no_field_no_template(self):
        """Should fall back to first string column."""
        sample = {"id": 1, "text": "hello", "score": 0.5}
        assert extract_text_from_sample(sample) == "hello"

    def test_no_text_raises(self):
        sample = {"id": 1, "score": 0.5}
        with pytest.raises(ValueError, match="Cannot extract text"):
            extract_text_from_sample(sample)

    def test_list_field(self):
        sample = {"messages": ["hello", "world"]}
        result = extract_text_from_sample(sample, fields="messages", separator=" ")
        assert result == "hello world"

    def test_list_of_dicts(self):
        """List of dicts in single-field mode: only string items are extracted."""
        sample = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
        # In single-field mode, list items that are dicts are skipped (only str items kept)
        result = extract_text_from_sample(sample, fields="messages", separator=" ")
        assert result == ""

    def test_list_of_dicts_with_template(self):
        """Template mode extracts 'content' from list of dicts."""
        sample = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
        result = extract_text_from_sample(sample, template="{messages}", separator=" ")
        assert result == "hi hello"

    def test_template_with_list_field(self):
        sample = {"messages": ["hello", "world"]}
        result = extract_text_from_sample(sample, template="{messages}", separator=" | ")
        assert result == "hello | world"

    def test_template_takes_precedence_over_fields(self):
        """Template should take precedence over fields."""
        sample = {"question": "Q", "answer": "A", "other": "X"}
        result = extract_text_from_sample(sample, fields=["question", "answer"], template="{question}")
        assert result == "Q"


# ---------------------------------------------------------------------------
# CalibDataset dataclass
# ---------------------------------------------------------------------------
class TestCalibDataset:
    def test_minimal(self):
        spec = CalibDataset("my/ds")
        assert spec.name == "my/ds"
        assert spec.split is None
        assert spec.num is None
        assert spec.concat is False
        assert spec.fields is None
        assert spec.template is None
        assert spec.separator == "\n\n"
        assert spec.apply_chat_template is False
        assert spec.system_prompt is None
        assert spec.timeout == 300

    def test_single_field_string(self):
        spec = CalibDataset("my/ds", fields="text")
        assert spec.fields == "text"

    def test_multiple_fields_list(self):
        spec = CalibDataset("my/ds", fields=["question", "answer"])
        assert spec.fields == ["question", "answer"]

    def test_to_spec_string_minimal(self):
        spec = CalibDataset("my/ds")
        assert spec.to_spec_string() == "my/ds"

    def test_to_spec_string_with_split(self):
        spec = CalibDataset("my/ds", split="train")
        assert spec.to_spec_string() == "my/ds:split=train"

    def test_to_spec_string_with_num(self):
        spec = CalibDataset("my/ds", num=100)
        assert spec.to_spec_string() == "my/ds:num=100"

    def test_to_spec_string_with_fields(self):
        spec = CalibDataset("my/ds", fields=["q", "a"])
        assert spec.to_spec_string() == "my/ds:fields=q+a"

    def test_to_spec_string_single_field(self):
        spec = CalibDataset("my/ds", fields="text")
        assert spec.to_spec_string() == "my/ds:fields=text"

    def test_to_spec_string_with_template(self):
        spec = CalibDataset("my/ds", template="{q} {a}")
        assert spec.to_spec_string() == "my/ds:template={q} {a}"

    def test_to_spec_string_with_separator(self):
        spec = CalibDataset("my/ds", fields=["q", "a"], separator=" ")
        assert spec.to_spec_string() == "my/ds:fields=q+a:separator= "

    def test_to_spec_string_with_timeout(self):
        spec = CalibDataset("my/ds", timeout=600)
        assert spec.to_spec_string() == "my/ds:timeout=600"

    def test_to_spec_string_default_timeout_omitted(self):
        """Default timeout (300) should not appear in spec string."""
        spec = CalibDataset("my/ds", timeout=300)
        assert spec.to_spec_string() == "my/ds"

    def test_to_spec_string_full(self):
        spec = CalibDataset(
            "my/ds",
            split="train",
            num=500,
            fields=["q", "a"],
            separator=" ",
            apply_chat_template=True,
            system_prompt="You are helpful.",
            timeout=600,
        )
        assert (
            spec.to_spec_string()
            == "my/ds:split=train:num=500:fields=q+a:separator= :apply_chat_template=true:system_prompt=You are helpful.:timeout=600"
        )

    def test_from_spec_string(self):
        spec = CalibDataset.from_spec_string("my/ds:split=train:num=100:fields=q+a")
        assert spec.name == "my/ds"
        assert spec.split == ["train"]
        assert spec.num == 100
        assert spec.fields == ["q", "a"]

    def test_from_spec_string_single_field(self):
        spec = CalibDataset.from_spec_string("my/ds:fields=text")
        assert spec.fields == "text"

    def test_from_spec_string_with_timeout(self):
        spec = CalibDataset.from_spec_string("my/ds:timeout=600")
        assert spec.timeout == 600

    def test_roundtrip(self):
        """build -> parse -> build should be identity."""
        spec = CalibDataset("my/ds", split="train", num=100, fields=["q", "a"], separator=" ")
        spec_str = spec.to_spec_string()
        parsed = CalibDataset.from_spec_string(spec_str)
        assert parsed.to_spec_string() == spec_str

    def test_roundtrip_template(self):
        spec = CalibDataset("my/ds", template="{question} {answer}", separator=" ")
        spec_str = spec.to_spec_string()
        parsed = CalibDataset.from_spec_string(spec_str)
        assert parsed.template == "{question} {answer}"
        assert parsed.separator == " "

    def test_roundtrip_with_timeout(self):
        spec = CalibDataset("my/ds", num=100, timeout=600)
        spec_str = spec.to_spec_string()
        parsed = CalibDataset.from_spec_string(spec_str)
        assert parsed.timeout == 600
        assert parsed.to_spec_string() == spec_str


# ---------------------------------------------------------------------------
# build_dataset_spec
# ---------------------------------------------------------------------------
class TestBuildDatasetSpec:
    def test_minimal(self):
        assert build_dataset_spec("my/ds") == "my/ds"

    def test_with_split(self):
        assert build_dataset_spec("my/ds", split="train") == "my/ds:split=train"

    def test_with_num(self):
        assert build_dataset_spec("my/ds", num=100) == "my/ds:num=100"

    def test_with_split_and_num(self):
        assert build_dataset_spec("my/ds", split="train", num=100) == "my/ds:split=train:num=100"

    def test_with_fields_string(self):
        assert build_dataset_spec("my/ds", fields="text") == "my/ds:fields=text"

    def test_with_fields_list(self):
        assert build_dataset_spec("my/ds", fields=["q", "a"]) == "my/ds:fields=q+a"

    def test_with_template(self):
        assert build_dataset_spec("my/ds", template="{q} {a}") == "my/ds:template={q} {a}"

    def test_with_separator(self):
        assert build_dataset_spec("my/ds", fields=["q", "a"], separator=" ") == "my/ds:fields=q+a:separator= "

    def test_with_chat_template(self):
        assert build_dataset_spec("my/ds", apply_chat_template=True) == "my/ds:apply_chat_template=true"

    def test_with_timeout(self):
        assert build_dataset_spec("my/ds", timeout=600) == "my/ds:timeout=600"

    def test_with_default_timeout_omitted(self):
        """Default timeout (300) should not appear in spec string."""
        assert build_dataset_spec("my/ds", timeout=300) == "my/ds"

    def test_full_spec(self):
        assert (
            build_dataset_spec(
                "my/ds", split="train", num=100, fields=["q", "a"], separator=" ", apply_chat_template=True
            )
            == "my/ds:split=train:num=100:fields=q+a:separator= :apply_chat_template=true"
        )

    def test_full_spec_with_timeout(self):
        assert (
            build_dataset_spec("my/ds", split="train", num=100, fields=["q", "a"], separator=" ", timeout=600)
            == "my/ds:split=train:num=100:fields=q+a:separator= :timeout=600"
        )


# ---------------------------------------------------------------------------
# parse_dataset_spec
# ---------------------------------------------------------------------------
class TestParseDatasetSpec:
    def test_name_only(self):
        result = parse_dataset_spec("my/ds")
        assert result["name"] == "my/ds"
        assert result["split"] is None
        assert result["num"] is None
        assert result["concat"] is False
        assert result["fields"] is None
        assert result["template"] is None
        assert result["separator"] == "\n\n"
        assert result["apply_chat_template"] is False
        assert result["system_prompt"] is None

    def test_split(self):
        result = parse_dataset_spec("my/ds:split=train")
        assert result["name"] == "my/ds"
        assert result["split"] == ["train"]

    def test_multi_split(self):
        result = parse_dataset_spec("my/ds:split=train+test")
        assert result["split"] == ["train", "test"]

    def test_num(self):
        result = parse_dataset_spec("my/ds:num=100")
        assert result["num"] == 100

    def test_concat(self):
        result = parse_dataset_spec("my/ds:concat=true")
        assert result["concat"] is True

    def test_concat_false(self):
        result = parse_dataset_spec("my/ds:concat=false")
        assert result["concat"] is False

    def test_fields_single(self):
        result = parse_dataset_spec("my/ds:fields=text")
        assert result["fields"] == "text"

    def test_fields_multiple(self):
        result = parse_dataset_spec("my/ds:fields=q+a")
        assert result["fields"] == ["q", "a"]

    def test_template(self):
        result = parse_dataset_spec("my/ds:template={q} {a}")
        assert result["template"] == "{q} {a}"

    def test_separator(self):
        result = parse_dataset_spec("my/ds:fields=q+a:separator= ")
        assert result["separator"] == " "

    def test_apply_chat_template(self):
        result = parse_dataset_spec("my/ds:apply_chat_template=true")
        assert result["apply_chat_template"] is True

    def test_system_prompt(self):
        result = parse_dataset_spec("my/ds:system_prompt=You are helpful.")
        assert result["system_prompt"] == "You are helpful."

    def test_combined(self):
        result = parse_dataset_spec("my/ds:split=train+test:num=100:fields=q+a")
        assert result["name"] == "my/ds"
        assert result["split"] == ["train", "test"]
        assert result["num"] == 100
        assert result["fields"] == ["q", "a"]

    def test_timeout(self):
        result = parse_dataset_spec("my/ds:timeout=600")
        assert result["timeout"] == 600

    def test_timeout_default(self):
        result = parse_dataset_spec("my/ds")
        assert result["timeout"] == 300

    def test_roundtrip(self):
        """build -> parse should be identity."""
        spec = build_dataset_spec("my/ds", split="train", num=100, fields=["q", "a"], separator=" ")
        parsed = parse_dataset_spec(spec)
        assert parsed["name"] == "my/ds"
        assert parsed["split"] == ["train"]
        assert parsed["num"] == 100
        assert parsed["fields"] == ["q", "a"]
        assert parsed["separator"] == " "

    def test_roundtrip_with_timeout(self):
        spec = build_dataset_spec("my/ds", num=100, timeout=600)
        parsed = parse_dataset_spec(spec)
        assert parsed["timeout"] == 600


# ---------------------------------------------------------------------------
# normalize_dataset_spec
# ---------------------------------------------------------------------------
class TestNormalizeDatasetSpec:
    def test_none_raises(self):
        with pytest.raises(TypeError, match="must be a str"):
            normalize_dataset_spec(None)

    def test_empty_string(self):
        assert normalize_dataset_spec("") == ""

    def test_single_string(self):
        assert normalize_dataset_spec("my/ds") == "my/ds"

    def test_single_calib_dataset(self):
        spec = CalibDataset("my/ds", num=100)
        assert normalize_dataset_spec(spec) == "my/ds:num=100"

    def test_list_of_strings(self):
        result = normalize_dataset_spec(["my/ds", "other/ds"])
        assert result == "my/ds,other/ds"

    def test_list_of_calib_datasets(self):
        specs = [CalibDataset("my/ds", num=100), CalibDataset("other/ds", num=200)]
        result = normalize_dataset_spec(specs)
        assert result == "my/ds:num=100,other/ds:num=200"

    def test_mixed_list(self):
        """Mix of strings and CalibDataset objects."""
        specs = ["my/ds", CalibDataset("other/ds", num=200, timeout=60)]
        result = normalize_dataset_spec(specs)
        assert result == "my/ds,other/ds:num=200:timeout=60"

    def test_tuple(self):
        specs = ("my/ds", CalibDataset("other/ds", num=100))
        result = normalize_dataset_spec(specs)
        assert result == "my/ds,other/ds:num=100"

    def test_empty_list(self):
        assert normalize_dataset_spec([]) == ""
