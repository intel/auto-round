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
"""Tests for ``auto_round/export/export_to_autogptq/export.py``."""

from auto_round.export.export_to_autogptq.export import (
    convert_from_autogptq_dynamic,
    convert_to_autogptq_dynamic,
    GPTQ_REQUIRED_CONFIG_KEYS,
    BLOCK_PATTERNS,
)


# ---------------------------------------------------------------------------
# GPTQ_REQUIRED_CONFIG_KEYS / BLOCK_PATTERNS
# ---------------------------------------------------------------------------
class TestModuleConstants:
    def test_required_config_keys(self):
        assert "bits" in GPTQ_REQUIRED_CONFIG_KEYS
        assert "group_size" in GPTQ_REQUIRED_CONFIG_KEYS
        assert "sym" in GPTQ_REQUIRED_CONFIG_KEYS

    def test_block_patterns_list(self):
        assert isinstance(BLOCK_PATTERNS, list)
        assert "model.layers" in BLOCK_PATTERNS
        assert "transformer.h" in BLOCK_PATTERNS


# ---------------------------------------------------------------------------
# convert_to_autogptq_dynamic
# ---------------------------------------------------------------------------
class TestConvertToAutogptqDynamic:
    def test_quantize_match(self):
        """bits < 16 -> positive match `+:regex`."""
        cfg = {"name1": {"bits": 4, "group_size": 128, "sym": True}}
        out = convert_to_autogptq_dynamic(cfg)
        # The key should start with `+:`
        positive_keys = [k for k in out if k.startswith("+:")]
        assert len(positive_keys) == 1
        # Required keys copied over
        pos = out[positive_keys[0]]
        assert pos["bits"] == 4
        assert pos["group_size"] == 128
        assert pos["sym"] is True

    def test_skip_match(self):
        """bits == 16 -> negative match `-:regex` with empty config."""
        cfg = {"name1": {"bits": 16, "group_size": 128, "sym": True}}
        out = convert_to_autogptq_dynamic(cfg)
        negative_keys = [k for k in out if k.startswith("-:")]
        assert len(negative_keys) == 1
        assert out[negative_keys[0]] == {}

    def test_bits_none_ignored(self):
        """bits is None -> entry skipped."""
        cfg = {"name1": {"bits": None}}
        out = convert_to_autogptq_dynamic(cfg)
        assert out == {}

    def test_bits_gt_16_skipped(self):
        """bits > 16 should also fall into the skip branch (negative match)."""
        cfg = {"name1": {"bits": 32}}
        out = convert_to_autogptq_dynamic(cfg)
        negative_keys = [k for k in out if k.startswith("-:")]
        assert len(negative_keys) == 1
        assert out[negative_keys[0]] == {}

    def test_multiple_entries(self):
        cfg = {
            "regex1": {"bits": 4, "group_size": 64, "sym": False},
            "regex2": {"bits": 8, "group_size": 128, "sym": True},
            "regex3": {"bits": 16, "group_size": -1, "sym": True},
        }
        out = convert_to_autogptq_dynamic(cfg)
        positives = [k for k in out if k.startswith("+:")]
        negatives = [k for k in out if k.startswith("-:")]
        assert len(positives) == 2
        assert len(negatives) == 1


# ---------------------------------------------------------------------------
# convert_from_autogptq_dynamic
# ---------------------------------------------------------------------------
class TestConvertFromAutogptqDynamic:
    def test_positive_match(self):
        cfg = {"+:model.layers": {"bits": 4, "group_size": 128, "sym": True}}
        out = convert_from_autogptq_dynamic(cfg)
        assert "model.layers" in out
        assert out["model.layers"]["bits"] == 4
        assert out["model.layers"]["group_size"] == 128
        assert out["model.layers"]["sym"] is True

    def test_negative_match(self):
        cfg = {"-:model.layers": {}}
        out = convert_from_autogptq_dynamic(cfg)
        assert "model.layers" in out
        assert out["model.layers"]["bits"] == 16
        assert out["model.layers"]["act_bits"] == 16

    def test_unknown_prefix_ignored(self):
        cfg = {"no_prefix": {"bits": 4}}
        out = convert_from_autogptq_dynamic(cfg)
        # Entries without +/- prefix are silently dropped
        assert "no_prefix" not in out

    def test_mixed_entries(self):
        cfg = {
            "+:a": {"bits": 4, "group_size": 64, "sym": True},
            "-:b": {},
            "uncategorized": {"bits": 4},
        }
        out = convert_from_autogptq_dynamic(cfg)
        assert "a" in out
        assert "b" in out
        assert "uncategorized" not in out

    def test_empty(self):
        assert convert_from_autogptq_dynamic({}) == {}