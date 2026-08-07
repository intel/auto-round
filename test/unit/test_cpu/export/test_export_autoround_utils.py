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
"""Tests for ``auto_round/export/export_to_autoround/utils.py``."""

from dataclasses import fields

import pytest

from auto_round.export.export_to_autoround.utils import check_neq_config
from auto_round.schemes import QuantizationScheme


class TestCheckNeqConfig:
    def test_no_mismatches(self):
        """All keys match the expected values -> empty list."""
        # Build a config dict that matches every scheme field
        config = {f.name: None for f in fields(QuantizationScheme)}
        # Provide an expected value for every key
        expected = {f.name: None for f in fields(QuantizationScheme)}
        result = check_neq_config(config, **expected)
        assert result == []

    def test_some_mismatches(self):
        config = {f.name: None for f in fields(QuantizationScheme)}
        # Differ on at least one key
        scheme_keys = [f.name for f in fields(QuantizationScheme)]
        first = scheme_keys[0]
        config[first] = 4  # actual value differs from expected
        expected = {f.name: None for f in fields(QuantizationScheme)}
        result = check_neq_config(config, **expected)
        assert first in result

    def test_missing_expected_key_raises(self):
        config = {f.name: None for f in fields(QuantizationScheme)}
        scheme_keys = [f.name for f in fields(QuantizationScheme)]
        # Drop one expected value
        incomplete = {f.name: None for f in fields(QuantizationScheme)[:-1]}
        with pytest.raises(ValueError, match="Missing expected"):
            check_neq_config(config, **incomplete)

    def test_config_value_none_not_mismatch(self):
        """If config.get(key) is None, it's treated as not-a-mismatch."""
        config = {f.name: None for f in fields(QuantizationScheme)}
        expected = {f.name: 999 for f in fields(QuantizationScheme)}
        # None != 999, but None is treated as 'not set' so no mismatch
        result = check_neq_config(config, **expected)
        assert result == []
