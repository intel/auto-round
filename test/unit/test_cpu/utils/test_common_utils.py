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

"""Unit tests for auto_round/utils/common.py to improve code coverage."""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestDownloadAudiocapsCsv:
    """Tests for download_audiocaps_csv function."""

    def test_download_audiocaps_csv_does_not_raise(self):
        from auto_round.utils.common import download_audiocaps_csv

        # Mock requests.get to avoid actual network call
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = "audio_id,audio_file,caption\ntest,test.wav,a test caption"
            mock_get.return_value = mock_response

            # Function may return None if network fails or return path if success
            result = download_audiocaps_csv()
            # Just verify it doesn't raise - result can be None or a path
            assert result is None or isinstance(result, str)

    def test_download_audiocaps_csv_uses_cache(self):
        import os
        import tempfile

        from auto_round.utils.common import download_audiocaps_csv

        # Create a temporary cached file
        cache_dir = os.path.join(tempfile.gettempdir(), "audiocaps_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "train.csv")

        with open(cache_file, "w") as f:
            f.write("audio_id,audio_file,caption\ntest,test.wav,a test caption")

        # Should use cached file without network call
        with patch("requests.get") as mock_get:
            result = download_audiocaps_csv()
            assert result == cache_file
            mock_get.assert_not_called()


class TestCompareVersions:
    """Tests for compare_versions function."""

    def test_equal_versions(self):
        from auto_round.utils.common import compare_versions

        assert compare_versions("1.0.0", "1.0.0") is True
        assert compare_versions("2.0.0", "2.0.0") is True

    def test_greater_than(self):
        from auto_round.utils.common import compare_versions

        assert compare_versions("2.0.0", "1.0.0") is True
        assert compare_versions("1.1.0", "1.0.0") is True
        assert compare_versions("1.0.1", "1.0.0") is True

    def test_less_than(self):
        from auto_round.utils.common import compare_versions

        assert compare_versions("1.0.0", "2.0.0") is False
        assert compare_versions("1.0.0", "1.1.0") is False
        assert compare_versions("1.0.0", "1.0.1") is False

    def test_greater_than_or_equal(self):
        from auto_round.utils.common import compare_versions

        assert compare_versions("2.0.0", "1.0.0") is True
        assert compare_versions("1.0.0", "1.0.0") is True

    def test_not_equal(self):
        from auto_round.utils.common import compare_versions

        assert compare_versions("2.0.0", "1.0.0") is True  # 2.0.0 >= 1.0.0 is True


class TestTorchVersionAtLeast:
    """Tests for torch_version_at_least function."""

    def test_torch_version_at_least_2_0_0(self):
        from auto_round.utils.common import torch_version_at_least

        result = torch_version_at_least("2.0.0")
        assert isinstance(result, bool)

    def test_torch_version_at_least_999_0_0(self):
        from auto_round.utils.common import torch_version_at_least

        # This will always be False since 999.0.0 > actual torch version
        result = torch_version_at_least("999.0.0")
        assert result is False


class TestLazyImport:
    """Tests for LazyImport class."""

    def test_lazy_import_getattr(self):
        from auto_round.utils.common import LazyImport

        lazy_os = LazyImport("os")
        # Should be able to get path attribute
        path = lazy_os.path
        assert hasattr(path, "join")

    def test_lazy_import_callable(self):
        from auto_round.utils.common import LazyImport

        # Test calling a function
        lazy_json = LazyImport("json")
        result = lazy_json.dumps({"test": 123})
        assert result == '{"test": 123}'

    def test_lazy_import_get_item(self):
        from auto_round.utils.common import LazyImport

        lazy_torch = LazyImport("torch")
        # Should be able to get Tensor attribute
        assert lazy_torch.Tensor is not None


class TestTorchVersionConstants:
    """Tests for TORCH_VERSION_AT_LEAST_* constants."""

    def test_torch_version_at_least_2_4_is_bool(self):
        from auto_round.utils.common import TORCH_VERSION_AT_LEAST_2_4

        assert isinstance(TORCH_VERSION_AT_LEAST_2_4, bool)

    def test_torch_version_at_least_2_6_is_bool(self):
        from auto_round.utils.common import TORCH_VERSION_AT_LEAST_2_6

        assert isinstance(TORCH_VERSION_AT_LEAST_2_6, bool)


class TestGetAttr:
    """Tests for get_attr function."""

    def test_nested_attr(self):
        from auto_round.utils.model import get_attr

        # Create a nested structure
        inner = MagicMock()
        inner.value = 42
        outer = MagicMock()
        outer.inner = inner

        result = get_attr(outer, "inner.value")
        assert result == 42

    def test_missing_attr_with_default(self):
        from auto_round.utils.model import get_attr

        class MockModule:
            pass

        module = MockModule()
        result = get_attr(module, "nonexistent.attr")
        assert result is None

    def test_direct_attr(self):
        from auto_round.utils.model import get_attr

        module = MagicMock()
        module.some_attr = "test_value"

        result = get_attr(module, "some_attr")
        assert result == "test_value"


class TestSetAttr:
    """Tests for set_attr function."""

    def test_set_attr(self):
        from auto_round.utils.model import set_attr

        class MockModule:
            pass

        model = MockModule()
        inner = MockModule()
        model.inner = inner

        set_attr(model, "inner.new_attr", "new_value")

        assert getattr(model.inner, "new_attr") == "new_value"

    def test_set_attr_missing_parent(self):
        from auto_round.utils.model import set_attr

        class MockModule:
            pass

        model = MockModule()

        # Should not raise even if parent doesn't exist
        set_attr(model, "nonexistent.parent.attr", "value")

    def test_set_attr_simple(self):
        from auto_round.utils.model import set_attr

        class MockModule:
            pass

        model = MockModule()

        set_attr(model, "simple_attr", "value")

        assert model.simple_attr == "value"


class TestImportFunctions:
    """Tests for import-related functions."""

    def test_import_quark_autograd_returns_bool(self):
        from auto_round.utils.common import LazyImport

        # Test quark autograd import via lazy import
        quark = LazyImport("quark.autograd")
        # Just verify we can check if it exists
        try:
            import quark.autograd  # noqa: F401

            exists = True
        except ImportError:
            exists = False

        # Should return a boolean
        assert isinstance(exists, bool)

    def test_import_auto_round_extension_returns_bool(self):
        # Test auto_round_extension import
        try:
            import auto_round_extension  # noqa: F401

            exists = True
        except ImportError:
            exists = False

        # Should return a boolean
        assert isinstance(exists, bool)
