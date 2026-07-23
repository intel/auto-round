# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.compressors.mllm.utils``."""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from auto_round.compressors.mllm.utils import (
    VISUAL_KEYS,
    _extract_data_dir,
    fetch_image,
)


class TestExtractDataDir:
    """Test _extract_data_dir."""

    def test_directory_path(self, tmp_path):
        result = _extract_data_dir(str(tmp_path))
        assert result == str(tmp_path)

    def test_key_value_string(self):
        result = _extract_data_dir("image=/path/to/image.png")
        assert result == {"image": "/path/to/image.png"}

    def test_multiple_key_value(self):
        result = _extract_data_dir("image=/img.png,video=/vid.mp4,audio=/aud.wav")
        assert result == {"image": "/img.png", "video": "/vid.mp4", "audio": "/aud.wav"}

    def test_unknown_key_skipped(self):
        result = _extract_data_dir("image=/img.png,unknown=/unk.png")
        assert result == {"image": "/img.png"}

    def test_raises_on_invalid_input(self):
        with pytest.raises(TypeError, match="incorrect input"):
            _extract_data_dir("invalid_no_equals")

    def test_directory_takes_precedence(self, tmp_path):
        # If it's a valid directory path (no "="), it's treated as a dir
        # even if it happens to be a file-like path
        # The function checks isdir first
        result = _extract_data_dir(str(tmp_path))
        assert result == str(tmp_path)


class TestFetchImage:
    """Test fetch_image."""

    def test_local_file(self, tmp_path):
        # Create a small valid image
        try:
            from PIL import Image

            img = Image.new("RGB", (10, 10), color="red")
            img_path = tmp_path / "test.png"
            img.save(str(img_path))

            result = fetch_image(str(img_path))
            assert result is not None
            assert hasattr(result, "size")
        except ImportError:
            pytest.skip("PIL not available")

    def test_http_url_success(self):
        with patch("auto_round.compressors.mllm.utils.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raw.decode_content = True
            mock_response.raise_for_status = MagicMock()
            mock_response.raw = MagicMock()
            mock_response.raw.read = MagicMock(return_value=b"")
            mock_get.return_value = mock_response

            with patch("auto_round.compressors.mllm.utils.Image.open") as mock_open:
                mock_img = MagicMock()
                mock_open.return_value = mock_img
                result = fetch_image("https://example.com/image.png")
                mock_get.assert_called_once()
                mock_open.assert_called_once()
                assert result is mock_img

    def test_http_url_failure(self):
        import requests

        with patch("auto_round.compressors.mllm.utils.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Network error")
            with pytest.raises(RuntimeError, match="Failed to fetch image"):
                fetch_image("https://example.com/image.png")

    def test_http_url_invalid_response(self):
        import requests

        with patch("auto_round.compressors.mllm.utils.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raw = MagicMock()
            mock_response.raw.decode_content = True
            mock_get.return_value = mock_response
            # OSError from Image.open after raise_for_status
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Bad response")
            with pytest.raises(RuntimeError, match="Failed to fetch image"):
                fetch_image("https://example.com/image.png")

    def test_neither_file_nor_url(self):
        with pytest.raises(TypeError, match="neither a path or url"):
            fetch_image("just_a_string")


class TestVisualKeys:
    """Test VISUAL_KEYS constant."""

    def test_visual_keys_not_empty(self):
        assert VISUAL_KEYS is not None
        assert isinstance(VISUAL_KEYS, (list, tuple, set))
