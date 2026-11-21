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

"""The tiny llama4 fixture needs a hub token: its config lives behind a gated repo."""

import pytest

from test import fixtures


def test_llama4_fixture_skips_without_hub_token(monkeypatch):
    monkeypatch.setattr(fixtures, "_tiny_model_ready", lambda *args, **kwargs: False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    generator = fixtures.tiny_llama4_model_path.__wrapped__()
    with pytest.raises(pytest.skip.Exception) as skip:
        next(generator)
    assert "gated" in str(skip.value)


def test_llama4_fixture_accepts_either_token_variable(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "dummy")
    assert fixtures._hub_token_available() is True

    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "dummy")
    assert fixtures._hub_token_available() is True

    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert fixtures._hub_token_available() is False
