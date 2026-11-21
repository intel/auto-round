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

from test import fixtures

import pytest


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


def test_llama4_fixture_skips_when_the_token_cannot_read_the_repo(monkeypatch):
    """A token that exists but is not authorised must skip too.

    ``_hub_token_available`` only proves a variable is set; an expired token, or
    one scoped to another org, fails later with ``OSError: You are trying to
    access a gated repo.``. Reproduced locally with a dummy ``HF_TOKEN`` before
    this was handled, which is what the CI org secret looks like when it lapses.
    """
    monkeypatch.setattr(fixtures, "_tiny_model_ready", lambda *args, **kwargs: False)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_token")

    def _gated(*args, **kwargs):
        raise OSError("You are trying to access a gated repo.")

    monkeypatch.setattr(fixtures, "get_model_path", _gated)

    generator = fixtures.tiny_llama4_model_path.__wrapped__()
    with pytest.raises(pytest.skip.Exception) as skip:
        next(generator)
    assert "not readable" in str(skip.value)


def test_other_failures_still_propagate(monkeypatch):
    """Only the gated-repo error becomes a skip; anything else stays a failure."""
    monkeypatch.setattr(fixtures, "_tiny_model_ready", lambda *args, **kwargs: False)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_token")

    def _broken(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(fixtures, "get_model_path", _broken)

    generator = fixtures.tiny_llama4_model_path.__wrapped__()
    with pytest.raises(OSError, match="No space left on device"):
        next(generator)
