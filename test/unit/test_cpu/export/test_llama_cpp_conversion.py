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
"""Tests for the small helpers in
``auto_round/export/export_to_gguf/llama_cpp_conversion.py``.
"""

from __future__ import annotations

import pytest

from auto_round.export.export_to_gguf import llama_cpp_conversion as lcc
from auto_round.export.export_to_gguf.config import ModelType as AutoRoundModelType


# ---------------------------------------------------------------------------
# ConversionContext.model_type
# ---------------------------------------------------------------------------
class TestConversionContextModelType:
    def test_mmproj_returns_module_mmp_proj(self):
        class _MT:
            MMPROJ = AutoRoundModelType.MMPROJ
            TEXT = AutoRoundModelType.TEXT

        ctx = lcc.ConversionContext(
            module=type("M", (), {"ModelType": _MT, "ModelBase": None})(),
            source="",
        )
        assert ctx.model_type(AutoRoundModelType.MMPROJ) == _MT.MMPROJ

    def test_text_returns_module_text(self):
        class _MT:
            MMPROJ = AutoRoundModelType.MMPROJ
            TEXT = AutoRoundModelType.TEXT

        ctx = lcc.ConversionContext(
            module=type("M", (), {"ModelType": _MT, "ModelBase": None})(),
            source="",
        )
        assert ctx.model_type(AutoRoundModelType.TEXT) == _MT.TEXT

    def test_other_model_type_returns_text(self):
        class _MT:
            MMPROJ = AutoRoundModelType.MMPROJ
            TEXT = AutoRoundModelType.TEXT

        ctx = lcc.ConversionContext(
            module=type("M", (), {"ModelType": _MT, "ModelBase": None})(),
            source="",
        )
        assert ctx.model_type(99999) == _MT.TEXT


class TestConversionContextIsSupported:
    def test_supported_returns_true(self):
        class _MT:
            MMPROJ = AutoRoundModelType.MMPROJ
            TEXT = AutoRoundModelType.TEXT

        class _Cls:
            pass

        module = type("M", (), {
            "ModelType": _MT,
            "ModelBase": type("MB", (), {
                "from_model_architecture": classmethod(lambda cls, arch, model_type=None: _Cls)
            }),
        })()

        ctx = lcc.ConversionContext(module=module, source="")
        assert ctx.is_supported("TestArch") is True

    def test_unsupported_returns_false(self):
        class _MT:
            MMPROJ = AutoRoundModelType.MMPROJ
            TEXT = AutoRoundModelType.TEXT

        def _raise(*args, **kwargs):
            raise NotImplementedError("x")

        module = type("M", (), {
            "ModelType": _MT,
            "ModelBase": type("MB", (), {
                "from_model_architecture": classmethod(_raise)
            }),
        })()

        ctx = lcc.ConversionContext(module=module, source="")
        assert ctx.is_supported("BogusArch") is False


# ---------------------------------------------------------------------------
# _literal_map_from_init
# ---------------------------------------------------------------------------
class TestLiteralMapFromInit:
    def test_annotated_assignment(self):
        src = """
some_dict: dict[str, str] = {
    "A": "1",
    "B": "2",
}
"""
        result = lcc._literal_map_from_init(src, "some_dict")
        assert result == {"A": "1", "B": "2"}

    def test_unannotated_assignment(self):
        src = """
my_map = {
    "x": "y",
}
"""
        result = lcc._literal_map_from_init(src, "my_map")
        assert result == {"x": "y"}

    def test_missing_returns_empty(self):
        assert lcc._literal_map_from_init("foo = 1", "missing") == {}

    def test_non_literal_value_raises(self):
        with pytest.raises(ValueError):
            lcc._literal_map_from_init("my_map = unknown_thing", "my_map")


# ---------------------------------------------------------------------------
# _conversion_dependencies
# ---------------------------------------------------------------------------
class TestConversionDependencies:
    def test_relative_module_import(self, tmp_path):
        src = "from .llama import LlamaModel\n"
        p = tmp_path / "__init__.py"
        p.write_text(src)
        deps = lcc._conversion_dependencies(p)
        assert "conversion/llama.py" in deps

    def test_relative_import_only_names_excluded(self, tmp_path):
        src = "from . import foo\n"
        p = tmp_path / "__init__.py"
        p.write_text(src)
        deps = lcc._conversion_dependencies(p)
        assert "conversion/foo.py" not in deps

    def test_absolute_root_import_not_matched(self, tmp_path):
        src = "from conversion import bar\n"
        p = tmp_path / "__init__.py"
        p.write_text(src)
        deps = lcc._conversion_dependencies(p)
        assert deps == set()

    def test_nested_absolute_import(self, tmp_path):
        src = "from conversion.sub import baz\n"
        p = tmp_path / "__init__.py"
        p.write_text(src)
        deps = lcc._conversion_dependencies(p)
        assert "conversion/sub.py" in deps

    def test_unrelated_import_excluded(self, tmp_path):
        src = "import os\nimport numpy\nfrom . import x\nfrom conversion.sub import y\n"
        p = tmp_path / "__init__.py"
        p.write_text(src)
        deps = lcc._conversion_dependencies(p)
        assert not any("os" in d for d in deps)
        assert "conversion/sub.py" in deps

    def test_plain_import_conversion(self, tmp_path):
        src = "import conversion.foo\n"
        p = tmp_path / "__init__.py"
        p.write_text(src)
        deps = lcc._conversion_dependencies(p)
        assert "conversion/foo.py" in deps

    def test_self_init_discarded(self, tmp_path):
        src = "from conversion.sub import x\n"
        p = tmp_path / "__init__.py"
        p.write_text(src)
        deps = lcc._conversion_dependencies(p)
        assert "conversion/__init__.py" not in deps


# ---------------------------------------------------------------------------
# _architecture_from_hparams
# ---------------------------------------------------------------------------
class TestArchitectureFromHparams:
    def test_top_level_architectures(self):
        assert lcc._architecture_from_hparams({"architectures": ["LlamaForCausalLM"]}) == "LlamaForCausalLM"

    def test_text_config(self):
        hparams = {"text_config": {"architectures": ["MistralForCausalLM"]}}
        assert lcc._architecture_from_hparams(hparams) == "MistralForCausalLM"

    def test_llm_config_fallback(self):
        assert lcc._architecture_from_hparams({"llm_config": {"architectures": ["X"]}}) == "X"

    def test_language_config_fallback(self):
        assert lcc._architecture_from_hparams({"language_config": {"architectures": ["Y"]}}) == "Y"

    def test_mmproj_uses_vision_config(self):
        hparams = {
            "text_config": {"architectures": ["Wrong"]},
            "vision_config": {"architectures": ["SiglipVisionModel"]},
        }
        assert lcc._architecture_from_hparams(hparams, model_type=AutoRoundModelType.MMPROJ) == "SiglipVisionModel"

    def test_mmproj_vision_encoder_key(self):
        hparams = {"vision_encoder": {"architectures": ["X"]}}
        assert lcc._architecture_from_hparams(hparams, model_type=AutoRoundModelType.MMPROJ) == "X"

    def test_returns_none_when_no_architectures(self):
        assert lcc._architecture_from_hparams({"foo": "bar"}) is None

    def test_non_dict_text_config_falls_through(self):
        hparams = {"text_config": "not a dict", "architectures": ["Good"]}
        assert lcc._architecture_from_hparams(hparams) == "Good"


# ---------------------------------------------------------------------------
# URL constants
# ---------------------------------------------------------------------------
class TestUrlConstants:
    def test_llama_cpp_raw_url(self):
        assert lcc.LLAMA_CPP_RAW_URL.startswith("https://")

    def test_llama_cpp_api_url(self):
        assert lcc.LLAMA_CPP_API_URL.startswith("https://")

    def test_request_timeout_positive(self):
        assert lcc.REQUEST_TIMEOUT > 0


# ---------------------------------------------------------------------------
# GGUFConversionError
# ---------------------------------------------------------------------------
class TestGGUFConversionError:
    def test_subclass_of_import_error(self):
        assert issubclass(lcc.GGUFConversionError, ImportError)