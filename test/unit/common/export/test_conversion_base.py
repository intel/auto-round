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
"""Tests for the model registry in
``auto_round/export/export_to_gguf/conversion/base.py``.
"""

import gguf
import pytest

from auto_round.export.export_to_gguf.conversion.base import (
    ModelBase,
    ModelType,
    SentencePieceTokenTypes,
)


# ---------------------------------------------------------------------------
# ModelType enum
# ---------------------------------------------------------------------------
class TestModelType:
    def test_two_values(self):
        assert len(ModelType) == 2
        assert ModelType.TEXT.value == 1
        assert ModelType.MMPROJ.value == 2

    def test_int_enum_comparisons(self):
        assert int(ModelType.TEXT) < int(ModelType.MMPROJ)


# ---------------------------------------------------------------------------
# SentencePieceTokenTypes
# ---------------------------------------------------------------------------
class TestSentencePieceTokenTypes:
    def test_values(self):
        assert SentencePieceTokenTypes.NORMAL == 1
        assert SentencePieceTokenTypes.UNKNOWN == 2
        assert SentencePieceTokenTypes.CONTROL == 3
        assert SentencePieceTokenTypes.USER_DEFINED == 4
        assert SentencePieceTokenTypes.UNUSED == 5
        assert SentencePieceTokenTypes.BYTE == 6


# ---------------------------------------------------------------------------
# ModelBase registry
# ---------------------------------------------------------------------------
class TestModelBaseRegistry:
    def _make_text_class(self, name):
        """Return a fresh ModelBase subclass registered under ``name``."""
        arch = gguf.MODEL_ARCH.LLAMA

        @ModelBase.register(name)
        class _FakeTextModel(ModelBase):
            model_arch = arch

        return _FakeTextModel

    def test_register_text_model(self):
        from auto_round.export.export_to_gguf.conversion.base import ModelBase

        name = "_test_register_text_model_uniq"
        # Save registry state for restoration
        saved = ModelBase._model_classes[ModelType.TEXT].get(name)
        try:
            cls = self._make_text_class(name)
            assert ModelBase._model_classes[ModelType.TEXT][name] is cls
        finally:
            # Cleanup
            if (
                name in ModelBase._model_classes[ModelType.TEXT]
                and ModelBase._model_classes[ModelType.TEXT][name] is not cls
            ):
                del ModelBase._model_classes[ModelType.TEXT][name]
            elif saved is not None:
                ModelBase._model_classes[ModelType.TEXT][name] = saved

    def test_register_multiple_aliases(self):
        @ModelBase.register("_test_arch_a", "_test_arch_b")
        class _FakeMulti(ModelBase):
            model_arch = gguf.MODEL_ARCH.LLAMA

        try:
            assert ModelBase._model_classes[ModelType.TEXT]["_test_arch_a"] is _FakeMulti
            assert ModelBase._model_classes[ModelType.TEXT]["_test_arch_b"] is _FakeMulti
        finally:
            ModelBase._model_classes[ModelType.TEXT].pop("_test_arch_a", None)
            ModelBase._model_classes[ModelType.TEXT].pop("_test_arch_b", None)

    def test_register_mmproj(self):
        @ModelBase.register("_test_mmproj_uniq")
        class _FakeMmproj(ModelBase):
            model_arch = gguf.MODEL_ARCH.MMPROJ

        try:
            assert ModelBase._model_classes[ModelType.MMPROJ]["_test_mmproj_uniq"] is _FakeMmproj
        finally:
            ModelBase._model_classes[ModelType.MMPROJ].pop("_test_mmproj_uniq", None)

    def test_register_returns_class_unchanged(self):
        @ModelBase.register("_test_return_uniq")
        class _FakeReturn(ModelBase):
            model_arch = gguf.MODEL_ARCH.LLAMA

        try:
            # Decorator must return the class unchanged so callers can use it
            assert _FakeReturn.__name__ == "_FakeReturn"
        finally:
            ModelBase._model_classes[ModelType.TEXT].pop("_test_return_uniq", None)

    def test_register_asserts_at_least_one_name(self):
        with pytest.raises(AssertionError):
            ModelBase.register()


class TestModelBaseFromArchitecture:
    def test_from_model_architecture_returns_class(self):
        @ModelBase.register("_test_lookup_arch")
        class _FakeLookup(ModelBase):
            model_arch = gguf.MODEL_ARCH.LLAMA

        try:
            cls = ModelBase.from_model_architecture("_test_lookup_arch")
            assert cls is _FakeLookup
        finally:
            ModelBase._model_classes[ModelType.TEXT].pop("_test_lookup_arch", None)

    def test_unknown_arch_raises(self):
        with pytest.raises(NotImplementedError):
            ModelBase.from_model_architecture("definitely_not_a_real_arch_xyz")


class TestModelBaseDirectInstantiation:
    def test_cannot_instantiate_base_directly(self):
        # ModelBase.__init__ explicitly forbids direct instantiation of
        # ModelBase / TextModel / MmprojModel.
        with pytest.raises(TypeError):
            ModelBase.__init__(
                ModelBase.__new__(ModelBase),
                dir_model=None,
                ftype=None,
                fname_out=None,
            )


# ---------------------------------------------------------------------------
# Mistral module-level constants / helpers (light coverage)
# ---------------------------------------------------------------------------
class TestMistralFallbackConstants:
    def test_dataset_mean_default(self):
        # When mistral_common isn't installed, fallback defaults are loaded.
        from auto_round.export.export_to_gguf.conversion.base import _MISTRAL_COMMON_DATASET_MEAN

        assert isinstance(_MISTRAL_COMMON_DATASET_MEAN, tuple)
        assert len(_MISTRAL_COMMON_DATASET_MEAN) == 3

    def test_dataset_std_default(self):
        from auto_round.export.export_to_gguf.conversion.base import _MISTRAL_COMMON_DATASET_STD

        assert isinstance(_MISTRAL_COMMON_DATASET_STD, tuple)
        assert len(_MISTRAL_COMMON_DATASET_STD) == 3
