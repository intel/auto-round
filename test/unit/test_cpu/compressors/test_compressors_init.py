# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.compressors.__init__``."""

import pytest

import auto_round.compressors as compressors


class TestCompressorsLazyImports:
    """Test the lazy import __getattr__ function."""

    def test_auto_round_lazy_import(self):
        AutoRound = compressors.AutoRound
        assert AutoRound is not None
        assert callable(AutoRound)

    def test_auto_round_compatible_lazy_import(self):
        AutoRoundCompatible = compressors.AutoRoundCompatible
        assert AutoRoundCompatible is not None
        assert callable(AutoRoundCompatible)

    def test_base_compressor_lazy_import(self):
        BaseCompressor = compressors.BaseCompressor
        assert BaseCompressor is not None
        assert isinstance(BaseCompressor, type)

    def test_data_driven_compressor_lazy_import(self):
        DataDrivenCompressor = compressors.DataDrivenCompressor
        assert DataDrivenCompressor is not None

    def test_calibrated_rtn_compressor_lazy_import(self):
        CalibratedRTNCompressor = compressors.CalibratedRTNCompressor
        assert CalibratedRTNCompressor is not None

    def test_zero_shot_compressor_lazy_import(self):
        ZeroShotCompressor = compressors.ZeroShotCompressor
        assert ZeroShotCompressor is not None

    def test_model_free_compressor_lazy_import(self):
        ModelFreeCompressor = compressors.ModelFreeCompressor
        assert ModelFreeCompressor is not None

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(compressors, "UnknownClass123")

    def test_all_contains_expected(self):
        assert "AutoRound" in compressors.__all__
        assert "BaseCompressor" in compressors.__all__
        assert "DataDrivenCompressor" in compressors.__all__
        assert "CalibratedRTNCompressor" in compressors.__all__
        assert "ZeroShotCompressor" in compressors.__all__
        assert "ModelFreeCompressor" in compressors.__all__
        assert "AutoRoundCompatible" in compressors.__all__

    def test_caching_same_object(self):
        """Second access returns the same object."""
        ar1 = compressors.AutoRound
        ar2 = compressors.AutoRound
        assert ar1 is ar2
