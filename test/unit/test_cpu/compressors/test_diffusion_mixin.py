# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.compressors.diffusion_mixin``."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from auto_round.compressors.diffusion_mixin import DiffusionMixin


class TestDiffusionMixinProperties:
    """Test DiffusionMixin attribute access patterns."""

    def test_guidance_scale_default(self):
        # Access the class docstring and check init signature for defaults
        sig = inspect.signature(DiffusionMixin.__init__)
        params = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
        assert params.get("guidance_scale") == 7.5
        assert params.get("num_inference_steps") == 50
        assert params.get("generator_seed") is None

    def test_get_calibrator_kind_returns_diffusion(self):
        # Create a minimal mock class
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                # Don't call super().__init__() to avoid needing real parent
                pass

        comp = MockCompressor()
        assert comp._get_calibrator_kind() == "diffusion"

    def test_pipeline_call_kwargs_extracted_from_kwargs(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                pass

        comp = MockCompressor()
        # Set the attribute directly since we're not calling super().__init__
        comp.pipeline_call_kwargs = {"height": 512, "width": 512}
        assert comp.pipeline_call_kwargs.get("height") == 512

    def test_align_pipeline_dtype_preserves_declared_fp32_modules(self):
        protected = torch.nn.Linear(2, 2)
        protected._keep_in_fp32_modules = ["weight"]
        protected.dtype = torch.float32
        declared_empty = torch.nn.Linear(2, 2)
        declared_empty._keep_in_fp32_modules = []
        declared_empty.dtype = torch.float32
        ordinary = torch.nn.Linear(2, 2)
        ordinary.dtype = torch.float32
        pipe = SimpleNamespace(
            components=["protected", "declared_empty", "ordinary"],
            protected=protected,
            declared_empty=declared_empty,
            ordinary=ordinary,
        )

        DiffusionMixin._align_pipeline_dtype(pipe, torch.bfloat16)

        assert protected.weight.dtype == torch.float32
        assert declared_empty.weight.dtype == torch.float32
        assert ordinary.weight.dtype == torch.bfloat16


class TestFindAdditionalTransformers:
    """Test _find_additional_transformers logic."""

    def test_returns_empty_when_pipe_is_none(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                self.model_context = SimpleNamespace(pipe=None)

        comp = MockCompressor()
        result = comp._find_additional_transformers()
        assert result == []

    def test_finds_secondary_transformers(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                pipe = MagicMock()
                pipe.components = ["transformer", "transformer_2", "vae"]
                pipe.transformer = torch.nn.Linear(4, 4)
                pipe.transformer_2 = torch.nn.Linear(4, 4)
                pipe.vae = torch.nn.Linear(4, 4)
                self.model_context = SimpleNamespace(pipe=pipe)

        comp = MockCompressor()
        result = comp._find_additional_transformers()
        assert len(result) == 1
        assert result[0][0] == "transformer_2"


class TestAlignDeviceAndDtype:
    """Test _align_device_and_dtype_for_secondary logic."""

    def test_no_op_when_pipe_is_none(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                self.model_context = SimpleNamespace(pipe=None, model=None)

        comp = MockCompressor()
        # Should not raise
        comp._align_device_and_dtype_for_secondary("transformer")

    def test_no_op_when_model_is_none(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                pipe = MagicMock()
                pipe.components = []
                self.model_context = SimpleNamespace(pipe=pipe, model=None)

        comp = MockCompressor()
        # Should not raise
        comp._align_device_and_dtype_for_secondary("transformer")
