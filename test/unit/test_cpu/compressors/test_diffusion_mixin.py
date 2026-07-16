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
from unittest.mock import MagicMock, patch

import pytest
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


class TestRequiresCalibrationImage:
    """Test _requires_calibration_image logic."""

    def test_false_when_no_image_param(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                pipe = MagicMock()
                # Mock signature with no image parameter
                pipe_sig = MagicMock()
                pipe_sig.parameters = {"prompt": MagicMock(default="test")}
                pipe._sig = pipe_sig
                self.model_context = SimpleNamespace(pipe=pipe)

        comp = MockCompressor()
        with patch.object(
            inspect, "signature", return_value=MagicMock(parameters={"prompt": MagicMock(default="test")})
        ):
            result = comp._requires_calibration_image()
        assert result is False

    def test_true_when_image_param_required(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                pipe = MagicMock()
                self.model_context = SimpleNamespace(pipe=pipe)

        comp = MockCompressor()
        with patch.object(
            inspect,
            "signature",
            return_value=MagicMock(
                parameters={
                    "prompt": MagicMock(default="test"),
                    "image": MagicMock(default=inspect.Parameter.empty),
                }
            ),
        ):
            result = comp._requires_calibration_image()
        assert result is True


class TestGetCalibrationImage:
    """Test _get_calibration_image logic."""

    def test_returns_single_image_for_batch_size_1(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                pipe = MagicMock()
                self.model_context = SimpleNamespace(pipe=pipe)

        comp = MockCompressor()
        with patch.object(
            inspect, "signature", return_value=MagicMock(parameters={"prompt": MagicMock(default="test")})
        ):
            image = comp._get_calibration_image(1)
        assert image is not None

    def test_returns_list_for_batch_size_greater_than_1(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                pipe = MagicMock()
                self.model_context = SimpleNamespace(pipe=pipe)

        comp = MockCompressor()
        with patch.object(
            inspect, "signature", return_value=MagicMock(parameters={"prompt": MagicMock(default="test")})
        ):
            images = comp._get_calibration_image(3)
        assert isinstance(images, list)
        assert len(images) == 3


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


class TestBuildPipelineCallKwargs:
    """Test _build_pipeline_call_kwargs logic."""

    def test_basic_kwargs(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                self.guidance_scale = 3.5
                self.num_inference_steps = 10
                self.generator_seed = 99
                self.pipeline_call_kwargs = {}
                # Set up model_context to avoid AttributeError
                self.model_context = SimpleNamespace(pipe=MagicMock())

        comp = MockCompressor()
        pipe = MagicMock()
        pipe.device = torch.device("cpu")
        comp.model_context.pipe = pipe

        with patch.object(comp, "_requires_calibration_image", return_value=False):
            kwargs = comp._build_pipeline_call_kwargs(pipe, ["test prompt"])
        assert kwargs["guidance_scale"] == 3.5
        assert kwargs["num_inference_steps"] == 10
        assert kwargs["generator"] is not None

    def test_no_generator_when_seed_none(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                self.guidance_scale = 3.5
                self.num_inference_steps = 10
                self.generator_seed = None
                self.pipeline_call_kwargs = {}
                # Set up model_context to avoid AttributeError
                self.model_context = SimpleNamespace(pipe=MagicMock())

        comp = MockCompressor()
        pipe = MagicMock()
        pipe.device = torch.device("cpu")
        comp.model_context.pipe = pipe

        with patch.object(comp, "_requires_calibration_image", return_value=False):
            kwargs = comp._build_pipeline_call_kwargs(pipe, ["test prompt"])
        assert kwargs["generator"] is None

    def test_adds_calibration_image_for_i2v(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                self.guidance_scale = 3.5
                self.num_inference_steps = 10
                self.generator_seed = None
                self.pipeline_call_kwargs = {}
                self.model_context = SimpleNamespace(pipe=MagicMock())

            def _requires_calibration_image(self):
                return True

            def _get_calibration_image(self, batch_size):
                return MagicMock()

        comp = MockCompressor()
        pipe = MagicMock()
        pipe.device = torch.device("cpu")

        kwargs = comp._build_pipeline_call_kwargs(pipe, ["test prompt"])
        assert "image" in kwargs
        assert "prompt" in kwargs
