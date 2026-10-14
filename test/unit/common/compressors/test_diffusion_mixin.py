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

import pytest
import torch

from auto_round.compressors.diffusion_mixin import DiffusionMixin
from auto_round.context.model import ModelContext


class TestDiffusionMixinProperties:
    """Test DiffusionMixin attribute access patterns."""

    def test_guidance_scale_default(self):
        # Access the class docstring and check init signature for defaults
        sig = inspect.signature(DiffusionMixin.__init__)
        params = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
        assert params.get("guidance_scale") == 7.5
        assert params.get("num_inference_steps") == 50
        assert params.get("calib_num_inference_steps") == 8
        assert params.get("generator_seed") is None
        assert params.get("diffusion_tuning_cache_size") == 0

    @pytest.mark.parametrize("budget", [2, "auto"])
    def test_generation_and_calibration_steps_are_independent(self, budget):
        class Parent:
            def __init__(self, *args, **kwargs):
                self.model_context = SimpleNamespace(pipe=None, model=None)

        class MockCompressor(DiffusionMixin, Parent):
            pass

        comp = MockCompressor(num_inference_steps=20, calib_num_inference_steps=7, diffusion_tuning_cache_size=budget)

        assert comp.num_inference_steps == 20
        assert comp.calib_num_inference_steps == 7
        assert comp.model_context.diffusion_tuning_cache_size == budget

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"calib_num_inference_steps": 0}, "calib_num_inference_steps"),
            ({"num_inference_steps": 0}, "num_inference_steps"),
            ({"diffusion_tuning_cache_size": -1}, "diffusion_tuning_cache_size"),
            ({"diffusion_tuning_cache_size": float("nan")}, "diffusion_tuning_cache_size"),
            ({"diffusion_tuning_cache_size": float("inf")}, "diffusion_tuning_cache_size"),
            ({"diffusion_tuning_cache_size": "invalid"}, "diffusion_tuning_cache_size"),
        ],
    )
    def test_rejects_non_positive_inference_steps(self, kwargs, match):
        class Parent:
            def __init__(self, *args, **kwargs):
                self.model_context = SimpleNamespace(pipe=None, model=None)

        class MockCompressor(DiffusionMixin, Parent):
            pass

        with pytest.raises(ValueError, match=match):
            MockCompressor(**kwargs)

    def test_get_calibrator_kind_returns_diffusion(self):
        # Create a minimal mock class
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                # Don't call super().__init__() to avoid needing real parent
                pass

        comp = MockCompressor()
        assert comp._get_calibrator_kind() == "diffusion"

    def test_quantize_does_not_use_removed_quantizer_attribute(self):
        assert "self.quantizer" not in inspect.getsource(DiffusionMixin.quantize)

    def test_pipeline_call_kwargs_extracted_from_kwargs(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                pass

        comp = MockCompressor()
        # Set the attribute directly since we're not calling super().__init__
        comp.pipeline_call_kwargs = {"height": 512, "width": 512}
        assert comp.pipeline_call_kwargs.get("height") == 512

    def test_align_pipeline_dtype_preserves_only_declared_fp32_tensors(self, monkeypatch):
        protected = torch.nn.Linear(2, 2)
        protected._keep_in_fp32_modules = ["weight"]
        with torch.no_grad():
            protected.weight.fill_(1.001)
        expected_weight = protected.weight.detach().clone()
        ordinary = torch.nn.Linear(2, 2)
        pipe = SimpleNamespace(components=["protected", "ordinary"], protected=protected, ordinary=ordinary)

        # AMP runs before pipeline alignment; restoring FP32 afterwards is too late.
        protected.dtype = torch.float32
        context = SimpleNamespace(model=protected, amp=True, device="cpu")
        device = MagicMock()
        device.prefers_bf16.return_value = True
        device.supports_bf16.return_value = True
        monkeypatch.setattr("auto_round.context.model.get_ar_device", lambda _: device)
        ModelContext._set_amp_dtype(context)
        DiffusionMixin._align_pipeline_dtype(pipe, torch.bfloat16)

        assert protected.weight.dtype == torch.float32
        assert torch.equal(protected.weight, expected_weight)
        assert protected.bias.dtype == torch.bfloat16
        assert ordinary.weight.dtype == torch.bfloat16
        assert ordinary.bias.dtype == torch.bfloat16


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


@pytest.mark.parametrize("fail_secondary", [False, True])
def test_zero_shot_quantizes_secondary_and_restores_primary(monkeypatch, fail_secondary):
    primary = torch.nn.Linear(4, 4)
    secondary = torch.nn.Linear(4, 4)
    calls = []

    class Compressor(DiffusionMixin):
        def __init__(self):
            self.model_context = SimpleNamespace(
                model=primary,
                quantized=False,
                pipe=SimpleNamespace(
                    components=["transformer", "transformer_2"], transformer=primary, transformer_2=secondary
                ),
            )
            self.compress_context = SimpleNamespace(
                low_cpu_mem_usage=True, is_immediate_packing=True, is_immediate_saving=True
            )
            self.layer_config = {"primary": {}}
            self.quant_block_list = [["primary"]]
            self._diffusion_user_block_selector = None
            self._diffusion_user_layer_config = None
            self.compression_plan = SimpleNamespace(regex_config={"primary": {}}, has_qlayer_outside_block=False)
            self.to_quant_block_names = "primary"

        def post_init(self):
            assert self.quant_block_list == [["secondary"]]
            assert self.to_quant_block_names is None
            self.compression_plan = SimpleNamespace(regex_config={"secondary": {}}, has_qlayer_outside_block=True)
            self.layer_config = {"secondary": {}}
            self.compress_context.is_immediate_packing = True
            self.compress_context.is_immediate_saving = True

        def _quantize_zero_shot(self):
            assert not self.compress_context.is_immediate_packing
            assert not self.compress_context.is_immediate_saving
            calls.append(self.model_context.model)
            if fail_secondary and self.model_context.model is secondary:
                raise RuntimeError("secondary failed")

    monkeypatch.setattr("auto_round.utils.get_block_names", lambda _: [["secondary"]])
    monkeypatch.setattr("auto_round.utils.find_matching_blocks", lambda model, blocks, pattern: blocks)
    comp = Compressor()
    if fail_secondary:
        with pytest.raises(RuntimeError, match="secondary failed"):
            comp._quantize_all_transformers_zero_shot()
    else:
        comp._quantize_all_transformers_zero_shot()
        assert comp._quantized_transformers == {"transformer_2": (secondary, {"secondary": {}})}
    assert calls == [primary, secondary]
    assert comp.model_context.model is primary
    assert comp.model_context.quantized
    assert comp.compression_plan.regex_config == {"primary": {}}
    assert not comp.compression_plan.has_qlayer_outside_block
    assert comp.to_quant_block_names == "primary"
    assert comp.layer_config == {"primary": {}}
    assert comp.quant_block_list == [["primary"]]
    assert comp.compress_context.low_cpu_mem_usage
    assert comp.compress_context.is_immediate_packing
    assert comp.compress_context.is_immediate_saving


@pytest.mark.parametrize("selector", [None, r"\.1$"])
def test_zero_shot_resolves_component_plan_before_initialization(monkeypatch, selector):
    from auto_round.compressors.base import BaseOrchestrator
    from auto_round.compressors.config_resolution.contracts import (
        FormatResolution,
        ResolvedQuantizationConfig,
        ResolvedScheme,
    )
    from auto_round.schemes import QuantizationScheme

    primary, secondary = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    scheme = ResolvedScheme.from_scheme(QuantizationScheme(bits=8))
    primary_blocks = [["blocks.0", "blocks.1"]] if selector is None else [["blocks.1"]]
    secondary_blocks = [["layers.0", "layers.1", "layers.2"]] if selector is None else [["layers.1"]]
    primary_plan = ResolvedQuantizationConfig(
        scheme=scheme,
        formats=(),
        layer_config={"primary": {"bits": 8}},
        regex_config={"primary": {"bits": 8}},
        quant_block_list=primary_blocks,
    )
    secondary_plan = ResolvedQuantizationConfig(
        scheme=scheme,
        formats=(),
        layer_config={"secondary": {"bits": 8}},
        regex_config={"secondary": {"bits": 8}},
        quant_block_list=secondary_blocks,
        has_qlayer_outside_block=True,
    )

    class Compressor(DiffusionMixin, BaseOrchestrator):
        def __init__(self):
            self.model_context = SimpleNamespace(
                model=primary,
                quantized=False,
                pipe=SimpleNamespace(
                    components=["transformer", "transformer_2"], transformer=primary, transformer_2=secondary
                ),
            )
            self.compress_context = SimpleNamespace(
                low_cpu_mem_usage=False, is_immediate_packing=False, is_immediate_saving=False
            )
            self.compression_plan = primary_plan
            self._format_resolution = FormatResolution(formats=(), scheme=scheme, quant_block_list=primary_blocks)
            self._diffusion_user_block_selector = selector
            self._diffusion_user_layer_config = None
            self.to_quant_block_names = selector or "blocks"
            self._alg_composer = object()

        def post_init(self):
            assert self.quant_block_list == secondary_blocks
            assert self._format_resolution.quant_block_list == tuple(tuple(group) for group in secondary_blocks)
            self.compression_plan = secondary_plan
            self._alg_composer = object()

        def _quantize_zero_shot(self):
            self.model_context.quantized = True

    monkeypatch.setattr("auto_round.utils.get_block_names", lambda model: [["layers.0", "layers.1", "layers.2"]])
    comp = Compressor()
    primary_composer = comp._alg_composer
    comp._quantize_all_transformers_zero_shot()
    assert comp.compression_plan == primary_plan
    assert comp._alg_composer is primary_composer
    assert comp._quantized_transformer_plans["transformer_2"] == secondary_plan


@pytest.mark.parametrize("fail_save", [False, True])
def test_secondary_export_selects_and_restores_full_plan(tmp_path, monkeypatch, fail_save):
    primary, secondary = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    primary_plan, secondary_plan = object(), object()
    calls = []

    class Parent:
        def save_quantized(self, **kwargs):
            calls.append((self.model_context.model, self.compression_plan))
            if self.model_context.model is secondary and fail_save:
                raise RuntimeError("export failed")
            return self.model_context.model

    class Compressor(DiffusionMixin, Parent):
        def __init__(self):
            self.model_context = SimpleNamespace(
                model=primary,
                pipe=SimpleNamespace(
                    components={"transformer": primary, "transformer_2": secondary},
                    transformer=primary,
                    transformer_2=secondary,
                    save_config=lambda path: None,
                ),
            )
            self.compress_context = SimpleNamespace(is_immediate_saving=True)
            self.compression_plan = primary_plan
            self.layer_config = {"primary": {}}
            self._quantized_transformers = {"transformer_2": (secondary, {"secondary": {}})}
            self._quantized_transformer_plans = {"transformer_2": secondary_plan}

    monkeypatch.setattr("auto_round.compressors.diffusion_mixin.rename_weights_files", lambda path: None)
    comp = Compressor()
    if fail_save:
        with pytest.raises(RuntimeError, match="export failed"):
            comp.save_quantized(str(tmp_path), format=[])
    else:
        comp.save_quantized(str(tmp_path), format=[])
    assert calls == [(primary, primary_plan), (secondary, secondary_plan)]
    assert comp.model_context.model is primary
    assert comp.compression_plan is primary_plan
    assert comp.layer_config == {"primary": {}}
    assert comp.compress_context.is_immediate_saving
