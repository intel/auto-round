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

from auto_round.compressors.base import BaseCompressor
from auto_round.compressors.config_resolution import ResolvedQuantizationConfig, ResolvedScheme
from auto_round.compressors.diffusion_mixin import DiffusionMixin
from auto_round.context.model import ModelContext
from auto_round.schemes import MXFP8


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

    @pytest.mark.parametrize("need_calib", [False, True])
    @pytest.mark.parametrize("failing_component", [None, "transformer_2"])
    def test_quantizes_and_exports_both_transformers(self, need_calib, failing_component, tmp_path, monkeypatch):
        class MockCompressor(DiffusionMixin, BaseCompressor):
            pass

        saved = []

        def export(_, model, serialization_dict, **kwargs):
            saved.append(serialization_dict)
            if failing_component and model is getattr(pipe, failing_component):
                raise RuntimeError("export failed")
            return model

        fmt = SimpleNamespace(is_fake=lambda: False, save_quantized=export)
        primary = torch.nn.ModuleDict({"blocks": torch.nn.ModuleList([torch.nn.Linear(4, 4)])})
        secondary = torch.nn.ModuleDict({"layers": torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])})
        pipe = SimpleNamespace(
            components={"transformer": primary, "transformer_2": secondary},
            transformer=primary,
            transformer_2=secondary,
            save_config=MagicMock(),
        )
        primary._autoround_pipeline_subfolder = "transformer"
        secondary._autoround_pipeline_subfolder = "transformer_2"
        comp = MockCompressor.__new__(MockCompressor)
        comp.model_context = SimpleNamespace(pipe=pipe, model=primary, tokenizer=None, quantized=False)
        comp.quantize_config = SimpleNamespace(act_bits=8, optimizer="sgd")
        comp.compress_context = SimpleNamespace(
            low_cpu_mem_usage=False, is_immediate_packing=False, is_immediate_saving=False
        )
        comp.need_calib = need_calib
        comp.calib_num_inference_steps = 2
        comp.calibration = None
        comp.calibration_context = SimpleNamespace(nsamples=1)
        comp.has_variable_block_shape = False
        comp.try_cache_inter_data_gpucpu = MagicMock(return_value={})
        comp.quant_block_list = [["blocks.0"]]
        comp.to_quant_block_names = "blocks"
        comp._scheme_resolved = True
        comp.is_auto_scheme = False

        def configure_layer_config(**kwargs):
            comp.compression_plan = ResolvedQuantizationConfig(
                scheme=ResolvedScheme.from_scheme(MXFP8),
                formats=(fmt,),
                layer_config={name: {"bits": 8} for group in comp.quant_block_list for name in group},
                regex_config={comp.to_quant_block_names: {"bits": 8}},
                quant_block_list=comp.quant_block_list,
            )

        comp.configure_layer_config = configure_layer_config
        comp.post_init = lambda: BaseCompressor._scheme_post_init(comp)
        quantize = MagicMock(side_effect=lambda: (comp.model_context.model, comp.layer_config))
        monkeypatch.setattr(BaseCompressor, "quantize", quantize)
        monkeypatch.setattr("auto_round.compressors.diffusion_mixin.clear_memory", lambda **kwargs: None)
        model, _ = comp.quantize()
        assert model is primary
        assert quantize.call_count == 2
        assert comp.try_cache_inter_data_gpucpu.call_count == (2 if need_calib else 0)
        primary_state = comp._capture_transformer_quantization_state()
        monkeypatch.setattr("auto_round.compressors.base._get_save_folder_name", lambda _: str(tmp_path))
        monkeypatch.setattr("auto_round.compressors.diffusion_mixin.rename_weights_files", lambda _: None)
        comp.compress_context.is_immediate_saving = True
        if failing_component:
            with pytest.raises(RuntimeError, match="export failed"):
                comp.save_quantized(str(tmp_path), format=[fmt])
        else:
            comp.save_quantized(str(tmp_path), format=[fmt])
        for metadata, block_names in zip(saved, (["blocks.0"], ["layers.0", "layers.1"])):
            assert metadata["quant_block_list"] == [block_names]
            assert metadata["to_quant_block_names"] == block_names[0].rsplit(".", 1)[0]
            assert metadata["regex_config"] == {metadata["to_quant_block_names"]: {"bits": 8}}
        assert len(saved) == 2
        assert comp.model_context.model is primary
        assert comp._capture_transformer_quantization_state() == primary_state
        assert comp.compress_context.is_immediate_saving
        assert primary._autoround_pipeline_subfolder == "transformer"
        assert secondary._autoround_pipeline_subfolder == "transformer_2"


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
