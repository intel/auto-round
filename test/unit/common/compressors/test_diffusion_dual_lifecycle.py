# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Small lifecycle tests through the real orchestrator post-init/quantize boundary."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from accelerate.hooks import ModelHook, add_hook_to_module

from auto_round.compressors.diffusion_mixin import DiffusionMixin
from auto_round.compressors.orchestrator import CompressionOrchestrator


class TinyPipe:
    def __init__(self):
        self.transformer = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.transformer_2 = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.vae = torch.nn.Linear(2, 2)
        self.components = {name: getattr(self, name) for name in ("transformer", "transformer_2", "vae")}
        self.config = SimpleNamespace(boundary_ratio=0.9)
        self.to = MagicMock(side_effect=AssertionError("whole pipeline transfer"))

    def register_to_config(self, **kwargs):
        vars(self.config).update(kwargs)


class TinyCompressor(DiffusionMixin, CompressionOrchestrator):
    def __init__(self):
        self.model_context = SimpleNamespace(model=None, pipe=TinyPipe(), quantized=False, amp_dtype=torch.float32)
        self.model_context.model = self.model_context.pipe.transformer
        self.compress_context = SimpleNamespace(
            low_cpu_mem_usage=True,
            low_gpu_mem_usage=True,
            is_immediate_packing=True,
            is_immediate_saving=True,
            device_map="cuda:0",
            device_list=["cuda:0"],
        )
        self.calibration_context = SimpleNamespace(nsamples=1)
        self.quantize_config = SimpleNamespace(is_act_quantize=False)
        self.low_gpu_mem_usage = True
        self.need_calib = True
        self._post_init_done = False
        self.calibration = None
        self.calib_num_inference_steps = 1
        self.layer_config = {}
        self.quant_block_list = [["0"]]
        self.has_variable_block_shape = False
        self.events = []
        self.fail_at = None
        self.to_quant_block_names = None

    def _resolve_scheme(self):
        pass

    def _resolve_formats(self):
        self.compress_context.is_immediate_packing = True
        self.compress_context.is_immediate_saving = True

    def _patch_model(self):
        pass

    def _build_layer_config(self):
        if self.fail_at == ("init", self.model):
            raise RuntimeError("initialization failure")
        self.layer_config = {
            "0": {"bits": self.layer_config.get("0", {}).get("bits", 4), "initialized": id(self.model)}
        }
        self.quant_block_list = [["0"]]

    def _apply_rotations(self):
        pass

    def _hardware_setup(self):
        pass

    def _build_composer(self):
        self._alg_composer = SimpleNamespace(model=self.model, block_forward=SimpleNamespace())

    def _collect_inputs(self, *args, **kwargs):
        # The real calibrator binds model/steps at construction, not dynamically.
        assert self.calibration.model is self.model
        assert self.calibration.calib_num_inference_steps == 2
        assert not self.compress_context.is_immediate_packing
        if self.fail_at == ("cache", self.model):
            raise RuntimeError("cache failure")
        for component in self.model_context.pipe.components.values():
            add_hook_to_module(component, ModelHook())
        self.calibration._cpu_offload_mode = "model"
        self.events.append(("cache", self.model))
        return {"0": torch.ones(1, 2)}

    def _quantize_data_driven(self):
        assert not hasattr(self.model, "_hf_hook"), "calibration hook reached block tuning"
        assert self.layer_config["0"]["initialized"] == id(self.model)
        assert self.layer_config["0"]["bits"] == getattr(self, "expected_bits", 4)
        assert not self.compress_context.is_immediate_saving
        assert self.cache_data(["0"], 1) is self.inputs
        if self.fail_at == ("tune", self.model):
            raise RuntimeError("tuning failure")
        self.layer_config = {"0": {"bits": 4, "tuned": id(self.model)}}
        self.model.tuned = True
        self.model_context.quantized = True
        self.events.append(("tune", self.model))
        return self.model, self.layer_config


@pytest.fixture
def comp(monkeypatch):
    class TinyCalibrator:
        def __init__(self, compressor):
            self.compressor = compressor
            self.model = compressor.model
            self.calib_num_inference_steps = compressor.calib_num_inference_steps
            self.batch_size = 1
            self.seqlen = 1
            self.batch_dim = 0
            self.dataset = []
            self.is_only_supported_bs1 = False

        def __call__(self, *args, **kwargs):
            return self.compressor._collect_inputs(*args, **kwargs)

    monkeypatch.setattr("auto_round.calibration.get_calibrator", lambda kind: TinyCalibrator)
    monkeypatch.setattr("auto_round.utils.get_block_names", lambda model: [["0"]])
    monkeypatch.setattr("auto_round.utils.find_matching_blocks", lambda model, blocks, names: blocks)
    return TinyCompressor()


def test_low_gpu_secondary_does_not_move_pipeline(comp):
    comp.model_context.model = comp.model_context.pipe.transformer_2
    comp._align_device_and_dtype_for_secondary("transformer_2")
    comp.model_context.pipe.to.assert_not_called()


def test_dual_lifecycle_preserves_tuned_configs_and_primary_runtime(comp):
    primary = comp.model
    secondary = comp.model_context.pipe.transformer_2
    model, config = comp.quantize()
    assert model is primary
    assert config == {"0": {"bits": 4, "tuned": id(primary)}}
    assert comp._quantized_transformers["transformer_2"] == (secondary, {"0": {"bits": 4, "tuned": id(secondary)}})
    assert comp.alg_composer.model is primary
    assert comp.calibration.model is primary
    assert comp.model_context.quantized
    assert comp.calib_num_inference_steps == 1
    assert comp.model_context.pipe.config.boundary_ratio == 0.9
    assert comp.compress_context.low_cpu_mem_usage
    assert comp.compress_context.is_immediate_packing
    assert comp.compress_context.is_immediate_saving
    assert comp.events == [("cache", primary), ("tune", primary), ("cache", secondary), ("tune", secondary)]


@pytest.mark.parametrize("stage", ["cache", "tune"])
@pytest.mark.parametrize("expert", ["transformer", "transformer_2"])
def test_dual_failure_restores_primary_context_without_success(comp, stage, expert):
    primary = comp.model
    comp.fail_at = (stage, getattr(comp.model_context.pipe, expert))
    with pytest.raises(RuntimeError, match="failure"):
        comp.quantize()
    assert comp.model is primary
    assert not comp.model_context.quantized
    assert comp.alg_composer.model is primary
    assert comp.calibration.model is primary
    assert comp._post_init_done
    assert not comp._inputs_cached
    assert comp.calib_num_inference_steps == 1
    assert comp.model_context.pipe.config.boundary_ratio == 0.9
    assert comp.compress_context.low_cpu_mem_usage
    assert comp.compress_context.is_immediate_packing
    assert comp.compress_context.is_immediate_saving


@pytest.fixture
def saved_comp(comp, monkeypatch):
    from safetensors.torch import save_file

    from auto_round.compressors.base import BaseOrchestrator

    comp.model_context.quantized = True
    comp.layer_config = {"primary": {"bits": 4}}
    comp._quantized_transformers = {
        "transformer_2": (comp.model_context.pipe.transformer_2, {"secondary": {"bits": 4}})
    }
    comp.formats = [SimpleNamespace(format_name="svdquant_nunchaku")]
    for name in ("transformer", "transformer_2"):
        getattr(comp.model_context.pipe, name)._autoround_pipeline_subfolder = name
    comp.model_context.pipe.config = {
        "_class_name": "WanPipeline",
        "boundary_ratio": 0.9,
        "transformer": ["diffusers", "WanTransformer3DModel"],
        "transformer_2": ["diffusers", "WanTransformer3DModel"],
    }
    comp.model_context.pipe.vae = None
    comp.save_adapter = "NunchakuWanTransformer3DModel"
    comp.fail_save = False
    comp.saved_configs = []

    def save_quantized(self, output_dir, **kwargs):
        from pathlib import Path

        self.saved_configs.append(self.layer_config)
        if self.fail_save and self.model is self.model_context.pipe.transformer_2:
            raise RuntimeError("export failure")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_file(
            {"test": torch.ones(1)},
            str(Path(output_dir) / "diffusion_pytorch_model.safetensors"),
            metadata={
                "model_class": self.save_adapter,
                "config": "{}",
                "quantization_config": json.dumps({"method": "svdquant"}),
            },
        )
        return self.model

    monkeypatch.setattr(BaseOrchestrator, "save_quantized", save_quantized)
    return comp


@pytest.mark.parametrize("format_name", ["svdquant_nunchaku", "auto_round"])
@pytest.mark.parametrize("adapter", ["NunchakuWanTransformer3DModel", "NunchakuFluxTransformer2dModel"])
def test_save_index_uses_actual_wan_nunchaku_export_only(saved_comp, tmp_path, format_name, adapter):
    comp = saved_comp
    comp.save_adapter = adapter
    comp.formats = [SimpleNamespace(format_name=format_name)]
    original_config = dict(comp.model_context.pipe.config)
    comp.save_quantized(str(tmp_path))
    index = json.loads((tmp_path / "model_index.json").read_text())
    expected = (
        ["nunchaku", "NunchakuWanTransformer3DModel"]
        if (format_name == "svdquant_nunchaku" and adapter == "NunchakuWanTransformer3DModel")
        else ["diffusers", "WanTransformer3DModel"]
    )
    assert index["transformer"] == expected
    assert index["transformer_2"] == expected
    assert comp.model_context.pipe.config == original_config
    assert comp.saved_configs == [{"primary": {"bits": 4}}, {"secondary": {"bits": 4}}]


def test_secondary_export_failure_restores_context(saved_comp, tmp_path):
    comp = saved_comp
    primary = comp.model
    primary_config = comp.layer_config
    comp.fail_save = True
    with pytest.raises(RuntimeError, match="export failure"):
        comp.save_quantized(str(tmp_path))
    assert comp.model is primary
    assert comp.layer_config == primary_config
    assert comp.compress_context.is_immediate_saving
    assert primary._autoround_pipeline_subfolder == "transformer"
    assert comp.model_context.pipe.transformer_2._autoround_pipeline_subfolder == "transformer_2"


def test_secondary_initialization_failure_restores_primary(comp):
    primary = comp.model
    comp.fail_at = ("init", comp.model_context.pipe.transformer_2)
    with pytest.raises(RuntimeError, match="initialization failure"):
        comp.quantize()
    assert comp.model is primary
    assert comp.calibration.model is primary
    assert not comp.model_context.quantized
    assert comp.model_context.pipe.config.boundary_ratio == 0.9
    assert comp.compress_context.is_immediate_saving


def test_secondary_preserves_requested_layer_overrides(comp):
    comp.layer_config = {"0": {"bits": 8}}
    comp.expected_bits = 8
    comp.quantize()
