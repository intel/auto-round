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
"""Tests for ``auto_round/calibration/diffusion.py``."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from auto_round.calibration.diffusion import DiffusionCalibrator
from auto_round.calibration.register import get_calibrator
from auto_round.utils.device_manager import device_manager
from auto_round.utils.model import wrap_block_forward_positional_to_kwargs


class FakeTqdm:
    """Minimal stand-in for ``tqdm`` used in diffusion calibration."""

    def __init__(self, iterable, desc=None):
        self._iterable = list(iterable)

    def __iter__(self):
        yield from self._iterable

    def update(self, step):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakePipeline:
    """Object that can be both called and moved to a device."""

    def __init__(self, device=torch.device("cpu"), fn=None):
        self.device = device
        self._fn = fn or (lambda *args, **kwargs: None)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def to(self, device):
        self.device = torch.device(device)
        return self


class TestDiffusionCalibrator:
    """Mocks keep everything CPU-only and fast."""

    def test_is_registered_as_diffusion(self):
        assert get_calibrator("diffusion") is DiffusionCalibrator

    @pytest.fixture()
    def calibrator(self, monkeypatch):
        compressor = SimpleNamespace(
            dataset="mock",
            batch_size=2,
            seed=0,
            nsamples=4,
            gradient_accumulate_steps=1,
            guidance_scale=7.5,
            num_inference_steps=1,
            generator_seed=None,
            seqlen=128,
            inputs={},
            dataloader=None,
            model=SimpleNamespace(hf_device_map={"cpu": 0}, device="cpu"),
            model_context=SimpleNamespace(pipe=FakePipeline()),
        )

        calib = DiffusionCalibrator(SimpleNamespace(compress_context=SimpleNamespace()))
        calib.compressor = compressor

        monkeypatch.setattr(device_manager, "device", "cpu")
        monkeypatch.setattr("auto_round.calibration.diffusion.logger.warning", lambda *args, **kwargs: None)
        monkeypatch.setattr("auto_round.calibration.diffusion.logger.error", lambda *args, **kwargs: None)

        return calib

    def test_should_stop_never_stops(self, calibrator):
        assert calibrator.should_stop("any_block") is False

    def test_wrap_block_forward_delegates_to_utility(self, calibrator):
        seen = []

        def base_hook(m, hidden_states, *args, **kwargs):
            seen.append((hidden_states, kwargs))
            return (hidden_states,)

        wrapped = calibrator.wrap_block_forward(base_hook)

        class DummyBlock:
            def forward(self, hidden_states, encoder_hidden_states, temb=None):
                return (hidden_states, encoder_hidden_states, temb)

            def __call__(self, hidden_states, encoder_hidden_states=None, temb=None, **kwargs):
                return self.forward(hidden_states, encoder_hidden_states, temb=temb, **kwargs)

        module = DummyBlock()
        module.orig_forward = module.forward
        result = wrapped(module, torch.ones(1), torch.ones(1), temb=torch.ones(1))
        assert result == (torch.ones(1),)
        assert seen == [(torch.ones(1), {"encoder_hidden_states": torch.ones(1), "temb": torch.ones(1)})]

    def test_calib_raises_when_pipeline_missing(self, calibrator):
        calibrator.compressor.model_context.pipe = None

        with pytest.raises(ValueError, match="Diffusion pipeline not found"):
            calibrator.calib(nsamples=1, bs=1)

    def test_calib_string_dataset_reloads_dataloader(self, calibrator):
        new_dataloader = [("id0", ["p1", "p2"])]
        calibrator.compressor.dataset = "mock_dataset"
        calibrator.compressor.model_context.pipe = FakePipeline(
            fn=lambda *args, **kwargs: None
        )
        calibrator.compressor._requires_calibration_image = lambda: False
        calibrator.compressor._autoround_pipeline_fn = None

        def fake_calib(nsamples, bs):
            calibrator.compressor.inputs = {"block_0": {"positional_inputs": ["p1", "p2"]}}

        calibrator.compressor.calib = fake_calib

        with patch(
            "auto_round.compressors.diffusion.dataset.get_diffusion_dataloader",
            return_value=(new_dataloader, 2, 1),
        ), patch("auto_round.calibration.diffusion.tqdm", FakeTqdm), patch(
            "auto_round.calibration.diffusion.device_manager",
            SimpleNamespace(device="cpu"),
        ):
            calibrator.calib(nsamples=2, bs=1)

        assert calibrator.compressor.dataloader is new_dataloader
        assert calibrator.compressor.batch_size == 2

    def test_calib_non_string_dataset_keeps_existing_dataloader(self, calibrator):
        calibrator.compressor.dataset = [("id0", ["p1", "p2"])]
        calibrator.compressor.model_context.pipe = FakePipeline(
            fn=lambda *args, **kwargs: None
        )
        calibrator.compressor._requires_calibration_image = lambda: False
        calibrator.compressor._autoround_pipeline_fn = None

        def fake_calib(nsamples, bs):
            calibrator.compressor.inputs = {"block_0": {"positional_inputs": ["p1", "p2"]}}

        calibrator.compressor.calib = fake_calib

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=2, bs=1)

        assert calibrator.compressor.dataloader is calibrator.compressor.dataset

    def test_calib_uses_dataloader_len_when_available(self, calibrator):
        class FakeDataloader:
            def __len__(self):
                return 1

            def __iter__(self):
                return iter([("id0", ["p1"])])

        calibrator.compressor.dataset = FakeDataloader()
        calibrator.compressor.model_context.pipe = FakePipeline(
            fn=lambda *args, **kwargs: None
        )
        calibrator.compressor._requires_calibration_image = lambda: False

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=1, bs=1)

        assert len(calibrator.compressor.inputs) == 0

    def test_calib_exits_on_multi_device_offload(self, calibrator):
        calibrator.compressor.model.hf_device_map = {"cpu": 0, "cuda:0": 1}
        calibrator.compressor.model.device = "cpu"
        calibrator.compressor.model_context.pipe = FakePipeline(
            device=torch.device("cpu"),
            fn=lambda *args, **kwargs: None,
        )
        calibrator.compressor.dataset = "mock"

        with patch(
            "auto_round.compressors.diffusion.dataset.get_diffusion_dataloader",
            return_value=([], 2, 1),
        ), patch(
            "auto_round.calibration.diffusion.device_manager",
            SimpleNamespace(device="cuda:0"),
        ):
            with pytest.raises(SystemExit):
                calibrator.calib(nsamples=1, bs=1)

    def test_calib_moves_pipeline_to_target_device(self, calibrator):
        seen = []
        calibrator.compressor.dataset = [("id0", ["p1", "p2"])]
        calibrator.compressor.model_context.pipe = FakePipeline(
            device=torch.device("cpu"),
            fn=lambda *args, **kwargs: None,
        )

        def fake_to(device):
            seen.append(device)
            return calibrator.compressor.model_context.pipe

        calibrator.compressor.model_context.pipe.to = fake_to

        def fake_calib(nsamples, bs):
            calibrator.compressor.inputs = {"block_0": {"positional_inputs": ["p1", "p2"]}}

        calibrator.compressor.calib = fake_calib

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm), patch(
            "auto_round.calibration.diffusion.device_manager",
            SimpleNamespace(device="cuda:0"),
        ):
            calibrator.calib(nsamples=2, bs=1)

        assert seen == ['cuda:0']

    def test_calib_uses_autoround_pipeline_fn_when_available(self, calibrator):
        calls = []

        def pipeline_fn(pipe, prompts, **kwargs):
            calls.append((pipe, prompts, kwargs))

        calibrator.compressor.dataset = [("id0", ["p1", "p2"])]
        calibrator.compressor.model_context.pipe = FakePipeline(
            fn=lambda *args, **kwargs: None
        )
        calibrator.compressor.model_context.pipe._autoround_pipeline_fn = pipeline_fn

        def fake_calib(nsamples, bs):
            calibrator.compressor.inputs = {"block_0": {"positional_inputs": ["p1", "p2"]}}

        calibrator.compressor.calib = fake_calib

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=2, bs=1)

        assert len(calls) == 1
        assert calls[0][1] == ["p1", "p2"]
        assert calls[0][2]["guidance_scale"] == pytest.approx(7.5)
        assert calls[0][2]["generator"] is None

    def test_calib_falls_back_to_pipe_when_no_pipeline_fn(self, calibrator):
        calls = []

        def fake_pipe(prompts, **kwargs):
            calls.append((prompts, kwargs))

        calibrator.compressor.dataset = [("id0", ["p1", "p2"])]
        calibrator.compressor.model_context.pipe = FakePipeline(fn=fake_pipe)
        calibrator.compressor._requires_calibration_image = lambda: False
        calibrator.compressor.model_context.pipe._autoround_pipeline_fn = None

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=2, bs=1)

        assert len(calls) == 1
        assert calls[0][0] == ["p1", "p2"]

    def test_calib_passes_image_when_required(self, calibrator):
        calls = []
        seen_images = []

        def fake_pipe(image, prompt=None, **kwargs):
            calls.append((image, prompt, kwargs))
            seen_images.append(image)

        calibrator.compressor.dataset = [("id0", ["p1"])]
        calibrator.compressor.model_context.pipe = FakePipeline(fn=fake_pipe)
        calibrator.compressor._requires_calibration_image = lambda: True
        calibrator.compressor._get_calibration_image = lambda batch_size: torch.randn(
            batch_size, 4, 64, 64
        )
        calibrator.compressor.model_context.pipe._autoround_pipeline_fn = None

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=1, bs=1)

        assert seen_images[0].shape == (1, 4, 64, 64)

    def test_calib_not_implemented_error_is_swallowed(self, calibrator):
        def failing_pipe(*args, **kwargs):
            raise NotImplementedError("unsupported op")

        calibrator.compressor.dataset = [("id0", ["p1"])]
        calibrator.compressor.model_context.pipe = FakePipeline(fn=failing_pipe)
        calibrator.compressor._requires_calibration_image = lambda: False
        calibrator.compressor.model_context.pipe._autoround_pipeline_fn = None

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=1, bs=1)

        assert calibrator.compressor.inputs == {}

    def test_calib_other_exceptions_propagate(self, calibrator):
        def failing_pipe(*args, **kwargs):
            raise RuntimeError("unexpected")

        calibrator.compressor.dataset = [("id0", ["p1"])]
        calibrator.compressor.model_context.pipe = FakePipeline(fn=failing_pipe)
        calibrator.compressor._requires_calibration_image = lambda: False
        calibrator.compressor.model_context.pipe._autoround_pipeline_fn = None

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            with pytest.raises(RuntimeError, match="unexpected"):
                calibrator.calib(nsamples=1, bs=1)

    def test_calib_single_sample_stops_early(self, calibrator):
        seen = []

        def fake_pipe(prompts, **kwargs):
            seen.append(len(prompts) if isinstance(prompts, list) else 1)

        calibrator.compressor.dataset = [("id0", ["p1", "p2"])]
        calibrator.compressor.model_context.pipe = FakePipeline(fn=fake_pipe)
        calibrator.compressor._requires_calibration_image = lambda: False
        calibrator.compressor.model_context.pipe._autoround_pipeline_fn = None

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=2, bs=2)

        assert seen == [2]

    def test_calib_zero_samples_exits(self, calibrator):
        calibrator.compressor.dataset = [("id0", [])]
        calibrator.compressor.model_context.pipe = FakePipeline(fn=lambda *args, **kwargs: None)
        calibrator.compressor._requires_calibration_image = lambda: False
        calibrator.compressor.model_context.pipe._autoround_pipeline_fn = None

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            with pytest.raises(SystemExit):
                calibrator.calib(nsamples=1, bs=1)

    def test_calib_insufficient_samples_warns_and_truncates(self, calibrator):
        def fake_pipe(prompts, **kwargs):
            return None

        calibrator.compressor.dataset = [("id0", ["p1"]), ("id1", ["p2"])]
        calibrator.compressor.model_context.pipe = FakePipeline(fn=fake_pipe)
        calibrator.compressor._requires_calibration_image = lambda: False
        calibrator.compressor.model_context.pipe._autoround_pipeline_fn = None

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=3, bs=2)

        assert all(len(v["positional_inputs"]) == 2 for v in calibrator.compressor.inputs.values())

    def test_calib_insufficient_below_batch_size_raises(self, calibrator):
        def fake_pipe(prompts, **kwargs):
            return None

        calibrator.compressor.dataset = [("id0", ["p1"])]
        calibrator.compressor.model_context.pipe = FakePipeline(fn=fake_pipe)
        calibrator.compressor._requires_calibration_image = lambda: False
        calibrator.compressor.model_context.pipe._autoround_pipeline_fn = None

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            with pytest.raises(ValueError, match="valid samples is less than batch_size"):
                calibrator.calib(nsamples=3, bs=2)
