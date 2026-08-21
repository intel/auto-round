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
        self._autoround_pipeline_fn = None

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def to(self, device):
        self.device = torch.device(device)
        return self


class ImagePipeline(FakePipeline):
    """I2V-style pipeline whose ``__call__`` requires a positional ``image``."""

    def __call__(self, image, prompt=None, **kwargs):
        return self._fn(image, prompt=prompt, **kwargs)


class TestDiffusionCalibrator:
    """Mocks keep everything CPU-only and fast."""

    def test_is_registered_as_diffusion(self):
        assert get_calibrator("diffusion") is DiffusionCalibrator

    @pytest.fixture()
    def calibrator(self, monkeypatch):
        # DiffusionCalibrator copies compressor state at __init__ (see
        # Calibrator.__init__ / DiffusionCalibrator.__init__), so build a
        # compressor namespace exposing every attribute those constructors read.
        model = SimpleNamespace(hf_device_map={"cpu": 0}, device="cpu")
        pipe = FakePipeline()
        compressor = SimpleNamespace(
            dataset="mock",
            seed=0,
            low_gpu_mem_usage=False,
            has_variable_block_shape=False,
            guidance_scale=7.5,
            num_inference_steps=1,
            generator_seed=None,
            max_cached_calibration_inputs=None,
            pipe=pipe,
            model=model,
            model_context=SimpleNamespace(
                model=model,
                tokenizer=None,
                shared_cache_keys=(),
            ),
            calibration_context=SimpleNamespace(
                batch_size=2,
                batch_dim=0,
                seqlen=128,
            ),
        )

        calib = DiffusionCalibrator(compressor)

        monkeypatch.setattr(device_manager, "device", "cpu")
        monkeypatch.setattr("auto_round.calibration.diffusion.logger.warning", lambda *args, **kwargs: None)
        monkeypatch.setattr("auto_round.calibration.diffusion.logger.error", lambda *args, **kwargs: None)

        return calib

    def test_should_stop_never_stops(self, calibrator):
        # DiffusionCalibrator inherits the base always-False stop policy so all
        # denoising steps execute during calibration.
        assert calibrator._should_stop_cache_forward("any_block") is False

    def test_cache_input_limit_bounds_retained_tensors(self, calibrator):
        calibrator.max_cached_calibration_inputs = 3
        calibrator.batch_size = 1

        block = torch.nn.Identity()
        block.orig_forward = block.forward
        capture = calibrator._make_block_forward_func("block")

        for value in range(10):
            capture(block, torch.tensor([[float(value)]]))

        retained = calibrator.inputs["block"]["hidden_states"]
        assert len(retained) == 3
        assert any(tensor.item() >= 3 for tensor in retained)

    def test_detects_required_i2v_image_from_pipeline_signature(self, calibrator):
        calibrator.pipe = ImagePipeline()
        assert calibrator._requires_calibration_image() is True

        calibrator.pipe = FakePipeline()
        assert calibrator._requires_calibration_image() is False

    def test_wrap_block_forward_delegates_to_utility(self, calibrator):
        seen = []

        def base_hook(m, hidden_states, *args, **kwargs):
            seen.append((hidden_states, kwargs))
            return (hidden_states,)

        wrapped = calibrator._wrap_block_forward(base_hook)

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
        calibrator.pipe = None

        with pytest.raises(ValueError, match="Diffusion pipeline not found"):
            calibrator.calib(nsamples=1, bs=1)

    def test_calib_string_dataset_reloads_dataloader(self, calibrator):
        new_dataloader = [("id0", ["p1", "p2"])]
        calibrator.dataset = "mock_dataset"
        calibrator.pipe = FakePipeline(fn=lambda *args, **kwargs: None)
        calibrator._requires_calibration_image = lambda: False

        with patch(
            "auto_round.compressors.diffusion.dataset.get_diffusion_dataloader",
            return_value=(new_dataloader, 2),
        ), patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=2, bs=1)

        assert calibrator.dataloader is new_dataloader
        assert calibrator.batch_size == 2

    def test_calib_non_string_dataset_keeps_existing_dataloader(self, calibrator):
        calibrator.dataset = [("id0", ["p1", "p2"])]
        calibrator.pipe = FakePipeline(fn=lambda *args, **kwargs: None)
        calibrator._requires_calibration_image = lambda: False

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=2, bs=1)

        assert calibrator.dataloader is calibrator.dataset

    def test_calib_uses_dataloader_len_when_available(self, calibrator):
        class FakeDataloader:
            def __len__(self):
                return 1

            def __iter__(self):
                return iter([("id0", ["p1"])])

        calibrator.dataset = FakeDataloader()
        calibrator.pipe = FakePipeline(fn=lambda *args, **kwargs: None)
        calibrator._requires_calibration_image = lambda: False

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=1, bs=1)

        # No block hooks fire against the fake pipe, so inputs stays empty.
        assert calibrator.inputs == {}

    def test_calib_exits_on_multi_device_offload(self, calibrator):
        calibrator.model.hf_device_map = {"cpu": 0, "cuda:0": 1}
        calibrator.model.device = "cuda:0"
        calibrator.pipe = FakePipeline(
            device=torch.device("cpu"),
            fn=lambda *args, **kwargs: None,
        )
        calibrator.dataset = "mock"

        with patch(
            "auto_round.compressors.diffusion.dataset.get_diffusion_dataloader",
            return_value=([], 2),
        ):
            with pytest.raises(SystemExit):
                calibrator.calib(nsamples=1, bs=1)

    def test_calib_moves_pipeline_to_target_device(self, calibrator):
        seen = []
        calibrator.dataset = [("id0", ["p1", "p2"])]
        calibrator.pipe = FakePipeline(
            device=torch.device("cpu"),
            fn=lambda *args, **kwargs: None,
        )

        def fake_to(device):
            seen.append(device)
            return calibrator.pipe

        calibrator.pipe.to = fake_to

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm), patch(
            "auto_round.calibration.diffusion.device_manager",
            SimpleNamespace(device="cuda:0"),
        ):
            calibrator.calib(nsamples=2, bs=1)

        assert seen == [torch.device("cuda:0")]

    def test_calib_uses_autoround_pipeline_fn_when_available(self, calibrator):
        calls = []

        def pipeline_fn(pipe, prompts, **kwargs):
            calls.append((pipe, prompts, kwargs))

        calibrator.dataset = [("id0", ["p1", "p2"])]
        calibrator.pipe = FakePipeline(fn=lambda *args, **kwargs: None)
        calibrator.pipe._autoround_pipeline_fn = pipeline_fn
        calibrator._requires_calibration_image = lambda: False

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

        calibrator.dataset = [("id0", ["p1", "p2"])]
        calibrator.pipe = FakePipeline(fn=fake_pipe)
        calibrator._requires_calibration_image = lambda: False

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=2, bs=1)

        assert len(calls) == 1
        assert calls[0][0] == ["p1", "p2"]

    def test_calib_passes_image_when_required(self, calibrator):
        seen_images = []
        calibrator.dataset = [("id0", ["p1"])]
        calibrator.pipe = ImagePipeline(fn=lambda image, prompt=None, **kwargs: seen_images.append(image))
        calibrator._requires_calibration_image = lambda: True
        calibrator._get_calibration_image = lambda batch_size: torch.randn(batch_size, 4, 64, 64)

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=1, bs=1)

        assert seen_images[0].shape == (1, 4, 64, 64)

    def test_calib_not_implemented_error_is_swallowed(self, calibrator):
        def failing_pipe(*args, **kwargs):
            raise NotImplementedError("unsupported op")

        calibrator.dataset = [("id0", ["p1"])]
        calibrator.pipe = FakePipeline(fn=failing_pipe)
        calibrator._requires_calibration_image = lambda: False

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=1, bs=1)

        assert calibrator.inputs == {}

    def test_calib_other_exceptions_propagate(self, calibrator):
        def failing_pipe(*args, **kwargs):
            raise RuntimeError("unexpected")

        calibrator.dataset = [("id0", ["p1"])]
        calibrator.pipe = FakePipeline(fn=failing_pipe)
        calibrator._requires_calibration_image = lambda: False

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            with pytest.raises(RuntimeError, match="unexpected"):
                calibrator.calib(nsamples=1, bs=1)

    def test_calib_single_sample_stops_early(self, calibrator):
        seen = []

        def fake_pipe(prompts, **kwargs):
            seen.append(len(prompts) if isinstance(prompts, list) else 1)

        calibrator.dataset = [("id0", ["p1", "p2"])]
        calibrator.pipe = FakePipeline(fn=fake_pipe)
        calibrator._requires_calibration_image = lambda: False

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            calibrator.calib(nsamples=2, bs=2)

        assert seen == [2]

    def test_calib_zero_samples_exits(self, calibrator):
        calibrator.dataset = [("id0", [])]
        calibrator.pipe = FakePipeline(fn=lambda *args, **kwargs: None)
        calibrator._requires_calibration_image = lambda: False

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            with pytest.raises(SystemExit):
                calibrator.calib(nsamples=1, bs=1)

    def test_calib_insufficient_samples_warns_and_truncates(self, calibrator):
        def fake_pipe(prompts, **kwargs):
            return None

        calibrator.dataset = [("id0", ["p1"]), ("id1", ["p2"])]
        calibrator.pipe = FakePipeline(fn=fake_pipe)
        calibrator._requires_calibration_image = lambda: False

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            # total_cnt (2) < nsamples (3) but >= batch_size (2): the warning /
            # truncation path runs without raising.
            calibrator.calib(nsamples=3, bs=2)

        assert calibrator.inputs == {}

    def test_calib_insufficient_below_batch_size_raises(self, calibrator):
        def fake_pipe(prompts, **kwargs):
            return None

        calibrator.dataset = [("id0", ["p1"])]
        calibrator.pipe = FakePipeline(fn=fake_pipe)
        calibrator._requires_calibration_image = lambda: False

        with patch("auto_round.calibration.diffusion.tqdm", FakeTqdm):
            with pytest.raises(ValueError, match="valid samples is less than batch_size"):
                calibrator.calib(nsamples=3, bs=2)
