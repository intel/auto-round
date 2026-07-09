from types import SimpleNamespace

import torch

from auto_round.calibration.diffusion import DiffusionCalibrator


def test_diffusion_calibrator_forwards_pipeline_call_kwargs():
    calls = []

    pipe = SimpleNamespace(device=torch.device("cpu"))
    pipe.to = lambda device: pipe

    def _pipeline_fn(pipe_obj, prompts, **kwargs):
        calls.append((prompts, kwargs))

    pipe._autoround_pipeline_fn = _pipeline_fn

    compressor = SimpleNamespace(
        model_context=SimpleNamespace(pipe=pipe),
        model=SimpleNamespace(device="cpu"),
        dataset=[(0, ["robot arm cleans a plate"])],
        batch_size=1,
        seed=42,
        nsamples=1,
        gradient_accumulate_steps=1,
        guidance_scale=6.0,
        num_inference_steps=2,
        generator_seed=123,
        inputs={},
        seqlen=1,
        _requires_calibration_image=lambda: False,
        _build_pipeline_call_kwargs=lambda current_pipe, prompts: {
            "guidance_scale": 6.0,
            "num_inference_steps": 2,
            "generator": torch.Generator(device=current_pipe.device).manual_seed(123),
            "height": 480,
            "width": 832,
            "negative_prompt": "bad output",
        },
    )

    calibrator = DiffusionCalibrator(compressor)
    calibrator.calib(nsamples=1, bs=1)

    assert len(calls) == 1
    prompts, kwargs = calls[0]
    assert prompts == ["robot arm cleans a plate"]
    assert kwargs["height"] == 480
    assert kwargs["width"] == 832
    assert kwargs["negative_prompt"] == "bad output"
    assert kwargs["guidance_scale"] == 6.0
    assert kwargs["num_inference_steps"] == 2
    assert isinstance(kwargs["generator"], torch.Generator)
