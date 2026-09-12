# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Download-free dual-Wan SVDQuant export and native Nunchaku inference."""

import json

import pytest


def test_tiny_dual_wan_svdquant_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA and the Nunchaku Wan runtime")
    device_index = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device_index) not in {(12, 0), (12, 1)}:
        pytest.skip("The current Nunchaku MXFP4 kernel requires SM120/SM121")
    diffusers = pytest.importorskip("diffusers")
    nunchaku = pytest.importorskip("nunchaku")
    runtime_class = getattr(nunchaku, "NunchakuWanTransformer3DModel", None)
    if runtime_class is None:
        pytest.skip("Requires a Nunchaku branch exposing NunchakuWanTransformer3DModel")

    from nunchaku.models.linear import SVDQW4A4Linear
    from safetensors import safe_open

    from auto_round import AutoRound
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
    from auto_round.algorithms.transforms.svdquant import SVDQuantConfig
    from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear

    original_threads = torch.get_num_threads()
    # AutoRound seeds all CUDA devices, so preserve their RNG states as well as CPU RNG.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        try:
            torch.set_num_threads(4)
            torch.manual_seed(0)
            device = torch.device("cuda", device_index)

            def expert():
                return diffusers.WanTransformer3DModel(
                    num_attention_heads=2,
                    attention_head_dim=64,
                    in_channels=16,
                    out_channels=16,
                    text_dim=128,
                    freq_dim=32,
                    ffn_dim=256,
                    num_layers=1,
                    rope_max_seq_len=64,
                ).to(torch.bfloat16)

            pipe = diffusers.WanPipeline(
                tokenizer=None,
                text_encoder=None,
                vae=None,
                transformer=expert(),
                transformer_2=expert(),
                scheduler=diffusers.UniPCMultistepScheduler(
                    prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0
                ),
                boundary_ratio=0.875,
            )
            source_path = tmp_path / "source-config"
            pipe.save_config(source_path)
            pipe.register_to_config(_name_or_path=str(source_path))
            embeddings = torch.randn(1, 4, 128, dtype=torch.bfloat16, device=device)

            def pipeline_fn(pipeline, prompts, **kwargs):
                return pipeline(
                    prompt_embeds=embeddings,
                    negative_prompt_embeds=torch.zeros_like(embeddings),
                    **kwargs,
                )

            pipe._autoround_pipeline_fn = pipeline_fn
            compressor = AutoRound(
                pipe,
                scheme="MXFP4",
                alg_configs=[
                    SVDQuantConfig(
                        rank=16,
                        smooth_enabled=True,
                        smooth_num_grids=2,
                        smooth_max_calibration_calls=1,
                        residual_iters=1,
                        low_rank_dtype="bf16",
                        model_adapter="wan",
                    ),
                    SignRoundConfig(iters=1, nblocks=1, enable_quanted_input=True),
                ],
                dataset=[([0], ["tiny test"])],
                nsamples=1,
                batch_size=1,
                device_map=device_index,
                low_gpu_mem_usage=True,
                low_cpu_mem_usage=False,
                model_dtype="bf16",
                calib_num_inference_steps=2,
                pipeline_call_kwargs={
                    "height": 16,
                    "width": 16,
                    "num_frames": 5,
                    "output_type": "latent",
                    "guidance_scale_2": 3.0,
                },
                guidance_scale=4.0,
                generator_seed=0,
                seed=0,
                format="svdquant_nunchaku",
            )
            compressor.quantize()
            names = ("transformer", "transformer_2")
            protected_tensors = {}
            for name in names:
                model = getattr(pipe, name)
                assert sum(isinstance(module, SVDQuantLinear) for module in model.modules()) == 10
                protected_tensors[name] = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                    if any(part in key for part in model._keep_in_fp32_modules)
                }
                assert protected_tensors[name]
                assert all(tensor.dtype == torch.float32 for tensor in protected_tensors[name].values())
            assert pipe.config.boundary_ratio == 0.875
            original_config = dict(pipe.config)
            export_path = tmp_path / "export"
            compressor.save_quantized(str(export_path), format="svdquant_nunchaku")
            assert dict(pipe.config) == original_config
            source_index = json.loads((source_path / "model_index.json").read_text())
            export_index = json.loads((export_path / "model_index.json").read_text())
            for name in names:
                assert source_index[name] == ["diffusers", "WanTransformer3DModel"]
                assert export_index[name] == ["nunchaku", "NunchakuWanTransformer3DModel"]
                with safe_open(export_path / name / "diffusion_pytorch_model.safetensors", framework="pt") as artifact:
                    for key, expected in protected_tensors[name].items():
                        actual = artifact.get_tensor(key)
                        assert actual.dtype == torch.float32, (name, key)
                        assert torch.equal(actual, expected), (name, key)
                loaded = runtime_class.from_pretrained(export_path / name, torch_dtype=torch.bfloat16)
                assert sum(isinstance(module, SVDQW4A4Linear) for module in loaded.modules()) == 10
                for key, expected in protected_tensors[name].items():
                    actual = loaded.state_dict()[key]
                    assert actual.dtype == torch.float32, (name, key)
                    assert torch.equal(actual.cpu(), expected), (name, key)
                setattr(pipe, name, loaded)

            pipe.enable_model_cpu_offload(device=device)
            calls = dict.fromkeys(names, 0)
            handles = []
            for name in names:

                def record(module, args, name=name):
                    calls[name] += 1

                handles.append(getattr(pipe, name).register_forward_pre_hook(record))
            steps = []

            def check_step(pipeline, index, timestep, callback_kwargs):
                assert torch.isfinite(callback_kwargs["latents"]).all(), (index, timestep)
                steps.append(index)
                return callback_kwargs

            try:
                latents = pipe(
                    prompt_embeds=embeddings,
                    negative_prompt_embeds=torch.zeros_like(embeddings),
                    height=16,
                    width=16,
                    num_frames=5,
                    num_inference_steps=4,
                    guidance_scale=4.0,
                    guidance_scale_2=3.0,
                    output_type="latent",
                    generator=torch.Generator(device=device).manual_seed(0),
                    callback_on_step_end=check_step,
                ).frames
            finally:
                for handle in handles:
                    handle.remove()
                pipe.remove_all_hooks()
                pipe.to("cpu")
            assert calls == {"transformer": 4, "transformer_2": 4}
            assert steps == [0, 1, 2, 3]
            assert tuple(latents.shape) == (1, 16, 2, 2, 2)
            assert torch.isfinite(latents).all()
        finally:
            torch.set_num_threads(original_threads)
