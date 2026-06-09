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
"""End-to-end diffusion-model quantization tests.

Each test quantizes a real, user-facing text-to-image model with
``auto-round``, reloads the resulting checkpoint through
``diffusers.AutoPipelineForText2Image``, runs a single inference pass
and checks that the produced image has the expected shape and dtype.

The test is structurally similar to :mod:`test.unit.test_cpu.models.test_diffusion`
but uses the *real* model rather than the 1-layer sliced variant, and
exercises a few extra model families.

Diffusion model downloads are large (≈12 GiB for FLUX.1-dev), so each
case is gated on free RAM via :class:`~.conftest.ModelCase.min_ram_gib`
and the test is skipped (not failed) if the host can't fit it.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Optional

import pytest
import torch

from test.e2e.test_cpu.conftest import (  # noqa: E402
    EvalResult,
    record,
)


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

# (hf_id, scheme, min_ram_gib, num_inference_steps, guidance_scale)
DIFFUSION_CASES = [
    # FLUX.1-dev - the canonical diffusion quant target.
    (
        "black-forest-labs/FLUX.1-dev",
        "W4A16",
        24,
        2,
        3.5,
    ),
    # FLUX.1-schnell - distilled / fewer steps, fits on smaller hosts.
    (
        "black-forest-labs/FLUX.1-schnell",
        "W4A16",
        24,
        2,
        0.0,
    ),
    # MXFP4 - low-bit float path.
    (
        "black-forest-labs/FLUX.1-schnell",
        "MXFP4",
        24,
        2,
        0.0,
    ),
    # NVFP4 - same family.
    (
        "black-forest-labs/FLUX.1-schnell",
        "NVFP4",
        24,
        2,
        0.0,
    ),
    # SDXL - older but still widely deployed.
    (
        "stabilityai/stable-diffusion-xl-base-1.0",
        "W4A16",
        16,
        2,
        7.5,
    ),
]


def _case_id(model_id: str, scheme: str) -> str:
    return f"{model_id.split('/')[-1].lower()}-{scheme.lower()}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_ram(min_gib: int) -> None:
    import psutil  # type: ignore

    avail = psutil.virtual_memory().available / 1024**3
    if avail < min_gib:
        pytest.skip(f"only {avail:.1f} GiB free RAM, need {min_gib} GiB for this diffusion case")


def _quantize_diffusion(model_id: str, scheme: str, output_dir: str, num_inference_steps: int) -> None:
    """Quantize a diffusion model with ``auto-round`` and write to ``output_dir``."""
    from auto_round import AutoRound

    from test.helpers import get_model_path

    model_id = get_model_path(model_id)
    shutil.rmtree(output_dir, ignore_errors=True)

    ar = AutoRound(
        model=model_id,
        tokenizer=None,
        scheme=scheme,
        iters=0,  # diffusion paths are typically RTN
        disable_opt_rtn=(scheme not in ("W4A16",)),
        num_inference_steps=num_inference_steps,
    )
    ar.quantize_and_save(output_dir)


def _reload_and_generate(saved_dir: str, prompt: str, num_inference_steps: int, guidance_scale: float):
    """Reload the quantized pipeline and run a single inference pass."""
    from diffusers import AutoPipelineForText2Image
    from PIL import Image

    pipe = AutoPipelineForText2Image.from_pretrained(saved_dir, torch_dtype=torch.bfloat16)
    try:
        gen = torch.Generator(device="cpu").manual_seed(0)
        kwargs = dict(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            generator=gen,
        )
        if guidance_scale and getattr(pipe, "guidance_scale", 0) != 0:
            kwargs["guidance_scale"] = guidance_scale
        image = pipe(**kwargs).images[0]
        return image
    finally:
        del pipe
        import gc

        gc.collect()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiffusionQuantizeE2E:
    """Quantize a real text-to-image model and run a single inference pass."""

    @pytest.mark.parametrize(
        "model_id,scheme,min_ram,num_steps,guidance",
        DIFFUSION_CASES,
        ids=[_case_id(m, s) for m, s, *_ in DIFFUSION_CASES],
    )
    def test_quantize_and_generate(
        self,
        model_id: str,
        scheme: str,
        min_ram: int,
        num_steps: int,
        guidance: float,
        tmp_path,
        require_diffusers,
    ):
        _require_ram(min_ram)

        save_dir = str(tmp_path / "diffusion_out")
        prompt = "a photo of an astronaut riding a horse on the moon"

        # ---- 1. quantize + save ----
        t0 = time.perf_counter()
        _quantize_diffusion(model_id, scheme, save_dir, num_inference_steps=num_steps)
        quant_time = time.perf_counter() - t0

        # Sanity: the saved pipeline must contain the expected files.
        assert os.path.isfile(os.path.join(save_dir, "model_index.json")), (
            f"diffusion pipeline not exported correctly: missing model_index.json in {save_dir}"
        )
        # The transformer (the thing we actually quantized) must carry a
        # quantization_config.json so downstream tools can dequantize.
        assert os.path.isfile(
            os.path.join(save_dir, "transformer", "quantization_config.json")
        ), f"transformer/quantization_config.json missing in {save_dir}"

        # ---- 2. reload + generate ----
        t0 = time.perf_counter()
        image = _reload_and_generate(save_dir, prompt, num_inference_steps=num_steps, guidance_scale=guidance)
        gen_time = time.perf_counter() - t0

        # ---- 3. shape/dtype sanity ----
        import numpy as np

        arr = np.array(image)
        assert arr.ndim == 3 and arr.shape[-1] in (3, 4), f"unexpected image shape {arr.shape}"
        assert arr.dtype == np.uint8, f"expected uint8 image, got {arr.dtype}"
        # A correctly-dequantized image is not a single solid color.
        assert arr.std() > 1.0, f"image appears flat / all-one-color (std={arr.std()})"

        record(
            EvalResult(
                test=self.__class__.__name__,
                model=model_id,
                fmt=scheme,
                bits=0,  # varies per scheme
                group_size=0,
                sym=True,
                task="image_generate",
                metric="image_std",
                value=float(arr.std()),
                wall_time_s=quant_time + gen_time,
                extra={
                    "quant_time_s": quant_time,
                    "gen_time_s": gen_time,
                    "image_shape": list(arr.shape),
                },
            )
        )
        print(
            f"\n[Diffusion-e2e] {model_id} {scheme} -> "
            f"quant={quant_time:.0f}s, gen={gen_time:.0f}s, img_std={arr.std():.1f}"
        )


class TestDiffusionLoadOnly:
    """Lighter cases that only verify load + dry-run, no full generation.

    These exist because diffusion generation is the slowest part of the
    pipeline, and a load-only check is enough to catch serialization
    regressions on a 1.5B-class diffusion model.
    """

    @pytest.mark.parametrize(
        "model_id,scheme",
        [
            ("black-forest-labs/FLUX.1-schnell", "W4A16"),
            ("stabilityai/stable-diffusion-xl-base-1.0", "W4A16"),
        ],
        ids=[_case_id(m, s) for m, s in [("FLUX.1-schnell", "W4A16"), ("SDXL", "W4A16")]],
    )
    def test_quantize_and_load(self, model_id, scheme, tmp_path, require_diffusers):
        _require_ram(20)

        save_dir = str(tmp_path / "diffusion_load_only")
        _quantize_diffusion(model_id, scheme, save_dir, num_inference_steps=2)

        from diffusers import AutoPipelineForText2Image

        pipe = AutoPipelineForText2Image.from_pretrained(save_dir, torch_dtype=torch.bfloat16)
        try:
            # Just touching the components is enough - we don't want to
            # wait for a full generation here.
            assert pipe.transformer is not None
        finally:
            del pipe
            import gc

            gc.collect()
