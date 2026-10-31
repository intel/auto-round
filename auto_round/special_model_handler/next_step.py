# Copyright (c) 2023 Intel Corporation
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
import torch


def load_next_step_diffusion(pretrained_model_name_or_path, device_str):
    try:
        from models.gen_pipeline import NextStepPipeline  # pylint: disable=E0401
    except ImportError:
        raise ImportError(
            "NextStepPipeline module not found. "
            + "Please navigate to the model file path and add it to your PYTHONPATH."
        )
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(pretrained_model_name_or_path, local_files_only=True, trust_remote_code=True)
    # The model is loaded onto the device because more than one block requires input data.
    pipe = NextStepPipeline(tokenizer=tokenizer, model=model).to(device=device_str, dtype=torch.bfloat16)

    def _nextstep_pipeline_fn(pipe, prompts, guidance_scale=7.5, num_inference_steps=28, generator=None, **kwargs):
        """Default pipeline_fn for NextStep models.

        Maps standard :class:`DiffusionCompressor` parameters to NextStep's
        ``generate_image`` API.  Pass a custom ``pipeline_fn`` to
        :class:`DiffusionCompressor` to override defaults or supply
        model-specific kwargs (e.g. ``hw``, ``positive_prompt``,
        ``cfg_schedule``, ``timesteps_shift``).
        """
        for prompt in (prompts if isinstance(prompts, list) else [prompts]):
            pipe.generate_image(
                prompt,
                cfg=guidance_scale,
                num_sampling_steps=num_inference_steps,
                **kwargs,
            )

    pipe._autoround_pipeline_fn = _nextstep_pipeline_fn
    return pipe, model
