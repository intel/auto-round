# Copyright (c) 2025 Intel Corporation
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

import os
from collections import defaultdict
from copy import deepcopy
from typing import Union

import torch
from tqdm import tqdm

from auto_round.compressors.base import BaseCompressor
from auto_round.compressors.diffusion.dataset import get_diffusion_dataloader
from auto_round.compressors.utils import block_forward
from auto_round.formats import OutputFormat
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import (
    LazyImport,
    clear_memory,
    copy_python_files_from_model_cache,
    diffusion_load_model,
    dispatch_model_by_all_available_devices,
    extract_block_names_to_str,
    find_matching_blocks,
    get_block_names,
    merge_block_output_keys,
    wrap_block_forward_positional_to_kwargs,
)

pipeline_utils = LazyImport("diffusers.pipelines.pipeline_utils")

output_configs = {
    "FluxTransformerBlock": ["encoder_hidden_states", "hidden_states"],
    "FluxSingleTransformerBlock": ["encoder_hidden_states", "hidden_states"],
}


class DiffusionCompressor(BaseCompressor):
    """Class for automatic rounding-based quantization with Diffusion models.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data, is not used for diffusion models.
        platform (str): The platform to load pretrained moded, options: ["hf", "model_scope"]
        guidance_scale (float): Control how much the image generation process follows the text prompt.
                                The more it is, the more closely it follows the prompt (default is 7.5).
        num_inference_steps (int): The reference number of denoising steps (default is 50).
        generator_seed (int): A sees that controls the initial noise from which an image is generated (default is None).
        pipeline_fn (callable, optional): Custom callable to run the pipeline during calibration.
            Signature: ``fn(pipe, prompts, *, guidance_scale, num_inference_steps, generator, **kwargs)``.
            Use this to support models whose inference API differs from the standard convention
            (e.g. NextStep). If ``None``, the standard ``pipe(prompts, ...)`` call is used unless
            the loaded pipeline already exposes an ``_autoround_pipeline_fn`` attribute.
        scheme: (str| dict | QuantizationScheme ): A preset scheme that defines the quantization configurations.
        layer_config (dict): Configuration for weight quantization (default is None).
        dataset: The path or name of the calib dataset.
        iters (int): Number of iterations (default is 200).
        seqlen (int): Length of the sequence.
        nsamples (int): Number of samples (default is 128).
        batch_size (int): Batch size for training (default is 8).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is False).
        device_map (str | dict | int | torch.device, optional): Device placement map. Defaults to 0.
        enable_torch_compile (bool): Whether to enable torch compile to optimize quant_block/layer
        **kwargs: Additional keyword arguments.
    """

    bits: int | None
    group_size: int | None
    sym: bool | None
    data_type: str | None
    act_bits: int | None
    act_group_size: int | None
    act_sym: bool | None
    act_data_type: str | None
    act_dynamic: bool | None
    super_bits: int | None
    super_group_size: int | None
    is_diffusion: bool = True

    def __init__(
        self,
        model: Union[object, str],
        tokenizer=None,
        platform: str = "hf",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator_seed: int = None,
        scheme: Union[str, dict, QuantizationScheme] = "W4A16",
        layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "coco2014",
        iters: int = 200,
        seqlen: int = 2048,
        nsamples: int = 128,
        batch_size: int = 8,
        gradient_accumulate_steps: int = 1,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        pipeline_fn: callable = None,
        **kwargs,
    ):
        logger.warning("Diffusion model quantization is experimental and is only validated on Flux models.")
        if dataset == "NeelNanda/pile-10k":
            dataset = "coco2014"
            logger.warning(
                "Dataset 'NeelNanda/pile-10k' is not suitable for diffusion model quantization, use coco2014 dataset instead."
            )
        model_dtype = kwargs.pop("model_dtype", None)

        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generator_seed = generator_seed

        to_quant_block_names: Union[str, list, None] = kwargs.pop("to_quant_block_names", None)
        if device_map is None:
            device_map = 0
        self._set_device(device_map)

        pipe, model = diffusion_load_model(model, platform=platform, device=self.device, model_dtype=model_dtype)

        self.model = model
        self.pipe = pipe
        # Use explicit pipeline_fn; fall back to whatever diffusion_load_model attached to the pipe
        self.pipeline_fn = pipeline_fn or getattr(pipe, "_autoround_pipeline_fn", None)

        all_blocks = get_block_names(model)
        self.quant_block_list = find_matching_blocks(model, all_blocks, to_quant_block_names)
        if to_quant_block_names is None:
            to_quant_block_names = extract_block_names_to_str(self.quant_block_list)

        if iters > 0 and batch_size != 1:
            logger.warning(
                f"reset batch_size({batch_size}) to 1 and "
                f"gradient_accumulate_steps({gradient_accumulate_steps}) "
                f"to {batch_size * gradient_accumulate_steps}, "
                f"because batch_size={batch_size} cannot be used for calibrating non-text modules."
            )
            gradient_accumulate_steps = batch_size * gradient_accumulate_steps
            batch_size = 1

        seqlen = 2048 if seqlen is None else seqlen

        if nsamples % batch_size != 0:
            nsamples = (nsamples // batch_size + 1) * batch_size
            logger.warning(f"'nsamples' is not divisible by 'batch_size', will adjusted to {nsamples}")

        kwargs["diffusion"] = True
        super(DiffusionCompressor, self).__init__(
            model=model,
            tokenizer=None,
            platform=platform,
            scheme=scheme,
            layer_config=layer_config,
            dataset=dataset,
            iters=iters,
            seqlen=seqlen,
            nsamples=nsamples,
            batch_size=batch_size,
            gradient_accumulate_steps=gradient_accumulate_steps,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            to_quant_block_names=to_quant_block_names,
            **kwargs,
        )
        self._align_device_and_dtype()

    def _update_inputs(self, inputs: dict, q_inputs: dict) -> tuple[dict, dict]:
        # flux transformer model's blocks will update hidden_states and encoder_hidden_states
        input_id_str = [key for key in inputs.keys() if "hidden_state" in key]
        if q_inputs is not None:
            q_inputs = {k: q_inputs.pop(k, None) for k in input_id_str}
        return inputs, q_inputs

    def _get_block_forward_func(self, name):
        return wrap_block_forward_positional_to_kwargs(super()._get_block_forward_func(name))

    def _split_inputs(self, inputs: dict, first_input_name: str) -> tuple[dict, dict]:
        input_id_str = [key for key in inputs.keys() if "hidden_state" in key]
        input_ids = {k: inputs.pop(k, None) for k in input_id_str}
        input_others = inputs
        return input_ids, input_others

    def _get_current_output(self, output: dict, indices: list[int]) -> torch.Tensor:
        assert "hidden_states" in output
        current_output = [output["hidden_states"][x] for x in indices]
        current_output = torch.cat(current_output, dim=self.batch_dim)
        return current_output

    def _get_current_q_output(
        self,
        block: torch.nn.Module,
        input_ids: dict,
        input_others: dict,
        indices: list[int],
        device: str,
        cache_device: str = "cpu",
    ) -> torch.Tensor:
        output_config = output_configs.get(block.__class__.__name__, ["hidden_states"])
        idx = None if "hidden_states" not in output_config else output_config.index("hidden_states")
        current_input_ids, current_input_others = self._sampling_inputs(
            input_ids,
            input_others,
            indices,
            seqlen=self.seqlen,
            batch_dim=self.batch_dim,
            share_cache_keys=self.shared_cache_keys,
        )
        if isinstance(current_input_ids, dict):
            hidden_states = current_input_ids.pop("hidden_states")
            merge_block_output_keys(block, current_input_others, current_input_ids)
            current_input_ids = hidden_states
        output_q = block_forward(block, current_input_ids, current_input_others, self.amp, self.amp_dtype, device, idx)
        return output_q.to(cache_device)

    @torch.no_grad()
    def _get_block_outputs(
        self,
        block: torch.nn.Module,
        input_ids: Union[torch.Tensor, dict],
        input_others: torch.Tensor,
        bs: int,
        device: Union[str, torch.device],
        cache_device: Union[str, torch.device],
        save_output: bool = True,
    ):
        """Compute the output of a given block of the model for a given input.

        Args:
        block: The block of the model.
        input_ids: The input tensor containing tokenized input ids.
        input_others: A dictionary containing additional input data.
        bs: The batch size for computing the output.
        device: The device for computation.
        cache_device: The device for storing the output.
        batch_dim: The batch dimension of the output tensor.

        Returns:
        The output tensor of the block.
        """

        output = defaultdict(list)
        output_config = output_configs.get(block.__class__.__name__, ["hidden_states"])
        if isinstance(input_ids, dict):
            nsamples = len(input_ids["hidden_states"])
        else:
            nsamples = len(input_ids)

        for i in range(0, nsamples, bs):
            end_index = min(nsamples, i + bs)
            indices = torch.arange(i, end_index).to(torch.long)
            tmp_input_ids, tmp_input_others = self._sampling_inputs(
                input_ids, input_others, indices, self.seqlen, self.batch_dim, share_cache_keys=self.shared_cache_keys
            )
            if isinstance(tmp_input_ids, dict):
                hidden_states = tmp_input_ids.pop("hidden_states")
                merge_block_output_keys(block, tmp_input_others, tmp_input_ids)
                tmp_input_ids = hidden_states

            tmp_output = block_forward(block, tmp_input_ids, tmp_input_others, self.amp, self.amp_dtype, device, None)
            if isinstance(tmp_output, torch.Tensor):
                tmp_output = [tmp_output]
            assert len(output_config) == len(tmp_output)
            tmp_output = dict(zip(output_config, tmp_output))

            if save_output:
                for name, out in tmp_output.items():
                    if self.batch_size == 1:
                        output[name].append(out.to(cache_device))
                    else:
                        output[name].extend(list(torch.split(out.to(cache_device), 1, dim=self.batch_dim)))
        if self.low_gpu_mem_usage:
            clear_memory()

        return output

    def _get_current_num_elm(
        self,
        input_ids: list[torch.Tensor],
        indices: list[int],
    ) -> int:
        current_input_ids = [input_ids["hidden_states"][i] for i in indices]
        return sum(id.numel() for id in current_input_ids)

    def _run_pipeline(self, prompts: list) -> None:
        """Execute one full diffusion pipeline forward pass for calibration input capture.

        This drives all transformer blocks so that their intermediate inputs are recorded
        by the hooks installed during calibration.

        **Extending for custom models** – choose whichever approach is simpler:

        * Pass a ``pipeline_fn`` to the constructor (no subclassing required).  The
          callable receives ``(pipe, prompts, *, guidance_scale, num_inference_steps,
          generator, **kwargs)`` and must trigger a full forward pass.
        * Subclass :class:`DiffusionCompressor` and override this method directly for
          full control over the inference logic.

        Example – NextStep model::

            def nextstep_fn(pipe, prompts, guidance_scale=7.5,
                            num_inference_steps=28, generator=None,
                            hw=(1024, 1024), **kwargs):
                for prompt in (prompts if isinstance(prompts, list) else [prompts]):
                    pipe.generate_image(
                        prompt,
                        cfg=guidance_scale,
                        num_sampling_steps=num_inference_steps,
                        hw=hw,
                        **kwargs,
                    )

            compressor = DiffusionCompressor(
                model="path/to/nextstep",
                pipeline_fn=nextstep_fn,
                pipeline_fn_kwargs={"hw": (512, 512)},
            )

        Args:
            prompts (list[str]): Text prompts for the current calibration batch.
        """
        generator = (
            None
            if self.generator_seed is None
            else torch.Generator(device=self.pipe.device).manual_seed(self.generator_seed)
        )
        if self.pipeline_fn is not None:
            self.pipeline_fn(
                self.pipe,
                prompts,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=generator,
            )
        else:
            self.pipe(
                prompts,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=generator,
            )

    def calib(self, nsamples, bs):
        """Perform calibration for quantization.

        This method calibrates the model for quantization by processing a specified
        number of samples from the calibration dataset. It ensures that the data is
        properly formatted and feeds it to the model. If the number of samples processed
        is less than the specified number, it logs a warning. If no samples are processed,
        it logs an error and exits.
        Args:
            nsamples (int): The number of samples to use for calibration.
            bs (int): The number of samples to use for calibration
        """
        logger.warning(
            "Diffusion model will catch nsamples * num_inference_steps inputs, "
            "you can reduce nsamples or num_inference_steps if OOM or take too much time."
        )
        raw_num_inference_steps = self.num_inference_steps
        self.num_inference_steps = 1
        logger.info(
            f"Set num_inference_steps to 1 for calibration, original num_inference_steps is {raw_num_inference_steps}"
        )
        if isinstance(self.dataset, str):
            dataset = self.dataset.replace(" ", "")
            self.dataloader, self.batch_size, self.gradient_accumulate_steps = get_diffusion_dataloader(
                dataset=dataset,
                bs=self.batch_size,
                seed=self.seed,
                nsamples=self.nsamples,
                gradient_accumulate_steps=self.gradient_accumulate_steps,
            )
        else:
            self.dataloader = self.dataset
        total_cnt = 0

        total = nsamples if not hasattr(self.dataloader, "len") else min(nsamples, len(self.dataloader))

        with tqdm(range(1, total + 1), desc="cache block inputs") as pbar:
            for ids, prompts in self.dataloader:
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                try:
                    self._run_pipeline(prompts)
                except NotImplementedError:
                    pass
                except Exception as error:
                    raise error
                step = len(prompts)
                total_cnt += step
                pbar.update(step)
                if total_cnt >= nsamples:
                    break
        if total_cnt == 0:
            logger.error(
                f"no data has been cached, please provide more data with sequence length >={self.seqlen} in the "
                f"dataset or decease the sequence length"
            )
            exit(-1)
        elif total_cnt < nsamples:
            logger.warning(
                f"Insufficient number of samples collected may affect the quantization. "
                f"target samples count is {nsamples}, while valid samples count is {total_cnt}"
            )
            if total_cnt < self.batch_size:
                raise ValueError(
                    f"valid samples is less than batch_size({self.batch_size}),"
                    " please adjust self.batch_size or seqlen."
                )
            max_len = (total_cnt // self.batch_size) * self.batch_size
            for k, v in self.inputs.items():
                for key in v:
                    if isinstance(v[key], list) and len(v[key]) == total_cnt:
                        self.inputs[k][key] = v[key][:max_len]

        # torch.cuda.empty_cache()
        self.num_inference_steps = raw_num_inference_steps
        logger.info(f"Restore num_inference_steps to {self.num_inference_steps} after calibration")

    def _should_stop_cache_forward(self, name: str) -> bool:
        """Determine whether current forward pass can stop after caching `name`."""
        # diffusion model needs to run all steps to collect input
        return False

    def _get_save_folder_name(self, format: OutputFormat) -> str:
        """Generates the save folder name based on the provided format string.

        If there are multiple formats to handle, the function creates a subfolder
        named after the format string with special characters replaced. If there's
        only one format, it returns the original output directory directly.

        Args:
            format_str (str): The format identifier (e.g., 'gguf:q2_k_s').

        Returns:
            str: The path to the folder where results should be saved.
        """
        # Replace special characters to make the folder name filesystem-safe
        sanitized_format = format.get_backend_name().replace(":", "-").replace("_", "-")
        if hasattr(self.model, "config") and getattr(self.model.config, "model_type", None) == "nextstep":
            # Use a subfolder only if there are multiple formats
            if len(self.formats) > 1:
                return os.path.join(self.orig_output_dir, sanitized_format)

            return self.orig_output_dir

        # Use a subfolder only if there are multiple formats
        if len(self.formats) > 1:
            return (
                os.path.join(self.orig_output_dir, sanitized_format, "transformer")
                if self.is_immediate_saving
                else os.path.join(self.orig_output_dir, sanitized_format, "transformer")
            )

        # if use is_immediate_saving, we need to save model in self.orig_output_dir/transformer folder
        return os.path.join(self.orig_output_dir, "transformer") if self.is_immediate_saving else self.orig_output_dir

    def save_quantized(self, output_dir=None, format="auto_round", inplace=True, **kwargs):
        """Save the quantized model to the specified output directory in the specified format.

        Args:
            output_dir (str, optional): The directory to save the quantized model. Defaults to None.
            format (str, optional): The format in which to save the model. Defaults to "auto_round".
            inplace (bool, optional): Whether to modify the model in place. Defaults to True.
            **kwargs: Additional keyword arguments specific to the export format.

        Returns:
            object: The compressed model object.
        """
        if output_dir is None:
            return super().save_quantized(output_dir, format=format, inplace=inplace, **kwargs)

        compressed_model = None
        if hasattr(self.model, "config") and getattr(self.model.config, "model_type", None) == "nextstep":
            compressed_model = super().save_quantized(
                output_dir=output_dir,
                format=format,
                inplace=inplace,
                **kwargs,
            )
            self.pipe.tokenizer.save_pretrained(output_dir)
            copy_python_files_from_model_cache(self.model, output_dir, copy_folders=["models", "vae", "utils"])
            return compressed_model
        for name in self.pipe.components.keys():
            val = getattr(self.pipe, name)
            sub_module_path = (
                os.path.join(output_dir, name) if os.path.basename(os.path.normpath(output_dir)) != name else output_dir
            )
            if (
                hasattr(val, "config")
                and hasattr(val.config, "_name_or_path")
                and val.config._name_or_path == self.model.config._name_or_path
            ):
                compressed_model = super().save_quantized(
                    output_dir=sub_module_path if not self.is_immediate_saving else output_dir,
                    format=format,
                    inplace=inplace,
                    **kwargs,
                )
            elif val is not None and hasattr(val, "save_pretrained"):
                val.save_pretrained(sub_module_path)
        self.pipe.config.save_pretrained(output_dir)
        return compressed_model

    def _align_device_and_dtype(self):
        if hasattr(self.model, "config") and getattr(self.model.config, "model_type", None) == "nextstep":
            return
        if (
            hasattr(self.model, "hf_device_map")
            and len(self.model.hf_device_map) > 0
            and type(self.pipe.device) != type(self.model.device)
            and self.pipe.device != self.model.device
            and torch.device(self.model.device).type in ["cuda", "xpu"]
        ):
            logger.error(
                "Diffusion model is activated sequential model offloading, it will crash during moving to GPU/XPU. "
                "Please use model path for quantization or "
                "move the pipeline object to GPU/XPU before passing them into API."
            )
            exit(-1)

        self.pipe.to(self.model.dtype)

        dispatch_model_by_all_available_devices(self.pipe, self.device_map)
