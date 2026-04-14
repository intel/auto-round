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

import inspect
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
    diffusion_load_model,
    dispatch_model_block_wise,
    extract_block_names_to_str,
    find_matching_blocks,
    get_block_names,
    is_auto_device_mapping,
    merge_block_output_keys,
    wrap_block_forward_positional_to_kwargs,
)

pipeline_utils = LazyImport("diffusers.pipelines.pipeline_utils")

output_configs = {
    "FluxTransformerBlock": ["encoder_hidden_states", "hidden_states"],
    "FluxSingleTransformerBlock": ["encoder_hidden_states", "hidden_states"],
    "WanTransformerBlock": ["hidden_states"],
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
        **kwargs,
    ):
        logger.warning("Diffusion model quantization is experimental and is only validated on Flux models.")
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

    def _update_inputs(self, inputs: dict, q_inputs: dict) -> tuple[dict, dict]:
        if self._uses_single_hidden_state_input():
            if q_inputs is not None:
                q_inputs = q_inputs.pop("hidden_states", None)
            return inputs, q_inputs

        # flux transformer model's blocks will update hidden_states and encoder_hidden_states
        input_id_str = [key for key in inputs.keys() if "hidden_state" in key]
        if q_inputs is not None:
            q_inputs = {k: q_inputs.pop(k, None) for k in input_id_str}
        return inputs, q_inputs

    def _get_block_forward_func(self, name):
        return wrap_block_forward_positional_to_kwargs(super()._get_block_forward_func(name))

    def _uses_single_hidden_state_input(self) -> bool:
        if not self.quant_block_list:
            return False
        first_block_name = self.quant_block_list[0][0]
        first_block = self.model.get_submodule(first_block_name)
        return output_configs.get(first_block.__class__.__name__, []) == ["hidden_states"]

    def _requires_calibration_image(self) -> bool:
        image_param = inspect.signature(self.pipe.__call__).parameters.get("image")
        return image_param is not None and image_param.default is inspect._empty

    def _get_calibration_image(self, batch_size: int):
        from PIL import Image

        params = inspect.signature(self.pipe.__call__).parameters
        width = params.get("width").default if "width" in params else 832
        height = params.get("height").default if "height" in params else 480
        image = Image.new("RGB", (width, height), color=(127, 127, 127))
        if batch_size == 1:
            return image
        return [image.copy() for _ in range(batch_size)]

    def _split_inputs(self, inputs: dict, first_input_name: str) -> tuple[dict, dict]:
        if self._uses_single_hidden_state_input():
            input_ids = inputs.pop("hidden_states", None)
            input_others = inputs
            return input_ids, input_others

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
        output_config = output_configs.get(block.__class__.__name__, [])
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
        output_config = output_configs.get(block.__class__.__name__, [])
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
                if len(output_config) != 1:
                    raise AssertionError(
                        f"Expected a single output name for {block.__class__.__name__}, got {output_config}"
                    )
                tmp_output = {output_config[0]: tmp_output}
            else:
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

    def cache_inter_data(self, block_names, nsamples, layer_names=None, last_cache_name=None):
        """Dispatch multi-device before caching so accelerate hooks are added before _replace_forward."""
        multi_device_diffusion = is_auto_device_mapping(self.device_map) and len(self.device_list) > 1
        if multi_device_diffusion:
            if not (hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1):
                self.model = dispatch_model_block_wise(self.model, self.device_map)
                self.pipe.transformer = self.model
        return super().cache_inter_data(block_names, nsamples, layer_names, last_cache_name)

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
        multi_device_diffusion = is_auto_device_mapping(self.device_map) and len(self.device_list) > 1

        if multi_device_diffusion:
            for name, component in self.pipe.components.items():
                if component is None or name == "transformer" or component is self.model:
                    continue
                if hasattr(component, "to"):
                    component.to(self.device)
                    if hasattr(component, "dtype") and component.dtype != self.model.dtype:
                        component.to(dtype=self.model.dtype)
        elif self.pipe.dtype != self.model.dtype:
            self.pipe.to(self.model.dtype)

        if (
            hasattr(self.model, "hf_device_map")
            and len(self.model.hf_device_map) > 0
            and self.pipe.device != self.model.device
            and torch.device(self.model.device).type in ["cuda", "xpu"]
        ):
            logger.error(
                "Diffusion model is activated sequential model offloading, it will crash during moving to GPU/XPU. "
                "Please use model path for quantization or "
                "move the pipeline object to GPU/XPU before passing them into API."
            )
            exit(-1)

        if not multi_device_diffusion and self.pipe.device != self.model.device:
            self.pipe.to(self.model.device)
        if not multi_device_diffusion:
            self.pipe.to(self.model.dtype)
        with tqdm(range(1, total + 1), desc="cache block inputs") as pbar:
            for ids, prompts in self.dataloader:
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                pipe_kwargs = {
                    "prompt": prompts,
                    "guidance_scale": self.guidance_scale,
                    "num_inference_steps": self.num_inference_steps,
                    "generator": (
                        None
                        if self.generator_seed is None
                        else torch.Generator(device=self.pipe.device).manual_seed(self.generator_seed)
                    ),
                }
                if self._requires_calibration_image():
                    pipe_kwargs["image"] = self._get_calibration_image(len(prompts))
                try:
                    self.pipe(**pipe_kwargs)
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
        if hasattr(self.pipe, "save_config"):
            self.pipe.save_config(output_dir)
        return compressed_model
