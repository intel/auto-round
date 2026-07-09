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
import inspect
import os
from typing import Any, Optional, Union

import torch
from tqdm import tqdm

from auto_round.logger import logger
from auto_round.utils import clear_memory
from auto_round.utils.device import (
    dispatch_model_block_wise,
    dispatch_model_by_all_available_devices,
    get_major_device,
)
from auto_round.utils.device_manager import device_manager, is_auto_device_mapping
from auto_round.utils.model import rename_weights_files


class DiffusionMixin:
    """Diffusion-specific functionality mixin.

    This mixin adds diffusion model-specific functionality to any compressor
    (DataDrivenCompressor, ZeroShotCompressor, ImatrixCompressor, etc). It handles
    diffusion models (like Stable Diffusion, FLUX) that require special pipeline
    handling and data generation logic.

    Can be combined with:
    - DataDrivenCompressor (for AutoRound with calibration)
    - ImatrixCompressor (for RTN with importance matrix)
    - ZeroShotCompressor (for basic RTN)

    Diffusion-specific parameters:
        guidance_scale: Control how much image generation follows text prompt
        num_inference_steps: Reference number of denoising steps
        generator_seed: Seed for initial noise generation

    Design note:
        ``ModelContext._load_model()`` loads the diffusion pipeline and sets
        ``model_context.pipe`` and ``model_context.model`` (the unet/transformer).
        This mixin reads ``self.model_context.pipe`` directly during calibration and
        saving so that ``model_context`` remains the single source of truth.
    """

    def __init__(
        self,
        *args,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        # Store diffusion-specific attributes
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generator_seed = generator_seed
        self.pipeline_call_kwargs = dict(kwargs.pop("pipeline_call_kwargs", {}) or {})

        iters = kwargs.get("iters", None)
        _alg_cfg = args[0] if args else None
        if iters is None and _alg_cfg is not None:
            cfgs = _alg_cfg if isinstance(_alg_cfg, list) else [_alg_cfg]
            for cfg in cfgs:
                if hasattr(cfg, "iters") and cfg.iters is not None:
                    iters = cfg.iters
                    break
        if iters is None:
            iters = 200

        if iters > 0:
            # ``batch_size`` is owned by the compressor / shared
            # CalibrationState; it only comes from kwargs now (entry.py forwards
            # it explicitly).  AlgConfig no longer carries it.  Treat a missing
            # value as ``BaseCompressor``'s default (8) so the reset path always
            # triggers when the user didn't explicitly opt out.
            batch_size = kwargs.get("batch_size", 8)
            if batch_size != 1:
                grad_acc = kwargs.get("gradient_accumulate_steps", 1)
                if _alg_cfg is not None:
                    cfgs = _alg_cfg if isinstance(_alg_cfg, list) else [_alg_cfg]
                    for cfg in cfgs:
                        if hasattr(cfg, "gradient_accumulate_steps") and cfg.gradient_accumulate_steps is not None:
                            grad_acc = cfg.gradient_accumulate_steps
                            break
                new_grad_acc = batch_size * grad_acc
                kwargs["gradient_accumulate_steps"] = new_grad_acc
                kwargs["batch_size"] = 1
                if _alg_cfg is not None:
                    cfgs = _alg_cfg if isinstance(_alg_cfg, list) else [_alg_cfg]
                    for cfg in cfgs:
                        if hasattr(cfg, "gradient_accumulate_steps"):
                            cfg.gradient_accumulate_steps = new_grad_acc
                logger.warning(
                    f"reset batch_size({batch_size}) to 1 and "
                    f"gradient_accumulate_steps to {new_grad_acc} "
                    f"because batch_size={batch_size} cannot be used for calibrating non-text modules."
                )

        # Call parent class __init__ (will be DataDrivenCompressor, ImatrixCompressor, etc)
        super().__init__(*args, **kwargs)

        pipe = getattr(self.model_context, "pipe", None)
        model = getattr(self.model_context, "model", None)
        if pipe is not None and model is not None:
            is_nextstep = hasattr(model, "config") and getattr(model.config, "model_type", None) == "nextstep"
            if not is_nextstep:
                pipe.to(model.dtype)

    def _get_calibrator_kind(self) -> str:
        """Select the diffusion calibration strategy.

        ``DiffusionCalibrator`` lives at
        :mod:`auto_round.calibration.diffusion` and owns what used to be
        ``DiffusionMixin.calib`` / ``_get_block_forward_func`` /
        ``_should_stop_cache_forward``.
        """
        return "diffusion"

    def _build_pipeline_call_kwargs(self, pipe, prompts):
        """Build kwargs for pipeline.__call__."""
        call_kwargs = {
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "generator": (
                None
                if self.generator_seed is None
                else torch.Generator(device=pipe.device).manual_seed(self.generator_seed)
            ),
        }
        call_kwargs.update(self.pipeline_call_kwargs)

        if self._requires_calibration_image():
            call_kwargs.setdefault(
                "image", self._get_calibration_image(len(prompts) if isinstance(prompts, list) else 1)
            )
            call_kwargs.setdefault("prompt", prompts)

        return call_kwargs

    def _requires_calibration_image(self) -> bool:
        """Return True when the pipeline's __call__ has a required 'image' parameter.

        I2V (image-to-video) pipelines like WanImageToVideoPipeline require a PIL/torch
        image as input. This is detected by checking whether 'image' is a positional-or-
        keyword parameter without a default value.
        """
        image_param = inspect.signature(self.model_context.pipe.__call__).parameters.get("image")
        return image_param is not None and image_param.default is inspect.Parameter.empty

    def _get_calibration_image(self, batch_size: int):
        """Return a synthetic PIL Image for I2V pipeline calibration."""
        params = inspect.signature(self.model_context.pipe.__call__).parameters
        width_param = params.get("width")
        height_param = params.get("height")
        width = (
            832
            if width_param is None or width_param.default in (inspect.Parameter.empty, None)
            else width_param.default
        )
        height = (
            480
            if height_param is None or height_param.default in (inspect.Parameter.empty, None)
            else height_param.default
        )
        from PIL import Image  # pylint: disable=E0401

        image = Image.new("RGB", (int(width), int(height)), color=(127, 127, 127))
        if batch_size == 1:
            return image
        return [image.copy() for _ in range(batch_size)]

    def _find_additional_transformers(self):
        """Find transformer components beyond the primary one (e.g. transformer_2 in WAN)."""
        pipe = getattr(self.model_context, "pipe", None)
        if pipe is None:
            return []
        result = []
        for comp_name in pipe.components:
            comp = getattr(pipe, comp_name, None)
            if (
                comp_name.startswith("transformer")
                and comp_name != "transformer"
                and comp is not None
                and isinstance(comp, torch.nn.Module)
            ):
                result.append((comp_name, comp))
        return result

    def _align_device_and_dtype_for_secondary(self, transformer_name: str):
        """Align dtype and dispatch secondary transformer for multi-transformer pipelines."""
        pipe = getattr(self.model_context, "pipe", None)
        model = getattr(self.model_context, "model", None)
        if pipe is None or model is None:
            return

        # Cast full pipeline to transformer's dtype
        is_nextstep = hasattr(model, "config") and getattr(model.config, "model_type", None) == "nextstep"
        if not is_nextstep:
            pipe.to(model.dtype)

        # Dispatch secondary transformer to GPU(s)
        device_map = getattr(self.compress_context, "device_map", None)
        device_list = getattr(self.compress_context, "device_list", [])
        multi_device = is_auto_device_mapping(device_map) and len(device_list) > 1

        if multi_device:
            comp_device = device_list[-1]
            for comp_name in pipe.components:
                comp = getattr(pipe, comp_name, None)
                if comp is None or comp is model or not hasattr(comp, "to"):
                    continue
                is_other_transformer = (
                    comp_name.startswith("transformer")
                    and isinstance(comp, torch.nn.Module)
                    and next(comp.parameters()).device.type == "cpu"
                )
                is_other_component = not comp_name.startswith("transformer")
                if is_other_transformer or is_other_component:
                    if isinstance(comp, torch.nn.Module) and hasattr(comp, "dtype") and comp.dtype != model.dtype:
                        comp.to(dtype=model.dtype)
                    try:
                        comp.to(comp_device)
                    except (NotImplementedError, RuntimeError):
                        continue

            self.model_context.model = dispatch_model_block_wise(model, device_map)
            setattr(pipe, transformer_name, self.model_context.model)
        elif device_map is not None:
            target_device = get_major_device(device_map)
            pipe.to(target_device)

    @torch.no_grad()
    def calib(self, nsamples: int, bs: int) -> None:
        """Perform diffusion-specific calibration for quantization.

        Override parent's calib method to use diffusion dataset loading logic.
        The diffusion pipeline is read from ``self.model_context.pipe``.
        """
        from auto_round.compressors.diffusion.dataset import get_diffusion_dataloader

        pipe = self.model_context.pipe
        if pipe is None:
            raise ValueError(
                "Diffusion pipeline not found in model_context. " "Ensure the model was loaded as a diffusion model."
            )

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

        # NOTE: we intentionally skip the sequential offloading check here (the guard that
        # exits when pipe is already dispatched).  In new-arch diffusion, the pipe may
        # already be dispatched to multi-device in a prior calib call.  The dispatch
        # state is preserved across calls, so re-dispatching or moving is unnecessary and
        # would break the existing placement.
        if (
            hasattr(self.model, "hf_device_map")
            and len(self.model.hf_device_map) > 1
            and pipe.device != self.model.device
            and torch.device(self.model.device).type in ["cuda", "xpu"]
        ):
            logger.warning(
                "Diffusion model is activated sequential model offloading. "
                "Pipe may already be dispatched from a prior calib call. "
                "Skipping re-dispatch to avoid breaking the existing placement."
            )

        if pipe.device != self.model.device:
            pipe.to(self.model.device)

        device_map = getattr(self.compress_context, "device_map", None)
        device_list = getattr(self.compress_context, "device_list", [])
        # Skip dispatch for secondary transformers
        if (
            not getattr(self, "_inputs_cached", False)
            and device_map is not None
            and is_auto_device_mapping(device_map)
            and len(device_list) > 1
        ):
            pipe_transformer = getattr(pipe, "transformer", None)
            if self.model_context.model is not pipe_transformer:
                pass  # secondary transformer — skip pipeline dispatch
            else:
                pipe = dispatch_model_by_all_available_devices(pipe, device_map)

        with tqdm(range(1, total + 1), desc="cache block inputs") as pbar:
            for ids, prompts in self.dataloader:
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                pipe_kwargs = self._build_pipeline_call_kwargs(pipe, prompts)
                try:
                    if self._requires_calibration_image() or "prompt" in pipe_kwargs:
                        # I2V pipeline: 'image' is the first positional arg, so pass
                        # 'prompt' as keyword to avoid "multiple values for argument 'image'".
                        pipe(**pipe_kwargs)
                    else:
                        pipe(prompts, **pipe_kwargs)
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

    def try_cache_inter_data_gpucpu(self, *args, **kwargs) -> Any:
        """Skip re-caching when DiffusionMixin.quantize has already populated self.inputs.

        CalibCompressor.quantize() always calls try_cache_inter_data_gpucpu, but for
        diffusion models the inputs were already collected by the diffusion pipeline
        in DiffusionMixin.quantize().  Return the cached data directly.
        """
        if getattr(self, "_inputs_cached", False):
            self._inputs_cached = False
            return self.inputs
        if hasattr(super(), "try_cache_inter_data_gpucpu"):
            return super().try_cache_inter_data_gpucpu(*args, **kwargs)

    def quantize(self) -> tuple[torch.nn.Module, dict]:
        """Quantize the diffusion model.

        Overrides the parent to use diffusion-specific cache_inter_data instead of
        the LLM-specific calib path.  The diffusion pipeline forward is used to collect
        block inputs (via _replace_forward hooks), then those inputs are passed to the
        standard CalibCompressor quantization loop.

        For dual-transformer pipelines (e.g. WAN with transformer + transformer_2),
        this method quantizes all transformers sequentially.
        """
        from auto_round.utils import find_matching_blocks, get_block_names
        from auto_round.utils.common import flatten_list

        self.post_init()

        # Get block names and call cache_inter_data to populate self.inputs
        if bool(self.quantizer.quant_block_list):
            all_blocks = self.quantizer.quant_block_list
        else:
            all_blocks = get_block_names(self.model_context.model)
        if len(all_blocks) == 0:
            logger.warning("could not find blocks, exit with original model")
            return self.model_context.model, self.quantizer.layer_config

        if not self.has_variable_block_shape:
            to_cache_block_names = [block[0] for block in all_blocks]
        else:
            to_cache_block_names = flatten_list(all_blocks)

        # Check for additional transformers (e.g. transformer_2 in WAN)
        additional = self._find_additional_transformers()
        if not additional:
            # Single-transformer path: let calib() own pipeline dispatch.
            pipe = self.model_context.pipe
            device_map = getattr(self.compress_context, "device_map", None)
            if device_map is not None and not is_auto_device_mapping(device_map) and not isinstance(device_map, int):
                target_device = get_major_device(device_map)
                # Skip if the transformer is already on the target device to avoid
                # redundant full-model transfer that exhausts GPU memory.
                transformer = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
                param = next(transformer.parameters(), None) if transformer else None
                skip_move = param is not None and get_major_device(str(param.device)) == target_device
                if not skip_move:
                    pipe.to(target_device)

            logger.info("start to cache block inputs")
            all_inputs = self.try_cache_inter_data_gpucpu(
                to_cache_block_names,
                self.nsamples,
                layer_names=[],
            )
            self.inputs = all_inputs
            clear_memory(device_list=device_manager.device_list)
            self._inputs_cached = True
            return super().quantize()

        # Dual-transformer path: quantize all transformers sequentially
        logger.info("Detected multi-transformer diffusion pipeline, quantizing all transformers")

        # Ensure at least 2 inference steps so both transformers are exercised during calibration
        orig_steps = getattr(self, "num_inference_steps", None) or 50
        if orig_steps < 2:
            logger.warning(
                f"num_inference_steps={orig_steps} is too low for dual-transformer "
                f"quantization — increasing to 2 so all transformers receive calibration data."
            )
            self.num_inference_steps = 2

        # Disable low_cpu_mem_usage so quantized models stay in memory during multi-transformer
        # quantization.
        orig_low_cpu = self.compress_context.low_cpu_mem_usage
        orig_immediate_saving = self.compress_context.is_immediate_saving
        self.compress_context.low_cpu_mem_usage = False
        # Defer shard writing until all transformers are quantized. Immediate saving
        # offloads the primary transformer to meta during the first pass, which breaks
        # later calibration when the pipeline switches to transformer_2.
        self.compress_context.is_immediate_saving = False

        # Store primary transformer state
        primary_model = self.model_context.model
        primary_layer_config = dict(self.layer_config) if self.layer_config else {}
        primary_quant_block_list = list(self.quant_block_list) if self.quant_block_list else []
        quantized_extras = {}

        # Quantize primary transformer
        logger.info("start to cache block inputs for primary transformer")
        all_inputs = self.try_cache_inter_data_gpucpu(
            to_cache_block_names,
            self.nsamples,
            layer_names=[],
        )
        self.inputs = all_inputs
        clear_memory(device_list=device_manager.device_list)
        self._inputs_cached = True
        super().quantize()

        # Clear stale hf_device_map so cache_inter_data can re-dispatch for next transformer
        if hasattr(primary_model, "hf_device_map"):
            delattr(primary_model, "hf_device_map")

        # Quantize additional transformers
        for comp_name, transformer in additional:
            logger.info(f"Quantizing {comp_name}")

            # Reset quantization state for new transformer
            self.model_context.model = transformer
            self.model_context.quantized = False
            self._post_init_done = False

            # Re-align device/dtype and dispatch pipeline for secondary transformer
            self._align_device_and_dtype_for_secondary(comp_name)

            # Re-run post_init to set up quantizer for new model
            self.post_init()

            # Get block names for new transformer
            all_blocks = get_block_names(self.model_context.model)
            self.quant_block_list = find_matching_blocks(self.model_context.model, all_blocks, None)
            self.layer_config = {}

            # Get new block names for caching
            if bool(self.quantizer.quant_block_list):
                all_blocks = self.quantizer.quant_block_list
            else:
                all_blocks = get_block_names(self.model_context.model)
            if len(all_blocks) == 0:
                logger.warning(f"could not find blocks in {comp_name}, skipping")
                continue

            if not self.has_variable_block_shape:
                to_cache_block_names = [block[0] for block in all_blocks]
            else:
                to_cache_block_names = flatten_list(all_blocks)

            logger.info(f"start to cache block inputs for {comp_name}")
            all_inputs = self.try_cache_inter_data_gpucpu(
                to_cache_block_names,
                self.nsamples,
                layer_names=[],
            )
            self.inputs = all_inputs
            clear_memory(device_list=device_manager.device_list)
            self._inputs_cached = True
            super().quantize()

            # Store quantized transformer
            quantized_extras[comp_name] = (self.model_context.model, dict(self.layer_config))
            # Also update the pipeline to reference the quantized transformer
            setattr(self.model_context.pipe, comp_name, self.model_context.model)

        # Restore primary transformer state
        self.model_context.model = primary_model
        self.model_context.quantized = True
        self.layer_config = primary_layer_config
        self.quant_block_list = primary_quant_block_list
        self._quantized_transformers = quantized_extras
        self.compress_context.low_cpu_mem_usage = orig_low_cpu
        self.compress_context.is_immediate_saving = orig_immediate_saving
        self.num_inference_steps = orig_steps

        return self.model_context.model, self.quantizer.layer_config

    def save_quantized(
        self,
        output_dir: Optional[str] = None,
        format: Union[str, list] = "auto_round",
        inplace: bool = True,
        return_folders: bool = False,
        **kwargs,
    ) -> Any:
        """Save the quantized model to the specified output directory in the specified format.

        For multi-transformer pipelines, all quantized transformers are saved.

        Args:
            output_dir (str, optional): The directory to save the quantized model. Defaults to None.
            format (str, optional): The format in which to save the model. Defaults to "auto_round".
            inplace (bool, optional): Whether to modify the model in place. Defaults to True.
            return_folders (bool, optional): Whether to return the save folder paths. Defaults to False.
            **kwargs: Additional keyword arguments specific to the export format.

        Returns:
            object: The compressed model object, or (compressed_model, folders) if return_folders is True.
        """
        if output_dir is None:
            return super().save_quantized(
                output_dir, format=format, inplace=inplace, return_folders=return_folders, **kwargs
            )

        pipe = self.model_context.pipe
        quantized_transformers = getattr(self, "_quantized_transformers", {})
        compressed_model = None
        folders = []
        has_multiple_quantized_transformers = bool(quantized_transformers)

        # Handle multi-format (convert string to list if needed)
        _format = format
        if isinstance(_format, str):
            from auto_round.formats import get_formats

            _format = get_formats(_format, self)

        for name in pipe.components.keys():
            val = getattr(pipe, name)
            sub_module_path = (
                os.path.join(output_dir, name) if os.path.basename(os.path.normpath(output_dir)) != name else output_dir
            )
            target_output_dir = (
                sub_module_path
                if has_multiple_quantized_transformers or not self.compress_context.is_immediate_saving
                else output_dir
            )
            if name in quantized_transformers:
                # Save secondary quantized transformer
                saved_model = self.model_context.model
                saved_lc = self.layer_config
                saved_immediate_saving = self.compress_context.is_immediate_saving
                self.model_context.model, self.layer_config = quantized_transformers[name]
                saved_subfolder = getattr(self.model_context.model, "_autoround_pipeline_subfolder", None)
                if has_multiple_quantized_transformers:
                    self.compress_context.is_immediate_saving = False
                    if saved_subfolder is not None:
                        delattr(self.model_context.model, "_autoround_pipeline_subfolder")
                compressed_model = super().save_quantized(
                    output_dir=target_output_dir,
                    format=_format,
                    inplace=inplace,
                    return_folders=False,
                    **kwargs,
                )
                self.compress_context.is_immediate_saving = saved_immediate_saving
                if saved_subfolder is not None:
                    self.model_context.model._autoround_pipeline_subfolder = saved_subfolder
                self.model_context.model = saved_model
                self.layer_config = saved_lc
            elif val is self.model_context.model:
                # Save primary quantized transformer
                saved_immediate_saving = self.compress_context.is_immediate_saving
                saved_subfolder = getattr(self.model_context.model, "_autoround_pipeline_subfolder", None)
                if has_multiple_quantized_transformers:
                    self.compress_context.is_immediate_saving = False
                    if saved_subfolder is not None:
                        delattr(self.model_context.model, "_autoround_pipeline_subfolder")
                compressed_model = super().save_quantized(
                    output_dir=target_output_dir,
                    format=_format,
                    inplace=inplace,
                    return_folders=False,
                    **kwargs,
                )
                self.compress_context.is_immediate_saving = saved_immediate_saving
                if saved_subfolder is not None:
                    self.model_context.model._autoround_pipeline_subfolder = saved_subfolder
            elif val is not None and hasattr(val, "save_pretrained"):
                val.save_pretrained(sub_module_path)
                continue

            if name.startswith("transformer"):
                rename_weights_files(target_output_dir)

            folders.append(target_output_dir)

        if hasattr(pipe, "save_config"):
            pipe.save_config(output_dir)
        elif hasattr(pipe.config, "save_pretrained"):
            pipe.config.save_pretrained(output_dir)
        else:
            # FrozenDict / plain dict — write model_index.json manually
            import json

            model_index_path = os.path.join(output_dir, "model_index.json")
            with open(model_index_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(dict(pipe.config), indent=2, sort_keys=True) + "\n")

        if return_folders:
            return compressed_model, folders
        return compressed_model
