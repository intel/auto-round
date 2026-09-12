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
import copy
import json
import os
from typing import Any, Optional, Union

import torch

from auto_round.logger import logger
from auto_round.utils import clear_memory
from auto_round.utils.device import (
    dispatch_model_block_wise,
    get_major_device,
)
from auto_round.utils.device_manager import device_manager, is_auto_device_mapping
from auto_round.utils.model import rename_weights_files


class DiffusionMixin:
    """Diffusion-specific functionality mixin.

    This mixin adds diffusion model-specific functionality to any compressor
    (Compressor, ImatrixCompressor, etc). It handles
    diffusion models (like Stable Diffusion, FLUX) that require special pipeline
    handling and data generation logic.

    Can be combined with:
    - Compressor (for AutoRound with calibration, or basic RTN)
    - ImatrixCompressor (for RTN with importance matrix)

    Diffusion-specific parameters:
        guidance_scale: Control how much image generation follows text prompt
        num_inference_steps: Number of denoising steps for diffusion generation or evaluation
        calib_num_inference_steps: Number of denoising steps used to collect calibration inputs
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
        calib_num_inference_steps: int = 8,
        generator_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be a positive integer.")
        if calib_num_inference_steps < 1:
            raise ValueError("calib_num_inference_steps must be a positive integer.")

        # Store diffusion-specific attributes
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.calib_num_inference_steps = calib_num_inference_steps
        self.generator_seed = generator_seed
        self.pipeline_call_kwargs = dict(kwargs.pop("pipeline_call_kwargs", {}) or {})

        # Default dataset for diffusion models is "coco2014", not "NeelNanda/pile-10k"
        if kwargs.get("dataset") in (None, "NeelNanda/pile-10k"):
            kwargs["dataset"] = "coco2014"

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
            # CalibrationContext; it only comes from kwargs now (entry.py forwards
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

        # Call parent class __init__ (will be Compressor, ImatrixCompressor, etc)
        super().__init__(*args, **kwargs)

        pipe = getattr(self.model_context, "pipe", None)
        model = getattr(self.model_context, "model", None)
        if (
            getattr(self.model_context, "preloaded_diffusion_pipeline", False)
            and pipe is not None
            and model is not None
        ):
            self._align_pipeline_dtype(pipe, model.dtype)

    @staticmethod
    def _align_pipeline_dtype(pipe, target_dtype) -> None:
        """Align a preloaded pipeline without casting declared FP32 tensors."""

        for component_name in pipe.components:
            component = getattr(pipe, component_name, None)
            if not isinstance(component, torch.nn.Module):
                continue

            fp32_modules = getattr(component, "_keep_in_fp32_modules", None) or []
            tensors = list(component.named_parameters()) + list(component.named_buffers())
            for tensor_name, tensor in tensors:
                if not tensor.is_floating_point():
                    continue
                keep_in_fp32 = any(module_name in tensor_name for module_name in fp32_modules)
                desired_dtype = torch.float32 if keep_in_fp32 else target_dtype
                if tensor.dtype != desired_dtype:
                    tensor.data = tensor.data.to(dtype=desired_dtype)

    def _get_calibrator_kind(self) -> str:
        """Select the diffusion calibration strategy.

        ``DiffusionCalibrator`` lives at
        :mod:`auto_round.calibration.diffusion` and owns diffusion-specific
        calibration input collection.
        """
        return "diffusion"

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

    def _defer_multi_transformer_serialization(self) -> None:
        """Keep every transformer executable until all calibration passes finish.

        Immediate packing replaces tuned ``nn.Linear`` modules with export-only
        ``QuantLinear`` modules. A multi-transformer pipeline such as WAN can
        still execute the primary transformer while collecting inputs for
        ``transformer_2``, so both packing and shard writing must be deferred.
        """
        self.compress_context.is_immediate_packing = False
        self.compress_context.is_immediate_saving = False

    def _align_device_and_dtype_for_secondary(self, transformer_name: str):
        """Dispatch a secondary transformer without changing component dtypes."""
        pipe = getattr(self.model_context, "pipe", None)
        model = getattr(self.model_context, "model", None)
        if pipe is None or model is None:
            return

        # Calibration owns component offload; moving the entire pipeline here
        # defeats low_gpu_mem_usage before calibration can install its hooks.
        if getattr(self.compress_context, "low_gpu_mem_usage", False):
            return

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
                    try:
                        comp.to(comp_device)
                    except (NotImplementedError, RuntimeError):
                        continue

            self.model_context.model = dispatch_model_block_wise(model, device_map)
            setattr(pipe, transformer_name, self.model_context.model)
        elif device_map is not None:
            target_device = get_major_device(device_map)
            pipe.to(target_device)

    def _release_calibration_components(self) -> None:
        """Release component-offload hooks and weights before block-wise tuning."""
        if getattr(self.calibration, "_cpu_offload_mode", None) != "model":
            return
        from accelerate.hooks import remove_hook_from_submodules

        pipe = self.model_context.pipe
        for name in pipe.components:
            component = getattr(pipe, name, None)
            if isinstance(component, torch.nn.Module):
                remove_hook_from_submodules(component)
                component.to("cpu")

    def cache_data(self, *args, **kwargs) -> Any:
        """Consume diffusion inputs at the current orchestrator caching boundary."""
        if getattr(self, "_inputs_cached", False):
            self._inputs_cached = False
            return self.inputs
        return super().cache_data(*args, **kwargs)

    def try_cache_inter_data_gpucpu(self, *args, **kwargs) -> Any:
        """Compatibility entry point for the calibrator-owned caching path."""
        return self.cache_data(*args, **kwargs)

    def quantize(self) -> tuple[torch.nn.Module, dict]:
        """Quantize the diffusion model.

        Overrides the parent to use diffusion-specific cache_inter_data instead of
        the LLM-specific calib path.  The diffusion pipeline forward is used to collect
        block inputs (via _replace_forward hooks), then those inputs are passed to the
        standard CalibCompressor quantization loop.

        For dual-transformer pipelines (e.g. WAN with transformer + transformer_2),
        this method quantizes all transformers sequentially.
        """
        from auto_round.utils import get_block_names
        from auto_round.utils.common import flatten_list

        requested_layer_config = copy.deepcopy(self.layer_config or {})
        self.post_init()

        # Zero-shot (RTN) path: no calibration data needed
        if not self.need_calib:
            return self._quantize_zero_shot()

        # Get block names and call cache_inter_data to populate self.inputs
        if bool(self.quant_block_list):
            all_blocks = self.quant_block_list
        else:
            all_blocks = get_block_names(self.model_context.model)
        if len(all_blocks) == 0:
            logger.warning("could not find blocks, exit with original model")
            return self.model, self.layer_config

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
            if (
                device_map is not None
                and not is_auto_device_mapping(device_map)
                and not isinstance(device_map, int)
                and not getattr(self.compress_context, "low_gpu_mem_usage", False)
            ):
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
                self.calibration_context.nsamples,
                layer_names=[],
            )
            self.inputs = all_inputs
            self._release_calibration_components()
            clear_memory()
            self._inputs_cached = True
            return super().quantize()

        # Dual-transformer path: quantize all transformers sequentially
        logger.info("Detected multi-transformer diffusion pipeline, quantizing all transformers")

        # Each post_init builds model-bound plans, preprocessors and calibrators.
        # Retain the primary runtime after tuning, not its pre-tuning config.
        runtime_names = (
            "compression_plan",
            "_alg_composer",
            "calibration",
            "_format_resolution",
            "_post_init_done",
            "to_quant_block_names",
            "has_variable_block_shape",
        )
        missing = object()

        def snapshot_runtime():
            return {name: getattr(self, name, missing) for name in runtime_names}

        primary_runtime = snapshot_runtime()
        primary_context = vars(self.model_context).copy()
        primary_layer_config = self.layer_config
        primary_quant_block_list = self.quant_block_list
        original_settings = vars(self.compress_context).copy()
        orig_steps = self.calib_num_inference_steps
        pipe = self.model_context.pipe
        original_boundary_ratio = getattr(pipe.config, "boundary_ratio", missing)
        quantized_extras = {}
        succeeded = False
        self._quantized_transformers = {}
        try:
            if self.calib_num_inference_steps < 2:
                logger.warning("Increasing calib_num_inference_steps to 2 for multi-transformer calibration.")
                self.calib_num_inference_steps = 2
            # The primary calibrator was constructed by post_init above.
            self.calibration.calib_num_inference_steps = self.calib_num_inference_steps
            self.compress_context.low_cpu_mem_usage = False
            self._defer_multi_transformer_serialization()

            for index, (comp_name, transformer) in enumerate([("transformer", self.model_context.model), *additional]):
                if index:
                    self.model_context.model = transformer
                    self.model_context.quantized = False
                    self._post_init_done = False
                    self.calibration = None
                    self._inputs_cached = False
                    self.inputs = {}
                    # Reset discovery before post_init. Clearing the resolved
                    # layer config afterwards discards the secondary quantizers.
                    self.layer_config = copy.deepcopy(requested_layer_config)
                    self.quant_block_list = None
                    self._align_device_and_dtype_for_secondary(comp_name)
                    self.post_init()
                    self._defer_multi_transformer_serialization()
                    all_blocks = self.quant_block_list or get_block_names(self.model_context.model)
                    if not all_blocks:
                        raise ValueError(f"could not find blocks in {comp_name}")
                    to_cache_block_names = (
                        flatten_list(all_blocks)
                        if self.has_variable_block_shape
                        else [block[0] for block in all_blocks]
                    )

                # Wan routes all steps to the active expert. Other pipeline
                # families do not acquire a synthetic boundary_ratio config.
                if original_boundary_ratio is not missing:
                    pipe.register_to_config(boundary_ratio=0.0 if index == 0 else 1.1)
                logger.info(f"start to cache block inputs for {comp_name}")
                self.inputs = self.try_cache_inter_data_gpucpu(
                    to_cache_block_names, self.calibration_context.nsamples, layer_names=[]
                )
                self._release_calibration_components()
                clear_memory(device_list=device_manager.device_list)
                self._inputs_cached = True
                super().quantize()
                self._release_calibration_components()
                setattr(pipe, comp_name, self.model_context.model)
                if hasattr(self.model_context.model, "hf_device_map"):
                    delattr(self.model_context.model, "hf_device_map")
                if index == 0:
                    primary_context = vars(self.model_context).copy()
                    primary_layer_config = self.layer_config
                    primary_quant_block_list = self.quant_block_list
                    primary_runtime = snapshot_runtime()
                else:
                    quantized_extras[comp_name] = (self.model_context.model, self.layer_config)
            succeeded = True
        finally:
            # A failed cache/tune must not leave the public compressor pointing
            # at an expert or advertise a partially quantized pipeline as done.
            try:
                self._release_calibration_components()
            finally:
                vars(self.model_context).clear()
                vars(self.model_context).update(primary_context)
                for name, value in primary_runtime.items():
                    if value is missing:
                        self.__dict__.pop(name, None)
                    else:
                        setattr(self, name, value)
                self.layer_config = primary_layer_config
                self.quant_block_list = primary_quant_block_list
                self.model_context.quantized = succeeded
                self._inputs_cached = False
                self.inputs = {}
                self._quantized_transformers = quantized_extras
                vars(self.compress_context).clear()
                vars(self.compress_context).update(original_settings)
                self.calib_num_inference_steps = orig_steps
                self.calibration.calib_num_inference_steps = orig_steps
                if original_boundary_ratio is not missing:
                    pipe.register_to_config(boundary_ratio=original_boundary_ratio)

        return self.model_context.model, self.layer_config

    def save_quantized(
        self,
        output_dir: Optional[str] = None,
        format: Optional[Union[str, list]] = None,
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
        saved_transformer_paths = {}
        has_multiple_quantized_transformers = bool(quantized_transformers)

        # Handle multi-format (convert string to list if needed)
        _format = format if format is not None else getattr(self, "formats", None) or "auto_round"
        if isinstance(_format, str):
            _format = self._resolve_format_string(_format)

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
            if name in quantized_transformers or val is self.model_context.model:
                saved_model = self.model_context.model
                saved_lc = getattr(self, "layer_config", None)
                saved_immediate_saving = self.compress_context.is_immediate_saving
                if name in quantized_transformers:
                    self.model_context.model, self.layer_config = quantized_transformers[name]
                component_model = self.model_context.model
                saved_subfolder = getattr(component_model, "_autoround_pipeline_subfolder", None)
                try:
                    if has_multiple_quantized_transformers:
                        self.compress_context.is_immediate_saving = False
                        if saved_subfolder is not None:
                            delattr(component_model, "_autoround_pipeline_subfolder")
                    compressed_model = super().save_quantized(
                        output_dir=target_output_dir,
                        format=_format,
                        inplace=inplace,
                        return_folders=False,
                        **kwargs,
                    )
                    if compressed_model is not None and name.startswith("transformer"):
                        saved_transformer_paths[name] = target_output_dir
                finally:
                    self.compress_context.is_immediate_saving = saved_immediate_saving
                    if saved_subfolder is not None:
                        component_model._autoround_pipeline_subfolder = saved_subfolder
                    self.model_context.model = saved_model
                    self.layer_config = saved_lc
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
            model_index_path = os.path.join(output_dir, "model_index.json")
            with open(model_index_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(dict(pipe.config), indent=2, sort_keys=True) + "\n")

        # The source pipeline remains a Diffusers/QDQ pipeline. Only the saved
        # runtime export should advertise Nunchaku loaders, and only when the
        # artifact header confirms the Wan architecture adapter was used.
        if len(_format) == 1 and getattr(_format[0], "format_name", None) == "svdquant_nunchaku":
            from safetensors import safe_open

            from auto_round.export.svdquant_nunchaku import NUNCHAKU_WEIGHT_FILENAME

            replacements = {}
            for name, path in saved_transformer_paths.items():
                weights_path = os.path.join(path, NUNCHAKU_WEIGHT_FILENAME)
                if not os.path.isfile(weights_path):
                    continue
                with safe_open(weights_path, framework="pt", device="cpu") as artifact:
                    metadata = artifact.metadata() or {}
                if (
                    metadata.get("model_class") == "NunchakuWanTransformer3DModel"
                    and isinstance(json.loads(metadata.get("config", "null")), dict)
                    and json.loads(metadata.get("quantization_config", "{}")).get("method") == "svdquant"
                ):
                    replacements[name] = ["nunchaku", "NunchakuWanTransformer3DModel"]
            if replacements:
                model_index_path = os.path.join(output_dir, "model_index.json")
                with open(model_index_path, encoding="utf-8") as f:
                    model_index = json.load(f)
                model_index.update(replacements)
                with open(model_index_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(model_index, indent=2, sort_keys=True) + "\n")

        if return_folders:
            return compressed_model, folders
        return compressed_model
