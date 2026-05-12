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
from typing import Union

import torch
from PIL import Image
from tqdm import tqdm

from auto_round.logger import logger
from auto_round.utils.model import wrap_block_forward_positional_to_kwargs
from auto_round.utils.device import dispatch_model_by_all_available_devices, is_auto_device_mapping
from auto_round.utils import clear_memory


class DiffusionMixin:
    """Diffusion-specific functionality mixin.

    This mixin adds diffusion model-specific functionality to any compressor
    (CalibCompressor, ZeroShotCompressor, ImatrixCompressor, etc). It handles
    diffusion models (like Stable Diffusion, FLUX) that require special pipeline
    handling and data generation logic.

    Can be combined with:
    - CalibCompressor (for AutoRound with calibration)
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

    def __init__(self, *args, guidance_scale=7.5, num_inference_steps=50, generator_seed=None, **kwargs):
        # Store diffusion-specific attributes
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generator_seed = generator_seed
        self.pipeline_call_kwargs = dict(kwargs.pop("pipeline_call_kwargs", {}) or {})

        # Mirror old-arch DiffusionCompressor.__init__: when iters > 0, diffusion calibration
        # cannot use batch_size > 1 for non-text modules; fold the extra batch into
        # gradient_accumulate_steps so the effective sample count is unchanged.
        # The authoritative batch_size lives on the AlgConfig (args[0]); kwargs may also
        # carry it from AutoRoundCompatible. Patch BOTH (same pattern as MLLMMixin).
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
            batch_size = kwargs.get("batch_size", None)
            if batch_size is None and _alg_cfg is not None:
                cfgs = _alg_cfg if isinstance(_alg_cfg, list) else [_alg_cfg]
                for cfg in cfgs:
                    if hasattr(cfg, "batch_size") and cfg.batch_size is not None:
                        batch_size = cfg.batch_size
                        break
            if batch_size is not None and batch_size != 1:
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
                        if hasattr(cfg, "batch_size"):
                            cfg.batch_size = 1
                        if hasattr(cfg, "gradient_accumulate_steps"):
                            cfg.gradient_accumulate_steps = new_grad_acc
                logger.warning(
                    f"reset batch_size({batch_size}) to 1 and "
                    f"gradient_accumulate_steps to {new_grad_acc} "
                    f"because batch_size={batch_size} cannot be used for calibrating non-text modules."
                )

        # Call parent class __init__ (will be CalibCompressor, ImatrixCompressor, etc)
        super().__init__(*args, **kwargs)

        # Mirror old-arch DiffusionCompressor._align_device_and_dtype: unconditionally
        # cast the full diffusion pipeline (VAE, text encoder, etc.) to the transformer's
        # dtype so that calibration's pipe(...) call doesn't crash with dtype mismatches
        # when the transformer is force-cast to bf16 for activation quantization.
        # Note: pipe.dtype only reflects the primary component, so an equality check would
        # miss mixed-dtype pipelines where e.g. the VAE is still float32.
        pipe = getattr(self.model_context, "pipe", None)
        model = getattr(self.model_context, "model", None)
        if pipe is not None and model is not None:
            is_nextstep = hasattr(model, "config") and getattr(model.config, "model_type", None) == "nextstep"
            if not is_nextstep:
                pipe.to(model.dtype)

    def _build_pipeline_call_kwargs(self, pipe, prompts):
        call_kwargs = {
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "generator": (
                None if self.generator_seed is None else torch.Generator(device=pipe.device).manual_seed(self.generator_seed)
            ),
        }
        call_kwargs.update(self.pipeline_call_kwargs)

        if self._requires_calibration_image():
            call_kwargs.setdefault("image", self._get_calibration_image(len(prompts) if isinstance(prompts, list) else 1))
            call_kwargs.setdefault("prompt", prompts)

        return call_kwargs

    def _get_block_forward_func(self, name: str):
        """Diffusion models pass positional args; wrap the base forward func accordingly.

        The MRO guarantees that super() resolves to CalibCompressor._get_block_forward_func,
        mirroring the old-arch pattern in compressors/diffusion/compressor.py.
        """
        # For diffusion with multi-device dispatch, the pipeline manages device placement
        # automatically.
        if (
            self.model_context.is_diffusion
            and hasattr(self.model_context.model, "hf_device_map")
            and len(self.model_context.model.hf_device_map) > 1
        ):
            def passthrough_block_forward(block, hidden_states=None, *positional_inputs, **kwargs):
                # Call _true_orig_forward directly to avoid recursion (block.forward is
                # this wrapper) and to avoid the "multiple values" error that would occur
                # if hidden_states were passed as both positional and keyword arg.
                if hidden_states is not None:
                    return block._true_orig_forward(hidden_states, *positional_inputs, **kwargs)
                return block._true_orig_forward(*positional_inputs, **kwargs)
            return passthrough_block_forward
        return wrap_block_forward_positional_to_kwargs(super()._get_block_forward_func(name))

    def _should_stop_cache_forward(self, name: str) -> bool:
        """Diffusion models must run all denoising steps to collect enough inputs.

        Mirrors old-arch DiffusionCompressor._should_stop_cache_forward which always
        returns False so the pipeline never exits early after the first block hit.
        Without this, CalibCompressor._should_stop_cache_forward would stop after the
        first inference step, yielding only nsamples inputs instead of
        nsamples * num_inference_steps.
        """
        return False

    def _requires_calibration_image(self) -> bool:
        """Return True when the pipeline's __call__ has a required 'image' parameter.

        I2V (image-to-video) pipelines like WanImageToVideoPipeline require a PIL/torch
        image as input. This is detected by checking whether 'image' is a positional-or-
        keyword parameter without a default value.
        """
        image_param = inspect.signature(self.model_context.pipe.__call__).parameters.get("image")
        return image_param is not None and image_param.default is inspect.Parameter.empty

    def _get_calibration_image(self, batch_size: int):
        """Return a synthetic PIL Image for I2V pipeline calibration.

        Respects the pipeline's declared height/width defaults so that the resulting
        image satisfies any divisibility constraints (e.g. WanImageToVideoPipeline
        requires height % 16 == 0 and width % 16 == 0).

        Args:
            batch_size: Number of images to return.  If 1 a single Image is returned,
                        otherwise a list of Image copies is returned.
        """
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
        image = Image.new("RGB", (int(width), int(height)), color=(127, 127, 127))
        if batch_size == 1:
            return image
        return [image.copy() for _ in range(batch_size)]

    @torch.no_grad()
    def calib(self, nsamples, bs):
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

        # Mirror old-arch DiffusionCompressor.cache_inter_data: dispatch the full pipeline
        # across the requested device_map so that inference runs on GPU, not CPU.
        device_map = getattr(self.compress_context, "device_map", None)
        device_list = getattr(self.compress_context, "device_list", [])
        # Only dispatch if the pipe has not been dispatched yet (not yet in hf_device_map).
        # When _inputs_cached is set, we are in the second calib call and the pipe is
        # already dispatched from the first call.
        if not getattr(self, "_inputs_cached", False) and device_map is not None and is_auto_device_mapping(device_map) and len(device_list) > 1:
            pipe = dispatch_model_by_all_available_devices(pipe, device_map)

        with tqdm(range(1, total + 1), desc="cache block inputs") as pbar:
            for ids, prompts in self.dataloader:
                logger.info(f"[DiffusionMixin calib] step, prompts type={type(prompts)}, ids type={type(ids)}")
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                pipe_kwargs = self._build_pipeline_call_kwargs(pipe, prompts)
                logger.info(f"[NEW ARCH] pipe call START, pipe={type(pipe).__name__}, pipe_device={pipe.device}, model_device={self.model.device}, prompts_count={len(prompts)}")
                import time as _time_module
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t0 = _time_module.time()
                _t_pipe_call = None
                try:
                    logger.info(f"[NEW ARCH] calib: prompts={type(prompts).__name__}, pipe_kwargs={list(pipe_kwargs.keys())}, "
                                f"num_frames={pipe_kwargs.get('num_frames', 'DEFAULT')}, "
                                f"height={pipe_kwargs.get('height', 'DEFAULT')}, "
                                f"width={pipe_kwargs.get('width', 'DEFAULT')}, "
                                f"output_type={pipe_kwargs.get('output_type', 'DEFAULT')}, "
                                f"guidance_scale={pipe_kwargs.get('guidance_scale')}, "
                                f"num_inference_steps={pipe_kwargs.get('num_inference_steps')}, "
                                f"_requires_calibration_image={self._requires_calibration_image()}")
                    _t_before = _time_module.time()
                    if self._requires_calibration_image() or "prompt" in pipe_kwargs:
                        # I2V pipeline: 'image' is the first positional arg, so pass
                        # 'prompt' as keyword to avoid "multiple values for argument 'image'".
                        result = pipe(**pipe_kwargs)
                    else:
                        result = pipe(prompts, **pipe_kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _t_pipe_call = _time_module.time() - _t_before
                    logger.info(f"[NEW ARCH] pipe call END, elapsed={_t_pipe_call:.1f}s")
                except NotImplementedError:
                    _t_pipe_call = None
                except Exception as error:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _elapsed = _time_module.time() - _t0
                    logger.error(f"[NEW ARCH] pipe call failed after {_elapsed:.1f}s: {error}")
                    raise error
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _total = _time_module.time() - _t0
                logger.info(f"[NEW ARCH] calib step done, total={_total:.1f}s, pipe_call={_t_pipe_call:.1f}s, overhead={_total - (_t_pipe_call or 0):.3f}s")
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

    def try_cache_inter_data_gpucpu(self, *args, **kwargs):
        """Skip re-caching when DiffusionMixin.quantize has already populated self.inputs.

        CalibCompressor.quantize() always calls try_cache_inter_data_gpucpu, but for
        diffusion models the inputs were already collected by the diffusion pipeline
        in DiffusionMixin.quantize().  Return the cached data directly.
        """
        if getattr(self, "_inputs_cached", False):
            self._inputs_cached = False
            return self.inputs
        return super().try_cache_inter_data_gpucpu(*args, **kwargs)

    def quantize(self):
        """Quantize the diffusion model.

        Overrides the parent to use diffusion-specific cache_inter_data instead of
        the LLM-specific calib path.  The diffusion pipeline forward is used to collect
        block inputs (via _replace_forward hooks), then those inputs are passed to the
        standard CalibCompressor quantization loop.
        """
        from auto_round.utils import get_block_names, flatten_list

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

        logger.info("start to cache block inputs")
        all_inputs = self.try_cache_inter_data_gpucpu(
            to_cache_block_names,
            self.nsamples,
            layer_names=[],
        )
        self.inputs = all_inputs
        clear_memory(device_list=self.compress_context.device_list)

        # Signal that caching is done so super().quantize() skips the cache step
        self._inputs_cached = True
        return super().quantize()

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

        pipe = self.model_context.pipe
        compressed_model = None
        for name in pipe.components.keys():
            val = getattr(pipe, name)
            sub_module_path = (
                os.path.join(output_dir, name) if os.path.basename(os.path.normpath(output_dir)) != name else output_dir
            )
            if (
                hasattr(val, "config")
                and hasattr(val.config, "_name_or_path")
                and val.config._name_or_path == self.model.config._name_or_path
            ):
                compressed_model = super().save_quantized(
                    output_dir=sub_module_path if not self.compress_context.is_immediate_saving else output_dir,
                    format=format,
                    inplace=inplace,
                    **kwargs,
                )
            elif val is not None and hasattr(val, "save_pretrained"):
                val.save_pretrained(sub_module_path)
        pipe.save_config(output_dir)
        return compressed_model
