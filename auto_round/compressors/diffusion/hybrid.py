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

"""HybridCompressor for models with both AR and diffusion components.

This compressor handles models that have a hybrid architecture consisting of:
  - An autoregressive (AR) language model component
  - A diffusion transformer (DiT) component

It quantizes both components in a single workflow:
  Phase 1: Quantize the AR model using MLLM-style text calibration
  Phase 2: Quantize the DiT model using diffusion-style pipeline calibration

Supported hybrid pipelines are registered in ``HYBRID_AR_COMPONENTS``.
To add a new model, register its AR component attribute name and (optionally)
its DiT block output config in ``output_configs``.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Union

import torch

from auto_round.compressors.diffusion.compressor import DiffusionCompressor, output_configs
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import (
    LazyImport,
    clear_memory,
    extract_block_names_to_str,
    find_matching_blocks,
    get_block_names,
)

pipeline_utils = LazyImport("diffusers.pipelines.pipeline_utils")

# ---------------------------------------------------------------------------
# Registry: known AR component attribute names in hybrid diffusion pipelines.
# Each entry maps a pipeline attribute name to the component role.
# When a pipeline has *both* "transformer" and one of these attributes,
# it is recognised as a hybrid model.
# To support a new hybrid architecture, simply add its AR attribute name here.
# ---------------------------------------------------------------------------
HYBRID_AR_COMPONENTS = [
    "vision_language_encoder",   # GLM-Image
    # Add new AR component names here, e.g.:
    # "language_model",
    # "text_decoder",
]

# ---------------------------------------------------------------------------
# Register DiT block output configs for hybrid models.
# Maps block class name -> ordered list of output tensor names.
# Pure-diffusion blocks (Flux*) are already registered in DiffusionCompressor.
# ---------------------------------------------------------------------------
output_configs["GlmImageTransformerBlock"] = ["hidden_states", "encoder_hidden_states"]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _find_ar_component_name(model_or_path):
    """Return the AR component attribute name if model_or_path is a hybrid pipeline, else None."""
    if isinstance(model_or_path, str):
        index_path = os.path.join(model_or_path, "model_index.json")
        if not os.path.exists(index_path):
            from huggingface_hub import hf_hub_download
            try:
                index_path = hf_hub_download(model_or_path, "model_index.json")
            except Exception:
                return None

        import json
        with open(index_path) as f:
            data = json.load(f)
        if "transformer" not in data:
            return None
        for name in HYBRID_AR_COMPONENTS:
            if name in data:
                return name
        return None

    # Runtime pipeline object
    if hasattr(model_or_path, "transformer"):
        for name in HYBRID_AR_COMPONENTS:
            if hasattr(model_or_path, name) and getattr(model_or_path, name) is not None:
                return name
    return None


def is_hybrid_diffusion_model(model_or_path):
    """Return True if *model_or_path* represents a hybrid AR+Diffusion pipeline."""
    return _find_ar_component_name(model_or_path) is not None


class HybridCompressor(DiffusionCompressor):
    """Compressor for hybrid AR + diffusion models.

    Quantizes both the autoregressive component and the diffusion transformer
    component in a single workflow.  The AR component is discovered automatically
    from ``HYBRID_AR_COMPONENTS``.

    Args:
        model: Model name/path or DiffusionPipeline object.
        tokenizer: Tokenizer (auto-loaded from pipeline if None).
        guidance_scale: Guidance scale for diffusion calibration.
        num_inference_steps: Denoising steps for diffusion calibration.
        generator_seed: Seed for noise generator.
        scheme: Quantization scheme.
        dataset: Calibration dataset for DiT (default: "coco2014").
        ar_dataset: Calibration dataset for AR model (default: "NeelNanda/pile-10k").
        quant_nontext_module: Whether to also quantize vision encoder in AR model.
        iters: Optimization iterations.
        seqlen: Calibration sequence length for AR model.
        nsamples: Number of calibration samples.
        batch_size: Calibration batch size.
        quant_ar: Whether to quantize the AR component.
        quant_dit: Whether to quantize the DiT component.
        height: Image height passed to the pipeline during DiT calibration (required by some pipelines
            such as GLM-Image; ignored if the pipeline does not accept it).
        width: Image width passed to the pipeline during DiT calibration.
        **kwargs: Additional keyword arguments passed to base compressor.
    """

    def __init__(
        self,
        model: Union[object, str],
        tokenizer=None,
        platform: str = "hf",
        guidance_scale: float = 1.5,
        num_inference_steps: int = 10,
        generator_seed: int = None,
        scheme: Union[str, dict, QuantizationScheme] = "W4A16",
        layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "coco2014",
        ar_dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
        quant_nontext_module: bool = False,
        iters: int = 200,
        seqlen: int = 2048,
        nsamples: int = 128,
        batch_size: int = 8,
        gradient_accumulate_steps: int = 1,
        low_gpu_mem_usage: bool = True,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        quant_ar: bool = True,
        quant_dit: bool = True,
        height: int = None,
        width: int = None,
        **kwargs,
    ):
        logger.warning("Hybrid AR+Diffusion model quantization is experimental.")
        model_dtype = kwargs.pop("model_dtype", None)

        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generator_seed = generator_seed
        self.quant_ar = quant_ar
        self.quant_dit = quant_dit
        self.quant_nontext_module = quant_nontext_module
        self.ar_dataset = ar_dataset
        self.height = height
        self.width = width

        to_quant_block_names: Union[str, list, None] = kwargs.pop("to_quant_block_names", None)
        if device_map is None:
            device_map = 0
        self._set_device(device_map)

        # --- Load the pipeline ---
        if isinstance(model, str):
            from auto_round.utils.model import diffusion_load_model
            pipe, dit_model = diffusion_load_model(
                model, platform=platform, device=self.device, model_dtype=model_dtype
            )
        elif isinstance(model, pipeline_utils.DiffusionPipeline):
            pipe = model
            dit_model = pipe.transformer
        else:
            raise ValueError(
                f"HybridCompressor requires a model path or DiffusionPipeline, got {type(model)}"
            )

        # --- Discover the AR component dynamically ---
        self.ar_component_name = _find_ar_component_name(pipe)
        if self.ar_component_name is None and self.quant_ar:
            logger.warning(
                f"No AR component found in pipeline (checked: {HYBRID_AR_COMPONENTS}), "
                "skipping AR quantization."
            )
            self.quant_ar = False

        self.pipe = pipe
        self.dit_model = dit_model
        self.ar_model = (
            getattr(pipe, self.ar_component_name, None)
            if self.ar_component_name
            else None
        )

        if not self.quant_ar and not self.quant_dit:
            raise ValueError("At least one of quant_ar and quant_dit must be True.")

        model = dit_model

        # --- Detect DiT blocks ---
        all_blocks = get_block_names(model)
        dit_blocks = find_matching_blocks(model, all_blocks, to_quant_block_names)

        # Filter to only blocks whose class has a registered output_config.
        # get_block_names may discover non-transformer ModuleLists (e.g. MLP projectors)
        # that don't match the expected output format.
        if to_quant_block_names is None:
            filtered = []
            for group in dit_blocks:
                if not group:
                    continue
                parts = group[0].split(".")
                m = model
                for p in parts:
                    m = getattr(m, p)
                if m.__class__.__name__ in output_configs:
                    filtered.append(group)
            if filtered:
                dit_blocks = filtered
        self.dit_quant_block_list = dit_blocks

        # --- Detect AR blocks ---
        if self.quant_ar and self.ar_model is not None:
            from auto_round.special_model_handler import SPECIAL_MULTIMODAL_BLOCK
            model_type = getattr(getattr(self.ar_model, "config", None), "model_type", None)
            if model_type and model_type in SPECIAL_MULTIMODAL_BLOCK:
                self.ar_quant_block_list = SPECIAL_MULTIMODAL_BLOCK[model_type](
                    self.ar_model, quant_vision=quant_nontext_module
                )
            else:
                self.ar_quant_block_list = [get_block_names(self.ar_model)]
        else:
            self.ar_quant_block_list = []

        self.quant_block_list = self.dit_quant_block_list
        if to_quant_block_names is None:
            to_quant_block_names = extract_block_names_to_str(self.quant_block_list)

        # Force batch_size to 1 for diffusion calibration
        if iters > 0 and batch_size != 1:
            logger.warning(
                f"reset batch_size({batch_size}) to 1 and "
                f"gradient_accumulate_steps({gradient_accumulate_steps}) "
                f"to {batch_size * gradient_accumulate_steps}, "
                f"because batch_size > 1 cannot be used for diffusion calibration."
            )
            gradient_accumulate_steps = batch_size * gradient_accumulate_steps
            batch_size = 1

        seqlen = 2048 if seqlen is None else seqlen

        if nsamples % batch_size != 0:
            nsamples = (nsamples // batch_size + 1) * batch_size
            logger.warning(f"'nsamples' is not divisible by 'batch_size', adjusted to {nsamples}")

        kwargs["diffusion"] = True
        self._saved_pipe = pipe
        self._saved_dit_model = dit_model
        self._saved_ar_model = self.ar_model

        from auto_round.compressors.base import BaseCompressor
        BaseCompressor.__init__(
            self,
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

        # Restore references that BaseCompressor.__init__ may have overwritten
        self.pipe = self._saved_pipe
        self.dit_model = self._saved_dit_model
        self.ar_model = self._saved_ar_model

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize both AR and DiT components.

        Phase 1: AR model via MLLM-style text calibration.
        Phase 2: DiT model via diffusion pipeline calibration.
        """
        start_time = time.time()
        combined_layer_config = {}

        # =================== Phase 1: AR Model ===================
        if self.quant_ar and self.ar_model is not None:
            logger.info("=" * 60)
            logger.info(f"Phase 1: Quantizing AR model ({self.ar_component_name})")
            logger.info("=" * 60)

            ar_compressor = self._create_ar_compressor()
            ar_model, ar_layer_config = ar_compressor.quantize()

            self.ar_model = ar_model
            setattr(self.pipe, self.ar_component_name, ar_model)
            combined_layer_config.update(
                {f"ar.{k}": v for k, v in ar_layer_config.items()}
            )
            self.ar_layer_config = ar_layer_config

            # Preserve serialization-relevant attributes from the AR compressor
            # so save_quantized can build the correct serialization_dict.
            from auto_round.compressors.base import SERIALIZATION_KEYS
            self._ar_serialization = {
                k: getattr(ar_compressor, k, None) for k in SERIALIZATION_KEYS
            }

            # Move AR model to CPU to free GPU for Phase 2
            self.ar_model.to("cpu")
            clear_memory(device_list=self.device_list)
            logger.info(f"Phase 1 complete: AR model ({self.ar_component_name}) quantized")

        # =================== Phase 2: DiT Model ===================
        if self.quant_dit:
            logger.info("=" * 60)
            logger.info("Phase 2: Quantizing DiT model (transformer)")
            logger.info("=" * 60)

            # Move DiT to target device for calibration
            self.dit_model = self.dit_model.to(self.device)
            self.model = self.dit_model
            self.quant_block_list = self.dit_quant_block_list
            self.quantized = False
            self.batch_dim = None

            for n, m in self.model.named_modules():
                m.global_name = n

            dit_model, dit_layer_config = self._quantize_dit()

            self.dit_model = dit_model
            self.pipe.transformer = dit_model
            combined_layer_config.update(
                {f"dit.{k}": v for k, v in dit_layer_config.items()}
            )
            self.dit_layer_config = dit_layer_config

            logger.info("Phase 2 complete: DiT model quantized")

        end_time = time.time()
        logger.info(f"Total hybrid quantization time: {end_time - start_time:.1f}s")

        self.quantized = True
        self.layer_config = combined_layer_config
        self.model = self.dit_model
        return self.model, self.layer_config

    def _create_ar_compressor(self):
        """Create an MLLM compressor for the AR component."""
        from auto_round.compressors.mllm.compressor import MLLMCompressor

        processor = getattr(self.pipe, "processor", None)
        tokenizer = getattr(self.pipe, "tokenizer", None)

        ar = MLLMCompressor(
            model=self.ar_model,
            tokenizer=tokenizer,
            processor=processor,
            image_processor=None,
            platform=self.platform,
            scheme=copy.deepcopy(self.orig_scheme) if hasattr(self, "orig_scheme") else self.scheme,
            dataset=self.ar_dataset,
            quant_nontext_module=self.quant_nontext_module,
            iters=self.iters,
            seqlen=self.seqlen,
            nsamples=self.nsamples,
            batch_size=1,
            gradient_accumulate_steps=self.gradient_accumulate_steps,
            low_gpu_mem_usage=self.low_gpu_mem_usage,
            device_map=self.device_map,
            enable_torch_compile=self.enable_torch_compile,
            seed=self.seed,
        )
        if hasattr(self, "formats"):
            ar.formats = self.formats
        ar.inplace = False
        # Required by base.quantize() → _adjust_immediate_packing_and_saving();
        # None disables immediate packing (correct since we call quantize() directly).
        ar.orig_output_dir = None
        return ar

    def _quantize_dit(self):
        """Quantize the DiT model using the parent DiffusionCompressor's quantize flow."""
        return DiffusionCompressor.quantize(self)

    def calib(self, nsamples, bs):
        """Override calib to pass extra pipeline kwargs (e.g. height/width) if set.

        Pipelines like GLM-Image require explicit image dimensions; standard diffusion
        pipelines (FLUX etc.) accept but ignore them.
        """
        import inspect
        pipe_sig = inspect.signature(self.pipe.__call__)
        extra = {}
        if "height" in pipe_sig.parameters and self.height is not None:
            extra["height"] = self.height
        if "width" in pipe_sig.parameters and self.width is not None:
            extra["width"] = self.width

        if not extra:
            # No extra kwargs needed — delegate to parent as-is
            return DiffusionCompressor.calib(self, nsamples, bs)

        # Replicate parent calib() with extra kwargs injected into the pipe call
        from auto_round.compressors.diffusion.dataset import get_diffusion_dataloader
        from auto_round.utils import clear_memory
        from tqdm import tqdm

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
        if self.pipe.dtype != self.model.dtype:
            self.pipe.to(self.model.dtype)
        if self.pipe.device != self.model.device:
            self.pipe.to(self.model.device)

        with tqdm(range(1, total + 1), desc="cache block inputs") as pbar:
            for ids, prompts in self.dataloader:
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                try:
                    self.pipe(
                        prompt=prompts,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.num_inference_steps,
                        generator=(
                            None
                            if self.generator_seed is None
                            else torch.Generator(device=self.pipe.device).manual_seed(self.generator_seed)
                        ),
                        **extra,
                    )
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

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_quantized(self, output_dir=None, format="auto_round", inplace=True, **kwargs):
        """Save both quantized AR and DiT models into a pipeline directory structure.

        The output directory mirrors the original pipeline layout::

            output_dir/
              model_index.json
              <ar_component>/   (quantized AR model)
              transformer/      (quantized DiT model)
              ...               (unchanged auxiliary components)

        Args:
            format: Export format for both the AR and DiT components.
        """
        if output_dir is None:
            logger.warning("output_dir is None, skipping save")
            return

        from auto_round.formats import get_formats
        from auto_round.compressors.base import BaseCompressor

        saved_formats = self.formats  # preserve original

        # Save DiT
        if self.quant_dit:
            dit_subdir = "transformer"
            logger.info(f"Saving quantized DiT model ({dit_subdir}) [format={format}]")
            dit_output_dir = os.path.join(output_dir, dit_subdir)
            os.makedirs(dit_output_dir, exist_ok=True)

            self.model = self.dit_model
            if hasattr(self, "dit_layer_config"):
                self.layer_config = self.dit_layer_config

            self.formats = get_formats(format, self)
            BaseCompressor.save_quantized(
                self, output_dir=dit_output_dir, format=format, inplace=inplace, **kwargs
            )

        # Save AR
        if self.quant_ar and self.ar_model is not None:
            ar_subdir = self.ar_component_name  # e.g. "vision_language_encoder"
            logger.info(f"Saving quantized AR model ({ar_subdir}) [format={format}]")
            ar_output_dir = os.path.join(output_dir, ar_subdir)
            os.makedirs(ar_output_dir, exist_ok=True)

            self.model = self.ar_model
            if hasattr(self, "ar_layer_config"):
                self.layer_config = self.ar_layer_config

            # Swap serialization attributes from the AR compressor so that
            # BaseCompressor.save_quantized builds the correct config.
            ar_ser = getattr(self, "_ar_serialization", {})
            saved_attrs = {}
            for k, v in ar_ser.items():
                saved_attrs[k] = getattr(self, k, None)
                setattr(self, k, v)

            self.formats = get_formats(format, self)
            BaseCompressor.save_quantized(
                self, output_dir=ar_output_dir, format=format, inplace=inplace, **kwargs
            )

            # Restore DiT serialization attributes
            for k, v in saved_attrs.items():
                setattr(self, k, v)

        self.formats = saved_formats
        self._save_pipeline_metadata(output_dir)
        self.model = self.dit_model
        logger.info(f"Full hybrid quantized model saved to {output_dir}")

    def _save_pipeline_metadata(self, output_dir):
        """Save model_index.json and auxiliary pipeline components."""
        src_path = (
            getattr(getattr(self.pipe, "config", None), "_name_or_path", None)
            or getattr(self.pipe, "name_or_path", None)
        )
        if src_path and os.path.exists(os.path.join(src_path, "model_index.json")):
            import shutil
            dst_index = os.path.join(output_dir, "model_index.json")
            if not os.path.exists(dst_index):
                shutil.copy2(os.path.join(src_path, "model_index.json"), dst_index)

        # Save non-quantized pipeline components so the exported directory remains
        # loadable as a complete diffusers pipeline even when only one branch is quantized.
        component_names = [
            "scheduler",
            "tokenizer",
            "processor",
            "vae",
            "text_encoder",
        ]
        if not self.quant_ar and self.ar_component_name is not None:
            component_names.append(self.ar_component_name)
        if not self.quant_dit:
            component_names.append("transformer")

        for component_name in component_names:
            component = getattr(self.pipe, component_name, None)
            if component is None:
                continue
            component_dir = os.path.join(output_dir, component_name)
            if os.path.exists(component_dir):
                continue
            try:
                if hasattr(component, "save_pretrained"):
                    component.save_pretrained(component_dir)
            except Exception as e:
                logger.warning(f"Failed to save {component_name}: {e}")

    def quantize_and_save(
        self,
        output_dir: str = "tmp_autoround",
        format: str = "auto_round",
        inplace: bool = True,
        **kwargs,
    ):
        """Quantize both components and save the complete pipeline.

        Args:
            format: Export format for both the AR and DiT components.
        """
        from auto_round.formats import get_formats

        format_list = get_formats(format, self)
        self.formats = format_list
        self.orig_output_dir = output_dir  # required by base.quantize() → _adjust_immediate_packing_and_saving()
        self.inplace = inplace

        self.quantize()
        self.save_quantized(
            output_dir, format=format, inplace=inplace, **kwargs,
        )
        logger.info(f"Hybrid quantized model saved to {output_dir}")
        return self.model, [output_dir]
