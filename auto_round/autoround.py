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

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from auto_round.algorithms.config_resolver import _prepare_entry_kwargs
from auto_round.compressors.builder import _CompressorBuilder
from auto_round.parallel import ParallelConfig
from auto_round.recipes import Recipe, get_recipe, resolve_recipe
from auto_round.recipes.recipe import UNSET
from auto_round.recipes.resolver import merge_layer_defaults
from auto_round.scheme_entry import resolve_entry_scheme
from auto_round.schemes import QuantizationScheme
from auto_round.utils.dataset_utils import CalibDataset
from auto_round.utils.device_manager import normalize_default_device_map

if TYPE_CHECKING:
    from auto_round.algorithms.quantization.config import QuantizationConfig
    from auto_round.algorithms.transforms.base import BaseRotationConfig
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
    from auto_round.compressors.base import BaseCompressor


class AutoRound:
    """Low-bit model quantization with AutoRound.

    See :meth:`__new__` for parameter defaults, supported runtime keyword
    arguments, and the rules for algorithm-specific keyword arguments.

    Examples:
        Quantize a repository model and save it::

            from auto_round import AutoRound

            compressor = AutoRound("Qwen/Qwen3-0.6B", scheme="W4A16")
            compressor.quantize_and_save("./quantized", format="auto_round")

        Use a preloaded model/tokenizer and customize calibration::

            compressor = AutoRound(
                model, tokenizer=tokenizer, scheme="W4A16",
                seqlen=1024, nsamples=64, batch_size=4,
                gradient_accumulate_steps=2,
            )

        Select RTN, a floating-point preset, or a composed algorithm::

            from auto_round import AWQConfig, SignRoundConfig

            compressor = AutoRound("Qwen/Qwen3-0.6B", alg_configs="rtn")
            compressor = AutoRound("Qwen/Qwen3-0.6B", scheme="MXFP4", alg_configs="opt-rtn")
            compressor = AutoRound(
                "Qwen/Qwen3-0.6B", scheme="W4A16",
                alg_configs=[AWQConfig(), SignRoundConfig(iters=100)],
            )

        Choose adaptive mixed-bit candidates or supply an AutoScheme::

            from auto_round import AutoScheme

            compressor = AutoRound("Qwen/Qwen3-0.6B", scheme=("W2A16", "W4A16"), bits=3.0)
            compressor = AutoRound(
                "Qwen/Qwen3-0.6B",
                scheme=AutoScheme(options=("W2A16", "W4A16"), avg_bits=3.0),
            )

        Specify structured calibration data and a per-layer override::

            from auto_round.utils.dataset_utils import CalibDataset

            compressor = AutoRound(
                "Qwen/Qwen3-0.6B",
                dataset=CalibDataset("my-org/my-dataset", split="train", fields="text"),
                layer_config={"lm_head": {"bits": 16}},
            )

        Select ModelScope rather than Hugging Face::

            compressor = AutoRound("Qwen/Qwen3-0.6B", platform="model_scope")

        Enable **experimental** quantization-aware distillation (QAD) or training (QAT)::

            from auto_round.algorithms.quantization.qad.config import QADConfig
            from auto_round.algorithms.quantization.qat.config import QATConfig

            compressor = AutoRound("Qwen/Qwen3-0.6B", alg_configs=QADConfig(qad_iters=100))
            compressor = AutoRound("Qwen/Qwen3-0.6B", alg_configs=QATConfig(qad_iters=100))

        Enable **experimental** data parallelism (DP) in one process::

            from auto_round import ParallelConfig

            compressor = AutoRound(
                "Qwen/Qwen3-0.6B", device_map="0,1",
                parallel_config=ParallelConfig(data_parallel=2),
            )

        Automatically launch **experimental** tensor parallelism (TP) from a standalone script::

            from auto_round import ParallelConfig, quantize_and_save

            # Keep the main guard; no separate torchrun command is needed.
            if __name__ == "__main__":
                folders = quantize_and_save(
                    "Qwen/Qwen3-0.6B", "./quantized_tp", scheme="W4A16",
                    parallel_config=ParallelConfig(tensor_parallel=2),
                    save_kwargs={"format": "auto_round"},
                )
    """

    def __new__(
        cls,
        model: torch.nn.Module | str,
        tokenizer=None,
        scheme: str | dict | QuantizationScheme | AutoScheme | list | tuple = "W4A16",
        alg_configs: (
            str
            | QuantizationConfig
            | BaseRotationConfig
            | list[str | QuantizationConfig | BaseRotationConfig]
            | tuple[str | QuantizationConfig | BaseRotationConfig, ...]
            | None
        ) = None,
        bits: float | None = None,
        shared_layers: Iterable[Iterable[str]] | None = None,
        default_shared_layers: bool | None = None,
        layer_config: dict[str, str | dict | QuantizationScheme] | None = None,
        dataset: str | CalibDataset | list | tuple | torch.utils.data.DataLoader | None = UNSET,
        low_gpu_mem_usage: bool = False,
        device_map: str | torch.device | int | dict = 0,
        parallel_config: ParallelConfig | dict[str, int | str] | None = None,
        enable_torch_compile: bool | None = None,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        platform: str = "hf",
        recipe: str | Recipe = "auto-round",
        **kwargs,
    ) -> BaseCompressor:
        """Configure model quantization; see the class documentation for examples.

        Args:
            model: Repository ID, local model directory, or preloaded
                ``torch.nn.Module``. Required.
            tokenizer: Tokenizer instance, or ``None`` (default) to load from
                the model source. Supply one for a preloaded text model.
            scheme: Preset such as ``"W4A16"`` (default), ``"W8A16"``,
                ``"MXFP4"``, or ``"NVFP4"``; a scheme dict; QuantizationScheme;
                AutoScheme; or a list/tuple of candidate schemes. Candidates
                enable adaptive mixed-bit selection and require ``bits``.
            alg_configs: Algorithm alias, config instance, or list/tuple mixing
                either. Comma-separated aliases are also accepted. ``None``
                (default) selects SignRound, unless ``iters=0`` selects RTN.
                Aliases include ``"signround"``, ``"auto_round_v2"``, ``"rtn"``,
                ``"opt-rtn"``, ``"awq"``, ``"hadamard"``, ``"svdquant"``, and
                experimental ``"qat"``/``"qad"``. Preprocessors run before
                quantizers; quantizers retain their relative order. AWQ alone
                receives an RTN quantizer. Prefer config objects for stage-local
                settings and when composing algorithms.
            bits: Weight-bit override for fixed schemes (an integer, e.g. ``4``),
                or the average-bit target for candidates/AutoScheme (e.g. ``4.2``).
                ``None`` (default) keeps the scheme's value.
            layer_config: Layer-name/pattern mapping to presets, scheme dicts,
                or QuantizationScheme objects. ``None`` (default) uses the global
                scheme. Example: ``{"lm_head": {"bits": 16}}`` skips that layer.
            dataset: Dataset name/local file, CalibDataset, list/tuple of dataset
                specifications or samples, or ``torch.utils.data.DataLoader``.
                Omitted values inherit the recipe; explicit ``None`` lets the
                calibration flow select its dataset.
            low_gpu_mem_usage: ``False`` (default) or ``True`` to reduce GPU
                memory usage, potentially at the cost of tuning/calibration speed.
            device_map: Device index (default ``0``), device string such as
                ``"cuda:0"``/``"cpu"``, ``torch.device``, comma-separated device
                indices such as ``"0,1"``, ``"auto"``, or a module-to-device map.
            parallel_config: ParallelConfig, equivalent dict, or ``None``
                (default, no explicit parallelism). Experimental DP accepts
                ``data_parallel="off"`` (default), ``"auto"``, or a replica count.
                Experimental TP uses ``tensor_parallel=N``; naive sequential PP
                uses ``pipeline_parallel=N``. Degrees default to one; hybrid modes
                are unsupported. DP's ``parallel_batch`` is ``"split"`` (default)
                or ``"per_replica"``. All TP ranks must participate; module-level
                ``auto_round.quantize_and_save`` can launch workers automatically.
            enable_torch_compile: ``True``/``False`` to request/disable compilation.
                ``None`` (default) lets the runtime choose where supported.
            seed: Calibration/tuning random seed. Default: ``42``.
            low_cpu_mem_usage: ``True`` (default) reduces host-memory usage;
                ``False`` keeps more weights/data in host memory for reuse.
            platform: ``"hf"`` (default, Hugging Face) or ``"model_scope"``
                (ModelScope; requires the ``modelscope`` package).
                ``AR_USE_MODELSCOPE=1`` also selects ModelScope in the model loader.
            recipe: Registered recipe name or Recipe object. Default:
                ``"auto-round"``. Explicit ``alg_configs`` replaces its pipeline
                while retaining recipe calibration defaults; explicit calibration
                values and layer overrides take precedence over recipe defaults.
            **kwargs: The runtime keys below, scheme overrides, and fields of the
                selected algorithm configs. Applicability depends on model type,
                algorithm, scheme, and export format; not every combination is
                supported. Unknown/unmatched keys are reported and ignored,
                except strict model-free validation. Removed ``algorithm`` and
                ``schemes`` keys are rejected; use ``alg_configs`` and ``scheme``.

        Keyword Args:
            iters (int | None): Common optimization budget. Omitted/``None``
                retains the algorithm config's value; ``0`` selects RTN for
                SignRound. Use ``qad_iters`` in QADConfig/QATConfig for their own
                stage-local budget. Common values matching multiple stages are
                ignored as ambiguous: configure each stage explicitly instead.
            seqlen (int): Calibration sequence length. Default: ``2048``.
            nsamples (int): Calibration sample count. Default: ``128``.
            batch_size (int): Calibration/tuning batch size. Default: ``8``.
            gradient_accumulate_steps (int | None): Accumulation steps.
                Omitted/``None`` retains the selected algorithm config's value.
            calibration_data_device (str): Calibration-input cache placement:
                ``"auto"`` (default, GPU placement/sharding), ``"cpu"``, ``"off"``
                (disable placement machinery), or explicit devices such as
                ``"cuda:1,cuda:2"``. Does not select compute devices.
            format (str | list | None): Export format(s), e.g. ``"auto_round"``,
                ``"auto_gptq"``, or ``"gguf:q4_k_m"``. Default: ``None``; may be
                supplied later to ``quantize_and_save``.
            model_free (bool): Explicitly request supported model-free RTN
                quantization from a repository ID/local path. Default: ``False``.
                Incompatible with calibration-based preprocessors and parallelism.
            disable_model_free (bool): Prevent automatic model-free routing.
                Default: ``False``; eligible zero-iteration runs may auto-route.
            scale_dtype (str | torch.dtype | None): Scale dtype, e.g. ``"fp16"``,
                ``"bf16"``, or ``"fp32"``. ``None`` (default) chooses by format.
            ignore_layers (str): Layers excluded from quantization. Default: ``""``.
            quant_lm_head (bool): Quantize the output head where supported.
                Default: ``False``.
            to_quant_block_names (str | list[str] | None): Explicit transformer
                block selection. ``None`` (default) uses automatic detection.
            model_dtype (str | torch.dtype | None): Model-loading dtype.
                ``None`` (default) retains the loader's dtype policy.
            trust_remote_code (bool): Allow model-repository custom code.
                Default: ``True``; disable for repositories you do not trust.
            amp (bool): Enable automatic mixed precision. Default: ``True``.
            enable_deterministic_algorithms (bool): Request deterministic PyTorch
                algorithms; unsupported operations may raise. Default: ``False``.
            static_kv_dtype (str | torch.dtype | None): Experimental static KV
                cache dtype (e.g. ``"fp8"`` or supported ``"nvfp4"`` paths).
                ``None`` (default) disables static KV calibration.
            static_attention_dtype (str | torch.dtype | None): Experimental
                static attention dtype, e.g. ``"fp8"`` on supported models.
                ``None`` (default) disables static attention calibration.
            static_kv_granularity (str): Static FP8 KV scale granularity:
                ``"tensor"`` (default) or ``"head"``.
            static_attention_granularity (str): Static FP8 attention scale
                granularity: ``"tensor"`` (default) or ``"head"``.
            processor (object | None): Vision-language processor override.
                ``None`` (default) uses the model's processor.
            image_processor (object | None): Image-processor override.
                ``None`` (default) uses the model's image processor.
            template (str | None): Vision-language calibration template.
                ``None`` (default) selects the model's template.
            extra_data_dir (str | None): Additional image/audio/video data
                directory for multimodal calibration. Default: ``None``.
            quant_nontext_module (bool): Include supported vision/non-text
                modules in quantization. Default: ``False``.
            guidance_scale (float): Diffusion classifier-free guidance scale.
                Default: ``7.5``.
            num_inference_steps (int): Diffusion generation/evaluation steps.
                Default: ``50``; must be positive.
            calib_num_inference_steps (int): Diffusion calibration steps.
                Default: ``8``; must be positive.
            generator_seed (int | None): Diffusion generator seed override.
                Default: ``None``.
            pipeline_call_kwargs (dict | None): Additional diffusion pipeline
                arguments (e.g. image size). Default: ``None`` (empty mapping).
            diffusion_tuning_cache_size (float | str): Diffusion tuning cache
                budget in GiB, or ``"auto"``. Default: ``0``; numeric values must
                be finite and non-negative.
            shared_layers (Iterable[Iterable[str]] | None): Extra layer groups that
                must share one quantization scheme, for fused serving kernels.
                AutoScheme only; combined with the default vLLM fusion groups.
                Default: ``None``.
            default_shared_layers (bool | None): Whether to add the built-in vLLM
                fusion groups (``q/k/v``, MLA ``q_a``/``kv_a``, ``gate/up``) to
                ``shared_layers``. ``False`` lets AutoScheme pick per layer.
                AutoScheme only; ``None`` (default) keeps the AutoScheme default.
            ignore_scale_zp_bits (bool): Exclude scale/zero-point overhead from
                adaptive average-bit accounting. AutoScheme only; default: ``False``.
            rotation_config (object | dict | str | None): Rotation configuration
                or supported shorthand (e.g. ``"hadamard"``/``"quarot"``).
                Default: ``None``. Put rotation backend options inside this config.
            device (object | None): Compatibility alias for ``device_map`` when
                it is unset/default. Prefer ``device_map``; default: ``None``.
            options (str | list | tuple | None): Compatibility alias for adaptive
                candidates. Prefer a list/tuple in ``scheme``; default: ``None``.
            avg_bits (float | None): Compatibility alias for adaptive target bits.
                Prefer ``bits``; conflicting values are rejected. Default: ``None``.

        Algorithm Keyword Arguments:
            Scheme overrides are ``group_size``, ``sym``, ``data_type``,
            ``act_bits``, ``act_group_size``, ``act_sym``, ``act_data_type``,
            ``act_dynamic``, ``super_bits``, and ``super_group_size`` (plus the
            explicit ``bits`` argument and ``rotation_config`` above). Unset
            overrides retain the chosen scheme; direct scheme overrides apply
            to the shared run scheme. Algorithm config constructors do not accept
            these fields or ``scheme``; pass them directly to AutoRound instead.

            Other accepted keys are the declared constructor fields of the
            selected config classes (including inherited algorithm fields), or
            model fields for Pydantic configs. See SignRoundConfig/SignRoundV2Config/
            AdamRoundConfig (rounding), RTNConfig (round-to-nearest), AWQConfig,
            SVDQuantConfig, RotationConfig/SpinQuantConfig (transforms), and
            QADConfig/QATConfig (experimental whole-model tuning). Examples include
            ``lr``, ``minmax_lr``, ``nblocks``, ``lora_rank``, ``disable_opt_rtn``,
            ``qad_iters``, and ``qad_lr``. Registered extensions expose their own
            fields by the same rule; arbitrary undeclared config kwargs are not
            automatically accepted by this entry point.

            A direct non-None algorithm value overrides a config field only when
            it matches one selected config. Unmatched/ambiguous keys are ignored
            with a diagnostic (unsupported keys raise in strict model-free mode).
            Use config objects to give multiple stages different values. Flags
            such as ``enable_opt_rtn``/``disable_opt_rtn`` have dedicated RTN/AWQ
            handling. Names such as ``"rtn"``/``"opt-rtn"`` take precedence over
            contradicting RTN-variant flags. Parallel settings belong inside
            ``parallel_config``, never in top-level kwargs.

        Returns:
            BaseCompressor: Compressor exposing ``quantize_and_save`` and other
            model-specific quantization/export operations.
        """
        from auto_round.parallel.config import _reject_parallel_kwargs

        _reject_parallel_kwargs(kwargs)
        parallel_config = ParallelConfig.from_value(parallel_config)
        direct_kwargs = dict(kwargs)
        seqlen = direct_kwargs.pop("seqlen", UNSET)
        nsamples = direct_kwargs.pop("nsamples", UNSET)
        batch_size = direct_kwargs.pop("batch_size", UNSET)
        calibration_data_device = direct_kwargs.pop("calibration_data_device", "auto")
        strict_model_free = bool(direct_kwargs.get("model_free", False))
        legacy_device = direct_kwargs.pop("device", None)
        if legacy_device is not None and device_map in (None, 0):
            device_map = legacy_device
        if bits is not None:
            direct_kwargs["bits"] = bits
        # AutoScheme-only options: they are consumed by resolve_entry_scheme when
        # `scheme` holds candidates and must not reach the algorithm config otherwise.
        if shared_layers is not None:
            direct_kwargs["shared_layers"] = shared_layers
        if default_shared_layers is not None:
            direct_kwargs["default_shared_layers"] = default_shared_layers
        scheme, direct_kwargs = resolve_entry_scheme(scheme, direct_kwargs)

        definition = get_recipe(recipe) if isinstance(recipe, str) else recipe
        plan = resolve_recipe(
            definition,
            scheme=scheme,
            model=model,
            trust_remote_code=direct_kwargs.get("trust_remote_code", True),
        )
        if alg_configs is None:
            alg_configs = plan.build_configs()
        calibration = plan.calibration
        dataset = calibration.get("dataset") if dataset is UNSET else dataset
        seqlen = calibration["seqlen"] if seqlen is UNSET else seqlen
        nsamples = calibration["nsamples"] if nsamples is UNSET else nsamples
        batch_size = calibration["batch_size"] if batch_size is UNSET else batch_size
        if plan.layer_config:
            layer_config = merge_layer_defaults(plan.layer_config, layer_config)

        configs, runtime_kwargs = _prepare_entry_kwargs(alg_configs, direct_kwargs, strict=strict_model_free)
        runtime_kwargs["batch_size"] = batch_size

        compressor = _CompressorBuilder(
            model,
            scheme,
            configs,
            tokenizer=tokenizer,
            platform=platform,
            format=runtime_kwargs.pop("format", None),
            dataset=dataset,
            low_gpu_mem_usage=low_gpu_mem_usage,
            calibration_data_device=calibration_data_device,
            device_map=normalize_default_device_map(device_map),
            parallel_config=parallel_config,
            iters=None,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            low_cpu_mem_usage=low_cpu_mem_usage,
            layer_config=layer_config,
            nsamples=nsamples,
            seqlen=seqlen,
            **runtime_kwargs,
        )
        compressor.resolved_recipe = plan
        return compressor


# Keep legacy entry points available for downstream integrations such as
# Neural Compressor. They delegate to the unified entry point.
class AutoRoundLLM:
    def __new__(cls, *args, **kwargs):
        return AutoRound(*args, **kwargs)


class AutoRoundAdam:
    def __new__(cls, *args, **kwargs):
        kwargs.setdefault("enable_adam", True)
        return AutoRound(*args, **kwargs)


class AutoRoundMLLM:
    def __new__(cls, *args, **kwargs):
        return AutoRound(*args, **kwargs)


class AutoRoundDiffusion:
    def __new__(cls, *args, **kwargs):
        return AutoRound(*args, **kwargs)
