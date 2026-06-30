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

from __future__ import annotations

import logging
import copy
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from auto_round import AutoRound
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.compressors.utils import is_mx_fp
from auto_round.export.export_to_llmcompressor.remap_fused_names import (
    remap_fused_quantized_names,
    validate_moe_expert_scale_coverage,
)

from auto_round.vllm_adapters import ModuleAdapterRegistry
from auto_round.vllm_backend import VLLMModelBackend

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def _register_vllm_supported_layer_types() -> None:
    """Register vLLM parallel linear layers into AutoRound supported types."""
    try:
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            MergedColumnParallelLinear,
            QKVParallelLinear,
            RowParallelLinear,
        )

        import auto_round.utils as utils_module
        import auto_round.utils.common as common_module

        extra_types = (
            QKVParallelLinear,
            RowParallelLinear,
            ColumnParallelLinear,
            MergedColumnParallelLinear,
        )

        # Patch both export locations since many modules do "from ... import SUPPORTED_LAYER_TYPES".
        for module in (utils_module, common_module):
            current = tuple(getattr(module, "SUPPORTED_LAYER_TYPES", ()))
            setattr(module, "SUPPORTED_LAYER_TYPES", tuple(dict.fromkeys(current + extra_types)))
    except Exception as exc:
        logger.warning("Failed to register vLLM layer types for quantization: %s", exc)


class VLLMCalibrationRunner:
    """Runs inference-side prefill passes for optimized RTN workflows.

    Collects streaming statistics (imatrix, activation bounds) for RTN scale search.
    """

    def __init__(
        self, backend: VLLMModelBackend, adapter_registry: Optional[ModuleAdapterRegistry] = None
    ) -> None:
        self.backend = backend
        self.adapter_registry = adapter_registry or ModuleAdapterRegistry()
        self.statistics: dict[str, Any] = {}

    def run(self, prompts: list[str] | str, collect_stats: bool = False) -> Any:
        """Run prefill and optionally collect activation statistics."""
        # TODO: When block hooks are active, imatrix aggregation can happen here
        # For now, just run prefill to cache inputs in the block-forward hooks
        return self.backend.run_prefill(prompts)

    def get_statistics(self) -> dict[str, Any]:
        """Return collected statistics (imatrix, etc.)."""
        return self.statistics


class VLLMCompressor:
    """Single-GPU AutoRound compressor for vLLM 0.23.x runtime models.

    Quantizes a running vLLM engine and exports standard LLM-Compressor/compressed-tensors checkpoints.
    """

    ALGORITHM_MAP: dict[str, str] = {
        "rtn": "rtn",
        "optimized_rtn": "optimized_rtn",
    }

    def __init__(
        self,
        model_id_or_path: str,
        algorithm: str = "rtn",
        tensor_parallel_size: int = 1,
        adapter_registry: Optional[ModuleAdapterRegistry] = None,
        disable_immediate_saving: bool = True,
        **autoround_kwargs: Any,
    ) -> None:
        """Initialize VLLMCompressor.

        Args:
            model_id_or_path: Model name (HuggingFace ID) or local path.
            algorithm: Either 'rtn' (disable imatrix optimization) or 'optimized_rtn' (enable).
            tensor_parallel_size: Must be 1 (single-GPU only).
            adapter_registry: Custom module adapter registry (defaults to standard).
            disable_immediate_saving: Disable immediate_saving mode to avoid deepcopy issues with vLLM Parameters.
                Defaults to True (immediate_saving disabled).
            **autoround_kwargs: Additional kwargs to pass to AutoRound (e.g., format, nblocks, iters).
        """
        if algorithm not in self.ALGORITHM_MAP:
            raise ValueError(
                f"Unknown algorithm: {algorithm!r}. Must be one of {list(self.ALGORITHM_MAP.keys())}"
            )
        if tensor_parallel_size != 1:
            raise ValueError(f"tensor_parallel_size must be 1, got {tensor_parallel_size}")

        self.model_id_or_path = model_id_or_path
        self.algorithm = algorithm
        self.adapter_registry = adapter_registry or ModuleAdapterRegistry()
        self.disable_immediate_saving = disable_immediate_saving
        self.scheme = autoround_kwargs.pop("scheme", None)
        self.bits = autoround_kwargs.pop("bits", None)
        self.group_size = autoround_kwargs.pop("group_size", None)
        self.nblocks = autoround_kwargs.pop("nblocks", None)
        self.iters = autoround_kwargs.pop("iters", None)
        self.vllm_args = autoround_kwargs.pop("vllm_args", None)
        self.enable_post_export_remap = autoround_kwargs.pop("enable_post_export_remap", True)
        self.autoround_kwargs = autoround_kwargs
        self.backend: Optional[VLLMModelBackend] = None
        self.calibration_runner: Optional[VLLMCalibrationRunner] = None

    def _create_alg_config(self) -> RTNConfig:
        """Route algorithm to RTN config with appropriate disable_opt_rtn flag.
        
        For opt-rtn, enable imatrix to use CalibratedRTNCompressor for imatrix-based optimization.
        For rtn, use ZeroShotCompressor (no optimization).
        """
        from auto_round.schemes import parse_scheme
        from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig
        
        # Parse scheme to extract quantization parameters
        scheme_name = self.scheme if isinstance(self.scheme, str) else "W4A16"
        try:
            _, _, config_dict = parse_scheme(scheme_name, {})
        except Exception as e:
            logger.warning(f"Failed to parse scheme {scheme_name}: {e}, using defaults")
            config_dict = {}
        
        # Create config kwargs with scheme parameters
        config_kwargs = dict(config_dict) if config_dict else {}
        
        # Override with explicitly set parameters
        if self.bits is not None:
            config_kwargs["bits"] = self.bits
        if self.group_size is not None:
            config_kwargs["group_size"] = self.group_size
        
        # Set routing flag based on algorithm.
        if self.algorithm == "rtn":
            config_kwargs["disable_opt_rtn"] = True
            config = RTNConfig(**config_kwargs)
        else:  # optimized_rtn
            config_kwargs["disable_opt_rtn"] = False
            config = OptimizedRTNConfig(**config_kwargs)
            config.enable_imatrix = True
            logger.info("opt-rtn: enable_imatrix=True for imatrix-informed quantization")
        
        return config
    
    def _debug_iters(self) -> None:
        """Debug: print iters parameter for verification."""
        msg = f"===== iters={self.iters}, algorithm={self.algorithm}, nblocks={self.nblocks} ====="
        logger.info(msg)
        # Also write to stderr to ensure it's visible
        import sys
        print(msg, file=sys.stderr, flush=True)

    def _warmup_prefill_if_needed(self) -> None:
        """Run vLLM prefill warmup for optimized_rtn if prompts provided."""
        prompts = self.autoround_kwargs.pop("prefill_prompts", None)
        if self.algorithm != "optimized_rtn" or prompts is None:
            return
        if self.calibration_runner is None:
            raise RuntimeError("Calibration runner is not initialized.")
        logger.info("Running vLLM prefill for optimized_rtn calibration flow.")
        self.calibration_runner.run(prompts, collect_stats=True)

    def _build_autoround(self, model: Any, output_format: Optional[str] = None) -> tuple[AutoRound, str]:
        """Build the regular AutoRound compressor for the loaded vLLM model."""
        tokenizer = self.backend.get_tokenizer() if self.backend is not None else None
        if tokenizer is None:
            tokenizer = self.model_id_or_path

        # Mark this model as coming from vLLM compressor path so export-time
        # compatibility logic can be scoped to --use_vllm_compressor only.
        setattr(model, "_use_vllm_compressor", True)

        if not hasattr(model, "_name_or_path"):
            setattr(model, "_name_or_path", self.model_id_or_path)
        if not hasattr(model, "name_or_path"):
            setattr(model, "name_or_path", self.model_id_or_path)
        if not hasattr(model, "dtype"):
            import torch

            setattr(model, "dtype", torch.bfloat16)
        vllm_config = None
        if self.backend is not None and getattr(self.backend, "llm", None) is not None:
            llm_engine = getattr(self.backend.llm, "llm_engine", None)
            vllm_config = getattr(llm_engine, "vllm_config", None)
        if vllm_config is not None:
            for module in model.modules():
                setattr(module, "_vllm_config", vllm_config)
        
        # Ensure model has device attribute for quantization code
        if not hasattr(model, "device"):
            import torch
            setattr(model, "device", torch.device("cuda"))

        _register_vllm_supported_layer_types()
        
        self._patch_vllm_parameter_deepcopy()

        kwargs = dict(self.autoround_kwargs)
        if output_format is not None:
            kwargs["format"] = output_format
        kwargs.setdefault("format", "llm_compressor")
        
        # Register vLLM parallel linear types in supported_types for AutoRound
        try:
            from vllm.model_executor.layers.linear import (
                ColumnParallelLinear,
                MergedColumnParallelLinear,
                QKVParallelLinear,
                RowParallelLinear,
            )
            from auto_round.utils import SUPPORTED_LAYER_TYPES as base_types
            
            extra_types = (
                QKVParallelLinear,
                RowParallelLinear,
                ColumnParallelLinear,
                MergedColumnParallelLinear,
            )
            # Explicitly pass extended supported_types to AutoRound
            kwargs["supported_types"] = tuple(dict.fromkeys(base_types + extra_types))
            logger.info(f"Registered vLLM parallel linear types: {[t.__name__ for t in extra_types]}")
        except Exception as exc:
            logger.warning(f"Failed to register vLLM layer types in AutoRound: {exc}")
        
        # For new-arch entry point with alg_configs
        kwargs["alg_configs"] = self._create_alg_config()
        
        # Pass through quantization parameters from CLI
        if self.nblocks is not None:
            kwargs["nblocks"] = self.nblocks
        if self.iters is not None:
            kwargs["iters"] = self.iters

        if self.scheme is not None:
            kwargs["scheme"] = self.scheme
        if self.bits is not None:
            kwargs["bits"] = self.bits
        if self.group_size is not None:
            kwargs["group_size"] = self.group_size

        output_format = kwargs.pop("format", "llm_compressor")
        
        # Disable immediate_saving for vLLM compatibility
        if self.disable_immediate_saving:
            kwargs["low_cpu_mem_usage"] = False
            logger.info(f"Building AutoRound with low_cpu_mem_usage=False to disable immediate_saving")
        
        logger.info(f"Building AutoRound instance with algorithm={self.algorithm}")
        return AutoRound(model=model, tokenizer=tokenizer, **kwargs), output_format

    def _patch_vllm_parameter_deepcopy(self) -> None:
        """Make vLLM's custom Parameter types compatible with torch deepcopy.

        AutoRound clones modules during quantization. vLLM's custom Parameter types
        inherit from torch.nn.Parameter but have non-standard __deepcopy__ behavior.
        We patch a wrapper that delegates to torch's native deepcopy for the tensor
        part, then copies any custom attributes.
        """
        try:
            from vllm.model_executor.parameter import BasevLLMParameter
        except Exception:
            return

        # Store the original deepcopy if it exists
        original_deepcopy = getattr(BasevLLMParameter, '__deepcopy__', None)

        def _base_vllm_parameter_deepcopy(self, memo):
            import torch
            
            # Use torch.nn.Parameter's standard deepcopy for the tensor data
            # This handles the storage and all torch-specific bookkeeping correctly
            new_tensor = self.data.clone().detach()
            result = torch.nn.Parameter(new_tensor)
            
            # Copy over grad if it exists
            if self.grad is not None:
                result.grad = self.grad.clone().detach()
            
            # Copy over any custom attributes (excluding torch internals)
            skip_attrs = {'data', 'grad', 'grad_fn', '_is_view', '_version', '_base',
                         '_grad_indices', '_requires_grad', 'requires_grad'}
            for key, val in self.__dict__.items():
                if key not in skip_attrs:
                    try:
                        setattr(result, key, copy.deepcopy(val, memo))
                    except Exception:
                        pass  # Skip attributes that can't be deepcopied
            
            memo[id(self)] = result
            return result

        BasevLLMParameter.__deepcopy__ = _base_vllm_parameter_deepcopy  # type: ignore[attr-defined]

    def quantize(self, output_format: Optional[str] = None) -> dict[str, Any]:
        """Quantize the loaded vLLM model.

        Args:
            output_format: Export format for quantized model (e.g., 'llm_compressor').
                If None, uses value from autoround_kwargs or defaults to 'llm_compressor'.

        Returns:
            Quantization result dict from AutoRound.
        """
        self.backend = VLLMModelBackend.load(self.model_id_or_path, vllm_args=self.vllm_args)
        self.calibration_runner = VLLMCalibrationRunner(self.backend, self.adapter_registry)

        self._warmup_prefill_if_needed()

        model = self.backend.get_model()
        autoround, _ = self._build_autoround(model, output_format)

        logger.info(f"Starting quantization with algorithm={self.algorithm}")
        result = autoround.quantize()
        logger.info("Quantization completed.")
        return result

    def save_quantized(self, output_dir: str | Path, quantized_model: Optional[dict[str, Any]] = None) -> None:
        """Save the quantized model to disk.

        Args:
            output_dir: Directory to save the quantized model.
            quantized_model: Pre-computed quantization result from quantize().
                If None, quantize() is called first.
        """
        if self.backend is None:
            self.backend = VLLMModelBackend.load(self.model_id_or_path, vllm_args=self.vllm_args)
        if self.calibration_runner is None:
            self.calibration_runner = VLLMCalibrationRunner(self.backend, self.adapter_registry)

        self._warmup_prefill_if_needed()

        model = self.backend.get_model()
        autoround, output_format = self._build_autoround(model)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        autoround.quantize_and_save(output_dir=str(output_dir), format=output_format)

        # For vLLM compressor MXFP exports, restore fused names/shapes back to
        # HF-style split projections for reliable vLLM loadback.
        if self.enable_post_export_remap and self._needs_post_export_remap():
            remap_fused_quantized_names(
                quant_model_dir=output_dir,
                orig_model_dir=self.model_id_or_path,
                logger=logger,
            )
            validate_moe_expert_scale_coverage(
                quant_model_dir=output_dir,
                logger=logger,
            )

        logger.info(f"Quantized model saved to {output_dir}")

    def _needs_post_export_remap(self) -> bool:
        """Whether post-export fused-name remap should run for current scheme."""
        if self.scheme is None:
            return False

        scheme_name = str(self.scheme).upper()
        if "MXFP" in scheme_name:
            return True
        if "W" in scheme_name and "A16" in scheme_name:
            return True

        # Fall back to parsed data_type if available.
        try:
            from auto_round.schemes import parse_scheme

            _, _, config_dict = parse_scheme(str(self.scheme), {})
            data_type = config_dict.get("data_type") if isinstance(config_dict, dict) else None
            if data_type and is_mx_fp(str(data_type)):
                return True
            if isinstance(config_dict, dict):
                bits = config_dict.get("bits")
                act_bits = config_dict.get("act_bits")
                if bits is not None and act_bits is not None and int(bits) <= 8 and int(act_bits) >= 16:
                    return True
            return False
        except Exception:
            return False

    def quantize_and_save(self, output_dir: str | Path) -> Path:
        """Quantize and save atomically to a temporary directory, then move to output_dir.

        Args:
            output_dir: Final destination directory.

        Returns:
            Path to the saved model.
        """
        output_dir = Path(output_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Quantizing to temporary directory: {tmpdir}")
            self.save_quantized(tmpdir)

            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(tmpdir, str(output_dir))

        logger.info(f"Quantized model atomically saved to {output_dir}")
        return output_dir

    @contextmanager
    def managed_backend(self):
        """Context manager to automatically close the vLLM backend."""
        try:
            yield self
        finally:
            if self.backend is not None:
                self.backend.close()
                self.backend = None
