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
import importlib
import traceback
from collections import defaultdict
from typing import Union

import torch

from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.compressors_new.utils import (
    block_forward,
    check_need_act_calibration,
)
from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size
from auto_round.logger import logger
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    clear_memory,
    compile_func,
)


class BaseQuantizers:
    # Class-level attribute declarations for convenient access in quantization methods.
    # Scheme-related attrs (layer_config, scale_dtype, has_qlayer_outside_block, etc.)
    # are resolved by SchemeMixin in BaseCompressor and synced here after post_init().
    model_context = None
    compress_context = None
    dataset = None
    supported_types = SUPPORTED_LAYER_TYPES
    inner_supported_types = INNER_SUPPORTED_LAYER_TYPES
    enable_alg_ext = False
    # Subclasses that support diffusion models should override this with the
    # appropriate output key mapping, e.g.:
    #   DIFFUSION_OUTPUT_CONFIGS = {"FluxTransformerBlock": ["encoder_hidden_states", "hidden_states"]}
    DIFFUSION_OUTPUT_CONFIGS: dict = {}

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.layer_config = None
        self.bits = config.bits
        self.group_size = config.group_size
        self.sym = config.sym
        self.data_type = config.data_type
        self.act_bits = config.act_bits
        self.act_group_size = config.act_group_size
        self.act_sym = config.act_sym
        self.act_data_type = config.act_data_type
        self.act_dynamic = config.act_dynamic
        self.super_bits = config.super_bits
        self.super_group_size = config.super_group_size
        self.scale_dtype = config.scale_dtype
        self.ignore_layers = config.ignore_layers
        self.quant_lm_head = config.quant_lm_head
        self.to_quant_block_names = config.to_quant_block_names
        # Calibration / sampling attrs – synced from compressor in post_init.
        self.seqlen = 2048
        self.nsamples = 128
        self.batch_size = getattr(config, "batch_size", 8)
        self.batch_dim = getattr(config, "batch_dim", None)
        self.infer_bs_coeff = getattr(config, "infer_bs_coeff", 1)
        # Whether to feed quantized-block outputs as inputs to the next block.
        # Subclasses that support cascaded quantized-input (e.g. SignRoundQuantizer)
        # override this from their config.  Defaults to False for zero-shot algorithms
        # (RTN) where activations are not used during weight optimization.
        self.enable_quanted_input = getattr(config, "enable_quanted_input", False)

    @classmethod
    def from_config(cls, config: QuantizationConfig):
        if cls.__name__ == config._alg_cls:
            return cls(config)
        else:
            module = importlib.import_module("auto_round.algorithms.quantization")
            alg_cls = getattr(module, config._alg_cls)
            return alg_cls(config)

    @property
    def formats(self):
        return getattr(self.compress_context, "formats", None)

    @property
    def amp(self):
        return getattr(self.model_context, "amp", False)

    @property
    def amp_dtype(self):
        import torch

        return getattr(self.model_context, "amp_dtype", torch.float32)

    def _register_act_max_hook(self, model):

        def get_act_max_hook(module, input, output):
            if isinstance(input, (tuple, list)):
                input = input[0]
            if input.numel() == 0:
                return  # as no needs for act_max update
            input, _, _ = reshape_pad_tensor_by_group_size(input, self.act_group_size)
            act_max = torch.max(torch.abs(input), dim=-1).values
            if not hasattr(module, "act_max") or module.act_max.numel() == 0:
                module.act_max = act_max
                if self.config.is_act_nv_fp:  ## for nvfp per-tensor input_global_scale calculation usage
                    max_val = act_max.max()
                    module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
            else:
                act_max = act_max.to(module.act_max.device)
                if self.config.is_act_nv_fp:  ## for nvfp per-tensor input_global_scale calculation usage
                    max_val = torch.max(act_max.max(), module.act_max.max())
                    module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
                else:
                    module.act_max = torch.max(act_max, module.act_max)

        hook_handles = []
        # for single layers out of blocks, like lm_head
        if isinstance(model, SUPPORTED_LAYER_TYPES):
            m = model
            if (
                hasattr(m, "act_dynamic")
                and check_need_act_calibration(m.act_dynamic, m.act_data_type, m.act_bits)
                and check_to_quantized(m)
            ):
                hook = m.register_forward_hook(get_act_max_hook)
                hook_handles.append(hook)
            return hook_handles

        for n, m in model.named_modules():
            if (
                hasattr(m, "act_dynamic")
                and check_need_act_calibration(m.act_dynamic, m.act_data_type, m.act_bits)
                and check_to_quantized(m)
            ):
                hook = m.register_forward_hook(get_act_max_hook)
                hook_handles.append(hook)
                continue

            # for whole model, RTN
            if n in self.layer_config:
                config = self.layer_config[n]
                act_dynamic = config.get("act_dynamic", True)
                act_data_type = config.get("act_data_type", None)
                act_bits = config.get("act_bits", 16)
                if (
                    config["bits"] <= 8
                    and check_need_act_calibration(act_dynamic, act_data_type, act_bits)
                    and check_to_quantized(config)
                ):
                    hook = m.register_forward_hook(get_act_max_hook)
                    hook_handles.append(hook)
                    continue
        return hook_handles

    @torch.inference_mode()
    def _quantize_embedding_layer(self):
        """Quantizes embedding layers in the model according to the configuration.

        This method iterates through all modules in the model, identifies embedding
        layers specified in `self.quantizer.layer_config`, and applies the appropriate quantization
        function based on bit precision, grouping strategy, and dtype.

        Returns:
            bool: True if the quantization process completes without critical errors.
        """
        is_quantized = False
        for name, module in self.model_context.model.named_modules():
            # Skip non-Embedding modules or layers not in config
            if not isinstance(module, torch.nn.Embedding) or name not in self.layer_config:
                continue

            config = self.layer_config[name]

            # Skip layers that are not marked for quantization
            if not check_to_quantized(config):
                continue
            is_quantized = True
            config["scale_dtype"] = self.scale_dtype
            dtype = config["data_type"]

            # Determine quantization function key with symmetry/asymmetry
            if dtype not in QUANT_FUNC_WITH_DTYPE:
                dtype = f"{dtype}_{'sym' if config['sym'] else 'asym'}"

            quant_func = QUANT_FUNC_WITH_DTYPE[dtype]
            dtype = module.weight.dtype
            # As typically float32 are used in RTN to search scale zp,
            # to avoid cache a bf16 copy we'd better use float32
            if config.get("super_group_size", None) is not None:
                dtype = torch.float32

            # Attempt quantization on GPU, fall back to CPU if OOM
            try:
                weight, scale, zp = quant_func(
                    module.weight.to(dtype=dtype, device=self.compress_context.device),
                    **{
                        k: config.get(k, None)
                        for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]
                    },
                )
            except torch.OutOfMemoryError:
                cuda_error_msg = traceback.format_exc()
                try:
                    logger.error(cuda_error_msg)
                    logger.warning("falling back to CPU")
                    weight, scale, zp = quant_func(
                        module.weight.to("cpu"),
                        **{
                            k: config.get(k, None)
                            for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]
                        },
                    )
                except Exception as e:
                    raise

            # Overwrite the module's weights with the quantized version
            module.weight.data.copy_(weight.cpu())

            # Attach scale and zero point (zp) to the module
            for param_name, value in zip(["scale", "zp"], [scale, zp]):
                if isinstance(value, dict):
                    for k, v in value.items():
                        setattr(module, k if k == "scale" else f"w_{k}", v.cpu())
                elif isinstance(value, torch.Tensor):
                    setattr(module, param_name, value.cpu())
                else:
                    setattr(module, param_name, value)

            # Update config
            self.layer_config.setdefault(name, {}).update(config)
            del weight
            del scale
            del zp
            clear_memory(device_list=self.compress_context.device_list)

        return is_quantized

    def quantize_block(
        self, block: torch.nn.Module, input_ids=None, input_others=None, reference_output=None, **kwargs
    ) -> dict:
        """Apply the quantization algorithm to a prepared block.

        This is the **pure-algorithm** entry point called by the Compressor after
        all infrastructure work (device placement, data collection, act-max hook
        registration, DDP setup) has been completed.

        Implementations should:
        - Perform the algorithm-specific weight/activation quantization on ``block``.
        - Return a dict of best parameters (may be empty for zero-shot algorithms).

        Args:
            block: Module already placed on the correct device(s).
            input_ids: Calibration inputs on cache_device (None for zero-shot RTN).
            input_others: Additional inputs (None for zero-shot RTN).
            reference_output: FP reference outputs collected by Compressor
                (None for algorithms that don't need a reconstruction loss).
            **kwargs: Algorithm-specific keyword arguments (e.g. ``loss_device``,
                ``card_0_in_high_risk`` for SignRoundQuantizer).

        Returns:
            dict: Best quantization parameters found, or ``{}`` if not applicable.
        """
        raise NotImplementedError("quantize_block must be implemented in subclasses of BaseQuantizers")

    def quantize_layer(self, layer_name: str, **kwargs):
        """Quantizes a single layer of the model.

        Args:
            layer_name (str): The name of the layer to quantize. The layer module is
                retrieved internally via get_module(model, layer_name).
        """
        raise NotImplementedError("quantize_layer must be implemented in subclasses of BaseQuantizers")

    def quantize_layer_outside_block(self, layer_name: str, **kwargs):
        """Quantizes a single layer of the model outside of a block.

        Args:
            layer_name (str): The name of the layer to quantize. The layer module is
                retrieved internally via get_module(model, layer_name).
        """
        raise NotImplementedError("quantize_layer_outside_block must be implemented in subclasses of BaseQuantizers")

    @torch.no_grad()
    def _get_block_outputs(
        self,
        block: torch.nn.Module,
        input_ids,
        input_others,
        bs: int,
        save_output: bool = True,
    ):
        """Compute the output of a block for calibration inputs.

        Shared by SignRoundQuantizer and OptimizedRTNQuantizer.  Algorithm-specific
        block-forward selection (compile vs. plain) is handled here based on
        ``enable_alg_ext`` and act-quantization flags.
        """
        diffusion_fn = getattr(self, "_get_diffusion_block_outputs", None)
        if getattr(self.model_context, "is_diffusion", False):
            return self._get_diffusion_block_outputs(
                block,
                input_ids,
                input_others,
                bs,
                self.compress_context.device,
                self.compress_context.cache_device,
            )

        if (
            (self.config.is_act_quantize and (not self.config.act_dynamic or self.config.is_act_nv_fp))  # have hooks
            or self.enable_alg_ext  # Use imatrix
            # or not self.disable_opt_rtn  # Use imatrix
        ):
            _bf = block_forward
        else:
            _bf = self._resolve_block_forward()

        output = []
        nsamples = len(input_ids)
        for i in range(0, nsamples, bs):
            end_index = min(nsamples, i + bs)
            indices = torch.arange(i, end_index).to(torch.long)
            tmp_input_ids, tmp_input_others = self._sampling_inputs(
                input_ids,
                input_others,
                indices,
                self.seqlen,
                self.batch_dim,
                share_cache_keys=self.model_context.shared_cache_keys,
            )
            tmp_output = _bf(
                block,
                tmp_input_ids,
                tmp_input_others,
                self.model_context.amp,
                self.model_context.amp_dtype,
                self.compress_context.device,
            ).to(self.compress_context.cache_device)
            if save_output:
                if self.batch_size == 1:
                    output.append(tmp_output)
                else:
                    output.extend(list(torch.split(tmp_output, 1, dim=self.batch_dim)))
        self.compress_context.clear_memory()

        return output

    def _resolve_block_forward(self):
        """Resolve and cache the block forward function once.

        This avoids repeated attribute checks in the hot training loop
        (called thousands of times per block).
        """
        cached = self.__dict__.get("_resolved_block_forward")
        if cached is not None:
            return cached
        if self.compress_context.enable_torch_compile:
            compiled = self.__dict__.get("_compiled_block_forward")
            if compiled is None:
                compiled = compile_func(block_forward, self.compress_context.device)
                self._compiled_block_forward = compiled
            self._resolved_block_forward = compiled
        else:
            self._resolved_block_forward = block_forward
        return self._resolved_block_forward

    def _invalidate_block_forward_cache(self):
        """Clear the cached block forward function (call when block changes)."""
        self.__dict__.pop("_resolved_block_forward", None)

    def _get_current_q_output(
        self,
        block: torch.nn.Module,
        input_ids,
        input_others: dict,
        indices,
        device,
        cache_device: str = "cpu",
    ) -> torch.Tensor:
        """Compute block output for a mini-batch selected by *indices* (used during training).

        Handles both LLM and diffusion model block formats.  Uses the compiled
        block_forward when enable_torch_compile is True (same as _get_block_outputs),
        matching old-arch behaviour where self.block_forward was compiled at init.
        """
        current_input_ids, current_input_others = self._sampling_inputs(
            input_ids,
            input_others,
            indices,
            seqlen=self.seqlen,
            batch_dim=self.batch_dim,
            share_cache_keys=self.model_context.shared_cache_keys,
        )
        _bf = self._resolve_block_forward()

        if getattr(self.model_context, "is_diffusion", False):
            output_config = self.DIFFUSION_OUTPUT_CONFIGS.get(block.__class__.__name__, [])
            idx = None if "hidden_states" not in output_config else output_config.index("hidden_states")
            if isinstance(current_input_ids, dict):
                hidden_states = current_input_ids.pop("hidden_states")
                current_input_others.update(current_input_ids)
                current_input_ids = hidden_states
            output_q = _bf(
                block,
                current_input_ids,
                current_input_others,
                self.model_context.amp,
                self.model_context.amp_dtype,
                device,
                idx,
            )
        else:
            output_q = _bf(
                block,
                current_input_ids,
                current_input_others,
                self.model_context.amp,
                self.model_context.amp_dtype,
                device,
            )
        return output_q.to(cache_device)

    @classmethod
    @torch.no_grad()
    def _sampling_inputs(
        cls,
        input_ids: Union[list[torch.Tensor], dict],
        input_others: dict,
        indices,
        seqlen: int,
        batch_dim: int = 0,
        share_cache_keys: tuple = (),
    ):
        """Sample a mini-batch of calibration inputs by indices.

        Shared by SignRoundQuantizer and OptimizedRTNQuantizer.
        """
        if isinstance(input_ids, list):
            current_input_ids = [input_ids[i] for i in indices]
            current_input_ids = torch.cat(current_input_ids, dim=batch_dim)
        elif isinstance(input_ids, dict):
            current_input_ids = defaultdict(list)
            for k in input_ids.keys():
                current_input_ids[k].extend([input_ids[k][i] for i in indices])
                current_input_ids[k] = torch.cat(current_input_ids[k], dim=batch_dim)

        current_input_others = {"positional_inputs": input_others["positional_inputs"]}
        for key in input_others.keys():
            if "positional_inputs" in key:
                continue
            # Shared cache keys (e.g. position_embeddings, position_ids, cache_position) are stored
            # directly as-is (not wrapped in a per-sample list) when batch_size > 1.  Indexing such
            # values by sample index would incorrectly decompose them (e.g. (cos, sin)[0] == cos).
            # Always pass them through unchanged.
            if key in share_cache_keys or isinstance(input_others[key], (str, bool, type(None))):
                current_input_others[key] = input_others[key]
            elif input_others[key] is not None:
                current_input_others[key] = [input_others[key][i] for i in indices]
                if len(indices) == 1:
                    current_input_others[key] = current_input_others[key][0]
                else:
                    try:
                        current_input_others[key] = torch.cat(current_input_others[key], dim=0)
                    except TypeError as err:
                        logger.warning_once("Please check the model cache inputs or try setting batch_size to 1.")
            else:
                current_input_others[key] = None

        return current_input_ids, current_input_others

    @torch.no_grad()
    def _get_diffusion_block_outputs(
        self,
        block: torch.nn.Module,
        input_ids: Union[torch.Tensor, dict],
        input_others,
        bs: int,
        device: Union[str, torch.device],
        cache_device: Union[str, torch.device],
        save_output: bool = True,
    ):
        """Compute block outputs for diffusion models.

        Uses ``self.DIFFUSION_OUTPUT_CONFIGS`` to map block class names to their
        output keys.  Subclasses override ``DIFFUSION_OUTPUT_CONFIGS`` to add
        support for new diffusion architectures.
        """
        output = defaultdict(list)
        output_config = self.DIFFUSION_OUTPUT_CONFIGS.get(block.__class__.__name__, [])
        if isinstance(input_ids, dict):
            nsamples = len(input_ids["hidden_states"])
        else:
            nsamples = len(input_ids)

        for i in range(0, nsamples, bs):
            end_index = min(nsamples, i + bs)
            indices = torch.arange(i, end_index).to(torch.long)
            tmp_input_ids, tmp_input_others = self._sampling_inputs(
                input_ids,
                input_others,
                indices,
                self.seqlen,
                self.batch_dim,
                share_cache_keys=self.model_context.shared_cache_keys,
            )
            if isinstance(tmp_input_ids, dict):
                hidden_states = tmp_input_ids.pop("hidden_states")
                tmp_input_others.update(tmp_input_ids)
                tmp_input_ids = hidden_states

            tmp_output = block_forward(
                block,
                tmp_input_ids,
                tmp_input_others,
                self.model_context.amp,
                self.model_context.amp_dtype,
                device,
                None,
            )
            assert len(output_config) == len(tmp_output)
            tmp_output = dict(zip(output_config, tmp_output))

            if save_output:
                for name, out in tmp_output.items():
                    if self.batch_size == 1:
                        output[name].append(out.to(cache_device))
                    else:
                        output[name].extend(list(torch.split(out.to(cache_device), 1, dim=self.batch_dim)))
        self.compress_context.clear_memory()

        return output
