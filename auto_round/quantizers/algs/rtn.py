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

import traceback
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import accelerate
import torch
from accelerate.big_modeling import dispatch_model, infer_auto_device_map
from tqdm import tqdm

from auto_round.compressors.utils import (
    check_need_act_calibration,
    immediate_saving,
    is_nv_fp,
    is_static_wfp8afp8,
)
from auto_round.logger import logger
from auto_round.quantizers.algs.base import AlgsBaseQuantizer
from auto_round.quantizers.utils import (
    get_quantized_layer_names_outside_blocks,
    quantize_embedding_layer,
    register_act_max_hook,
)
from auto_round.utils import (
    check_to_quantized,
    clear_memory,
    convert_fp8_layer_to_linear,
    convert_fp8_model_to_16b_model,
    flatten_list,
    get_block_names,
    get_module,
    is_auto_device_mapping,
    is_fp8_linear,
    is_fp8_model,
    memory_monitor,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
    set_module,
    to_device,
    to_dtype,
)
from auto_round.utils.device import (
    clear_memory_if_reached_threshold,
    get_major_device,
    parse_available_devices,
    set_auto_device_map_for_block_with_tuning,
    set_non_auto_device_map,
)
from auto_round.wrapper import WrapperLinear, WrapperMultiblock, unwrapper_block, unwrapper_layer, wrapper_block

if TYPE_CHECKING:
    from auto_round.compressors.base import BaseCompressor


class RTNQuantizer(AlgsBaseQuantizer):

    def __init__(self, compressor: "BaseCompressor"):
        super().__init__(compressor)
        self.all_to_quantized_module_names: list[str] = [
            n for n, m in self.compressor.model.named_modules() if check_to_quantized(m)
        ]

    def _pre_quantize_impl(self, *args, **kwargs):
        if self.compressor.amp and self.compressor.model.dtype != self.compressor.amp_dtype:
            self.compressor.model.to(self.compressor.amp_dtype)

        if is_nv_fp(self.compressor.data_type):
            from auto_round.data_type.nvfp import calculate_gparam
            from auto_round.data_type.utils import update_fused_layer_global_scales

            pbar = tqdm(self.all_to_quantized_module_names)
            for name in pbar:
                pbar.set_description(f"Calculate weight global scale: {name}")
                m = get_module(self.compressor.model, name)
                if is_fp8_linear(m):
                    m = convert_fp8_layer_to_linear(m, self.compressor.amp_dtype, self.compressor.device)
                    set_module(self.compressor.model, name, m)
                weight_global_scale = calculate_gparam(m.weight, self.compressor.group_size)
                setattr(m, "weight_global_scale", weight_global_scale)

            logger.info("Start to update fused layer global scales, it may take some time.")
            for name, module in self.compressor.model.named_modules():
                update_fused_layer_global_scales(module)
            logger.info("Finished updating fused layer global scales.")

    @torch.inference_mode()
    def _quantize_impl(self, *args, **kwargs) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize all modules in the model using RTN (Round-To-Nearest) strategy.

        If the target format includes GGUF with `k`, and optimized RTN is enabled,
        blockwise quantization with input caching and imatrix is used.

        Returns:
            tuple[nn.Module, Dict[str, Any]]: The quantized model and the layer configuration.
        """
        if not (
            any(fmt.is_gguf() for fmt in getattr(self.compressor, "formats", []))
            or self.compressor.super_bits is not None
        ):
            quantize_embedding_layer(
                model=self.compressor.model,
                layer_config=self.compressor.layer_config,
                scale_dtype=self.compressor.scale_dtype,
                disable_opt_rtn=self.compressor.disable_opt_rtn,
                device=self.compressor.device,
                device_list=self.compressor.device_list,
            )  # leave to gguf itself to handle

        self.compressor.model.to("cpu")
        # Release memory
        clear_memory(device_list=self.compressor.device_list)

        if self.compressor.act_bits <= 8 and check_need_act_calibration(
            self.compressor.act_dynamic,
            self.compressor.act_data_type,
            self.compressor.act_bits,
            self.compressor.static_kv_dtype,
            self.compressor.static_attention_dtype,
        ):  # TODO, mixed datatype has bug
            hook_handles = register_act_max_hook(
                model=self.compressor.model,
                layer_config=self.compressor.layer_config,
                act_group_size=self.compressor.act_group_size,
                act_data_type=self.compressor.act_data_type,
            )
            try:
                self._quantize_via_rtn_blockwise()
            except torch.OutOfMemoryError:
                logger.warning("Fallback to CPU. Consider using more GPUs via `--device 0,1,2,3`.")
                self.model = self.model.to("cpu")
                clear_memory(device_list=self.compressor.device_list)
                if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                    import accelerate

                    accelerate.hooks.remove_hook_from_submodules(self.model)
                orig_device = self.compressor.device
                self.compressor.device = "cpu"
                self._quantize_via_rtn_blockwise()
                self.compressor.device = orig_device
            for handle in hook_handles:
                handle.remove()
        else:
            block_names_cnt = len(flatten_list(get_block_names(self.compressor.model, True)))
            clear_mem_freq = len(self.all_to_quantized_module_names) // block_names_cnt
            if clear_mem_freq == 0:
                clear_mem_freq = 1
            pbar = tqdm(self.all_to_quantized_module_names)
            cnt = 1
            for name in pbar:
                pbar.set_description(f"Quantizing {name}")
                self._quantize_layer_via_rtn(name)
                if cnt % clear_mem_freq == 0:
                    clear_memory(device_list=self.compressor.device_list)
                    memory_monitor.log_summary()
                    cnt = 1
                cnt += 1
        # Convert remaining fp8
        if is_fp8_model(self.compressor.model):
            convert_fp8_model_to_16b_model(self.compressor.model, self.compressor.amp_dtype, self.compressor.device)
        self.compressor.quantized = True
        return self.compressor.model, self.compressor.layer_config

    def _quantize_via_rtn_blockwise(self) -> None:
        """Quantize model layers block by block using cached inputs."""

        all_to_quantized_module_names = list(set(self.all_to_quantized_module_names))

        all_blocks = (
            self.compressor.quant_block_list
            if self.compressor.quant_block_list
            else get_block_names(self.compressor.model)
        )
        if not all_blocks:
            raise ValueError("Could not find any blocks. Check the model or quant_block_list.")

        all_first_block_names = [block[0] for block in all_blocks]
        layer_names = get_quantized_layer_names_outside_blocks(
            model=self.compressor.model,
            layer_config=self.compressor.layer_config,
            supported_types=self.compressor.supported_types,
            quant_block_list=self.compressor.quant_block_list,
        )
        if self.compressor.act_bits < 16 and (not self.compressor.act_dynamic or len(layer_names) > 0):
            if len(layer_names) > 0:
                logger.warning(
                    "quantize layers outside blocks for static activation quantizaiton"
                    " will significantly increase calibration time"
                )
            all_inputs = self.compressor.try_cache_inter_data_gpucpu(
                all_first_block_names, self.compressor.nsamples, layer_names
            )
        else:
            all_inputs = self.compressor.cache_inter_data(all_first_block_names, self.compressor.nsamples)

        # Clear hooks for multi-GPU setups
        if hasattr(self.compressor.model, "hf_device_map") and len(self.compressor.model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(self.compressor.model)

        pbar = tqdm(range(sum(len(block) for block in all_blocks)))

        for block_names in all_blocks:
            first_block = block_names[0]
            inputs = all_inputs.pop(first_block)
            input_keys = [k for k in inputs if k.startswith("hidden_state")]
            if len(input_keys) != 1:
                raise RuntimeError(
                    "hidden_states arg mismatch. Please file an issue at https://github.com/intel/auto-round/issues"
                )
            inputs["input_ids"] = inputs.pop(input_keys[0])

            clear_memory(self.compressor.inputs, device_list=self.compressor.device_list)

            total_samples = len(inputs["input_ids"])
            if total_samples < self.compressor.batch_size:
                self.compressor.batch_size = total_samples
                logger.warning(f"Forcing batch size to {total_samples}")

            input_ids = to_device(inputs.pop("input_ids"), self.compressor.cache_device)
            input_others = to_device(inputs, self.compressor.cache_device)

            tmp_dtype = self.compressor.amp_dtype if self.compressor.amp else torch.float32
            input_ids = [id_.to(tmp_dtype) for id_ in input_ids]

            for key, val in input_others.items():
                if isinstance(val, torch.Tensor) and val.dtype in (torch.float16, torch.bfloat16):
                    input_others[key] = val.to(tmp_dtype)
                elif isinstance(val, list):
                    input_others[key] = [to_dtype(v, tmp_dtype) for v in val]

            for block_name in block_names:
                pbar.set_description(f"Quantizing {block_name}")
                block = get_module(self.compressor.model, block_name)
                if is_fp8_model(self.compressor.model):
                    convert_fp8_model_to_16b_model(
                        block, dtype=self.compressor.amp_dtype, device=self.compressor.device
                    )

                if is_auto_device_mapping(self.compressor.device_map) and len(self.compressor.device_list) > 1:
                    set_auto_device_map_for_block_with_tuning(
                        block,
                        self.compressor.device_map,
                        input_ids,
                        self.compressor.low_gpu_mem_usage,
                        self.compressor.batch_size,
                        self.compressor.device,
                    )
                # Dispatch model if needed
                if len(self.compressor.device_list) > 1:
                    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                    for _, m in block.named_modules():
                        if len(list(m.children())) != 0 or not hasattr(m, "tuning_device"):
                            continue
                        hook = AlignDevicesHook(m.tuning_device, io_same_device=True)
                        add_hook_to_module(m, hook, True)
                else:
                    block = block.to(self.compressor.device)

                # TODO: refactor this part
                input_ids = self.compressor._get_block_outputs(
                    block,
                    input_ids,
                    input_others,
                    self.compressor.batch_size * self.compressor.infer_bs_coeff,
                    self.compressor.device,
                    self.compressor.cache_device,
                )

                if len(self.compressor.device_list) > 1:
                    accelerate.hooks.remove_hook_from_submodules(block)

                if is_nv_fp(self.compressor.act_data_type) or is_static_wfp8afp8(self.compressor):
                    # enable moe experts act_max automatic generation for Linear
                    set_amax_for_all_moe_layers(block, attr_name="act_max")
                if self.compressor.low_gpu_mem_usage:
                    block.to("cpu")
                    clear_memory(device_list=self.compressor.device_list)

                for _, m in block.named_modules():
                    if hasattr(m, "tmp_name") and m.tmp_name in all_to_quantized_module_names:
                        self._quantize_layer_via_rtn(m.tmp_name, to_cpu=self.compressor.low_gpu_mem_usage)
                        all_to_quantized_module_names.remove(m.tmp_name)
                if not self.compressor.immediate_saving:
                    mv_module_from_gpu(block)
                if block_name == block_names[-1]:
                    clear_memory(input_ids, device_list=self.compressor.device_list)
                else:
                    clear_memory(device_list=self.compressor.device_list)

                memory_monitor.log_summary()
                pbar.update(1)
        pbar.close()
        # Process remaining layers not in blocks
        for name in all_to_quantized_module_names:
            dtype = None
            if self.compressor.super_group_size is not None:
                dtype = torch.float32
            self._quantize_layer_via_rtn(name, dtype=dtype)
            # clear_memory(device_list=self.compressor.device_list)

    def _quantize_layer_via_rtn(self, name: str, dtype: torch.dtype = None, to_cpu=True) -> None:
        """Quantizes a layer using RTN (Round-To-Nearest) if available.

        This function attempts to quantize a layer by switching its data type to a
        `rtn_*` version if supported, then wraps and unwraps the module to apply
        quantization. If GPU memory is insufficient, it falls back to CPU.

        If packing is enabled (`immediate_packing`), the function will also export
        the quantized layer to the appropriate backend format.

        Args:
            name (str): Name of the layer to quantize.

        Raises:
            RuntimeError: If quantization fails for reasons unrelated to memory.
        """
        m = get_module(self.compressor.model, name)
        if dtype is not None:
            m = m.to(dtype)

        if is_fp8_linear(m):
            m = convert_fp8_layer_to_linear(m, self.compressor.amp_dtype, self.compressor.device)
            set_module(self.compressor.model, name, m)
        tuning_device = m.tuning_device if hasattr(m, "tuning_device") else self.compressor.device

        try:
            m = m.to(tuning_device)
            m = WrapperLinear(
                m,
                device=tuning_device,
                enable_minmax_tuning=False,
                enable_norm_bias_tuning=False,
                enable_round_tuning=False,
                enable_torch_compile=self.compressor.enable_torch_compile,
            )
            m = m.unwrapper({})
        except torch.OutOfMemoryError:
            cuda_error_msg = traceback.format_exc()
            m = m.orig_layer if hasattr(m, "orig_layer") else m
            try:
                logger.error(cuda_error_msg)
                logger.warning("falling back to CPU.")
                m.to("cpu")
                m = WrapperLinear(
                    m,
                    enable_minmax_tuning=False,
                    enable_norm_bias_tuning=False,
                    enable_round_tuning=False,
                    enable_torch_compile=self.compressor.enable_torch_compile,
                )
                m = m.unwrapper({})
            except Exception as e:
                raise

        # Step 2: Optional immediate packing/export
        if self.compressor.immediate_packing:  # For gguf, packing conducts on block level
            self.compressor._immediate_pack(name)
            if to_cpu:
                m = m.to("cpu")
                packed_m = get_module(self.compressor.model, name)
                set_module(self.compressor.model, name, packed_m.to("cpu"))
        else:
            if to_cpu:
                m = m.to("cpu")
            set_module(self.compressor.model, name, m)
        if self.compressor.immediate_saving:
            if hasattr(self.compressor, "all_to_quantized_module_names"):
                all_to_quantized_module_names = self.compressor.all_to_quantized_module_names
            else:
                all_to_quantized_module_names = [
                    n for n, m in self.compressor.model.named_modules() if check_to_quantized(m)
                ]
            last_module = (len(all_to_quantized_module_names) == 0) or (name == all_to_quantized_module_names[-1])
            m = get_module(self.compressor.model, name)
            immediate_saving(self.compressor, m, name, last_module)


class OptRTNQuantizer(RTNQuantizer):

    @staticmethod
    def register_act_hook(model, supported_types):
        """Registers hooks to accumulate activation squared norms into `imatrix`."""

        def get_imatrix_hook(module, input, output):
            input = input[0] if isinstance(input, (tuple, list)) else input
            flattened = input.reshape(-1, input.shape[-1]).to(torch.float32)
            squared = torch.sum(torch.pow(flattened, 2), dim=0).to(torch.float32)

            if not hasattr(module, "imatrix"):
                module.imatrix = squared
                module.imatrix_cnt = input.shape[0]
            else:
                module.imatrix += squared.to(module.imatrix.device)
                module.imatrix_cnt += input.shape[0]

        hook_handles = []
        for name, module in model.named_modules():
            if type(module) in supported_types and check_to_quantized(module):
                hook = module.register_forward_hook(get_imatrix_hook)
                hook_handles.append(hook)
        return hook_handles

    @torch.inference_mode()
    def _quantize_impl(self, *args, **kwargs) -> tuple[torch.nn.Module, dict[str, Any]]:
        enable_imatrix = False
        has_gguf_k = (
            any(fmt.is_gguf() and "k" in fmt.output_format for fmt in getattr(self.compressor, "formats", []))
            or self.compressor.super_bits is not None
        )
        if has_gguf_k:
            enable_imatrix = True
        elif self.compressor.data_type == "int" and self.compressor.sym:
            enable_imatrix = True
        if enable_imatrix:
            self._quant_rtn_with_imatrix(self.all_to_quantized_module_names)
            # Convert remaining fp8
            if is_fp8_model(self.compressor.model):
                convert_fp8_model_to_16b_model(self.compressor.model, self.compressor.amp_dtype, self.compressor.device)
            self.compressor.quantized = True
            return self.compressor.model, self.compressor.layer_config
        else:
            return super()._quantize_impl(*args, **kwargs)

    def _quant_rtn_with_imatrix(self, *args, **kwargs) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize all modules in the model using Optimized RTN strategy.

        This method applies optimized RTN quantization to all modules in the model
        that are marked for quantization. It leverages input caching and imatrix
        techniques for enhanced performance.

        Returns:
            tuple[nn.Module, Dict[str, Any]]: The quantized model and the layer configuration.
        """
        if not (
            any(fmt.is_gguf() for fmt in getattr(self.compressor, "formats", []))
            or self.compressor.super_bits is not None
        ):
            quantize_embedding_layer(
                model=self.compressor.model,
                layer_config=self.compressor.layer_config,
                scale_dtype=self.compressor.scale_dtype,
                disable_opt_rtn=self.compressor.disable_opt_rtn,
                device=self.compressor.device,
                device_list=self.compressor.device_list,
            )  # leave to gguf itself to handle

        self.compressor.model.to("cpu")
        # Release memory
        clear_memory(device_list=self.compressor.device_list)

        logger.info("start to compute imatrix")

        # Load dataset
        from auto_round.calib_dataset import get_dataloader

        if isinstance(self.compressor.dataset, str):
            if self.compressor.tokenizer is None:
                raise ValueError("A tokenizer must be set for the model when using a dataset string.")
            dataset_name = self.compressor.dataset.replace(" ", "")
            self.compressor.dataloader = get_dataloader(
                self.compressor.tokenizer,
                self.compressor.seqlen,
                dataset_name,
                self.compressor.seed,
                self.compressor.batch_size,
                self.compressor.nsamples,
            )
        else:
            self.compressor.dataloader = self.compressor.dataset

        model = self.compressor.model

        # Dispatch multi-GPU model if necessary
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            dispatch_model(model, model.hf_device_map)

        hooks = self.register_act_hook(model, self.compressor.supported_types)

        try:
            if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                import accelerate

                accelerate.hooks.remove_hook_from_submodules(model)
            model = model.to("cpu")
            clear_memory(device_list=self.compressor.device_list)
            self._quantize_via_rtn_blockwise()
        except torch.OutOfMemoryError:
            cuda_error_msg = traceback.format_exc()
            try:
                logger.error(cuda_error_msg)
                # Final fallback: warn and use CPU-only quantization
                logger.warning(
                    "Fallback to CPU. "
                    "Consider enabling `low_gpu_mem_usage` or using more GPUs via `--device 0,1,2,3`."
                )
                model = model.to("cpu")
                clear_memory(device_list=self.compressor.device_list)
                if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                    import accelerate

                    accelerate.hooks.remove_hook_from_submodules(model)

                orig_device = self.compressor.device
                self.compressor.device = "cpu"
                self._quantize_via_rtn_blockwise()
                self.compressor.device = orig_device
            except Exception as e:
                raise
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()

    def _quantize_layer_via_rtn(self, name: str, dtype: torch.dtype = None, to_cpu=True) -> None:
        """Quantizes a layer using RTN (Round-To-Nearest) if available.

        This function attempts to quantize a layer by switching its data type to a
        `rtn_*` version if supported, then wraps and unwraps the module to apply
        quantization. If GPU memory is insufficient, it falls back to CPU.

        If packing is enabled (`immediate_packing`), the function will also export
        the quantized layer to the appropriate backend format.

        Args:
            name (str): Name of the layer to quantize.

        Raises:
            RuntimeError: If quantization fails for reasons unrelated to memory.
        """
        m = get_module(self.compressor.model, name)
        if dtype is not None:
            m = m.to(dtype)

        if is_fp8_linear(m):
            m = convert_fp8_layer_to_linear(m, self.compressor.amp_dtype, self.compressor.device)
            set_module(self.compressor.model, name, m)
        tuning_device = m.tuning_device if hasattr(m, "tuning_device") else self.compressor.device
        # Step 1: Try quantization on GPU first, fall back to CPU if OOM
        if (
            self.compressor.immediate_packing
            and self.compressor.iters == 0
            and self.compressor.formats[0].is_gguf()
            and not self.compressor.disable_opt_rtn
        ):
            m = m.to(tuning_device)
            m.scale = None
            m.zp = None
        else:
            try:
                disable_opt_rtn = False
                if (
                    self.compressor.orig_disable_opt_rtn is None
                    and self.compressor.is_moe_model
                    and "expert" in m.tmp_name
                    and "shared_expert" not in m.tmp_name
                    and self.compressor.super_bits is None  # GGUF still uses the optimized RTN for MoE layers
                ):
                    disable_opt_rtn = True
                    logger.warning_once(
                        "MoE layer detected: optimized RTN is disabled for efficiency. "
                        "Use `--enable_opt_rtn` to force-enable it for MoE layers."
                    )
                m = m.to(tuning_device)
                m = WrapperLinear(
                    m,
                    device=tuning_device,
                    enable_minmax_tuning=False,
                    enable_norm_bias_tuning=False,
                    enable_round_tuning=False,
                    enable_torch_compile=self.compressor.enable_torch_compile,
                    disable_opt_rtn=disable_opt_rtn,
                )
                m = m.unwrapper({})
            except torch.OutOfMemoryError:
                cuda_error_msg = traceback.format_exc()
                m = m.orig_layer if hasattr(m, "orig_layer") else m
                try:
                    logger.error(cuda_error_msg)
                    logger.warning("falling back to CPU.")
                    m.to("cpu")
                    m = WrapperLinear(
                        m,
                        enable_minmax_tuning=False,
                        enable_norm_bias_tuning=False,
                        enable_round_tuning=False,
                        enable_torch_compile=self.compressor.enable_torch_compile,
                    )
                    m = m.unwrapper({})
                except Exception as e:
                    raise

        # Step 2: Optional immediate packing/export
        if self.compressor.immediate_packing:  # For gguf, packing conducts on block level
            self.compressor._immediate_pack(name)
            if to_cpu:
                m = m.to("cpu")
                packed_m = get_module(self.compressor.model, name)
                set_module(self.compressor.model, name, packed_m.to("cpu"))
        else:
            if to_cpu:
                m = m.to("cpu")
            set_module(self.compressor.model, name, m)
        if self.compressor.immediate_saving:
            if hasattr(self.compressor, "all_to_quantized_module_names"):
                all_to_quantized_module_names = self.compressor.all_to_quantized_module_names
            else:
                all_to_quantized_module_names = [
                    n for n, m in self.compressor.model.named_modules() if check_to_quantized(m)
                ]
            last_module = (len(all_to_quantized_module_names) == 0) or (name == all_to_quantized_module_names[-1])
            m = get_module(self.compressor.model, name)
            immediate_saving(self.compressor, m, name, last_module)
