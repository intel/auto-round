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
import copy
import time
import traceback
from typing import Any, Callable, Union

import accelerate
import torch
from accelerate.big_modeling import dispatch_model, infer_auto_device_map
from tqdm import tqdm

from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
from auto_round.export.export_to_gguf.config import GGUF_CONFIG, GGUF_INNER_CONFIG, ModelType
from auto_round.logger import logger
from auto_round.quantizers.base import BaseQuantizer
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_DTYPES,
    SUPPORTED_FORMATS,
    SUPPORTED_LAYER_TYPES,
    TORCH_VERSION_AT_LEAST_2_6,
    CpuInfo,
    _gguf_args_check,
    _is_fp8_linear,
    _is_fp8_model,
    block_forward,
    check_and_mark_fp8_model,
    check_is_cpu,
    check_need_act_calibration,
    check_seqlen_compatible,
    check_skippable_keywords,
    check_to_quantized,
    clear_memory,
    collect_best_params,
    compile_func,
    convert_dtype_str2torch,
    convert_fp8_layer_to_linear,
    convert_fp8_model_to_16b_model,
    copy_python_files_from_model_cache,
    detect_device,
    estimate_tuning_block_mem,
    find_matching_blocks,
    flatten_list,
    get_block_names,
    get_device_memory,
    get_fp_layer_names,
    get_layer_config_by_gguf_format,
    get_layer_features,
    get_layer_names_in_block,
    get_lm_head_name,
    get_max_vram,
    get_module,
    get_quant_keys,
    get_shared_keys,
    htcore,
    infer_bits_by_data_type,
    init_cache,
    is_debug_mode,
    is_mx_fp,
    is_nv_fp,
    is_optimum_habana_available,
    is_standard_fp,
    is_static_wfp8afp8,
    is_wfp8afp8,
    llm_load_model,
    mv_module_from_gpu,
    reset_params,
    set_amax_for_all_moe_layers,
    set_module,
    to_device,
    to_dtype,
    unsupport_meta_device,
)
from auto_round.wrapper import WrapperLinear, WrapperMultiblock, unwrapper_block, unwrapper_layer, wrapper_block


class RTNQuantizer(BaseQuantizer):
    @torch.inference_mode()
    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize all modules in the model using RTN (Round-To-Nearest) strategy.

        If the target format includes GGUF with `k`, and optimized RTN is enabled,
        blockwise quantization with input caching and imatrix is used.

        Returns:
            tuple[nn.Module, Dict[str, Any]]: The quantized model and the layer configuration.
        """
        if self.amp and self.model.dtype != self.amp_dtype:
            self.model.to(self.amp_dtype)

        all_to_quantized_module_names: list[str] = [n for n, m in self.model.named_modules() if check_to_quantized(m)]

        if is_nv_fp(self.data_type):
            from auto_round.data_type.nvfp import calculate_gparam
            from auto_round.data_type.utils import update_fused_layer_global_scales

            pbar = tqdm(all_to_quantized_module_names)
            for name in pbar:
                pbar.set_description(f"Calculate weight global scale: {name}")
                m = get_module(self.model, name)
                weight_global_scale = calculate_gparam(m.weight, self.group_size)
                setattr(m, "weight_global_scale", weight_global_scale)

            modules = list(self.model.modules())
            for module in tqdm(modules, desc="Update weight global scale for fuse module"):
                update_fused_layer_global_scales(module)

        has_gguf_k = any("gguf" in fmt and "k" in fmt for fmt in getattr(self, "formats", []))

        self._quantize_embedding_layer()

        self.model.to("cpu")
        if has_gguf_k and not self.disable_opt_rtn:
            self._quant_rtn_with_imatrix(all_to_quantized_module_names)
        elif self.act_bits <= 8 and check_need_act_calibration(
            self.act_dynamic, self.act_data_type, self.act_bits
        ):  # TODO, mixed datatype has bug
            hook_handles = self._register_act_max_hook(self.model)
            try:
                self._quantize_via_rtn_blockwise(all_to_quantized_module_names)
            except RuntimeError as e:
                logger.warning("Fallback to CPU. Consider using more GPUs via `--device 0,1,2,3`.")
                self.model = self.model.to("cpu")
                clear_memory()
                if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                    import accelerate

                    accelerate.hooks.remove_hook_from_submodules(self.model)
                orig_device = self.device
                self.device = "cpu"
                self._quantize_via_rtn_blockwise(all_to_quantized_module_names)
                self.device = orig_device
            for handle in hook_handles:
                handle.remove()
        else:
            block_names_cnt = len(flatten_list(get_block_names(self.model, True)))
            clear_mem_freq = len(all_to_quantized_module_names) // block_names_cnt
            if clear_mem_freq == 0:
                clear_mem_freq = 1
            pbar = tqdm(all_to_quantized_module_names)
            cnt = 1
            for name in pbar:
                pbar.set_description(f"Quantizing {name}")
                self._quantize_layer_via_rtn(name)
                if cnt % clear_mem_freq == 0:
                    clear_memory()
                    cnt = 1
                cnt += 1
        # Convert remaining fp8
        if _is_fp8_model(self.model):
            convert_fp8_model_to_16b_model(self.model, self.amp_dtype)
        self.quantized = True
        return self.model, self.layer_config

    def _quantize_layer_via_rtn(self, name: str) -> None:
        """Quantizes a layer using RTN (Round-To-Nearest) if available.

        This function attempts to quantize a layer by switching its data type to a
        `rtn_*` version if supported, then wraps and unwraps the module to apply
        quantization. If GPU memory is insufficient, it falls back to CPU.

        If packing is enabled (`is_packing_immediate`), the function will also export
        the quantized layer to the appropriate backend format.

        Args:
            name (str): Name of the layer to quantize.

        Raises:
            RuntimeError: If quantization fails for reasons unrelated to memory.
        """
        m = get_module(self.model, name)

        # if m.__class__.__name__ == "FP8Linear":
        if _is_fp8_linear(m):
            m = convert_fp8_layer_to_linear(m, self.amp_dtype)
            set_module(self.model, name, m)

        # Step 1: Use optimized RTN data type if available
        if not self.disable_opt_rtn and not m.data_type.startswith("rtn_"):
            from auto_round.data_type import QUANT_FUNC_WITH_DTYPE

            rtn_dtype = "rtn_" + m.data_type
            if rtn_dtype in QUANT_FUNC_WITH_DTYPE:
                m.data_type = rtn_dtype
                self.layer_config[name]["data_type"] = m.data_type

        # Step 2: Try quantization on GPU first, fall back to CPU if OOM
        # if only export gguf, using gguf-packing instead of rtn
        if self.is_packing_immediate and self.iters == 0 and "gguf" in self.formats[0] and not self.disable_opt_rtn:
            m.scale = None
            m.zp = None
        else:
            try:
                m.to(self.device)
                m = WrapperLinear(
                    m,
                    enable_minmax_tuning=False,
                    enable_norm_bias_tuning=False,
                    enable_round_tuning=False,
                )
                m = m.unwrapper({})
                m.to("cpu")
            except RuntimeError as e:
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
                    )
                    m = m.unwrapper({})
                except Exception as e:
                    raise

        # Step 3: Optional immediate packing/export
        if self.is_packing_immediate:
            from auto_round.export import PACKING_LAYER_WITH_FORMAT

            if check_to_quantized(m):
                target_backend = self.formats[0].split(":")[0] if ":" in self.formats[0] else self.formats[0]
                has_gguf = any("gguf" in fmt for fmt in self.formats)

                if has_gguf:
                    from auto_round.export.export_to_gguf.export import pack_gguf_layer

                    output_dir = self._get_save_folder_name(self.formats[0])
                    model_type = ModelType.MMPROJ if self.mllm else ModelType.TEXT
                    pack_gguf_layer(
                        name,
                        self.model,
                        self.formats[0],
                        output_dir,
                        self.layer_config,
                        self.tokenizer,
                        processor=self.processor if hasattr(self, "processor") else None,
                        image_processor=self.image_processor if hasattr(self, "image_processor") else None,
                        model_type=model_type,
                    )
                else:
                    PACKING_LAYER_WITH_FORMAT[target_backend](name, self.model, self.formats[0], device=self.device)

                # if self.low_gpu_mem_usage:
                #     clear_memory()
        else:
            set_module(self.model, name, m)

    def _quant_rtn_with_imatrix(self, all_to_quantized_module_names: list[str]) -> None:
        """Performs RTN quantization using input activation statistics (imatrix).

        This method accumulates per-channel second-moment activation statistics (imatrix)
        via forward hooks and uses them to perform RTN quantization. If CUDA memory runs out,
        it falls back to CPU-based blockwise quantization.

        Args:
            all_to_quantized_module_names (list[str]):
                A list of module names (e.g., 'model.layers.0.self_attn.q_proj') to be quantized.

        Returns:
            None
        """
        logger.info("start to compute imatrix for GGUF quantization")

        # Load dataset
        from auto_round.calib_dataset import get_dataloader

        if _is_fp8_model(self.model):
            convert_fp8_model_to_16b_model(self.model, self.amp_dtype)

        if isinstance(self.dataset, str):
            if self.tokenizer is None:
                raise ValueError("A tokenizer must be set for the model when using a dataset string.")
            dataset_name = self.dataset.replace(" ", "")
            self.dataloader = get_dataloader(
                self.tokenizer, self.seqlen, dataset_name, self.seed, self.batch_size, self.nsamples
            )
        else:
            self.dataloader = self.dataset

        model = self.model

        # Dispatch multi-GPU model if necessary
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            dispatch_model(model, model.hf_device_map)

        def register_act_hook(model):
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
                if isinstance(module, self.supported_types) and check_to_quantized(module):
                    hook = module.register_forward_hook(get_imatrix_hook)
                    hook_handles.append(hook)
            return hook_handles

        hooks = register_act_hook(model)

        try:
            # Move model to target device
            if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                dispatch_model(self.model, self.model.hf_device_map)
            else:
                model = model.to(self.device)
            cnt = 0

            # Run forward pass to accumulate imatrix
            for data in self.dataloader:
                cnt += data["input_ids"].shape[0]
                data = to_device(data, self.device)
                model(**data)
                if cnt >= self.nsamples:
                    break

            # Remove hooks after data collection
            for hook in hooks:
                hook.remove()

            # Normalize imatrix by count
            for _, module in model.named_modules():
                if hasattr(module, "imatrix"):
                    module.imatrix /= module.imatrix_cnt
            if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                import accelerate

                accelerate.hooks.remove_hook_from_submodules(model)
            # Perform quantization using RTN
            pbar = tqdm(all_to_quantized_module_names)
            block_names_cnt = len(flatten_list(get_block_names(self.model, True)))
            clear_mem_freq = len(all_to_quantized_module_names) // block_names_cnt
            if clear_mem_freq == 0:
                clear_mem_freq = 1
            cnt = 1
            for name in pbar:
                pbar.set_description(f"Quantizing {name}")
                self._quantize_layer_via_rtn(name)
                if cnt % clear_mem_freq == 0:
                    clear_memory()
                    cnt = 1
                cnt += 1
        except RuntimeError as e:
            try:
                if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                    import accelerate

                    accelerate.hooks.remove_hook_from_submodules(model)
                # Fallback: out-of-memory → try CPU blockwise quantization
                logger.warning("Out of VRAM, falling back to blockwise quantization. Accuracy may degrade.")
                model = model.to("cpu")
                clear_memory()
                self._quantize_via_rtn_blockwise(all_to_quantized_module_names)
            except RuntimeError as e:
                cuda_error_msg = traceback.format_exc()
                try:
                    logger.error(cuda_error_msg)
                    # Final fallback: warn and use CPU-only quantization
                    logger.warning(
                        "Fallback to CPU. "
                        "Consider enabling `low_gpu_mem_usage` or using more GPUs via `--device 0,1,2,3`."
                    )
                    model = model.to("cpu")
                    clear_memory()
                    if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                        import accelerate

                        accelerate.hooks.remove_hook_from_submodules(model)

                    orig_device = self.device
                    self.device = "cpu"
                    self._quantize_via_rtn_blockwise(all_to_quantized_module_names)
                    self.device = orig_device
                except Exception as e:
                    raise
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()

    def _quantize_via_rtn_blockwise(self, all_to_quantized_module_names: list[str]) -> None:
        """Quantize model layers block by block using cached inputs and imatrix.

        Args:
            all_to_quantized_module_names (list[str]): Names of layers to be quantized.
        """
        all_to_quantized_module_names = list(set(all_to_quantized_module_names))

        all_blocks = self.quant_block_list if self.quant_block_list else get_block_names(self.model)
        if not all_blocks:
            raise ValueError("Could not find any blocks. Check the model or quant_block_list.")

        all_first_block_names = [block[0] for block in all_blocks]
        if self.act_bits < 16 and not self.act_dynamic:
            layer_names = self._get_quantized_layer_names_outside_blocks()
            if len(layer_names) > 0:
                logger.warning(
                    "quantize layers outside blocks for static activation quantizaiton"
                    " will significantly increase calibration time"
                )
            all_inputs = self.try_cache_inter_data_gpucpu(all_first_block_names, self.nsamples, layer_names)
        else:
            all_inputs = self.cache_inter_data(all_first_block_names, self.nsamples)

        # Clear hooks for multi-GPU setups
        if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(self.model)

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

            clear_memory(self.inputs)

            total_samples = len(inputs["input_ids"])
            if total_samples < self.batch_size:
                self.batch_size = total_samples
                logger.warning(f"Forcing batch size to {total_samples}")

            input_ids = to_device(inputs.pop("input_ids"), self.cache_device)
            input_others = to_device(inputs, self.cache_device)

            tmp_dtype = self.amp_dtype if self.amp else torch.float32
            input_ids = [id_.to(tmp_dtype) for id_ in input_ids]

            for key, val in input_others.items():
                if isinstance(val, torch.Tensor) and val.dtype in (torch.float16, torch.bfloat16):
                    input_others[key] = val.to(tmp_dtype)
                elif isinstance(val, list):
                    input_others[key] = [to_dtype(v, tmp_dtype) for v in val]

            for block_name in block_names:
                pbar.set_description(f"Quantizing {block_name}")
                block = get_module(self.model, block_name)
                block = block.to(self.device)
                if _is_fp8_model(self.model):
                    convert_fp8_model_to_16b_model(block, dtype=self.amp_dtype)

                if self.device_map == "auto":
                    self._set_auto_device_map_in_block(block, input_ids)

                # Dispatch model if needed
                if self.device_map is not None:
                    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                    for _, m in block.named_modules():
                        if len(list(m.children())) != 0 or not hasattr(m, "tuning_device"):
                            continue
                        hook = AlignDevicesHook(m.tuning_device, io_same_device=True)
                        add_hook_to_module(m, hook, True)
                else:
                    block = block.to(self.device)
                input_ids = self._get_block_outputs(
                    block,
                    input_ids,
                    input_others,
                    self.batch_size * self.infer_bs_coeff,
                    self.device,
                    self.cache_device,
                )
                if self.device_map is not None:
                    accelerate.hooks.remove_hook_from_submodules(block)

                if (
                    is_nv_fp(self.act_data_type) and any("nv_fp" in format_ for format_ in self.formats)
                ) or is_static_wfp8afp8(self):
                    # enable moe experts act_max automatic generation for Linear
                    set_amax_for_all_moe_layers(block, attr_name="act_max")
                # Normalize imatrix and quantize layers
                for _, m in block.named_modules():
                    if hasattr(m, "imatrix"):
                        m.imatrix /= m.imatrix_cnt
                    if hasattr(m, "tmp_name") and m.tmp_name in all_to_quantized_module_names:
                        self._quantize_layer_via_rtn(m.tmp_name)
                        all_to_quantized_module_names.remove(m.tmp_name)

                mv_module_from_gpu(block, self.low_cpu_mem_usage)
                pbar.update(1)

        pbar.close()
        cnt = 1
        block_names_cnt = len(flatten_list(get_block_names(self.model, True)))
        clear_mem_freq = len(all_to_quantized_module_names) // block_names_cnt
        if clear_mem_freq == 0:
            clear_mem_freq = 1
        # Process remaining layers not in blocks
        for name in all_to_quantized_module_names:
            self._quantize_layer_via_rtn(name)
            if cnt % clear_mem_freq == 0:
                clear_memory()
                cnt = 1
            cnt += 1
