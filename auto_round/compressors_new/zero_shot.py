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
from typing import Any, Union

import accelerate
import torch
from tqdm import tqdm

from auto_round.algorithms.alg_config import AlgConfig
from auto_round.compressors_new.base import BaseCompressor
from auto_round.compressors_new.shard_writer import shard_writer
from auto_round.compressors_new.utils import (
    _get_quantized_layer_names_outside_blocks,
    check_need_act_calibration,
)
from auto_round.logger import logger
from auto_round.modeling.fused_moe.replace_modules import materialize_model_
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    clear_memory,
    convert_module_to_hp_if_necessary,
    flatten_list,
    get_block_names,
    get_lm_head_name,
    get_module,
    global_state,
    memory_monitor,
    set_module,
    to_device,
    to_dtype,
)


class ZeroShotCompressor(BaseCompressor):
    need_calib: bool = False

    def __init__(
        self,
        config: Union[AlgConfig, list[AlgConfig]],
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform="hf",
        format=None,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        enable_alg_ext: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        **kwargs,
    ):
        super().__init__(
            config=config,
            model=model,
            tokenizer=tokenizer,
            platform=platform,
            format=format,
            device_map=device_map,
            low_gpu_mem_usage=low_gpu_mem_usage,
            enable_torch_compile=enable_torch_compile,
            enable_alg_ext=enable_alg_ext,
            seed=seed,
            low_cpu_mem_usage=low_cpu_mem_usage,
            **kwargs,
        )
        self.lr = 5e-3

    def _quantize_via_rtn_blockwise(self) -> None:
        """Quantize model layers block by block using cached inputs and imatrix."""

        all_blocks = self.quantizer.quant_block_list if self.quantizer.quant_block_list else get_block_names(self.model)
        if not all_blocks:
            raise ValueError("Could not find any blocks. Check the model or quant_block_list.")

        all_first_block_names = [block[0] for block in all_blocks]
        layer_names = _get_quantized_layer_names_outside_blocks(
            model=self.model_context.model,
            layer_config=self.quantizer.layer_config,
            supported_types=SUPPORTED_LAYER_TYPES,
            quant_block_list=self.quantizer.quant_block_list,
        )
        if self.quantize_config.is_act_quantize and (not self.quantize_config.act_dynamic or len(layer_names) > 0):
            if len(layer_names) > 0:
                logger.warning(
                    "quantize layers outside blocks for static activation quantizaiton"
                    " will significantly increase calibration time"
                )
            all_inputs = self.try_cache_inter_data_gpucpu(
                all_first_block_names, self.quantize_config.nsamples, layer_names
            )
        else:
            all_inputs = self.cache_inter_data(all_first_block_names, self.quantize_config.nsamples)

        # Clear hooks for multi-GPU setups
        if hasattr(self.model_context.model, "hf_device_map") and len(self.model_context.model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(self.model_context.model)

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

            clear_memory(self.inputs, device_list=self.compress_context.device_list)

            total_samples = len(inputs["input_ids"])
            if total_samples < self.quantize_config.batch_size:
                self.quantize_config.batch_size = total_samples
                logger.warning(f"Forcing batch size to {total_samples}")

            input_ids = to_device(inputs.pop("input_ids"), self.compress_context.cache_device)
            input_others = to_device(inputs, self.compress_context.cache_device)

            tmp_dtype = self.model_context.amp_dtype if self.model_context.amp else torch.float32
            input_ids = [id_.to(tmp_dtype) for id_ in input_ids]

            for key, val in input_others.items():
                if isinstance(val, torch.Tensor) and val.dtype in (torch.float16, torch.bfloat16):
                    input_others[key] = val.to(tmp_dtype)
                elif isinstance(val, list):
                    input_others[key] = [to_dtype(v, tmp_dtype) for v in val]

            for block_name in block_names:
                pbar.set_description(f"Quantizing {block_name}")
                block = get_module(self.model_context.model, block_name)

                self.quantizer.quantize_block(
                    block,
                    input_ids,
                    input_others,
                )

                if self.low_cpu_mem_usage and not self.is_immediate_saving:
                    self._offloader.offload(self.model_context.model, block_name)
                if block_name == block_names[-1]:
                    clear_memory(input_ids, device_list=self.compress_context.device_list)
                else:
                    clear_memory(device_list=self.compress_context.device_list)

                memory_monitor.log_summary()
                pbar.update(1)
        pbar.close()
        # Process remaining layers not in blocks
        # Collect names of quantizable layers not belonging to any block
        remain_layer_names = []
        block_name_set = set(name for block in all_blocks for name in block)
        for n, m in self.model_context.model.named_modules():
            if not check_to_quantized(m):
                continue
            # Skip if this layer is part of any block (by prefix match)
            if any(n == block_name or n.startswith(f"{block_name}.") for block_name in block_name_set):
                continue
            remain_layer_names.append(n)

        for name in remain_layer_names:
            dtype = None
            if self.super_group_size is not None:
                dtype = torch.float32
            self.quantizer.quantize_layer(name, dtype=dtype)

    @torch.inference_mode()
    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """

        self.post_init()
        self.model_context.initialize(formats=self.formats, is_act_quantize=self.config.is_act_quantize)

        formats = self.formats if isinstance(self.formats, list) else []
        if not (any(fmt.is_gguf() for fmt in formats) or self.super_bits is not None):
            self._quantize_embedding_layer()  # leave to gguf itself to handle

        # Release memory
        clear_memory(device_list=self.device_list)

        if self.quantize_config.is_act_quantize and check_need_act_calibration(
            self.quantize_config.act_dynamic,
            self.quantize_config.act_data_type,
            self.quantize_config.act_bits,
            self.static_kv_dtype,
            self.static_attention_dtype,
        ):
            model = self.model_context.model
            hook_handles = self.quantizer._register_act_max_hook(model)
            try:
                self._quantize_via_rtn_blockwise()
            except torch.OutOfMemoryError:
                logger.warning("Fallback to CPU. Consider using more GPUs via `--device 0,1,2,3`.")
                model = model.to("cpu")
                self.model_context.model = model
                clear_memory(device_list=self.device_list)
                if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                    import accelerate

                    accelerate.hooks.remove_hook_from_submodules(model)
                orig_device = self.compress_context.device
                self.compress_context.device = "cpu"
                self._quantize_via_rtn_blockwise()
                self.compress_context.device = orig_device
            for handle in hook_handles:
                handle.remove()
        else:
            # By default, we go with layer-wise way if no replacement happened.
            # In RTN mode (iters == 0), force blockwise quantization to avoid
            # full-model materialization and linear CPU RAM growth.
            use_blockwise_quantization = global_state.replaced_module_count > 0
            if not use_blockwise_quantization:
                logger.info(
                    "RTN mode detected (iters=0): force blockwise quantization to avoid "
                    "layer-wise full-model materialization."
                )
                use_blockwise_quantization = True
            tied_weights_keys = getattr(self.model, "_tied_weights_keys", [])
            if tied_weights_keys is None:
                tied_weights_keys = []
            if isinstance(tied_weights_keys, dict):
                tied_weights_values = list(tied_weights_keys.values())
            else:
                tied_weights_values = list(tied_weights_keys)
            tied_weights_layers = [".".join(val.split(".")[:-1]) for val in tied_weights_values]  # rm weight/bias
            # In fact, we should detect whether it is is_separate_lm_head, to simplify, we don't do it
            if hasattr(self, "formats") and self.formats[0].is_gguf():
                lm_head_name = get_lm_head_name(self.model)
                if lm_head_name is not None:
                    tied_weights_layers.append(lm_head_name)

            if use_blockwise_quantization:  # The ram usage is a little higher

                all_blocks = self.quant_block_list if self.quant_block_list else get_block_names(self.model)
                pbar = tqdm(range(sum(len(block) for block in all_blocks)))
                for block_names in all_blocks:
                    for block_name in block_names:
                        pbar.set_description(f"Quantizing {block_name}")
                        block = get_module(self.model, block_name)
                        self.quantizer.quantize_block(block, block_name=block_name)

                        if self.low_cpu_mem_usage and not self.is_immediate_saving:
                            self._offloader.offload(self.model, block_name)
                        clear_memory(device_list=self.device_list)
                        memory_monitor.log_summary()
                        pbar.update(1)
                cnt = 1
                remain_layer_names = []
                block_name_set = set(name for block in all_blocks for name in block)
                for n, m in self.model_context.model.named_modules():
                    if not check_to_quantized(m):
                        continue
                    # Skip if this layer is part of any block (by prefix match)
                    if any(n == block_name or n.startswith(f"{block_name}.") for block_name in block_name_set):
                        continue
                    remain_layer_names.append(n)
                for name in remain_layer_names:
                    logger.info(f"Quantizing remaining layer {name} on CPU.")
                    self.quantizer.quantize_layer(name)
                    cnt += 1
                    if cnt % 10 == 0:
                        clear_memory(device_list=self.device_list)
                        memory_monitor.log_summary()
            else:
                all_to_quantized_module_names: list[str] = [
                    n for n, m in self.model.named_modules() if check_to_quantized(m)
                ]
                all_to_quantized_module_names = all_to_quantized_module_names
                materialize_model_(self.model)
                self.model.to("cpu")
                block_names_cnt = len(flatten_list(get_block_names(self.model, True)))
                clear_mem_freq = len(all_to_quantized_module_names) // block_names_cnt
                cnt = 0
                pbar = tqdm(all_to_quantized_module_names)

                for n, m in self.model.named_modules():
                    if hasattr(m, "global_name") and m.global_name in all_to_quantized_module_names:
                        pbar.set_description(f"Quantizing {m.global_name}")
                        self.quantizer.quantize_layer(m.global_name)
                        cnt += 1
                        pbar.update()
                        if cnt % clear_mem_freq == 0:
                            clear_memory(device_list=self.device_list)
                            memory_monitor.log_summary()

                    elif (
                        not any(m.children())
                        and len(m.state_dict()) > 0
                        and n not in tied_weights_layers
                        and self.is_immediate_saving
                    ):
                        set_module(self.model, n, copy.deepcopy(m))
                        shard_writer(self, name=n)
                        m.to("meta")

        # Convert remaining fp8
        convert_module_to_hp_if_necessary(self.model, self.amp_dtype, self.device)
        if self.low_cpu_mem_usage:
            self._offloader.reload(self.model)
        if self.is_immediate_saving:
            shard_writer(self, is_finalize=True)

        self.quantized = True
        return self.model, self.layer_config
