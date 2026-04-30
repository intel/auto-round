# Copyright (c) 2023 Intel Corporation
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
import os
import sys
import time
import traceback
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, fields
from functools import partial
from typing import Any, Callable, Optional, Union

import accelerate
import torch
from accelerate.big_modeling import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory, get_max_memory
from packaging import version
from torch import autocast
from tqdm import tqdm
from transformers import AutoConfig, set_seed

from auto_round import envs
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.compressors.base import BaseCompressor
from auto_round.compressors.shard_writer import shard_writer
from auto_round.compressors.utils import (
    IndexSampler,
    block_forward,
    check_need_act_calibration,
    check_skippable_keywords,
    collect_best_params,
    get_shared_keys,
    infer_bits_by_data_type,
    init_cache,
    is_block_wfp8,
    is_dynamic_afp8,
    is_dynamic_wint8aint8,
    is_mx_fp,
    is_nv_fp,
    is_static_wfp8afp8,
    is_wfp8afp8,
    reset_params,
    set_layer_config,
)
from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, update_block_global_scale_if_needed
from auto_round.experimental.transform.hadamard_config import HadamardConfig
from auto_round.export.export_to_gguf.config import GGUF_INNER_CONFIG
from auto_round.formats import OutputFormat, get_formats
from auto_round.logger import logger
from auto_round.modeling.replace_modules import materialize_model_, safe_to_cpu_
from auto_round.modeling.unfused_moe import apply_model_monkey_patches
from auto_round.schemes import (
    QuantizationScheme,
    _handle_special_schemes,
    get_gguf_scheme,
    preset_name_to_scheme,
)
from auto_round.special_model_handler import (
    _handle_special_model,
)
from auto_round.sign_sgd import SignSGD
from auto_round.special_model_handler import get_predefined_ignore_layers, update_module
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_DTYPES,
    SUPPORTED_LAYER_TYPES,
    TORCH_VERSION_AT_LEAST_2_6,
    CpuInfo,
    check_and_mark_quantized_module,
    check_seqlen_compatible,
    check_to_quantized,
    check_vllm_installed,
    clear_memory,
    compile_func,
    compress_layer_names,
    convert_dtype_str2torch,
    convert_module_to_hp_if_necessary,
    detect_device,
    extract_block_names_to_str,
    find_matching_blocks,
    flatten_list,
    get_block_names,
    get_layer_names_in_block,
    get_lm_head_name,
    get_module,
    global_state,
    htcore,
    is_auto_device_mapping,
    is_debug_mode,
    is_hpex_available,
    is_moe_model,
    is_moe_model_via_config,
    is_quantized_input_module,
    llm_load_model,
    memory_monitor,
    mv_module_from_gpu,
    normalize_no_split_modules,
    set_amax_for_all_moe_layers,
    set_module,
    to_device,
    to_dtype,
    unsupported_meta_device,
    vllm_load_model,
)
from auto_round.utils.device import (
    clear_memory_if_reached_threshold,
    get_major_device,
    parse_available_devices,
    set_auto_device_map_for_block_with_tuning,
    set_non_auto_device_map,
)
from auto_round.utils.distributed import setup_ddp_if_needed_
from auto_round.utils.offload import OffloadManager
from auto_round.wrapper import WrapperLinear, WrapperMultiblock, unwrapper_block, unwrapper_layer, wrapper_block

SERIALIZATION_KEYS = (
    "bits",
    "act_bits",
    "data_type",
    "act_data_type",
    "group_size",
    "act_group_size",
    "sym",
    "act_sym",
    "act_dynamic",
    "amp",
    "batch_size",
    "enable_minmax_tuning",
    "enable_norm_bias_tuning",
    "enable_quanted_input",
    "gradient_accumulate_steps",
    "iters",
    "lr",
    "low_gpu_mem_usage",
    "minmax_lr",
    "nsamples",
    "quant_block_list",
    "regex_config",
    "scale_dtype",
    "seqlen",
    "supported_types",
    "static_attention_dtype",
    "static_kv_dtype",
    "super_bits",
    "super_group_size",
    "to_quant_block_names",
    "hadamard_config",
)


class VllmCompressor(BaseCompressor):
    """Vllm compressor for vllm loading model quantization

    Attributes:
        model (torch.nn.Module): The loaded PyTorch model in eval mode.
        tokenizer: Tokenizer used to prepare input text for calibration/tuning.
        platform (str): The platform to load pretrained moded, options: ["hf", "model_scope"]
        bits (int): Weight quantization bits.
        group_size (int): Per-group size for weight quantization.
        sym (bool): Whether to use symmetric weight quantization.
        layer_config (dict): Per-layer quantization configuration.
        nsamples (int): Number of calibration samples.
        enable_torch_compile (bool): Whether to enable compile_func for quant blocks/layers.
    """

    bits: int | None
    group_size: int | None
    sym: bool | None
    data_type: str | None
    act_bits: int | None
    act_group_size: int | None
    act_sym: bool | None
    act_data_type: str | None
    act_dynamic: bool | None
    super_bits: int | None
    super_group_size: int | None

    def __init__(
        self,
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform="hf",
        scheme: Union[str, dict, QuantizationScheme, AutoScheme] = "W4A16",
        layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
        iters: int = 200,
        seqlen: int = 2048,
        nsamples: int = 128,
        batch_size: int = 8,
        gradient_accumulate_steps: int = 1,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        """Initialize AutoRound with quantization and tuning configuration.

        Args:
            model (torch.nn.Module | str): Model object or model name to load.
            tokenizer: Tokenizer for text processing. Required if `model` is not a string and `iters > 0`.
            scheme (str| dict | QuantizationScheme ): A preset scheme that defines the quantization configurations
            bits (int, optional): Weight quantization bits. Defaults to 4.
            group_size (int, optional): Weight quantization group size. Defaults to 128.
            sym (bool, optional): Symmetric weight quantization. Defaults to True.
            layer_config (dict, optional): Layer-wise quantization config. Defaults to None.
            batch_size (int, optional): Calibration batch size. Defaults to 8.
            amp (bool, optional): Use AMP for tuning. Defaults to True.
            device (str | torch.device | int, optional): Compute device. Defaults to 0.
            dataset (str | list | tuple | DataLoader, optional): Calibration data. Defaults to "NeelNanda/pile-10k".
            enable_minmax_tuning (bool, optional): Enable weight min-max tuning. Defaults to True.
            lr (float, optional): Learning rate; if None, set to 1.0 / iters except when iters==0.
            minmax_lr (float, optional): Learning rate for min-max tuning; defaults to `lr`.
            low_gpu_mem_usage (bool, optional): Lower GPU memory mode. Defaults to False.
            low_cpu_mem_usage (bool, optional): Lower CPU memory mode. Defaults to False.
            iters (int, optional): Optimization iterations. Defaults to 200.
            seqlen (int, optional): Calibration sequence length. Defaults to 2048.
            nsamples (int, optional): Number of calibration samples. Defaults to 128.
            seed (int, optional): Random seed. Defaults to 42.
            gradient_accumulate_steps (int, optional): Gradient accumulation steps. Defaults to 1.
            data_type (str, optional): Weight data type string, e.g., "int". Defaults to "int".
            act_bits (int, optional): Activation quantization bits. Defaults to 16.
            act_group_size (int, optional): Activation group size. Defaults to None.
            act_sym (bool, optional): Symmetric activation quantization. Defaults to None.
            act_data_type (str, optional): Activation data type; inherits weight dtype if None and act_bits < 16.
            act_dynamic (bool, optional): Dynamic activation quantization. Defaults to True.
            enable_torch_compile (bool, optional): Enable torch.compile for quant blocks/layers. Defaults to False.
            device_map (str | dict, optional): Device placement map. Defaults to None.
            disable_opt_rtn (bool, optional): Disable RTN-mode optimization (iters=0). Defaults to None.
            enable_alg_ext (bool, optional): Enable algorithm extension (primarily for INT2). Defaults to False.
            **kwargs: Backward compatible options:
                - enable_alg_ext, quant_lm_head, lr, lr_scheduler, not_use_best_mse, dynamic_max_gap,
                  super_group_size, super_bits, scale_dtype ("fp16" etc.),
                  nblocks, to_quant_block_names,
                  enable_norm_bias_tuning, enable_quanted_input, enable_opt_rtn,
                  disable_deterministic_algorithms, mllm, static_kv_dtype,enable_deterministic_algorithms,momentum
        Raises:
            ValueError: If invalid device is provided or tokenizer is missing for non-str model with iters > 0.
            RuntimeError: If model parameters are on meta device.
        Example:
            Layer-wise configuration structure:

            >>> layer_config = {
            ...     "layer1": {
            ...         "bits": 3,
            ...         "group_size": 128,
            ...         "sym": True,
            ...     },
            ...     "layer2": {
            ...         "W8A16"
            ...      }
            ...     # ...
            ... }
        """
        check_vllm_installed()
        from vllm import LLM

        logger.warning("vllm model quantization is experimental.")
        self.llm, model, tokenizer = vllm_load_model(model)
        check_and_mark_quantized_module(model)

        all_blocks = get_block_names(model)
        to_quant_block_names: Union[str, list, None] = kwargs.pop("to_quant_block_names", None)
        self.quant_block_list = find_matching_blocks(model, all_blocks, to_quant_block_names)
        if to_quant_block_names is None:
            to_quant_block_names = extract_block_names_to_str(self.quant_block_list)

        if iters != 0:
            logger.warning(
                "Currently vllm format model doesn't support tuning (iters > 0)"
            )
            iters = 0

        batch_size = 1
        model = _handle_special_model(model)
        kwargs["vllm_loading"] = True
        super(VllmCompressor, self).__init__(
            model=model,
            tokenizer=tokenizer,
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
        logger.info("start to compute imatrix")

        # Load dataset
        from auto_round.calib_dataset import get_dataloader

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
                if type(module) in self.supported_types and check_to_quantized(module):
                    hook = module.register_forward_hook(get_imatrix_hook)
                    hook_handles.append(hook)
            return hook_handles

        hooks = register_act_hook(model)
        self.calib(self.nsamples, self.batch_size)
        for hook in hooks:
            hook.remove()

        all_to_quantized_module_names = list(set(all_to_quantized_module_names))

        all_blocks = self.quant_block_list if self.quant_block_list else get_block_names(self.model)
        pbar = tqdm(range(sum(len(block) for block in all_blocks)))

        if not all_blocks:
            raise ValueError("Could not find any blocks. Check the model or quant_block_list.")
        for block_names in all_blocks:
            for block_name in block_names:
                pbar.set_description(f"Quantizing {block_name}")
                block = get_module(self.model, block_name)
                if is_nv_fp(self.act_data_type) or is_static_wfp8afp8(self):
                    # enable moe experts act_max automatic generation for Linear
                    set_amax_for_all_moe_layers(block, attr_name="act_max")
                for _, m in block.named_modules():
                    # fix issue: Ling-flash-2.0-q2_k_s fail infer on cuda but well on cpu
                    # https://huggingface.co/Intel/Ling-flash-2.0-gguf-q2ks-mixed-AutoRound/discussions/1
                    if hasattr(m, "imatrix"):
                        m.imatrix /= m.imatrix_cnt
                    if hasattr(m, "tmp_name") and m.tmp_name in all_to_quantized_module_names:
                        self._quantize_layer_via_rtn(m.tmp_name, to_cpu=self.low_gpu_mem_usage)
                        all_to_quantized_module_names.remove(m.tmp_name)
                memory_monitor.log_summary()
                pbar.update(1)
        pbar.close()

        # Process remaining layers not in blocks
        for name in all_to_quantized_module_names:
            dtype = None
            if self.super_group_size is not None:
                dtype = torch.float32
            self._quantize_layer_via_rtn(name, dtype=dtype)

    # Use no_grad instead of inference mode
    # https://github.com/intel/auto-round/issues/1620
    @torch.no_grad()
    def _quantize_rtn(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize all modules in the model using RTN (Round-To-Nearest) strategy.

        If the target format includes GGUF with `k`, and optimized RTN is enabled,
        blockwise quantization with input caching and imatrix is used.

        Returns:
            tuple[nn.Module, Dict[str, Any]]: The quantized model and the layer configuration.
        """
        if self.amp and self.model.dtype != self.amp_dtype:
            self.model.to(self.amp_dtype)

        all_to_quantized_module_names: list[str] = [n for n, m in self.model.named_modules() if check_to_quantized(m)]
        self.all_to_quantized_module_names = all_to_quantized_module_names

        if not (any(fmt.is_gguf() for fmt in getattr(self, "formats", [])) or self.super_bits is not None):
            self._quantize_embedding_layer()  # leave to gguf itself to handle

        # Release memory
        clear_memory(device_list=self.device_list)

        enable_imatrix = False
        if not self.disable_opt_rtn:
            has_gguf_k = (
                any(fmt.is_gguf() and "k" in fmt.output_format for fmt in getattr(self, "formats", []))
                or self.super_bits is not None
            )
            if has_gguf_k:
                enable_imatrix = True
            elif self.data_type == "int" and self.sym:
                enable_imatrix = True
        if enable_imatrix:
            self._quant_rtn_with_imatrix(all_to_quantized_module_names)
        elif self.act_bits <= 8 and check_need_act_calibration(
            self.act_dynamic,
            self.act_data_type,
            self.act_bits,
            self.static_kv_dtype,
            self.static_attention_dtype,
        ):  # TODO, mixed datatype has bug
            hook_handles = self._register_act_max_hook(self.model)
            all_blocks = self.quant_block_list or get_block_names(self.model)
            pbar = tqdm(range(sum(len(block) for block in all_blocks)))
            for block_names in all_blocks:
                for block_name in block_names:
                    pbar.set_description(f"Quantizing {block_name}")
                    block = get_module(self.model, block_name)

                    block = convert_module_to_hp_if_necessary(block, dtype=self.amp_dtype, device=self.device)
                    update_block_global_scale_if_needed(block, self.data_type, self.group_size)

            self.calib(self.nsamples, self.batch_size)
            for handle in hook_handles:
                handle.remove()

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

        # materialize_model_(self.model)
        block_names_cnt = len(flatten_list(get_block_names(self.model, True)))
        clear_mem_freq = len(all_to_quantized_module_names) // block_names_cnt
        cnt = 0
        pbar = tqdm(all_to_quantized_module_names)

        for n, m in self.model.named_modules():
            if hasattr(m, "global_name") and m.global_name in all_to_quantized_module_names:
                pbar.set_description(f"Quantizing {m.global_name}")
                self._quantize_layer_via_rtn(m.global_name)
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

    @torch.no_grad()
    def calib(self, nsamples, bs):
        """Perform calibration for quantization.

        This method calibrates the model for quantization by processing a specified
        number of samples from the calibration dataset. It ensures that the data is
        properly formatted and feeds it to the model. If the number of samples processed
        is less than the specified number, it logs a warning. If no samples are processed,
        it logs an error and exits.
        Args:
            nsamples (int): The number of samples to use for calibration.
            bs (int): The number of samples to use for calibration
        """
        from auto_round.calib_dataset import get_dataloader

        need_attention_mask = True
        if isinstance(self.dataset, str):
            need_attention_mask = False  # all supported datasets does not use pad
            dataset = self.dataset.replace(" ", "")  ##remove all whitespaces

            # slow here
            self.dataloader = get_dataloader(
                self.tokenizer,
                self.seqlen,
                dataset,
                self.seed,
                bs,
                self.nsamples,
            )
        else:
            self.dataloader = self.dataset
        total_cnt = 0
        total = nsamples if not hasattr(self.dataloader, "len") else min(nsamples, len(self.dataloader))

        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        with tqdm(range(1, total + 1), desc="cache block inputs") as pbar:
            for prompts in self.dataloader:
                prompts = self.tokenizer.batch_decode(prompts["input_ids"])
                try:
                    self.llm.generate(prompts, sampling_params)
                except NotImplementedError:
                    pass
                except Exception as error:
                    raise error
                if isinstance(prompts, list):
                    step = len(prompts)
                elif isinstance(prompts, str):
                    step = 1
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

    def save_quantized(
        self,
        output_dir: str = None,
        format: Union[str, list[OutputFormat]] = "auto_round",
        inplace: bool = True,
        **kwargs,
    ) -> torch.nn.Module:
        """Save the quantized model to the specified output directory in the specified format.

        Args:
            output_dir (str, optional): The directory to save the quantized model. Defaults to None.
            format (str, optional): The format in which to save the model. Defaults to "auto_round".
            inplace (bool, optional): Whether to modify the model in place. Defaults to True.
            **kwargs: Additional keyword arguments specific to the export format.

        Returns:
            object: The compressed model object.
        """
        import os
        import json
        from functools import partial
        from huggingface_hub import split_torch_state_dict_into_shards
        from safetensors.torch import save_file, _remove_duplicate_names

        def save_pretrained(model_to_save, save_directory, max_shard_size="5GB", safe_serialization=True, **kwargs):
            if (
                hasattr(model_to_save.config, "rope_parameters")
                and model_to_save.config.rope_parameters.get("type", None) is not None
                and model_to_save.config.rope_parameters.get("rope_type", None) is not None
                and model_to_save.config.rope_parameters["type"] != model_to_save.config.rope_parameters["rope_type"]
            ):
                model_to_save.config.rope_parameters.pop("rope_type")
            model_to_save.config.save_pretrained(save_directory)

            state_dict = model_to_save.state_dict()
            duplicates = _remove_duplicate_names(state_dict)
            metadata = {}
            metadata["format"] = "pt"

            for kept_name, duplicate_group in duplicates.items():
                for duplicate_name in duplicate_group:
                    state_dict[duplicate_name] = state_dict[kept_name].clone()

            filename_pattern = "model{suffix}.safetensors"

            state_dict_split = split_torch_state_dict_into_shards(
                state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
            )

            if state_dict_split.is_sharded:
                total_params = list(model_to_save.named_parameters())
                total_numel = []
                for _, param in total_params:
                    total_numel.append(param.numel())
                index = {
                    "metadata": {"total_parameters": sum(total_numel), **state_dict_split.metadata},
                    "weight_map": state_dict_split.tensor_to_filename,
                }
                for shard_file, tensor_names in state_dict_split.filename_to_tensors.items():
                    filename = os.path.join(save_directory, shard_file)
                    shard_state_dict = {}
                    for tensor_name in tensor_names:
                        # Get the tensor, and remove it from state_dict to avoid keeping the ref
                        tensor = state_dict.pop(tensor_name)
                        shard_state_dict[tensor_name] = tensor.contiguous()
                    save_file(shard_state_dict, filename, metadata=metadata)
                del shard_state_dict

                save_index_file = os.path.join(save_directory, "model.safetensors.index.json")
                # Save the index as well
                with open(save_index_file, "w", encoding="utf-8") as f:
                    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                    f.write(content)
            else:
                filename = os.path.join(save_directory, list(state_dict_split.filename_to_tensors.keys())[0])
                save_file(state_dict, filename, metadata=metadata)

        print(self.model)
        self.model.save_pretrained = partial(save_pretrained, self.model)
        compressed_model = super().save_quantized(output_dir=output_dir, format=format, inplace=inplace, **kwargs)
        return compressed_model
