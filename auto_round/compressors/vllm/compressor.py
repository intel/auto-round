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

from collections import defaultdict
from copy import deepcopy
from typing import Union, Any

import torch
from tqdm import tqdm

from auto_round.compressors.base import BaseCompressor
from auto_round.compressors.utils import block_forward
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.compressors.utils import (
    IndexSampler,
    block_forward,
    check_need_act_calibration,
    check_skippable_keywords,
    collect_best_params,
    get_shared_keys,
    immediate_saving,
    infer_bits_by_data_type,
    init_cache,
    is_mx_fp,
    is_nv_fp,
    is_standard_fp,
    is_static_wfp8afp8,
    is_wfp8afp8,
    reset_params,
    set_layer_config,
)
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_DTYPES,
    SUPPORTED_LAYER_TYPES,
    TORCH_VERSION_AT_LEAST_2_6,
    CpuInfo,
    check_and_mark_fp8_model,
    check_seqlen_compatible,
    check_to_quantized,
    check_vllm_installed,
    clear_memory,
    compile_func,
    convert_dtype_str2torch,
    convert_fp8_layer_to_linear,
    convert_fp8_model_to_16b_model,
    copy_python_files_from_model_cache,
    detect_device,
    extract_block_names_to_str,
    find_matching_blocks,
    flatten_list,
    get_block_names,
    get_layer_names_in_block,
    get_module,
    is_auto_device_mapping,
    is_debug_mode,
    is_fp8_linear,
    is_fp8_model,
    is_moe_model,
    llm_load_model,
    memory_monitor,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
    set_module,
    to_device,
    to_dtype,
    unsupported_meta_device,
    vllm_load_model,
)


class VllmCompressor(BaseCompressor):
    """Class for automatic rounding-based quantization with Diffusion models.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data, is not used for diffusion models.
        platform (str): The platform to load pretrained moded, options: ["hf", "model_scope"]
        guidance_scale (float): Control how much the image generation process follows the text prompt.
                                The more it is, the more closely it follows the prompt (default is 7.5).
        num_inference_steps (int): The reference number of denoising steps (default is 50).
        generator_seed (int): A sees that controls the initial noise from which an image is generated (default is None).
        scheme: (str| dict | QuantizationScheme ): A preset scheme that defines the quantization configurations.
        layer_config (dict): Configuration for weight quantization (default is None).
        dataset: The path or name of the calib dataset.
        iters (int): Number of iterations (default is 200).
        seqlen (int): Length of the sequence.
        nsamples (int): Number of samples (default is 128).
        batch_size (int): Batch size for training (default is 8).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is False).
        device_map (str | dict | int | torch.device, optional): Device placement map. Defaults to 0.
        enable_torch_compile (bool): Whether to enable torch compile to optimize quant_block/layer
        **kwargs: Additional keyword arguments.
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
        model: Union[object, str],
        tokenizer=None,
        platform: str = "hf",
        scheme: Union[str, dict, QuantizationScheme] = "W4A16",
        layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "coco2014",
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
        check_vllm_installed()
        from vllm import LLM

        logger.warning("Vllm model quantization is experimental.")
        self.llm, self.model = vllm_load_model(model)

        to_quant_block_names: Union[str, list, None] = kwargs.pop("to_quant_block_names", None)

        all_blocks = get_block_names(self.model)
        self.quant_block_list = find_matching_blocks(model, all_blocks, to_quant_block_names)
        if to_quant_block_names is None:
            to_quant_block_names = extract_block_names_to_str(self.quant_block_list)

        if iters != 0:
            logger.warning(
                "Currently vllm format model doesn't support tuning (iters > 0)"
            )
            iters = 0

        if nsamples % batch_size != 0:
            nsamples = (nsamples // batch_size + 1) * batch_size
            logger.warning(f"'nsamples' is not divisible by 'batch_size', will adjusted to {nsamples}")

        seqlen = 2048 if seqlen is None else seqlen

        kwargs["vllm"] = True
        super(VllmCompressor, self).__init__(
            model=self.model,
            tokenizer=self.llm.llm_engine.input_processor.tokenizer,
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

    @torch.inference_mode()
    def _quantize_rtn(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize all modules in the model using RTN (Round-To-Nearest) strategy.

        If the target format includes GGUF with `k`, and optimized RTN is enabled,
        blockwise quantization with input caching and imatrix is used.

        Returns:
            tuple[nn.Module, Dict[str, Any]]: The quantized model and the layer configuration.
        """
        all_to_quantized_module_names: list[str] = [n for n, m in self.model.named_modules() if check_to_quantized(m)]
        if is_nv_fp(self.data_type):
            from auto_round.data_type.nvfp import calculate_gparam
            from auto_round.data_type.utils import update_fused_layer_global_scales

            pbar = tqdm(all_to_quantized_module_names)
            for name in pbar:
                pbar.set_description(f"Calculate weight global scale: {name}")
                m = get_module(self.model, name)
                if is_fp8_linear(m):
                    m = convert_fp8_layer_to_linear(m, self.amp_dtype, self.device)
                    set_module(self.model, name, m)
                weight_global_scale = calculate_gparam(m.weight, self.group_size)
                setattr(m, "weight_global_scale", weight_global_scale)

            logger.info("Start to update fused layer global scales, it may take some time.")
            for name, module in self.model.named_modules():
                update_fused_layer_global_scales(module)
            logger.info("Finished updating fused layer global scales.")

        # Release memory
        clear_memory(device_list=self.device_list)

        enable_imatrix = False
        if not self.disable_opt_rtn and self.data_type == "int" and self.sym:
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
            self.calib(self.nsamples, self.batch_size)
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
                    clear_memory(device_list=self.device_list)
                    cnt = 1
                cnt += 1
        # Convert remaining fp8
        if is_fp8_model(self.model):
            convert_fp8_model_to_16b_model(self.model, self.amp_dtype, self.device)
        self.quantized = True
        import pdb;pdb.set_trace()
        return self.model, self.layer_config

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

        # self._quantize_via_rtn_blockwise(all_to_quantized_module_names)
        all_to_quantized_module_names = list(set(all_to_quantized_module_names))
        pbar = tqdm(range(sum(len(block) for block in all_blocks)))

        all_blocks = self.quant_block_list if self.quant_block_list else get_block_names(self.model)
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
            # clear_memory(device_list=self.device_list)

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
            max_len = (total_cnt // self.batch_size) * self.batch_size
        # torch.cuda.empty_cache()

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
        compressed_model = super().save_quantized(output_dir=output_dir, format=format, inplace=inplace, **kwargs)
        return compressed_model
