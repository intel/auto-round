# Copyright (c) 2024 Intel Corporation
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

from typing import Optional, Union
from tqdm import tqdm
from copy import deepcopy
from transformers import set_seed
import torch
from tqdm import tqdm, trange

from ..utils import (
    detect_device,
    to_device,
    to_dtype,
    clear_memory,
    supported_layer_types
)

from wrapper_block import WapperBasicTransformerBlock, WapperResBlock, WapperQKMatMul, WapperSMVMatMul, WapperBasicTransformerBlock, WapperAttnBlock, get_specials
from ldm.modules.diffusionmodules.openaimodel import ResBlock
from ldm.modules.attention import BasicTransformerBlock
from ..wrapper_layer import WrapperMultiblock, wrapper_block, unwrapper_block, WrapperLinear, unwrapper_layer
from ..autoround import AutoRoundDM, AdaRoundDM

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import gc
import numpy as np
import logging
logger = logging.getLogger("autoround")

# class AutoRoundDiffusion(AutoRoundDM):
#     """Class for automatic rounding-based quantization with MLLMs.
    
#     Args:
#         model: The PyTorch model to be quantized.
#         tokenizer: An optional tokenizer for processing input data.
#         processor: Any multi-modal model will require an object to encode or
#                    decode the data that groups several modalities (among text, vision and audio).
#         image_processor: Image processor for special model like llava.
#         bits (int): Number of bits for quantization (default is 4).
#         group_size (int): Size of the quantization group (default is 128).
#         sym (bool): Whether sym to be used (default is True).
#         layer_config (dict): Configuration for weight quantization (default is None).
#         batch_size (int): Batch size for training (default is 8).
#         amp (bool): Whether to use automatic mixed precision (default is True).
#         device: The device to be used for training (default is "auto").
#         lr_scheduler: The learning rate scheduler to be used.
#         dataset: The path or name of the calib dataset.
#         extra_data_dir: The path of extra data such as images, audio and videos.
#         template: The path or name of template used to specify process for different MLLMs.
#         quant_nontext_module: Whether to quantize nontext module.
#         enable_quanted_input (bool): Whether to use quantized input data (default is True).
#         enable_minmax_tuning (bool): Whether to enable min-max tuning (default is True).
#         lr (float): The learning rate (default is 0.005).
#         minmax_lr (float): The learning rate for min-max tuning (default is None).
#         low_gpu_mem_usage (bool): Whether to use low GPU memory (default is False).
#         low_cpu_mem_usage (bool): Whether to use low CPU memory (default is False).
#         iters (int): Number of iterations (default is 200).
#         seqlen (int): Length of the sequence.
#         nsamples (int): Number of samples (default is 128).
#         sampler (str): The sampling method (default is "rand").
#         seed (int): The random seed (default is 42).s
#         nblocks (int): Number of blocks (default is 1).
#         gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
#         not_use_best_mse (bool): Whether to use mean squared error (default is False).
#         dynamic_max_gap (int): The dynamic maximum gap (default is -1).
#         data_type (str): The data type to be used (default is "int").
#         scale_dtype (str): The data type of quantization scale to be used (default is "float16"), different kernels
#                            have different choices.
#         act_bits (int): Number of bits for activation quantization. Default is 32.
#         act_group_size (int): Group size for activation quantization. Default is None.
#         act_sym (bool): Whether to use symmetric activation quantization. Default is None.
#         act_dynamic (bool): Whether to use dynamic activation quantization. Default is True.
#         to_quant_block_names (str|list): A string or list whose elements are list of 
#                             block's layer names to be quantized.
#         enable_torch_compile (bool): Whether to enable torch compile to optimize quant_block/layer
#         **kwargs: Additional keyword arguments.


#     """

#     def __init__(
#             self,
#             model: torch.nn.Module,
#             # tokenizer,
#             # processor = None,
#             # image_processor = None,
#             bits: int = 4,
#             level: str = 'group_size',
#             group_size: int = 128,
#             sym: bool = True,
#             layer_config: dict = None,
#             batch_size: int = 8,
#             amp: bool = True,
#             device: str = None,
#             lr_scheduler=None,
#             dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = None,
#             extra_data_dir: str = None,
#             quant_nontext_module: bool = False,
#             enable_quanted_input: bool = True,
#             enable_minmax_tuning: bool = True,
#             lr: float = None,
#             minmax_lr: float = None,
#             low_gpu_mem_usage: bool = False,
#             low_cpu_mem_usage: bool = False,
#             iters: int = 200,
#             seqlen: int = None,
#             nsamples: int = 128,
#             sampler: str = "rand",
#             seed: int = 42,
#             nblocks: int = 1,
#             gradient_accumulate_steps: int = 1,
#             not_use_best_mse: bool = False,
#             dynamic_max_gap: int = -1,
#             data_type: str = "int",
#             scale_dtype: str = "fp16",
#             act_bits: int = 32,
#             act_group_size: int = None,
#             act_sym: bool = None,
#             act_dynamic: bool = True,
#             to_quant_block_names: Union[str, list] = None,
#             enable_norm_bias_tuning: bool = False,
#             truncation: bool = None,
#             enable_torch_compile: bool = False,
#             model_kwargs: dict = None,
#             **kwargs,
#     ):
#         all_blocks = get_block_names(model, quant_nontext_module)
#         self.quant_block_list = find_matching_blocks(model, all_blocks, to_quant_block_names)
#         if to_quant_block_names is None:
#             to_quant_block_names = extract_block_names_to_str(self.quant_block_list)
#         self.to_quant_block_names = to_quant_block_names
#         self.extra_data_dir = extra_data_dir
#         self.quant_nontext_module = quant_nontext_module
#         self.processor = processor
#         self.image_processor = image_processor
#         self.template = template if template is not None else model.config.model_type
#         if not isinstance(dataset, torch.utils.data.DataLoader):
#             self.template = get_template(
#                 self.template, model=model, tokenizer=tokenizer, processor=processor, image_processor=image_processor)
#             dataset = self.template.default_dataset if dataset is None else dataset

#         model = _handle_special_model(model)

#         from ..calib_dataset import CALIB_DATASETS
#         from .mllm_dataset import MLLM_DATASET
#         if isinstance(dataset, str):
#             if quant_nontext_module or \
#                 (dataset in CALIB_DATASETS.keys() and not \
#                  _only_text_test(model, tokenizer, device, self.template.model_type)):
#                 if quant_nontext_module:
#                     logger.warning(f"Text only dataset cannot be used for calibrating non-text modules,"
#                                 "switching to liuhaotian/llava_conv_58k")
#                 else:
#                     logger.warning(f"{model.config.model_type} not support for {dataset},"
#                              " will use liuhaotian/llava_conv_58k with default config as an alternative.")
#                 dataset = "liuhaotian/llava_conv_58k"

#             if dataset in MLLM_DATASET.keys():
#                 truncation = False
#                 seqlen = 512 if seqlen is None else seqlen
#                 if batch_size != 1:
#                     logger.warning(
#                         f"reset batch_size({batch_size}) to 1 and "
#                         f"gradient_accumulate_steps({gradient_accumulate_steps}) "
#                         f"to {batch_size * gradient_accumulate_steps}, "
#                         f"because batch_size={batch_size} cannot be used for {dataset}")
#                     gradient_accumulate_steps = batch_size * gradient_accumulate_steps
#                     batch_size = 1
#         if quant_nontext_module and batch_size != 1:
#             logger.warning(
#                 f"reset batch_size({batch_size}) to 1 and "
#                 f"gradient_accumulate_steps({gradient_accumulate_steps}) "
#                 f"to {batch_size * gradient_accumulate_steps}, "
#                 f"because batch_size={batch_size} cannot be used for calibrating non-text modules.")
#             gradient_accumulate_steps = batch_size * gradient_accumulate_steps
#             batch_size = 1
#         seqlen = 2048 if seqlen is None else seqlen
#         truncation = True if truncation is None else truncation
#         self.truncation = truncation

#         if nsamples % batch_size != 0:
#             nsamples = (nsamples // batch_size + 1) * batch_size
#             logger.warning(f"'nsamples' is not divisible by 'batch_size', will adjusted to {nsamples}")

#         super(AutoRoundMLLM, self).__init__(
#             model=model,
#             tokenizer=tokenizer,
#             bits=bits,
#             group_size=group_size,
#             sym=sym,
#             layer_config=layer_config,
#             batch_size=batch_size,
#             amp=amp,
#             device=device,
#             lr_scheduler=lr_scheduler,
#             dataset=dataset,
#             enable_quanted_input=enable_quanted_input,
#             enable_minmax_tuning=enable_minmax_tuning,
#             lr=lr,
#             minmax_lr=minmax_lr,
#             low_gpu_mem_usage=low_gpu_mem_usage,
#             low_cpu_mem_usage=low_cpu_mem_usage,
#             iters=iters,
#             seqlen=seqlen,
#             nsamples=nsamples,
#             sampler=sampler,
#             seed=seed,
#             nblocks=nblocks,
#             gradient_accumulate_steps=gradient_accumulate_steps,
#             not_use_best_mse=not_use_best_mse,
#             dynamic_max_gap=dynamic_max_gap,
#             data_type=data_type,
#             scale_dtype=scale_dtype,
#             act_bits=act_bits,
#             act_group_size=act_group_size,
#             act_sym=act_sym,
#             act_dynamic=act_dynamic,
#             to_quant_block_names=self.to_quant_block_names,
#             enable_norm_bias_tuning=enable_norm_bias_tuning,
#             enable_torch_compile=enable_torch_compile,
#             **kwargs,
#         )

#     def calib(self, nsamples, bs):
#         """Perform calibration for quantization.

#         This method calibrates the model for quantization by processing a specified
#         number of samples from the calibration dataset. It ensures that the data is
#         properly formatted and feeds it to the model. If the number of samples processed
#         is less than the specified number, it logs a warning. If no samples are processed,
#         it logs an error and exits.
#         Args:
#             nsamples (int): The number of samples to use for calibration.
#             bs (int): The number of samples to use for calibration
#         """
#         if isinstance(self.dataset, str):
#             dataset = self.dataset.replace(" ", "")
#             self.dataloader, self.batch_size, self.gradient_accumulate_steps = get_mllm_dataloader(
#                 template=self.template,
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 processor=self.processor,
#                 image_processor=self.image_processor,
#                 dataset=dataset,
#                 extra_data_dir=self.extra_data_dir,
#                 seqlen=self.seqlen,
#                 bs=self.batch_size,
#                 seed=self.seed,
#                 truncation=self.truncation,
#                 nsamples=self.nsamples,
#                 gradient_accumulate_steps=self.gradient_accumulate_steps,
#                 quant_nontext_module=self.quant_nontext_module
#             )
#         else:
#             self.dataloader = self.dataset
#         total_cnt = 0

#         if self.low_cpu_mem_usage:
#             embed_layers = get_layers_before_block(self.model)
#             for n, m in embed_layers:
#                 m = m.to(self.device)

#         total = nsamples if not hasattr(self.dataloader, "len") else min(nsamples, len(self.dataloader))
#         with tqdm(range(1, total + 1), desc="cache block inputs") as pbar:
#             for data in self.dataloader:
#                 if data is None:
#                     pbar.update(1)
#                     continue
#                 if isinstance(data, torch.Tensor):
#                     input_ids = data.to(self.device)
#                     data_new = input_ids
#                 elif isinstance(data, str):
#                     if self.tokenizer is None:
#                         logger.error("please provide tokenizer for string input")
#                         exit()
#                     # data = self.template._encode(data)
#                     data = self.template.processor.get_input(
#                         text=data,
#                         images=None,
#                         max_length=self.seqlen,
#                         squeeze=False,
#                     )
#                     data_new = {}
#                     for key in data.keys():
#                         data_new[key] = data[key].to(self.device)
#                     input_ids = data_new["input_ids"]
#                 elif isinstance(data, dict) and "text" in data.keys():
#                     text = data['text']
#                     if isinstance(text, dict):
#                         text = [text]
#                     input_text = self.template._encode(text)
#                     data = self.template.processor.get_input(
#                         text=input_text,
#                         images=data["image"],
#                         max_length=self.seqlen,
#                         squeeze=False,
#                     )
#                     data_new = {}
#                     for key in data.keys():
#                         data_new[key] = torch.tensor(data[key])
#                         data_new[key] = to_device(data_new[key], self.model.device)
#                         if key == 'images':
#                             data_new[key] = to_dtype(data_new[key], self.model.dtype)
#                     input_ids = data_new["input_ids"]
#                 elif isinstance(data, tuple) or isinstance(data, list):
#                     data_new = data
#                     input_ids = data_new[0]
#                 else:
#                     data_new = {}
#                     for key in data.keys():
#                         data_new[key] = to_device(data[key], self.model.device)
#                         if key in ['images', 'pixel_values']:
#                             data_new[key] = to_dtype(data_new[key], self.model.dtype)
#                     if "input_ids" in data_new:
#                         input_ids = data_new["input_ids"]
#                     else:
#                         input_ids = data_new["inputs_embeds"]

#                 if input_ids.shape[-1] < self.seqlen:
#                     pbar.update(1)
#                     continue
#                 try:
#                     if isinstance(data_new, torch.Tensor):
#                         self.model(data_new)
#                     elif isinstance(data_new, tuple) or isinstance(data_new, list):
#                         self.model(*data_new)
#                     else:
#                         self.model(**data_new)
#                 except NotImplementedError:
#                     pass
#                 except Exception as error:
#                     raise error
#                 step = input_ids.shape[0] if len(input_ids.shape) > 1 else 1
#                 total_cnt += step
#                 pbar.update(step)
#                 if total_cnt >= nsamples:
#                     break
#         if total_cnt == 0:
#             logger.error(
#                 f"no data has been cached, please provide more data with sequence length >={self.seqlen} in the "
#                 f"dataset or decease the sequence length"
#             )
#             exit(-1)
#         elif total_cnt < nsamples:
#             logger.warning(
#                 f"Insufficient number of samples collected may affect the quantization. "
#                 f"target samples count is {nsamples}, while valid samples count is {total_cnt}"
#             )
#             if total_cnt < self.batch_size:
#                 raise ValueError(f"valid samples is less than batch_size({self.batch_size}),"
#                                  " please adjust self.batch_size or seqlen.")
#             max_len = (total_cnt // self.batch_size) * self.batch_size
#             for k, v in self.inputs.items():
#                 for key in v:
#                     if isinstance(v[key], list) and len(v[key]) == total_cnt:
#                         self.inputs[k][key] = v[key][:max_len]

#         # clean embed weight to save memory
#         if self.low_cpu_mem_usage:
#             for n, m in embed_layers:
#                 m = m.to("meta")
#         # torch.cuda.empty_cache()

#     def save_quantized(self, output_dir=None, format="auto_round", inplace=True, **kwargs):
#         """Save the quantized model to the specified output directory in the specified format.

#         Args:
#             output_dir (str, optional): The directory to save the quantized model. Defaults to None.
#             format (str, optional): The format in which to save the model. Defaults to "auto_round".
#             inplace (bool, optional): Whether to modify the model in place. Defaults to True.
#             **kwargs: Additional keyword arguments specific to the export format.

#         Returns:
#             object: The compressed model object.
#         """
#         if self.processor is not None and not hasattr(self.processor, "chat_template"):
#             self.processor.chat_template = None
#         compressed_model = super().save_quantized(
#             output_dir=output_dir, format=format, inplace=inplace, processor=self.processor, **kwargs)
#         return compressed_model

class DiffusionInputDataset(Dataset):

    def __init__(self, data_path):
        data_list = torch.load(data_path, map_location='cpu') ## its a list of tuples of tensors
        self.xt_list = []
        self.t_list = []
        self.y_list = []
        ## datalist[i][0].shape (B,4,32,32), flat B dimension
        for i in range(len(data_list)):
            for b in range(len(data_list[i][0])):
                self.xt_list.append(data_list[i][0][b])
                self.t_list.append(data_list[i][1][b])
                self.y_list.append(data_list[i][2][b])

    def __len__(self):
        return len(self.xt_list)
    
    def __getitem__(self, idx):
        return self.xt_list[idx], self.t_list[idx], self.y_list[idx]

class AdaRoundUnetDiffusion(object):
    """Class for adaptive rounding-based quantization with Diffusion Models.
    
    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data.
        processor: Any multi-modal model will require an object to encode or
                   decode the data that groups several modalities (among text, vision and audio).
        image_processor: Image processor for special model like llava.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        sym (bool): Whether sym to be used (default is True).
        layer_config (dict): Configuration for weight quantization (default is None).
        batch_size (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True).
        device: The device to be used for training (default is "auto").
        lr_scheduler: The learning rate scheduler to be used.
        dataset: The path or name of the calib dataset.
        extra_data_dir: The path of extra data such as images, audio and videos.
        template: The path or name of template used to specify process for different MLLMs.
        quant_nontext_module: Whether to quantize nontext module.
        enable_quanted_input (bool): Whether to use quantized input data (default is True).
        enable_minmax_tuning (bool): Whether to enable min-max tuning (default is True).
        lr (float): The learning rate (default is 0.005).
        minmax_lr (float): The learning rate for min-max tuning (default is None).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is False).
        low_cpu_mem_usage (bool): Whether to use low CPU memory (default is False).
        iters (int): Number of iterations (default is 200).
        seqlen (int): Length of the sequence.
        nsamples (int): Number of samples (default is 128).
        sampler (str): The sampling method (default is "rand").
        seed (int): The random seed (default is 42).s
        nblocks (int): Number of blocks (default is 1).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        not_use_best_mse (bool): Whether to use mean squared error (default is False).
        dynamic_max_gap (int): The dynamic maximum gap (default is -1).
        data_type (str): The data type to be used (default is "int").
        scale_dtype (str): The data type of quantization scale to be used (default is "float16"), different kernels
                           have different choices.
        act_bits (int): Number of bits for activation quantization. Default is 32.
        act_group_size (int): Group size for activation quantization. Default is None.
        act_sym (bool): Whether to use symmetric activation quantization. Default is None.
        act_dynamic (bool): Whether to use dynamic activation quantization. Default is True.
        to_quant_block_names (str|list): A string or list whose elements are list of 
                            block's layer names to be quantized.
        enable_torch_compile (bool): Whether to enable torch compile to optimize quant_block/layer
        **kwargs: Additional keyword arguments.


    """

    def __init__(
            self,
            model: torch.nn.Module,
            prompts_path: str = None, # cali prompts
            weight_bits: int = 4,
            sym_w: bool = False,
            w_quant_granularity: str = 'channel_wise',
            batch_size: int = 8,
            w_group_size: int = 128,
            data_type_w: str = 'int',
            w_scale_method: str = 'max',
            cali_iters_w: int = 20000,
            quant_act: bool = False,
            act_bits: int = 8,
            act_quant_granularity: str = 'channel_wise',
            act_group_size: int = None,
            sym_act: bool = None,
            act_dynamic: bool = True,
            data_type_act: str = 'int',
            act_scale_method: str = 'max',
            running_stat: bool = False,
            rs_sm_only: bool = False,
            sm_abit: int = 8,
            cali_iters_a: int = 5000,
            device: str = None,
            lr_scheduler = None,
            enable_quanted_input: bool = True,
            lr_a: float = None,
            lr_w: float = None,
            seed: int = 42,
            tune: bool = False,
            cali_data_path: str = None,
            resume_w: bool = False,
            split: bool = True,
            **kwargs,
    ):
        self.quantized = False
        self.model_orig_dtype = model.dtype
        self.prompts_path = prompts_path
        self.supported_types = supported_layer_types
        self.cali_data_path = cali_data_path
        self.resume_w = resume_w
        self.tune = tune
        self.seed = seed
        set_seed(self.seed)

        # weight quant params
        self.weight_bits = weight_bits
        self.sym_w = sym_w
        self.w_quant_granularity = w_quant_granularity
        self.w_group_size = w_group_size if self.w_quant_granularity == 'group_wise' else -1
        self.data_type_w = data_type_w
        self.cali_iters_w = cali_iters_w
        self.w_scale_method = w_scale_method
        self.lr_w = lr_w
        self.enable_quanted_input = enable_quanted_input
        self.optimizer = torch.optim.Adam
        self.lr_w = lr_w
        self.lr_scheduler = lr_scheduler
        self.quant_act = quant_act
    
        ## activation
        if self.quant_act:
            self.act_quant_granularity = act_quant_granularity
            self.act_group_size = act_group_size if self.act_quant_granularity == 'group_wise' else -1
            self.act_bits = act_bits if not (act_bits is None) else self.bits
            self.act_sym = sym_act
            self.act_dynamic = act_dynamic
            self.act_data_type = data_type_act
            self.act_scale_method = act_scale_method
            self.lr_a = lr_a
            self.running_stat_a = running_stat
            self.rs_sm_only_a = rs_sm_only
            self.sm_abit = sm_abit
            self.cali_iters_a = cali_iters_a

        self.layer_config = {} 
        self.batch_size = batch_size
        self.ldm = model.eval()
        self.model = self.ldm.model.diffusion_model
        self.device = detect_device(device)
        setattr(self.model, "split", True)
    
        torch.set_printoptions(precision=3, sci_mode=True)
        self.check_configs()

        self.serialization_keys = [
            "weight_bits",
            "sym_w",
            "w_quant_granularity",
            "batch_size",
            "w_group_size",
            "data_type_w",
            "w_scale_method",
            "lr_w",
            "cali_iters_w",
            "quant_act",
            "act_bits",
            "act_quant_granularity",
            "act_group_size",
            "sym_act",
            "act_dynamic",
            "data_type_act",
            "act_scale_method",
            "cali_iters_a",
            "lr_a",
            "running_stat_a",
            "sm_abit",
            "tune",
            "enable_quanted_input"
        ]

        self.set_layerwise_config(self.layer_config)  ##better place in the end
        # self.shared_cache_keys = get_shared_keys(self.model)

    def check_configs(self):

        """Checks if the configurations are valid.

        Raises:
        AssertionError: If any of the configurations are invalid.
        """
        assert isinstance(self.model, torch.nn.Module)
        assert self.weight_bits > 0, "bits must be positive"
        assert self.w_group_size == -1 or self.group_size >= 1, "only supports positive group_size or -1(per channel)"
        assert self.batch_size > 0, "batch size must be positive"
        if self.quant_act:
            self.act_bits > 0, "bits must be positive"
            assert self.act_group_size == -1 or self.act_group_size >= 1, \
            "only supports positive group_size or -1(per channel)"

    def quantize_and_save(self, output_dir: str = "tmp_autoround", format: str = "auto_round", inplace=True, **kwargs):
        """Quantizes the model and saves it in the specified format(s).

        This function checks the validity of the requested format(s), quantizes
        the model accordingly, and saves it to the specified output directory.
        If multiple formats are provided, the model is saved separately for each format.

        Args:
            output_dir (str, optional): The directory where the quantized model
                will be saved. Defaults to "tmp_autoround".
            format (str, optional): The quantization format(s) to use, separated
                by commas if multiple. Defaults to "auto_round".
            inplace (bool, optional): Whether to modify the model in place if only
                one format is used. Defaults to True.
            **kwargs: Additional arguments for the quantization and saving process.

        Returns:
            model: A qdq model or packed model based on the configurations
            folders: The folder paths where the quantized models are saved.

        Raises:
            ValueError: If an unsupported format is specified.
        """
        # Validate and process the specified formats
        formats = format.replace(' ', '').split(',')
        from auto_round.utils import supported_formats
        for format_ in formats:
            if format_ not in supported_formats:
                logger.error(f"Unsupported format {format_}, please choose from {supported_formats}")
                exit(-1)

        # only support to export afp8
        if self.act_bits <= 8:
            if "fp8" not in self.act_data_type:
                if len(formats) > 1 or "fake" not in formats:
                    logger.warning(
                        f"Currently only support to export auto_round format quantized model"
                        " with fp8 dtype activation for activation quantization."
                        " Change format to fake and save."
                    )
                    formats = ["fake"]
            else:
                if len(formats) > 1 or "auto_round" not in formats:
                    logger.warning(
                        f"Currently only support to export auto_round format for W{self.bits}AFP8 model,"
                        " change format to auto_round"
                    )
                    formats = ["auto_round"]

        # If multiple formats are specified, enforce inplace=False
        if len(formats) > 1:
            inplace = False
        inplace = kwargs.get("inplace", inplace)
        kwargs.pop("inplace", None)

        # Determine if immediate packing is required
        if (len(formats) == 1 and
                ("awq" in formats[0] or "gptq" in formats[0] or "auto_round" in formats[0]) and
                not self.has_qlayer_outside_block and inplace):  # TODO: Support more formats
            self.is_packing_immediate = True

        # Adjust format settings based on compatibility
        for index in range(len(formats)):
            format = formats[index]
            if "auto_round" in format:
                if (self.sym and ("gptq" not in format and "awq" not in format)) or self.bits == 3:
                    format = format.replace('auto_round', 'auto_round:auto_gptq')
                    formats[index] = format

        # Remove duplicates from formats list
        def remove_duplicates(lst):
            seen = set()
            return [x for x in lst if not (x in seen or seen.add(x))]

        formats = remove_duplicates(formats)
        self.formats = formats

        # # Check format compatibility
        # self._check_format_compatibility(formats)

        # Perform model quantization
        model, _ = self.quantize()

        # Save the quantized model in the specified formats
        folders = []
        for format in formats:
            if "gptq" in format and not self.sym:
                logger.warning(
                    "The asymmetrical kernel of the GPTQ format may result in a noticeable accuracy drop,"
                    " particularly for 2-bit quantization and smaller models."
                    " We recommend exporting to either the AutoAWQ format ( only 4 bits) or "
                    "the AutoRound format(2/4/8 bits)."
                )
            save_format_ = format.replace(":", "-").replace("_", "-")
            save_folder = os.path.join(output_dir, save_format_) if len(formats) > 1 else output_dir
            self.save_quantized(save_folder, format=format, inplace=inplace, **kwargs)

            folders.append(save_folder)

        return model, folders

    def dntc_sample(self, data_path):
        ddim_step = 51
        t_mean = 0.4
        t_std = 0.4
        num_samples = 128
        t_i = np.random.normal(t_mean, t_std, num_samples) * (ddim_step-1)
        t_i = np.clip(np.round(t_i), 0, ddim_step-1)
        
        dataset = DiffusionInputDataset(data_path)
        x = dataset.xt_list
        t = dataset.t_list
        y = dataset.y_list

        st = np.zeros((250, 8, ddim_step))

        calib_xt, calib_y, calib_t = [], [], []

        for i in range(t_i.shape[0]):
            ct = int(t_i[i])
            
            while True:
                c = np.random.randint(0, 250)
                idx = np.random.randint(0, 8)
        
                if st[c][idx][ct] == 0:
                    st[c][idx][ct] = 1
                    break
            
            j = ddim_step * 8 * c + (ddim_step-1-ct) * 8 + idx
            calib_xt.append(x[j].unsqueeze(0))
            calib_y.append(y[j].unsqueeze(0))
            calib_t.append(t[j].unsqueeze(0))

        cali_xt, cali_t, cali_y = torch.cat(calib_xt, dim=0), torch.cat(calib_t, dim=0), torch.cat(calib_y, dim=0)

        del(dataset)
        del(x)
        del(t)
        del(y)
        del(st)
        gc.collect()
        torch.cuda.empty_cache()

        return cali_xt, cali_t, cali_y

    def quant_module_refactor(self, module: torch.nn.Module):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        for name, child_module in module.named_children():
            if isinstance(child_module, tuple(self.supported_types)): # nn.Conv1d
                setattr(module, name, WrapperLinear(child_module, keys=self.serialization_keys, device=self.device))
            else:
                self.quant_module_refactor(child_module)

    def quant_block_refactor(self, module: torch.nn.Module):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock, QuantAttnBlock]:
                    setattr(module, name, self.specials[type(child_module)](child_module, keys=self.serialization_keys, device=self.device))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module, keys=self.serialization_keys, device=self.device))
            else:
                self.quant_block_refactor(child_module)

    def resume_cali_model(self):
        pass

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (WrapperLinear, )):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)
    
    def set_running_stat(self, running_stat: bool, sm_only=False):
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_q.running_stat = running_stat
                    m.attn2.act_quantizer_k.running_stat = running_stat
                    m.attn2.act_quantizer_v.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
            if isinstance(m, WrapperLinear) and not sm_only:
                m.set_running_stat(running_stat)

    def recon_model(self, model):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        global idx
        for name, module in model.named_children():
            # logger.info(f"{name} {isinstance(module, BaseQuantBlock)}")
            if name == 'output_blocks':
                logger.info("Finished calibrating input and mid blocks, saving temporary checkpoint...")
                in_recon_done = True
                # torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
            if name.isdigit() and int(name) >= 9:
                logger.info(f"Saving temporary checkpoint at {name}...")
                # torch.save(self.model.state_dict(), os.path.join(outpath, "ckpt.pth"))
                
            if isinstance(module, WrapperLinear):
                if module.ignore_reconstruction is True:
                    logger.info('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    logger.info('Reconstruction for layer {}'.format(name))
                    # layer_reconstruction(qnn, module, **kwargs)
                    idx += 1
                    print("idx: ", idx)
            elif isinstance(module, BaseWrapperBlock):
                if module.ignore_reconstruction is True:
                    logger.info('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    logger.info('Reconstruction for block {}'.format(name))
                    # block_reconstruction(qnn, module, **kwargs)
                    idx += 1
                    print("idx: ", idx)
            else:
                self.recon_model(module)

    def unwrap_model(self):
        pass

    def quantize(self):
        """Quantize the model and return the quantized model along with layer configurations.
        the entry of AutoRound.

        Returns:
        The quantized model and layer configurations.
        """

        # model refactor
        self.specials = get_specials(self.quant_act)
        self.quant_module_refactor(self.model)
        self.quant_block_refactor(self.model)

        # get cali data
        cali_xs, cali_ts, cali_cs = self.dntc_sample(self.cali_data_path)
        logger.info(f"Calibration data shape: {cali_xs.shape} {cali_ts.shape} {cali_cs.shape}")
        
        if self.resume_w:
            # set-max 
            # blabla 
            self.resume_cali_model() # include init forward
            # resume_cali_model(qnn, opt.cali_ckpt, cali_data, False, cond=opt.cond)
        else:
            # RTN initialization for weight quantization
            # logger.info("Quantizing model weight using RTN...")
            self.set_quant_state(True, False) # enable weight quantization, disable act quantization
            _ = self.model(cali_xs[:8].to(self.device), cali_ts[:8].to(self.device), cali_cs[:8].to(self.device))
            logger.info("RTN quantizing has done!") 

        if self.tune: 
            # Adaround tuning for weight quantization
            logger.info("Doing weight calibration...")
            self.recon_model(self.model)
            self.set_quant_state(weight_quant=True, act_quant=False)

        if self.quant_act:
            # RTN initialization for weight quantization
            logger.info("Doing activation calibration...")
            # Initialize activation quantization parameters
            self.set_quant_state(True, True)
            with torch.no_grad():
                inds = np.random.choice(cali_xs.shape[0], 8, replace=False)
                _ = self.model(cali_xs[inds].to(self.device), cali_ts[inds].to(self.device), cali_cs[inds].to(self.device))
                if self.running_stat_a:
                    logger.info('Running stat for activation quantization')
                    inds = np.arange(cali_xs.shape[0])
                    np.random.shuffle(inds)
                    self.set_running_stat(True, self.rs_sm_only_a)
                    for i in trange(int(cali_xs.size(0) / 8)):
                        _ = self.model(cali_xs[inds[i * 8:(i + 1) * 8]].cuda(), 
                            cali_ts[inds[i * 8:(i + 1) * 8]].cuda(),
                            cali_cs[inds[i * 8:(i + 1) * 8]].cuda())
                    self.set_running_stat(False, self.rs_sm_only_a)

            if self.tune: 
                # Adaround tuning for activation quantization
                pass
        
        self.quantized = True
        self.ldm.model.diffusion_model = self.model
        return self.ldm, self.layer_config


    def set_layerwise_config(self, layer_config):
        """
        Sets the layer-wise configuration based on the provided `layer_config`.
        By default, only quantize layers in blocks.

        Args:
            layer_config (dict): The configuration dictionary for each layer containing various configuration options.

        Returns:
            bool: Returns True if there are quantized layers outside the blocks (e.g., lm-head),
                  otherwise returns False.
        """
        # List of configuration keys
        keys = self.serialization_keys

        # Iterate through all modules in the model
        # supported_type = tuple(self.supported_types) + (ResBlock, BasicTransformerBlock)
        for n, m in self.model.named_modules():
            
            if not isinstance(m, tuple(self.supported_types) + (ResBlock, BasicTransformerBlock)):
                continue
            
            layer_config[n] = {}

            # Skip unsupported types
            if isinstance(m, tuple(self.supported_types)):
                for key in keys:
                    if hasattr(self, key):
                        layer_config[n][key] = getattr(self, key)
                        setattr(m, key, layer_config[n][key])
            elif isinstance(m, ResBlock):
                layer_config[n]["split"] = 0
            elif isinstance(m, BasicTransformerBlock):
                if hasattr(self, "sm_abit"):
                    layer_config[n]["sm_abit"] = getattr(self, "sm_abit")
                layer_config[n]["sm_always_zero_a"] = True

    def register_act_max_hook(self, model):
        def get_act_max_hook(module, input, output):
            if isinstance(input, (tuple, list)):
                input = input[0]
            if not hasattr(module, "act_max"):
                module.act_max = torch.abs(input).max().item()
            else:
                module.act_max = max(torch.abs(input).max().item(), module.act_max)

        hook_handles = []

        for n, m in model.named_modules():
            if hasattr(m, "act_dynamic") and m.act_dynamic == False and check_to_quantized(m):
                hook = m.register_forward_hook(get_act_max_hook)
                hook_handles.append(hook)
        return hook_handles

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
        if self.processor is not None and not hasattr(self.processor, "chat_template"):
            self.processor.chat_template = None
        compressed_model = super().save_quantized(
            output_dir=output_dir, format=format, inplace=inplace, processor=self.processor, **kwargs)
        return compressed_model
