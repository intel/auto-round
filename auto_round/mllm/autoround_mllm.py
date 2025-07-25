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

from copy import deepcopy
from typing import Union

import torch
from tqdm import tqdm

from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS, _handle_special_model

from ..autoround import AutoRound
from ..low_cpu_mem.utils import get_layers_before_block
from ..utils import (
    clear_memory,
    detect_device,
    extract_block_names_to_str,
    find_matching_blocks,
    get_block_names,
    logger,
    mllm_load_model,
    to_device,
    to_dtype,
)
from .mllm_dataset import get_mllm_dataloader
from .template import Template, get_template


def _only_text_test(model, tokenizer, device, model_type):
    """Test if the model whether can use text-only datasets."""

    if model_type in SUPPORT_ONLY_TEXT_MODELS:  # save time
        return True

    new_tokenizer = deepcopy(tokenizer)
    device = detect_device(device)
    text = ["only text", "test"]
    new_tokenizer.padding_side = "left"
    if new_tokenizer.pad_token is None:
        new_tokenizer.pad_token = new_tokenizer.eos_token
    inputs = new_tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    try:
        inputs = inputs.to(device)
        model = model.to(device)
        model(**inputs)
        return True
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            model = model.to("cpu")
            inputs = inputs.to("cpu")
            try:
                model(**inputs)
            except:
                return False
        return False
    except Exception as e:
        return False


class AutoRoundMLLM(AutoRound):
    """Class for automatic rounding-based quantization with MLLMs.

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
        tokenizer,
        processor=None,
        image_processor=None,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
        layer_config: dict = None,
        batch_size: int = 8,
        amp: bool = True,
        device: str = None,
        lr_scheduler=None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = None,
        extra_data_dir: str = None,
        template: Union[str, Template] = None,
        quant_nontext_module: bool = False,
        enable_quanted_input: bool = True,
        enable_minmax_tuning: bool = True,
        lr: float = None,
        minmax_lr: float = None,
        low_gpu_mem_usage: bool = False,
        low_cpu_mem_usage: bool = False,
        iters: int = 200,
        seqlen: int = None,
        nsamples: int = 128,
        sampler: str = "rand",
        seed: int = 42,
        nblocks: int = 1,
        gradient_accumulate_steps: int = 1,
        not_use_best_mse: bool = False,
        dynamic_max_gap: int = -1,
        data_type: str = "int",
        scale_dtype: str = "fp16",
        act_bits: int = 32,
        act_group_size: int = None,
        act_sym: bool = None,
        act_dynamic: bool = True,
        to_quant_block_names: Union[str, list] = None,
        enable_norm_bias_tuning: bool = False,
        truncation: bool = None,
        enable_torch_compile: bool = False,
        model_kwargs: dict = None,
        **kwargs,
    ):
        all_blocks = get_block_names(model, quant_nontext_module)
        self.quant_block_list = find_matching_blocks(model, all_blocks, to_quant_block_names)
        if to_quant_block_names is None:
            to_quant_block_names = extract_block_names_to_str(self.quant_block_list)
        self.to_quant_block_names = to_quant_block_names
        self.extra_data_dir = extra_data_dir
        self.quant_nontext_module = quant_nontext_module
        self.processor = processor
        self.image_processor = image_processor
        from transformers import PreTrainedModel

        if model.config.model_type == "llava" and isinstance(model, PreTrainedModel):
            template = "default"
        self.template = template if template is not None else model.config.model_type
        if not isinstance(dataset, torch.utils.data.DataLoader):
            self.template = get_template(
                self.template,
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                image_processor=image_processor,
                use_rtn=iters == 0,
            )
            dataset = self.template.default_dataset if dataset is None else dataset

        model = _handle_special_model(model)

        from ..calib_dataset import CALIB_DATASETS
        from .mllm_dataset import MLLM_DATASET

        if isinstance(dataset, str):
            if quant_nontext_module or (
                dataset in CALIB_DATASETS.keys()
                and not _only_text_test(model, tokenizer, device, self.template.model_type)
            ):
                if quant_nontext_module:
                    logger.warning(
                        "Text only dataset cannot be used for calibrating non-text modules,"
                        "switching to liuhaotian/llava_conv_58k"
                    )
                else:
                    logger.warning(
                        f"{model.config.model_type} not support for {dataset},"
                        " will use liuhaotian/llava_conv_58k with default config as an alternative."
                    )
                dataset = "liuhaotian/llava_conv_58k"

            if dataset in MLLM_DATASET.keys():
                truncation = False
                seqlen = 512 if seqlen is None else seqlen
                if batch_size != 1:
                    logger.warning(
                        f"reset batch_size({batch_size}) to 1 and "
                        f"gradient_accumulate_steps({gradient_accumulate_steps}) "
                        f"to {batch_size * gradient_accumulate_steps}, "
                        f"because batch_size={batch_size} cannot be used for {dataset}"
                    )
                    gradient_accumulate_steps = batch_size * gradient_accumulate_steps
                    batch_size = 1
        if quant_nontext_module and batch_size != 1:
            logger.warning(
                f"reset batch_size({batch_size}) to 1 and "
                f"gradient_accumulate_steps({gradient_accumulate_steps}) "
                f"to {batch_size * gradient_accumulate_steps}, "
                f"because batch_size={batch_size} cannot be used for calibrating non-text modules."
            )
            gradient_accumulate_steps = batch_size * gradient_accumulate_steps
            batch_size = 1
        seqlen = 2048 if seqlen is None else seqlen
        truncation = True if truncation is None else truncation
        self.truncation = truncation

        if nsamples % batch_size != 0:
            nsamples = (nsamples // batch_size + 1) * batch_size
            logger.warning(f"'nsamples' is not divisible by 'batch_size', will adjusted to {nsamples}")

        super(AutoRoundMLLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            layer_config=layer_config,
            batch_size=batch_size,
            amp=amp,
            device=device,
            lr_scheduler=lr_scheduler,
            dataset=dataset,
            enable_quanted_input=enable_quanted_input,
            enable_minmax_tuning=enable_minmax_tuning,
            lr=lr,
            minmax_lr=minmax_lr,
            low_gpu_mem_usage=low_gpu_mem_usage,
            low_cpu_mem_usage=low_cpu_mem_usage,
            iters=iters,
            seqlen=seqlen,
            nsamples=nsamples,
            sampler=sampler,
            seed=seed,
            nblocks=nblocks,
            gradient_accumulate_steps=gradient_accumulate_steps,
            not_use_best_mse=not_use_best_mse,
            dynamic_max_gap=dynamic_max_gap,
            data_type=data_type,
            scale_dtype=scale_dtype,
            act_bits=act_bits,
            act_group_size=act_group_size,
            act_sym=act_sym,
            act_dynamic=act_dynamic,
            to_quant_block_names=self.to_quant_block_names,
            enable_norm_bias_tuning=enable_norm_bias_tuning,
            enable_torch_compile=enable_torch_compile,
            vlm=True,
            **kwargs,
        )

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
        if isinstance(self.dataset, str):
            dataset = self.dataset.replace(" ", "")
            self.dataloader, self.batch_size, self.gradient_accumulate_steps = get_mllm_dataloader(
                template=self.template,
                model=self.model,
                tokenizer=self.tokenizer,
                processor=self.processor,
                image_processor=self.image_processor,
                dataset=dataset,
                extra_data_dir=self.extra_data_dir,
                seqlen=self.seqlen,
                bs=self.batch_size,
                seed=self.seed,
                truncation=self.truncation,
                nsamples=self.nsamples,
                gradient_accumulate_steps=self.gradient_accumulate_steps,
                quant_nontext_module=self.quant_nontext_module,
            )
        else:
            self.dataloader = self.dataset
        total_cnt = 0

        if self.low_cpu_mem_usage:
            embed_layers = get_layers_before_block(self.model)
            for n, m in embed_layers:
                m = m.to(self.device)

        total = nsamples if not hasattr(self.dataloader, "len") else min(nsamples, len(self.dataloader))
        with tqdm(range(1, total + 1), desc="cache block inputs") as pbar:
            for data in self.dataloader:
                if data is None:
                    pbar.update(1)
                    continue
                if isinstance(data, torch.Tensor):
                    input_ids = data.to(self.device)
                    data_new = input_ids
                elif isinstance(data, str):
                    if self.tokenizer is None:
                        logger.error("please provide tokenizer for string input")
                        exit()
                    # data = self.template._encode(data)
                    data = self.template.processor.get_input(
                        text=data,
                        images=None,
                        max_length=self.seqlen,
                        squeeze=False,
                    )
                    data_new = {}
                    for key in data.keys():
                        data_new[key] = data[key].to(self.device)
                    input_ids = data_new["input_ids"]
                elif isinstance(data, dict) and "text" in data.keys():
                    text = data["text"]
                    if isinstance(text, dict):
                        text = [text]
                    input_text = self.template._encode(text)
                    data = self.template.processor.get_input(
                        text=input_text,
                        images=data["image"],
                        max_length=self.seqlen,
                        squeeze=False,
                    )
                    data_new = {}
                    for key in data.keys():
                        data_new[key] = torch.tensor(data[key])
                        data_new[key] = to_device(data_new[key], self.model.device)
                        if key == "images":
                            data_new[key] = to_dtype(data_new[key], self.model.dtype)
                    input_ids = data_new["input_ids"]
                elif isinstance(data, tuple) or isinstance(data, list):
                    data_new = data
                    input_ids = data_new[0]
                else:
                    data_new = {}
                    for key in data.keys():
                        data_new[key] = to_device(data[key], self.model.device)
                        if key in ["images", "pixel_values"]:
                            data_new[key] = to_dtype(data_new[key], self.model.dtype)
                    if "input_ids" in data_new:
                        input_ids = data_new["input_ids"]
                    else:
                        input_ids = data_new["inputs_embeds"]

                if input_ids.shape[-1] < self.seqlen:
                    pbar.update(1)
                    continue
                try:
                    if isinstance(data_new, torch.Tensor):
                        self.model(data_new)
                    elif isinstance(data_new, tuple) or isinstance(data_new, list):
                        self.model(*data_new)
                    else:
                        self.model(**data_new)
                except NotImplementedError:
                    pass
                except Exception as error:
                    raise error
                step = input_ids.shape[0] if len(input_ids.shape) > 1 else 1
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

        # clean embed weight to save memory
        if self.low_cpu_mem_usage:
            for n, m in embed_layers:
                m = m.to("meta")
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
        if self.processor is not None and not hasattr(self.processor, "chat_template"):
            self.processor.chat_template = None
        compressed_model = super().save_quantized(
            output_dir=output_dir, format=format, inplace=inplace, processor=self.processor, **kwargs
        )
        return compressed_model
