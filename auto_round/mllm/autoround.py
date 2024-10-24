from typing import Optional, Union

import torch

from ..utils import (
    logger,
    to_device,
    to_dtype,
)
from ..autoround import AutoRound
from .template import get_template, Template
from .mllm_dataset import get_mllm_dataloader
from ..low_cpu_mem.utils import get_layers_before_block

class AutoRoundMLLM(AutoRound):
    def __init__(
            self,
            model,
            tokenizer,
            bits: int = 4,
            group_size: int = 128,
            sym: bool = False,
            layer_config: dict = None,
            batch_size: int = 8,
            amp: bool = True,
            device: str = None,
            lr_scheduler=None,
            dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = None,
            extra_data_dir: Union[str, torch.utils.data.DataLoader] = None,
            template: Union[str, Template] = None,
            quant_nontext_module: bool = False,
            enable_quanted_input: bool = True,
            enable_minmax_tuning: bool = True,
            lr: float = None,
            minmax_lr: float = None,
            low_gpu_mem_usage: bool = False,
            low_cpu_mem_usage: bool = False,
            iters: int = 200,
            seqlen: int = 2048,
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
            quant_block_list: list = None,
            enable_norm_bias_tuning: bool = False,
            **kwargs,
    ):
        if quant_block_list is None:
            quant_block_list = get_multimodal_block_names(model, quant_nontext_module)
        self.extra_data_dir = extra_data_dir
        self.quant_nontext_module = quant_nontext_module
        self.template = template
        if self.template is None:
            self.template = get_template(model.config.model_type)
        assert dataset is not None, "dataset should not be None"
        if isinstance(dataset, str):
            dataset = get_mllm_dataloader(self.template, model, tokenizer, dataset, extra_data_dir, seqlen, batch_size)
        
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
            enable_norm_bias_tuning=enable_norm_bias_tuning,
            quant_block_list=quant_block_list,
            **kwargs,
        )
        

    def calib(self, nsamples, bs):
        if isinstance(self.dataset, str):
            dataset = self.dataset.replace(" ", "")
            self.dataloader = get_mllm_dataloader(
                self.template, self.model, self.tokenizer, dataset, self.extra_data_dir, self.seqlen, bs)
        else:
            self.dataloader = self.dataset
        total_cnt = 0 

        if self.low_cpu_mem_usage:
            embed_layers = get_layers_before_block(self.model)
            for n, m in embed_layers:
                m = m.to(self.device)

        for data in self.dataloader:
            if data is None:
                continue  
            if isinstance(data, torch.Tensor):
                input_ids = data.to(self.device)
                data_new = input_ids
            elif isinstance(data, str):
                if self.tokenizer is None:
                    logger.error("please provide tokenizer for string input")
                    exit()
                data = self.template._encode(data)
                data = self.template.plugin.get_input(
                    self.model,
                    self.tokenizer,
                    text=data,
                    images=None,
                    max_length=self.seqlen,
                    )
                input_ids = data_new["input_ids"]
                data_new = {}
                for key in data.keys():
                    data_new[key] = data[key].to(self.device)
            elif isinstance(data, dict) and all([isinstance(v, str) for v in data.values()]):
                text = data['text']
                text = self.template._encode(data)
                image = None
                if "image" in data:
                    image = self.template.plugin.image_processor("image")
                data = self.template.plugin.get_input(
                    self.model,
                    self.tokenizer,
                    text=text,
                    images=image,
                    max_length=self.seqlen,
                    )
                data_new = {}
                for key in data.keys():
                    data_new[key] = to_device(data[key], self.model.device)
                    if key == 'images':
                        data_new[key] = to_dtype(data_new[key], self.model.dtype)
                input_ids = data_new["input_ids"]
            elif isinstance(data, tuple) or isinstance(data, list):
                data_new = data
                input_ids = data_new[0]
            else:
                data_new = {}
                for key in data.keys():
                    data_new[key] = to_device(data[key], self.model.device)
                    if key == 'images':
                        data_new[key] = to_dtype(data_new[key], self.model.dtype)
                input_ids = data_new["input_ids"]

            if input_ids.shape[-1] < self.seqlen:
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
            total_cnt += input_ids.shape[0] if len(input_ids.shape) > 1 else 1
            if total_cnt >= nsamples:
                break
        if total_cnt == 0:
            logger.error(
                f"no data has been cached, please provide more data with sequence length >={self.seqlen} in the "
                f"dataset or decease the sequence length"
            )
            exit()
        elif total_cnt < nsamples:
            logger.warning(
                f"Insufficient number of samples collected may affect the quantification. "
                f"Valid samples size:{total_cnt}, Target sample size:{nsamples}"
            )

        # clean embed weight to save memory
        if self.low_cpu_mem_usage:
            for n, m in embed_layers:
                m = m.to("meta")
        # torch.cuda.empty_cache()