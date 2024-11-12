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

import os
import json
from typing import Dict


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed

from .utils import _extract_data_dir
from .template import Template
from ..utils import logger


MLLM_DATASET : Dict[str, Dataset] = {}

def register_dataset(name):
    """Class decorator to register a DATASET subclass to the registry.

    Decorator function used before a Pattern subclass.

    Args:
        name: A string. Define the dataset type.

    Returns:
        cls: The class of register.
    """

    def register(dataset):
        MLLM_DATASET[name] = dataset
        return dataset
    return register



@register_dataset("liuhaotian/llava")
class LlavaDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    BASE_LLAVA_URL = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/"
    LLAVA_DATASET = {
        "llava_conv_58k": BASE_LLAVA_URL + "conversation_58k.json?download=true",
        "llava_instruct_80k": BASE_LLAVA_URL + "llava_instruct_80k.json?download=true",
        "llava_instruct_150k": BASE_LLAVA_URL + "llava_instruct_150k.json?download=true",
    }
    _COCO_DATA_URL = "http://images.cocodataset.org/train2017/"
    IMAGE_TOKEN = "<image>"

    def __init__(
            self,
            template,
            model,
            tokenzier,
            dataset_path,
            extra_data_dir=None,
            seqlen=None,
            padding=True,
            truncation=True,
            ) -> None:
        super().__init__()
        self.model = model
        self.model_type = template.model_type
        self.template = template
        self.tokenizer = tokenzier
        if os.path.exists(dataset_path):
            logger.info(f'use dataset {dataset_path}, loading from disk...')
            self.questions = json.load(open(dataset_path, "r"))
        else:
            import requests
            dataset_name = dataset_path.split('/')[-1]
            if dataset_name in self.LLAVA_DATASET:
                logger.info(f'use dataset {dataset_name}, downloading ...')
                self.questions = requests.get(self.LLAVA_DATASET[dataset_name], stream=True).json()
            else:
                raise KeyError(f"{dataset_path} is not support, we support {self.LLAVA_DATASET.keys()}.")
        self.seqlen = seqlen
        self.questions = self.check(self.questions, seqlen)
        self.padding = padding
        self.truncation = truncation
        self.extra_data_dir = extra_data_dir
        self.role_mapping = {"human": "user", "gpt": "assistant"}
        self.cached_data_dict = {}

        self.image_fold = None
        if extra_data_dir is not None:
            image_fold = _extract_data_dir(self.extra_data_dir)
            if isinstance(image_fold, dict):
                image_fold = image_fold['image']
            self.image_fold = image_fold


    def check(self, questions, seqlen):
        new_questions = []
        for source in questions:
            text_lenght = 0
            for text in source['conversations']:
                if self.IMAGE_TOKEN in text['value']:
                    text['value'] = self.IMAGE_TOKEN + text['value'].replace(self.IMAGE_TOKEN, '')
                text_lenght += len(text['value'].split(' '))
            if text_lenght >= seqlen:
                new_questions.append(source)
        assert len(new_questions) > 0, f"no data with length greater than {seqlen}, please check"
        return new_questions
    

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        text = self.questions[i]["conversations"]
        if self.template.model_type != "llava":
            text = self.covert_conversations(text)

        if self.image_fold is not None:
            image_path = os.path.join(
                self.image_fold, os.path.basename(self.questions[i]["image"]))
        else:
            image_path = self.questions[i]["image"]
            if not os.path.exists(image_path):
                image_path = self._COCO_DATA_URL + self.questions[i]["image"].split('/')[-1]
        # image = self.template.processor.image_processor(image_path)

        text = self.template._encode(text)

        max_length = self.seqlen
        truncation_strategy = "text"
        ret = self.template.processor.get_input(
            text=text, 
            images=image_path,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
            max_length = max_length,
            truncation_strategy=truncation_strategy
           )
        self.cached_data_dict[i] = ret
        return ret
    
    def covert_conversations(self, data):
        new_data = []
        for d in data:
            content = d["value"]
            if self.template.replace_tokens is not None:
                for old, new in self.template.replace_tokens:
                    content = content.replace(old, new)
            new_data.append({
                "role": self.role_mapping.get(d["from"], d["from"]),
                "content": content
            })
        return new_data


def get_mllm_dataloader(
        template,
        model,
        tokenizer, 
        image_processor=None,
        dataset="liuhaotian/llava_conv_58k",
        extra_data_dir=None,
        seqlen=512, 
        bs=1, 
        split=None,
        apply_template=None,
        seed=42,
):
    """Generate a DataLoader for calibration using specified parameters.

    Args:
        template (Template): The template to specify process for different mllms.
        model (Model): The model to quantized.
        tokenizer (Tokenizer): The tokenizer to use for tokenization.
        Dataset_name (str): The name or path of the dataset.
        extra_data_dir (str): The path for extra data such as images, audio or videos.
        seqlen (int): The exact sequence length. samples < seqlen will be dropped,
                      samples longer than seqlen will be truncated
        bs (int, optional): The batch size. Defaults to 4.
        split (str, optional): The data split to use. Defaults to None.
        apply_template: Whether to apply chat template in tokenization.
    
    Returns:
        DataLoader: The DataLoader for the calibrated datasets.
    """
    if isinstance(template, str):
        from .template import get_template
        template = get_template(template, model=model, tokenizer=tokenizer, image_processor=image_processor)

    if isinstance(dataset, str):
        if os.path.isfile(dataset):
            dataset = MLLM_DATASET['liuhaotian/llava'](
                template, model, tokenizer, dataset, extra_data_dir, 
                seqlen=min(seqlen, tokenizer.model_max_length))
        elif "liuhaotian/llava" in dataset:
            dataset = MLLM_DATASET["liuhaotian/llava"](
                template, model, tokenizer, dataset, extra_data_dir, 
                seqlen=min(seqlen, tokenizer.model_max_length))
        else:
            from datasets import load_dataset
            from ..calib_dataset import get_tokenizer_function
            dataset = load_dataset(dataset, split=split)
            tokenizer_function = get_tokenizer_function(tokenizer, seqlen, apply_template=apply_template)
            dataset = dataset.map(tokenizer_function, batched=True)

    
    set_seed(seed)
    dataloader_params = {
        "batch_size": bs,
        "shuffle": True,
        "collate_fn": dataset.template.processor.data_collator
    }

    return DataLoader(dataset, **dataloader_params)
