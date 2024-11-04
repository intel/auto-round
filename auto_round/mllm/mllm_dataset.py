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

from .utils import _extract_data_dir
from .template import Template


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


@register_dataset("llava")
class LlavaDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            template,
            model,
            tokenzier,
            dataset_path,
            extra_data_dir,
            max_length,
            padding=True,
            truncation=True,
            ) -> None:
        super().__init__()
        self.model = model
        self.model_type = template.model_type
        self.template = template
        self.tokenizer = tokenzier
        self.questions = json.load(open(dataset_path, "r"))
        self.padding = padding
        self.truncation = truncation
        self.extra_data_dir = extra_data_dir
        self.max_length = max_length
        self.role_mapping = {"human": "user", "gpt": "assistant"}
        self.cached_data_dict = {}
    

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        text = self.questions[i]["conversations"]
        text = self.covert_conversations(text)

        if self.extra_data_dir is not None:
            image_fold = _extract_data_dir(self.extra_data_dir)
            if isinstance(image_fold, dict):
                image_fold = image_fold['image']
            image_path = os.path.join(
                image_fold, os.path.basename(self.questions[i]["image"]))
        else:
            image_path = self.questions[i]["image"]
        image = self.template.processor.image_processor(image_path)

        text = self.template._encode(text)

        ret = self.template.processor.get_input(
            self.model,
            self.tokenizer,
            text=text, 
            images=image,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
            max_length = self.max_length
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
        dataset,
        extra_data_dir,
        seqlen=512, 
        bs=1, 
        split=None,
        apply_template=None,
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
    assert isinstance(template, Template)

    if isinstance(dataset, str):
        if os.path.isfile(dataset):
            dataset = MLLM_DATASET['llava'](
                template, model, tokenizer, dataset, extra_data_dir, 
                max_length=min(seqlen, tokenizer.model_max_length))
        else:
            from datasets import load_dataset
            from ..calib_dataset import get_tokenizer_function
            dataset = load_dataset(dataset, split=split)
            tokenizer_function = get_tokenizer_function(tokenizer, seqlen, apply_template=apply_template)
            dataset = dataset.map(tokenizer_function, batched=True)

    
    dataloader_params = {
        "batch_size": bs,
        "collate_fn": dataset.template.processor.data_collator
    }

    return DataLoader(dataset, **dataloader_params)
