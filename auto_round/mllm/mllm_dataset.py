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
from PIL import Image
from transformers.data.data_collator import default_data_collator

from .utils import _extract_data_dir
from .template import Template, TEMPLATES, load_template


MLLM_DATASET : Dict[str, Dataset] = {}

def register_dataset(name):
    def register(dataset):
        MLLM_DATASET[name] = dataset
        return dataset
    return register


@register_dataset("llava")
class LlavaDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            model_type_or_template,
            model,
            tokenzier,
            dataset_path,
            extra_data_dir,
            max_length,
            padding=True,
            truncation=True,
            ) -> None:
        super().__init__()
        if isinstance(model_type_or_template, str):
            assert model_type_or_template in TEMPLATES, f"{model_type_or_template} is not supported"
            self.model = model
            self.model_type = model_type_or_template
            self.template = TEMPLATES[model_type_or_template]
        elif isinstance(model_type_or_template, Template):
            self.model_type = model_type_or_template.model_type
            self.template = model_type_or_template
        else:
            raise TypeError
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

        text = self.template._encode(text)

        image_fold = _extract_data_dir(self.extra_data_dir)
        if isinstance(image_fold, dict):
            image_fold = image_fold['image']
        image = self.template.plugin.image_processor(os.path.join(image_fold, os.path.basename(self.questions[i]["image"])))

        ret = self.template.plugin.get_input(
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
            for old, new in self.template.replace_tokens:
                content = content.replace(old, new)
            new_data.append({
                "role": self.role_mapping.get(d["from"], d["from"]),
                "content": content
            })
        return new_data


def get_mllm_dataloader(
        template_or_path,
        model,
        tokenizer, 
        dataset_path,
        extra_data_dir,
        seqlen=512, 
        bs=1, 
):
    if os.path.isfile(template_or_path):
        model_type_or_template = load_template(template_or_path)
    else:
        model_type_or_template = template_or_path
    dataset = MLLM_DATASET['llava'](
        model_type_or_template, model, tokenizer, dataset_path, extra_data_dir, 
        max_length=min(seqlen, tokenizer.model_max_length))
    
    dataloader_params = {
        "batch_size": bs,
        "collate_fn": dataset.template.plugin.data_collator
    }

    return DataLoader(dataset, **dataloader_params)
