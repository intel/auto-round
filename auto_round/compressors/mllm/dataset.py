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

import json
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed

from auto_round.logger import logger
from auto_round.special_model_handler import check_mllm_model_batch

from .template import Template
from .utils import _extract_data_dir

MLLM_DATASET: Dict[str, Dataset] = {}


def register_dataset(name_list):
    """Class decorator to register a DATASET subclass to the registry.

    Decorator function used before a Pattern subclass.

    Args:
        name: A string. Define the dataset type.

    Returns:
        cls: The class of register.
    """

    def register(dataset):
        for name in name_list.replace(" ", "").split(","):
            MLLM_DATASET[name] = dataset

    return register


@register_dataset(
    "liuhaotian/llava,liuhaotian/llava_conv_58k,liuhaotian/llava_instruct_80k,liuhaotian/llava_instruct_150k"
)
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
        seqlen=512,
        padding=True,
        truncation=True,
        nsamples=512,
    ) -> None:
        super().__init__()
        self.model = model
        self.model_type = template.model_type
        self.template = template
        self.tokenizer = tokenzier
        if os.path.exists(dataset_path):
            logger.info(f"use dataset {dataset_path}, loading from disk...")
            self.questions = json.load(open(dataset_path, "r"))
        else:
            import requests

            if dataset_path == "liuhaotian/llava":
                dataset_path = "llava_conv_58k"
            else:
                dataset_path = dataset_path.split("/")[-1]
            dataset_name = dataset_path.split("/")[-1]
            if dataset_name in self.LLAVA_DATASET:
                logger.info(f"use dataset {dataset_name}, downloading...")
                self.questions = requests.get(self.LLAVA_DATASET[dataset_name], stream=True).json()
            else:
                raise KeyError(f"{dataset_path} is not support, we support {self.LLAVA_DATASET.keys()}.")

        self.seqlen = seqlen
        self.questions = self.check(self.questions, self.seqlen, nsamples)
        self.padding = padding
        self.truncation = truncation
        self.extra_data_dir = extra_data_dir
        self.role_mapping = {"human": "user", "gpt": "assistant"}
        self.cached_data_dict = {}

        self.image_fold = None
        if extra_data_dir is not None:
            image_fold = _extract_data_dir(self.extra_data_dir)
            if isinstance(image_fold, dict):
                image_fold = image_fold["image"]
            self.image_fold = image_fold

    def check(self, questions, word_len, nsamples):
        def _check(questions, min_word_len, max_word_len, nsamples):
            new_questions = []
            max_len = 0
            for source in questions:
                str_len = 0
                for text in source["conversations"]:
                    if self.IMAGE_TOKEN in text["value"]:
                        text["value"] = self.IMAGE_TOKEN + text["value"].replace(self.IMAGE_TOKEN, "")
                    str_len += len(text["value"].split(" "))
                if str_len > max_len:
                    max_len = str_len
                if min_word_len <= str_len < max_word_len:
                    new_questions.append(source)
                if len(new_questions) >= nsamples:
                    return new_questions
            if min_word_len > max_len:
                logger.debug(
                    f"seqlen={min_word_len} is greater than the max length of dataset {max_len},"
                    f" will change seqlen to {max_len - 128}"
                )
                new_min_word_len = max_len - 128
            else:
                logger.debug(
                    f"no enough sample for seqlen greater than {min_word_len},"
                    f" will decrease to {min_word_len - 128}"
                )
                new_min_word_len = min_word_len - 128
            return new_questions + _check(questions, new_min_word_len, min_word_len, nsamples - len(new_questions))

        return _check(questions, word_len, float("inf"), nsamples)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        text = self.questions[i]["conversations"]
        if self.template.model_type != "llava":
            text = self.covert_conversations(text)

        if self.image_fold is not None:
            image_path = os.path.join(self.image_fold, os.path.basename(self.questions[i]["image"]))
        else:
            image_path = self.questions[i]["image"]
            if not os.path.exists(image_path):
                image_path = self._COCO_DATA_URL + self.questions[i]["image"].split("/")[-1]
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
            max_length=max_length,
            truncation_strategy=truncation_strategy,
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
            new_data.append({"role": self.role_mapping.get(d["from"], d["from"]), "content": content})
        return new_data


def get_mllm_dataloader(
    template,
    model,
    tokenizer,
    processor,
    image_processor=None,
    dataset="liuhaotian/llava_conv_58k",
    extra_data_dir=None,
    seqlen=512,
    bs=1,
    split=None,
    apply_template=None,
    truncation=False,
    seed=42,
    nsamples=512,
    gradient_accumulate_steps=1,
    quant_nontext_module=False,
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

        template = get_template(
            template, model=model, tokenizer=tokenizer, processor=processor, image_processor=image_processor
        )

    if os.path.isfile(dataset) or dataset in MLLM_DATASET.keys():
        dataset = MLLM_DATASET["liuhaotian/llava"](
            template, model, tokenizer, dataset, extra_data_dir, seqlen=seqlen, truncation=truncation, nsamples=nsamples
        )

        bs, gradient_accumulate_steps = check_mllm_model_batch(
            model, batch_size=bs, gradient_accumulate_steps=gradient_accumulate_steps
        )

        set_seed(seed)
        dataloader_params = {"batch_size": bs, "shuffle": True, "collate_fn": dataset.template.processor.data_collator}

        return DataLoader(dataset, **dataloader_params), bs, gradient_accumulate_steps
    else:
        # try to load text calibration dataset
        from auto_round.calib_dataset import get_dataloader

        dataloader = get_dataloader(tokenizer, seqlen, dataset, seed, bs, nsamples)
        if quant_nontext_module:
            logger.error(
                "Text only dataset cannot be used for calibrating non-text modules,"
                " switching to liuhaotian/llava_conv_58k"
            )
            exit(-1)
        return dataloader, bs, gradient_accumulate_steps
