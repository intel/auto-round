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

import os
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed
from auto_round.calib_dataset import CALIB_DATASETS, select_dataset

from auto_round.utils import logger

DIFFUSION_DATASET: Dict[str, Dataset] = {}

def get_vllm_dataloader(
    name="NeelNanda/pile-10k",
    bs=1,
    seqlen=1024,
    seed=42,
    nsamples=128,
    gradient_accumulate_steps=1,
):
    """Generate a DataLoader for calibration using specified parameters.
    Args:
        Dataset_name (str): The name or path of the dataset.
        bs (int, optional): The batch size. Defaults to 1.
    Returns:
        DataLoader: The DataLoader for the calibrated datasets.
    """
    set_seed(seed)
    system_prompt = "You are a helpful assistant."
    split = None
    apply_chat_template = False

    if ":" in name:
        name, split_list = name.split(":")[0], name.split(":")[1:]
        for ele in split_list:
            key, values = ele.split("=")[0], ele.split("=")[1:]
            if key == "split":
                split = values[0].split("+")
            if key == "apply_chat_template":
                apply_chat_template = False if (len(values) > 0 and values[0].lower() == "false") else True
            if key == "system_prompt":
                system_prompt = values[0]
                apply_chat_template = True

    get_dataset = CALIB_DATASETS.get(dataset)
    if get_dataset is None:
            filtered_keys = [k for k in CALIB_DATASETS.keys() if "/" not in k]
            raise ValueError(
                f"Dataset '{dataset}' is not found. Please choose from the supported datasets: {filtered_keys}."
            )
    dataset = get_dataset(
            tokenizer=None,
            seqlen=seqlen,
            seed=seed,
            split=split,
            dataset_name=name,
            apply_chat_template=apply_chat_template,
            system_prompt=system_prompt,
        )

    if len(dataset) > nsamples:
        dataset = select_dataset(dataset, range(nsamples))
    return DataLoader(dataset["text"], batch_size=bs, shuffle=False), bs
