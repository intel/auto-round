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

from auto_round.utils import logger


class DiffusionDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    COCO_URL = {
        "coco2014": "https://github.com/mlcommons/inference/raw/refs/heads/master/text_to_image/coco2014/captions/captions_source.tsv"
    }

    def __init__(
        self,
        dataset_path,
        nsamples=128,
    ) -> None:
        super().__init__()
        self.captions = []
        self.caption_ids = []

        if os.path.exists(dataset_path):
            logger.info(f"use dataset {dataset_path}, loading from disk...")
            df = pd.read_csv(dataset_path, sep="\t")
        else:
            from io import StringIO

            import requests

            dataset_path = "coco2014"

            if dataset_path in self.COCO_URL:
                logger.info(f"use dataset {dataset_path}, downloading ...")
                text_data = requests.get(self.COCO_URL[dataset_path]).text
                df = pd.read_csv(StringIO(text_data), sep="\t")
            else:
                raise KeyError(f"{dataset_path} is not support, we support {self.COCO_URL.keys()}.")
        for index, row in df.iterrows():
            if nsamples > 0 and index + 1 > nsamples:
                break
            assert "id" in row and "caption" in row
            caption_id = row["id"]
            caption_text = row["caption"]
            self.caption_ids.append(caption_id)
            self.captions.append(caption_text)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.caption_ids[i], self.captions[i]


def get_diffusion_dataloader(
    dataset="coco2014",
    bs=1,
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
    dataset = DiffusionDataset(dataset, nsamples)
    set_seed(seed)
    dataloader_params = {"batch_size": bs, "shuffle": True}

    return DataLoader(dataset, **dataloader_params), bs, gradient_accumulate_steps