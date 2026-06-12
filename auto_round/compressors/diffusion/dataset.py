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
from io import StringIO
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed

from auto_round.utils import download_audiocaps_csv, logger

DIFFUSION_DATASET: Dict[str, Dataset] = {}


COCO_URL = {
    "coco2014": (
        "https://github.com/mlcommons/inference/raw/refs/heads/master/text_to_image/"
        "coco2014/captions/captions_source.tsv"
    )
}


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
            DIFFUSION_DATASET[name] = dataset

    return register


@register_dataset("local")
class Text2ImgDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        dataset_path: str,
        nsamples: int = 128,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__()
        self.captions = []
        self.caption_ids = []

        if dataframe is not None:
            df = dataframe
        else:
            logger.info(f"use dataset {dataset_path}, loading from disk...")
            df = pd.read_csv(dataset_path, sep="\t")

        required_cols = {"id", "caption"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"Invalid diffusion caption data from {dataset_path!r}: "
                f"expected columns {sorted(required_cols)}, got {list(df.columns)}"
            )

        for index, row in df.iterrows():
            if nsamples > 0 and index + 1 > nsamples:
                break
            caption_id = row["id"]
            caption_text = row["caption"]
            self.caption_ids.append(caption_id)
            self.captions.append(caption_text)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.caption_ids[i], self.captions[i]


@register_dataset("audiocaps")
class AudioCapsDataset(Dataset):
    """Dataset for AudioCaps caption-based calibration.

    AudioCaps CSV contains columns like ``audiocap_id``, ``youtube_id``,
    ``start_time``, and ``caption``. For diffusion calibration we use
    ``audiocap_id`` as the sample id and ``caption`` as the text prompt.
    """

    def __init__(self, dataset_path: str, nsamples: int = 128) -> None:
        super().__init__()
        self.captions = []
        self.caption_ids = []

        logger.info(f"use dataset {dataset_path}, loading from disk...")
        df = pd.read_csv(dataset_path)

        id_col = "audiocap_id" if "audiocap_id" in df.columns else "id"
        if "caption" not in df.columns:
            raise ValueError("AudioCaps dataset must contain a 'caption' column")

        for index, row in df.iterrows():
            if nsamples > 0 and index + 1 > nsamples:
                break
            caption_id = row.get(id_col, index)
            caption_text = str(row.get("caption", "")).strip()
            if not caption_text:
                continue
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
    if dataset in COCO_URL:
        import requests

        logger.info(f"use dataset {dataset}, downloading ...")
        response = requests.get(COCO_URL[dataset], timeout=30)
        response.raise_for_status()
        dataframe = pd.read_csv(StringIO(response.text), sep="\t")
        dataset = DIFFUSION_DATASET["local"](dataset, nsamples, dataframe=dataframe)

    if dataset in ("audiocaps",):
        dataset = download_audiocaps_csv()

    if isinstance(dataset, Dataset):
        pass
    elif isinstance(dataset, str) and os.path.exists(dataset):
        if dataset.endswith(".csv"):
            dataset = DIFFUSION_DATASET["audiocaps"](dataset, nsamples)
        else:
            dataset = DIFFUSION_DATASET["local"](dataset, nsamples)
    else:
        raise ValueError("Only support coco2014/audiocaps dataset or loading local tsv/csv file now.")
    set_seed(seed)
    dataloader_params = {"batch_size": bs, "shuffle": True}

    return DataLoader(dataset, **dataloader_params), bs, gradient_accumulate_steps
