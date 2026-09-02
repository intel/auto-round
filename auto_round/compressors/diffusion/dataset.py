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

import json
import os
import zipfile
from io import StringIO
from pathlib import Path
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

COCO_ANNOTATIONS_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO_CAPTIONS_MEMBER = "annotations/captions_val2014.json"
COCO_ALLOWED_LICENSE_ID = 4  # Creative Commons Attribution 2.0


def _get_coco_cache_dir() -> Path:
    """Return the persistent cache directory for COCO calibration data."""
    from auto_round import envs as _envs

    cache_root = (
        Path(_envs.AUTO_ROUND_CACHE).expanduser() if _envs.AUTO_ROUND_CACHE else Path.home() / ".cache" / "auto_round"
    )
    return cache_root / "datasets" / "coco2014"


def _download_to_cache(url: str, destination: Path, timeout: int) -> None:
    """Download a file atomically unless a non-empty cached copy exists."""
    if destination.is_file() and destination.stat().st_size > 0:
        return

    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.part")
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        content = response.content
        if not content:
            raise RuntimeError(f"Downloaded an empty file from {url}")
        temporary.write_bytes(content)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_cc_by_coco_manifest(dataframe: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """Join official COCO metadata and retain only CC BY 2.0 (license ID 4) images."""
    filtered_manifest = cache_dir / "captions_source_cc_by.tsv"
    if filtered_manifest.is_file() and filtered_manifest.stat().st_size > 0:
        filtered = pd.read_csv(filtered_manifest, sep="\t")
        if "license_id" in filtered and filtered["license_id"].eq(COCO_ALLOWED_LICENSE_ID).all():
            return filtered

    annotation_archive = cache_dir / "annotations_trainval2014.zip"
    _download_to_cache(COCO_ANNOTATIONS_URL, annotation_archive, timeout=300)
    try:
        with zipfile.ZipFile(annotation_archive) as archive, archive.open(COCO_CAPTIONS_MEMBER) as metadata_file:
            metadata = json.load(metadata_file)
    finally:
        # The archive is about 242 MB; the compact filtered manifest is all subsequent runs need.
        annotation_archive.unlink(missing_ok=True)

    licenses = {item["id"]: item for item in metadata["licenses"]}
    image_metadata = pd.DataFrame(
        {
            "image_id": image["id"],
            "license_id": image["license"],
            "flickr_url": image.get("flickr_url", ""),
        }
        for image in metadata["images"]
    )
    image_metadata["license_url"] = image_metadata["license_id"].map(
        lambda license_id: licenses.get(license_id, {}).get("url", "")
    )
    image_metadata["license_name"] = image_metadata["license_id"].map(
        lambda license_id: licenses.get(license_id, {}).get("name", "")
    )
    filtered = dataframe.merge(image_metadata, on="image_id", how="inner", validate="many_to_one")
    filtered = filtered.loc[filtered["license_id"] == COCO_ALLOWED_LICENSE_ID].copy()
    if filtered.empty:
        raise ValueError("The COCO manifest contains no images with CC BY 2.0 license ID 4.")

    filtered_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = filtered_manifest.with_name(f"{filtered_manifest.name}.{os.getpid()}.part")
    try:
        filtered.to_csv(temporary, sep="\t", index=False)
        temporary.replace(filtered_manifest)
    finally:
        temporary.unlink(missing_ok=True)
    return filtered


def _load_coco_dataframe(dataset: str, nsamples: int, image_required: bool) -> pd.DataFrame:
    """Load COCO metadata, caching the manifest and paired images only for I2V."""
    cache_dir = _get_coco_cache_dir()
    if not image_required:
        import requests

        response = requests.get(COCO_URL[dataset], timeout=30)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), sep="\t")

    manifest_path = cache_dir / "captions_source.tsv"
    _download_to_cache(COCO_URL[dataset], manifest_path, timeout=30)
    dataframe = pd.read_csv(manifest_path, sep="\t")

    required_cols = {"image_id", "coco_url", "file_name"}
    if not required_cols.issubset(dataframe.columns):
        raise ValueError(f"COCO I2V calibration requires columns {sorted(required_cols)}.")

    dataframe = _load_cc_by_coco_manifest(dataframe, cache_dir)
    selected = dataframe.iloc[:nsamples].copy() if nsamples > 0 else dataframe.copy()
    image_paths = []
    logger.info(f"Caching {len(selected)} CC BY 2.0 COCO images for I2V calibration in {cache_dir / 'images'}")
    for _, row in selected.iterrows():
        image_path = cache_dir / "images" / Path(str(row["file_name"])).name
        image_url = str(row["coco_url"]).replace(
            "http://images.cocodataset.org/", "https://s3.amazonaws.com/images.cocodataset.org/", 1
        )
        _download_to_cache(image_url, image_path, timeout=60)
        image_paths.append(str(image_path))
    selected["image"] = image_paths
    return selected


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
        self.image_paths = None

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

        if "image" in df.columns:
            selected_df = df.iloc[:nsamples] if nsamples > 0 else df
            image_values = selected_df["image"]
            if image_values.isna().any() or image_values.astype(str).str.strip().eq("").any():
                raise ValueError("Diffusion datasets with an 'image' column must provide an image for every sample.")
            dataset_dir = Path(dataset_path).expanduser().resolve().parent
            self.image_paths = []

        for index, row in df.iterrows():
            if nsamples > 0 and index + 1 > nsamples:
                break
            caption_id = row["id"]
            caption_text = row["caption"]
            self.caption_ids.append(caption_id)
            self.captions.append(caption_text)
            if self.image_paths is not None:
                image_path = Path(str(row["image"])).expanduser()
                if not image_path.is_absolute():
                    image_path = dataset_dir / image_path
                self.image_paths.append(str(image_path))

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.image_paths is not None:
            return self.caption_ids[i], self.captions[i], self.image_paths[i]
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


def get_diffusion_dataloader(dataset="coco2014", bs=1, seed=42, nsamples=128, image_required=False):
    """Generate a DataLoader for calibration using specified parameters.
    Args:
        Dataset_name (str): The name or path of the dataset.
        bs (int, optional): The batch size. Defaults to 1.
    Returns:
        DataLoader: The DataLoader for the calibrated datasets.
    """
    if dataset in COCO_URL:
        logger.info(f"use dataset {dataset}, loading calibration data...")
        dataframe = _load_coco_dataframe(dataset, nsamples, image_required)
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

    return DataLoader(dataset, **dataloader_params), bs
